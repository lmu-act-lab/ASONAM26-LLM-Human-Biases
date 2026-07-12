import sys
import re
from pydantic import ValidationError

from base import BaseBiasRunner

try:
    from ollama import AsyncClient as OllamaAsyncClient
    from ollama import ResponseError as OllamaResponseError
except ImportError:
    OllamaAsyncClient = None
    OllamaResponseError = Exception

class OllamaBiasRunner(BaseBiasRunner):
    def __init__(self, ollama_port: int = 11434, **kwargs):
        if OllamaAsyncClient is None:
            raise ImportError("ollama package is required for OllamaBiasRunner")
        super().__init__(**kwargs)
        self.ollama_port = ollama_port
        self._client = OllamaAsyncClient(host=f"http://localhost:{self.ollama_port}")

    async def _call_ollama(self, messages: list[dict], response_schema: dict | None = None):
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": self.temperature, "num_ctx": self.context_length},
            "keep_alive": "5m",
        }
        if response_schema is not None:
            kwargs["format"] = response_schema
        return await self._client.chat(**kwargs)

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        llm_thought_process = None

        try:
            if self.include_chained_prompts:
                step1_messages = [
                    {"role": "system", "content": self.system_prompt[0]},
                    {"role": "user", "content": prompt},
                ]
                step1_response = await self._call_ollama(
                    step1_messages,
                )

                llm_thought_process = step1_response["message"]["content"].strip()

                # Parse llm thought process to remove ```json if the model try to give the final answer in a JSON 
                # `like this ```json\n{...}\n```` use regex to extract the JSON part. If match, remove that part and keep the rest as thought process. If no match, keep the whole thing as thought process.
                json_pattern = r"```json\s*(\{.*?\})\s*```"
                match = re.search(json_pattern, llm_thought_process, re.DOTALL)
                if match:
                    llm_thought_process = re.sub(json_pattern, "", llm_thought_process, flags=re.DOTALL).strip()
                    
                step2_messages = [
                    {"role": "system", "content": self.system_prompt[1]},
                    {"role": "user", "content": f"Original article prompt:\n{prompt}"},
                    {"role": "assistant", "content": llm_thought_process},
                    {
                        "role": "user",
                        "content": (
                            "Using the reasoning above, return only valid JSON following the schema. "
                            "assessment must be exactly 'is-biased' or 'is-not-biased'."
                        ),
                    },
                ]
                step2_response = await self._call_ollama(
                    step2_messages,
                    self.assessment_model.model_json_schema(),
                )
                step2_text = step2_response["message"]["content"].strip()
                llm_data = self.assessment_model.model_validate_json(step2_text)
            else:
                single_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = await self._call_ollama(
                    single_messages,
                    self.assessment_model.model_json_schema(),
                )
                response_text = response["message"]["content"].strip()
                llm_data = self.assessment_model.model_validate_json(response_text)

        except ValidationError as exc:
            llm_error = str(exc)
            print(
                f"\n[Warning] Pydantic validation failed for row {index}. Error details: {exc}",
                file=sys.stderr,
            )
        except OllamaResponseError as exc:
            llm_error = str(exc)
            print(f"\n[Error] Ollama Response Error for row {index}: {exc}", file=sys.stderr)
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] General Ollama inference error for row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
            "llm_thought_process": llm_thought_process,
        }