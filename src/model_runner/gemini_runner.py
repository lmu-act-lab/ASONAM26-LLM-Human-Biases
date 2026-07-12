import asyncio
import os
import re
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from base import BaseBiasRunner

load_dotenv()
class GeminiBiasRunner(BaseBiasRunner):
    def __init__(self, gemini_api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        self._client = genai.Client(api_key=api_key)

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(5))
    def _sync_generate(
        self,
        prompt: str,
        system_prompt: str,
        schema: dict | None = None,
        max_output_tokens: int = 8192,
    ):
        config_kwargs = {
            "system_instruction": system_prompt,
            "temperature": self.temperature,
            "max_output_tokens": max_output_tokens,
            "safety_settings": [
                {"category": cat, "threshold": "BLOCK_NONE"}
                for cat in [
                    "HARM_CATEGORY_CIVIC_INTEGRITY",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
            ],
        }

        # Only force JSON schema when schema is provided (step 2 / non-chained mode).
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = schema

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        finish_reason = str(getattr(candidate, "finish_reason", "")).upper()
        if "MAX_TOKENS" in finish_reason:
            raise RuntimeError("Gemini output truncated by MAX_TOKENS")

        return response

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        llm_thought_process = None

        try:
            if self.include_chained_prompts:
                sys_reasoning = self.system_prompt[0] if isinstance(self.system_prompt, list) else self.system_prompt
                sys_judge = self.system_prompt[1] if isinstance(self.system_prompt, list) else self.system_prompt

                step1_response = await asyncio.to_thread(
                    self._sync_generate,
                    prompt,
                    sys_reasoning,
                    None,
                    8192,
                )

                llm_thought_process = (step1_response.text or "").strip()

                json_pattern = r"``json\s*(\{.*?\})\s*``"
                if re.search(json_pattern, llm_thought_process, re.DOTALL):
                    llm_thought_process = re.sub(
                        json_pattern, "", llm_thought_process, flags=re.DOTALL
                    ).strip()

                step2_prompt = (
                    f"Original article prompt:\n{prompt}\n\n"
                    f"Prior reasoning:\n{llm_thought_process}\n\n"
                    "Using the reasoning above, return only valid JSON following schema. "
                    "assessment must be exactly 'is-biased' or 'is-not-biased'."
                )

                step2_response = await asyncio.to_thread(
                    self._sync_generate,
                    step2_prompt,
                    sys_judge,
                    self.assessment_model.model_json_schema(),
                    8192,
                )
                step2_text = (step2_response.text or "").strip()
                llm_data = self.assessment_model.model_validate_json(step2_text)

            else:
                response = await asyncio.to_thread(
                    self._sync_generate,
                    prompt,
                    self.system_prompt,
                    self.assessment_model.model_json_schema(),
                    8192,
                )
                response_text = (response.text or "").strip()
                llm_data = self.assessment_model.model_validate_json(response_text)

        except ValidationError as exc:
            llm_error = str(exc)
            print(f"\n[Warning] Gemini validation failed for row {index}: {exc}", file=sys.stderr)
            print(f"Response text that caused validation error for row {index}:\n{response_text}\n", file=sys.stderr)
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] Gemini row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
            "llm_thought_process": llm_thought_process,
        }