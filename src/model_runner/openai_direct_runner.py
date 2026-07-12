import asyncio
import os
import re
import sys

from openai import OpenAI
from pydantic import ValidationError

from base import BaseBiasRunner


class OpenAIDirectBiasRunner(BaseBiasRunner):
    def __init__(
        self,
        openai_api_key: str | None = None,
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
        **kwargs,
    ):
        super().__init__(**kwargs)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI direct mode")

        self._client = OpenAI(api_key=api_key)
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary

    def _extract_native_cot(self, response) -> str | None:
        summaries: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for s in (getattr(item, "summary", None) or []):
                if getattr(s, "type", None) == "summary_text" and getattr(s, "text", None):
                    summaries.append(s.text)
        return "\n\n".join(summaries) if summaries else None

    def _extract_text(self, response) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if text_value:
                    chunks.append(text_value)
        return "".join(chunks)

    def _sync_generate_freeform(self, messages: list[dict]):
        request_kwargs = {
            "model": self.model_name,
            "input": messages,
        }
        if "gpt-5" not in self.model_name.lower():
            request_kwargs["temperature"] = self.temperature
        return self._client.responses.create(**request_kwargs)

    def _sync_parse_structured(self, messages: list[dict], text_format_model, use_reasoning: bool = False):
        request_kwargs = {
            "model": self.model_name,
            "input": messages,
            "text_format": text_format_model,
        }

        if use_reasoning and self.include_native_cot:
            request_kwargs["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            }

        if "gpt-5" not in self.model_name.lower():
            request_kwargs["temperature"] = self.temperature

        return self._client.responses.parse(**request_kwargs)

    async def _process_chained_prompts(self, prompt: str, index: int) -> tuple:
        sys_reasoning = self.system_prompt[0] if isinstance(self.system_prompt, list) else self.system_prompt
        sys_judge = self.system_prompt[1] if isinstance(self.system_prompt, list) else self.system_prompt

        step1_messages = [
            {"role": "system", "content": sys_reasoning},
            {"role": "user", "content": prompt},
        ]
        step1_response = await asyncio.to_thread(self._sync_generate_freeform, step1_messages)
        llm_thought_process = self._extract_text(step1_response).strip()

        fenced_json_pattern = rf"{chr(96)}{{3}}json\s*(\{{.*?\}})\s*{chr(96)}{{3}}"
        if re.search(fenced_json_pattern, llm_thought_process, re.DOTALL):
            llm_thought_process = re.sub(
                fenced_json_pattern,
                "",
                llm_thought_process,
                flags=re.DOTALL,
            ).strip()

        step2_messages = [
            {"role": "system", "content": sys_judge},
            {"role": "user", "content": f"Original article prompt:\n{prompt}"},
            {"role": "assistant", "content": llm_thought_process},
            {
                "role": "user",
                "content": (
                    "Using the reasoning above, return only valid JSON following schema. "
                    "assessment must be exactly 'is-biased' or 'is-not-biased'."
                ),
            },
        ]
        step2_response = await asyncio.to_thread(
            self._sync_parse_structured,
            step2_messages,
            self.assessment_model,
            True,
        )
        llm_data = step2_response.output_parsed
        llm_native_cot = self._extract_native_cot(step2_response) if self.include_native_cot else None
        return llm_data, llm_native_cot, llm_thought_process

    async def _process_single_prompt(self, prompt: str, index: int) -> tuple:
        single_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await asyncio.to_thread(
            self._sync_parse_structured,
            single_messages,
            self.assessment_model,
            True,
        )
        llm_data = response.output_parsed
        llm_native_cot = self._extract_native_cot(response) if self.include_native_cot else None
        return llm_data, llm_native_cot, None

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        llm_data = None
        llm_error = None
        llm_native_cot = None
        llm_thought_process = None

        try:
            if self.include_chained_prompts:
                llm_data, llm_native_cot, llm_thought_process = await self._process_chained_prompts(prompt, index)
            else:
                llm_data, llm_native_cot, llm_thought_process = await self._process_single_prompt(prompt, index)

        except ValidationError as exc:
            llm_error = str(exc)
            print(f"\n[Warning] Pydantic validation failed for row {index}: {exc}", file=sys.stderr)
        except Exception as exc:
            llm_error = str(exc)
            print(f"\n[Error] OpenAI direct inference error for row {index}: {exc}", file=sys.stderr)

        return {
            "index": index,
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
            "llm_native_cot": llm_native_cot,
            "llm_thought_process": llm_thought_process,
        }