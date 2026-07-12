import json
import os
import sys
import tempfile
import time

from openai import BadRequestError, OpenAI

from base import BaseBiasRunner
from utils import extract_text_content


class OpenAIBatchBiasRunner(BaseBiasRunner):
    MAX_REQUESTS_PER_BATCH = 500

    def __init__(
        self,
        batch_poll_interval: int = 30,
        openai_api_key: str | None = None,
        reasoning_effort: str = "medium", 
        **kwargs,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIBatchBiasRunner")
        super().__init__(workers=1, **kwargs)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI batch mode")

        self._client = OpenAI(api_key=api_key)
        self.batch_poll_interval = batch_poll_interval
        self.reasoning_effort = reasoning_effort
        self.url = "/v1/chat/completions"
        
        if self.include_chained_prompts:
            raise ValueError("Chained prompts are not supported in OpenAI batch mode. Use --openai-mode direct.")

    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        raise NotImplementedError("OpenAI batch runner does not use per-row async requests")

    def _build_batch_jsonl(self, df_to_process, jsonl_path: str) -> dict[str, dict]:
        custom_id_to_row: dict[str, dict] = {}

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for sequence_number, (_, row) in enumerate(df_to_process.iterrows()):
                row_data = row.drop("prompt").to_dict()
                custom_id = f"{row_data['article_id']}::{row_data['index']}::{sequence_number}"
                custom_id_to_row[custom_id] = row_data
                body = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "political_bias_assessment",
                            "schema": self.assessment_model.model_json_schema(),
                        },
                    },
                }
                if "gpt-5.4" not in self.model_name.lower():
                    body["temperature"] = self.temperature
                request_line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": self.url,
                    "body": body,
                }
                handle.write(json.dumps(request_line, ensure_ascii=False) + "\n")

        return custom_id_to_row

    def _poll_batch(self, batch_id: str):
        running_states = {"validating", "in_progress", "finalizing"}
        while True:
            batch = self._client.batches.retrieve(batch_id)
            print(f"Batch {batch.id} status: {batch.status}")
            if batch.status in running_states:
                time.sleep(self.batch_poll_interval)
                continue
            return batch

    def _download_file_text(self, file_id: str) -> str:
        content = self._client.files.content(file_id)
        if hasattr(content, "text"):
            return content.text
        if hasattr(content, "read"):
            return content.read().decode("utf-8")
        return str(content)

    def _parse_batch_output_line(
        self,
        line: str,
        custom_id_to_row: dict[str, dict],
    ) -> dict | None:
        if not line.strip():
            return None

        parsed_line = json.loads(line)
        custom_id = parsed_line.get("custom_id")
        row_data = custom_id_to_row.get(custom_id)
        if row_data is None:
            return None

        llm_data = None
        llm_error = None
        native_cot = None
        try:
            response = parsed_line.get("response", {})
            body = response.get("body", {})
            
            reasoning_obj = body.get("reasoning")
            if reasoning_obj and "summary" in reasoning_obj:
                summaries = reasoning_obj.get("summary", [])
                if summaries:
                    native_cot = summaries[0].get("text")


            choices = body.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content = extract_text_content(message.get("content", ""))
            llm_data = self.assessment_model.model_validate_json(content)
        except Exception as exc:
            llm_error = str(exc)
            print(f"[Warning] Failed to parse batch item {custom_id}: {exc}", file=sys.stderr)

        return {
            "index": row_data.get("index"),
            "row_data": row_data,
            "llm_data": llm_data,
            "llm_model": self.model_name,
            "llm_error": llm_error,
            "llm_native_cot": native_cot,
        }

    def _get_columns_and_process_mask(self, df, output_path):
        """Extract columns and identify rows to process."""
        original_columns = [column for column in df.columns.tolist() if column != "prompt"]
        all_columns = original_columns + self.get_llm_result_columns()
        processed_keys = self._initialize_output_file(output_path, all_columns)
        mask = df.apply(
            lambda row: (row["article_id"], row["index"]) not in processed_keys,
            axis=1,
        )
        return all_columns, df[mask], processed_keys

    def _log_processing_status(self, df, df_to_process, processed_keys):
        """Log processing status information."""
        print(f"Total prompts in file: {len(df)}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining to process: {len(df_to_process)}")

    def _create_batch_with_error_handling(self, input_file, jsonl_path, chunk_index, input_file_path, chunk_df):
        """Create batch with proper error handling."""
        try:
            batch = self._client.batches.create(
                input_file_id=input_file.id,
                endpoint=self.url,
                completion_window="24h",
                metadata={
                    "model": self.model_name,
                    "version": self.version,
                    "source_file": os.path.basename(input_file_path),
                    "chunk_index": str(chunk_index + 1),
                    "chunk_size": str(len(chunk_df)),
                },
            )
            return batch
        except BadRequestError as exc:
            self._handle_bad_request_error(exc)
            return None
        except Exception as exc:
            print(f"[Error] Unexpected error creating OpenAI batch: {exc}", file=sys.stderr)
            return None

    def _handle_bad_request_error(self, exc):
        """Handle BadRequestError from batch creation."""
        error_text = str(exc)
        if "billing_hard_limit_reached" in error_text:
            print(
                "[Error] OpenAI billing hard limit reached for this API key/account. "
                "Increase billing limit, add funds, or use a different key/account.",
                file=sys.stderr,
            )
        elif "Enqueued token limit reached" in error_text:
            print(
                "[Error] OpenAI Batch enqueued token limit reached for the organization. "
                "Wait for in-progress batches to complete, cancel stale in-progress batches, "
                "or use another organization/key with available batch capacity.",
                file=sys.stderr,
            )
        else:
            print(f"[Error] OpenAI batch creation failed: {exc}", file=sys.stderr)

    def _process_batch_results(self, batch, custom_id_to_row, all_columns, output_path):
        """Process batch results and save to output."""
        output_file_id = getattr(batch, "output_file_id", None)
        
        if output_file_id:
            self._process_output_results(batch.output_file_id, custom_id_to_row, all_columns, output_path)
        else:
            self._handle_missing_output_file(batch)

    def _process_output_results(self, output_file_id, custom_id_to_row, all_columns, output_path):
        """Process and save results from output file."""
        output_text = self._download_file_text(output_file_id)
        results_buffer = []
        
        for line in output_text.splitlines():
            parsed_result = self._parse_batch_output_line(line, custom_id_to_row)
            if parsed_result is None:
                continue
            
            results_buffer.append(parsed_result)
            if len(results_buffer) >= self.checkpoint_size:
                self._save_results_buffer(results_buffer, all_columns, output_path)
                results_buffer = []
        
        if results_buffer:
            self._save_results_buffer(results_buffer, all_columns, output_path)

    def _handle_missing_output_file(self, batch):
        """Handle case where batch completed but output_file_id is missing."""
        print("[Warning] Batch status is 'completed' but output_file_id is None.")
        print("This usually means all requests in the batch failed.")
        
        error_file_id = getattr(batch, 'error_file_id', None)
        if error_file_id:
            self._print_batch_errors(error_file_id)

    def _print_batch_errors(self, error_file_id):
        """Print errors from batch error file."""
        print(f"Downloading error details from: {error_file_id}")
        error_content = self._download_file_text(error_file_id)
        
        for line in error_content.splitlines():
            error_entry = json.loads(line)
            err_msg = error_entry.get("response", {}).get("body", {}).get("error", {}).get("message")
            print(f" - Request {error_entry.get('custom_id')}: {err_msg}")

    def _process_chunks(self, df_to_process, input_file_path, all_columns, output_path):
        """Process all chunks of data."""
        total_remaining = len(df_to_process)
        chunk_size = self.MAX_REQUESTS_PER_BATCH
        total_chunks = (total_remaining + chunk_size - 1) // chunk_size

        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, total_remaining)
            chunk_df = df_to_process.iloc[start:end]
            
            if not self._process_single_chunk(chunk_index, total_chunks, chunk_df, input_file_path, all_columns, output_path):
                return False
        
        return True

    def _process_single_chunk(self, chunk_index, total_chunks, chunk_df, input_file_path, all_columns, output_path):
        """Process a single chunk and return success status."""
        print(
            f"Submitting OpenAI batch chunk {chunk_index + 1}/{total_chunks} "
            f"with {len(chunk_df)} prompts"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_jsonl:
            jsonl_path = temp_jsonl.name

        custom_id_to_row = self._build_batch_jsonl(chunk_df, jsonl_path)

        with open(jsonl_path, "rb") as handle:
            input_file = self._client.files.create(file=handle, purpose="batch")

        batch = self._create_batch_with_error_handling(input_file, jsonl_path, chunk_index, input_file_path, chunk_df)
        if batch is None:
            return False

        print(f"Created batch job: {batch.id}")
        batch = self._poll_batch(batch.id)

        if batch.status == "completed":
            self._process_batch_results(batch, custom_id_to_row, all_columns, output_path)

        return True

    def _process_single_file(self, input_file_path: str) -> None:
        print(f"\n--- Processing File (OpenAI Batch): {os.path.basename(input_file_path)} ---")

        df = self._load_data(input_file_path)
        if df is None:
            print(f"Skipping {os.path.basename(input_file_path)}.")
            return

        output_path = self._setup_output_file(input_file_path)
        all_columns, df_to_process, processed_keys = self._get_columns_and_process_mask(df, output_path)

        self._log_processing_status(df, df_to_process, processed_keys)

        if len(df_to_process) == 0:
            print("All prompts already processed. Skipping file.")
            return

        self._process_chunks(df_to_process, input_file_path, all_columns, output_path)

        print(
            f"--- Finished processing {os.path.basename(input_file_path)}. "
            f"Results written to: {output_path} ---"
        )
