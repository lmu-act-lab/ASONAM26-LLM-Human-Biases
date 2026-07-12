import asyncio
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import sanitize_model_name
from models import PoliticalBiasAssessment, PoliticalBiasAssessmentNoExplanation, PoliticalBiasAssessmentWithCoT

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from prompt_generation.constants import Constants
from system_prompts import (
    SYSTEM_PROMPT_NO_EXPLANATION,
    SYSTEM_PROMPT_WITH_EXPLANATION,
    SYSTEM_PROMPT_WITH_COT,
    SYSTEM_PROMPT_REASONING_ONLY,
    SYSTEM_PROMPT_CHAINED_JUDGE,
)


class BaseBiasRunner(ABC):
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        version: str,
        workers: int,
        checkpoint_size: int,
        temperature: float,
        context_length: int = 2048,
        include_explanation: bool = False,
        include_prompted_cot: bool = False,
        include_native_cot: bool = False,
        include_chained_prompts: bool = False,
    ):
        self.model_name = model_name
        self.output_dir = os.path.join(
            output_dir,
            version.split(".")[0],
            version,
            sanitize_model_name(model_name),
            Constants.DEFAULT_LLM_OUTPUT_FOLDER,
        )
        self.workers = workers
        self.checkpoint_size = checkpoint_size
        self.temperature = temperature
        self.version = version
        self.include_explanation = include_explanation
        self.include_prompted_cot = include_prompted_cot
        self.include_native_cot = include_native_cot
        self.include_chained_prompts = include_chained_prompts
        self.context_length = context_length
        
        self.system_prompt, self.assessment_model = self._get_system_prompt_and_assessment_model()
        print("system prompt:", self.system_prompt)

    def _load_data(self, input_file: str) -> pd.DataFrame | None:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at '{input_file}'.")
            return None

        data = pd.read_csv(input_file)
        if data.empty:
            print(f"Error: Input file '{input_file}' is empty.")
            return None

        missing_required = [
            col for col in ("article_id", "index", "prompt") if col not in data.columns
        ]
        if missing_required:
            print(f"Error: Input CSV missing required columns: {missing_required}")
            return None

        return data
    
    def _get_system_prompt_and_assessment_model(self) -> tuple[str, type]:
        if self.include_chained_prompts:
            return [SYSTEM_PROMPT_REASONING_ONLY, SYSTEM_PROMPT_CHAINED_JUDGE],  PoliticalBiasAssessment
        if self.include_prompted_cot or self.include_native_cot:
            return SYSTEM_PROMPT_WITH_COT, PoliticalBiasAssessmentWithCoT
        if self.include_explanation:
            return SYSTEM_PROMPT_WITH_EXPLANATION, PoliticalBiasAssessment
        return SYSTEM_PROMPT_NO_EXPLANATION, PoliticalBiasAssessmentNoExplanation

    def _skip_errored_result(self, final_row: dict, llm_error) -> None:
        print(f"\n[Warning] Skipping row {final_row.get('index', 'N/A')} due to LLM error: {llm_error}")
        print("Row data:", final_row)
        print("LLM_data:", None)

    def _add_llm_data_to_row(self, final_row: dict, llm_data, result: dict) -> None:
        final_row.update(
            {
                "llm_assessment": llm_data.assessment,
                "llm_confidence": llm_data.confidence_score,
                "llm_model": result["llm_model"],
                "llm_error": None,
            }
        )
        if self.include_explanation or self.include_prompted_cot or self.include_native_cot or self.include_chained_prompts:
            final_row["llm_explanation"] = getattr(llm_data, "explanation", None)
        if self.include_prompted_cot or self.include_native_cot:
            final_row["llm_thought_process"] = getattr(llm_data, "thought_process", None)
        if self.include_chained_prompts:
            final_row["llm_thought_process"] = result.get("llm_thought_process", None)
        if self.include_native_cot:
            final_row["llm_native_cot"] = result.get("llm_native_cot", None)

    def _process_result(self, result: dict) -> tuple[dict, bool]:
        llm_error = result.get("llm_error")
        final_row = dict(result["row_data"])
        llm_data = result.get("llm_data")
        
        if llm_error or llm_data is None:
            self._skip_errored_result(final_row, llm_error)
            return final_row, True
        
        self._add_llm_data_to_row(final_row, llm_data, result)
        return final_row, False

    def _save_results_buffer(self, results_buffer, all_columns, output_path: str) -> None:
        print(f"Processing results buffer with {len(results_buffer)} entries...")
        final_rows = []
        skipped_error_count = 0
        
        for result in results_buffer:
            final_row, should_skip = self._process_result(result)
            if should_skip:
                skipped_error_count += 1
                continue
            final_rows.append(final_row)

        if skipped_error_count:
            print(f"[Skip] Dropped {skipped_error_count} errored results (not saved).")

        if not final_rows:
            return

        pd.DataFrame(final_rows, columns=all_columns).to_csv(
            output_path,
            index=False,
            mode="a",
            header=False,
        )

    def _setup_output_file(self, input_file_path: str) -> str:
        input_filename = os.path.basename(input_file_path).replace(".csv", "")
        output_filename = f"{input_filename}.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        os.makedirs(self.output_dir, exist_ok=True)
        return output_path

    def _initialize_output_file(self, output_path: str, columns: list[str]) -> set[tuple]:
        processed_keys: set[tuple] = set()

        if not os.path.exists(output_path):
            pd.DataFrame(columns=columns).to_csv(output_path, index=False, mode="w")
            print(f"Initialized output file: {output_path}")
            return processed_keys

        try:
            existing_df = pd.read_csv(output_path)
            missing_columns = [column for column in columns if column not in existing_df.columns]
            if missing_columns:
                for column in missing_columns:
                    existing_df[column] = None
                existing_df = existing_df[columns]
                existing_df.to_csv(output_path, index=False, mode="w")

            processed_df = existing_df[["article_id", "index"]]
            processed_keys = set(zip(processed_df["article_id"], processed_df["index"]))
            print(
                f"Appending to existing file: {output_path} "
                f"({len(processed_keys)} rows already processed)"
            )
        except Exception as exc:
            print(
                f"[Warning] Could not load existing output ({exc}). "
                f"Starting from scratch for {output_path}"
            )
            processed_keys = set()

        return processed_keys

    @abstractmethod
    async def _fetch_llm_response(self, prompt: str, row_data: dict, index: int) -> dict:
        raise NotImplementedError

    async def _process_one_experiment(
        self,
        df_to_process: pd.DataFrame,
        all_columns: list[str],
        output_path: str,
        initial_count: int = 0,
    ) -> None:
        semaphore = asyncio.Semaphore(self.workers)
        tasks = []

        for index, row in df_to_process.iterrows():
            prompt = row["prompt"]
            row_data = row.drop("prompt").to_dict()

            async def limited_fetch(idx=index, p=prompt, data=row_data):
                async with semaphore:
                    return await self._fetch_llm_response(p, data, idx)

            tasks.append(limited_fetch())

        print(
            f"Starting async inference with {self.workers} workers and "
            f"checkpoint size: {self.checkpoint_size}..."
        )

        results_buffer = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(df_to_process),
            initial=initial_count,
            desc="Async Inference",
        ):
            result_data = await future
            results_buffer.append(result_data)

            if len(results_buffer) >= self.checkpoint_size:
                print(f"\n[Checkpoint] Saving {len(results_buffer)} results to disk...")
                self._save_results_buffer(results_buffer, all_columns, output_path)
                results_buffer = []

        if results_buffer:
            print(f"\n[Final Save] Saving remaining {len(results_buffer)} results to disk...")
            self._save_results_buffer(results_buffer, all_columns, output_path)
            print("Final save complete.")

    def _process_single_file(self, input_file_path: str) -> None:
        print(f"\n--- Processing File: {os.path.basename(input_file_path)} ---")

        df = self._load_data(input_file_path)
        if df is None:
            print(f"Skipping {os.path.basename(input_file_path)}.")
            return

        output_path = self._setup_output_file(input_file_path)

        original_columns = [column for column in df.columns.tolist() if column != "prompt"]
        all_columns = original_columns + self.get_llm_result_columns()

        processed_keys = self._initialize_output_file(output_path, all_columns)
        mask = df.apply(
            lambda row: (row["article_id"], row["index"]) not in processed_keys,
            axis=1,
        )
        df_to_process = df[mask]

        print(f"Total prompts in file: {len(df)}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Remaining to process: {len(df_to_process)}")

        if len(df_to_process) == 0:
            print("All prompts already processed. Skipping file.")
            return

        asyncio.run(
            self._process_one_experiment(
                df_to_process=df_to_process,
                all_columns=all_columns,
                output_path=output_path,
                initial_count=len(processed_keys),
            )
        )

        print(
            f"--- Finished processing {os.path.basename(input_file_path)}. "
            f"Results written to: {output_path} ---"
        )
    
    def get_llm_result_columns(self) -> list[str]:
        # Start with the base 4 columns
        cols = ["llm_assessment", "llm_confidence", "llm_model", "llm_error"]
        
        if self.include_prompted_cot or self.include_native_cot or self.include_chained_prompts:
            cols.append("llm_thought_process")
        
        if self.include_explanation or self.include_prompted_cot or self.include_native_cot or self.include_chained_prompts:
            if "llm_explanation" not in cols:
                cols.append("llm_explanation")
                
        if self.include_native_cot:
            cols.append("llm_native_cot")
                
        return cols

    def run_experiment(self, file_type: str, prompt_dir: str) -> None:
        if file_type == "all":
            files_to_run = list(Constants.PROMPT_FILE_MAP.values())
        else:
            files_to_run = [Constants.PROMPT_FILE_MAP[file_type]]

        print(f"Targeting prompt files in: {prompt_dir}")
        print(
            f"Using model: {self.model_name} "
            f"(Temperature: {self.temperature}) "
            f"with {self.workers} workers "
            f"and context length {self.context_length}."
        )
        print(f"Output directory: {self.output_dir}")
        print(f"Files to process: {files_to_run}")

        prompt_version = self.version.split(".")[0]
        for file_name in files_to_run:
            input_file_path = os.path.join(prompt_dir, prompt_version, file_name)
            if not os.path.exists(input_file_path):
                print(f"\n[Skipping] Input file not found: {input_file_path}")
                continue
            self._process_single_file(input_file_path)
