import argparse

from gemini_runner import GeminiBiasRunner
from ollama_runner import OllamaBiasRunner
from openai_batch_runner import OpenAIBatchBiasRunner
from openai_direct_runner import OpenAIDirectBiasRunner

from prompt_generation.constants import Constants


def infer_provider(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized.startswith("gpt-") or normalized.startswith("openai/"):
        return "openai"
    if normalized.startswith("gemini-") or normalized.startswith("google/"):
        return "gemini"
    return "ollama"


def build_runner(args):
    provider = args.provider
    if provider == "auto":
        provider = infer_provider(args.model)

    common_kwargs = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "version": args.version,
        "workers": args.workers,
        "checkpoint_size": args.checkpoint_size,
        "temperature": args.model_temperature,
        "context_length": args.context_length,
        "include_explanation": args.include_explanation,
        "include_prompted_cot": args.include_prompted_cot,
        "include_native_cot": args.include_native_cot,
        "include_chained_prompts": args.include_chained_prompts,
    }

    if provider == "openai":
        openai_kwargs = {key: value for key, value in common_kwargs.items() if key != "workers"}
        # Auto-detect mode: native CoT requires direct Responses API; otherwise use Batch API.
        openai_direct_mode = args.openai_mode == "direct" or (args.include_native_cot)
        openai_batch_mode = args.openai_mode == "batch" or (not args.include_native_cot)
        openai_mode = "direct" if openai_direct_mode else "batch"
        if args.openai_mode == "auto":
            openai_mode = "direct" if args.include_native_cot else "batch"
        if openai_direct_mode and openai_batch_mode:
            print(
                "Warning: Both direct and batch modes are enabled for OpenAI. Defaulting to direct mode due to native CoT usage. To use batch mode without native CoT, disable native CoT or explicitly set --openai-mode to 'batch'."
            )
            
        if openai_mode == "direct":
            return OpenAIDirectBiasRunner(
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
                **common_kwargs,
            )
        else:
            return OpenAIBatchBiasRunner(
                batch_poll_interval=args.batch_poll_interval,
                reasoning_effort=args.reasoning_effort,
                **openai_kwargs,
            )

    if provider == "gemini":
        return GeminiBiasRunner(**common_kwargs)

    return OllamaBiasRunner(ollama_port=args.ollama_server_port, **common_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch LLM inference over generated prompts across Ollama/OpenAI/Gemini."
    )

    file_type_choices = ["all"] + list(Constants.PROMPT_FILE_MAP.keys())

    parser.add_argument("--file-type", type=str, default="all", choices=file_type_choices)
    parser.add_argument("--model", type=str, default=Constants.MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=Constants.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", type=str, default=Constants.VERSION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-size", type=int, default=100)
    parser.add_argument("--model-temperature", type=float, default=0.5)
    parser.add_argument("--prompt-dir", type=str, default=Constants.DEFAULT_PROMPT_DIR)
    parser.add_argument("--ollama-server-port", type=int, default=11434)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument(
        "--include-explanation",
        action="store_true",
        help="Include llm_explanation in model output and output CSV.",
    )
    parser.add_argument(
        "--include-prompted-cot",
        action="store_true",
        help="Enable Chain of Thought reasoning in a dedicated JSON field..",
    )
    parser.add_argument(
        "--include-chained-prompts",
        action="store_true",
        help="Enable additional chained prompts for follow-up questioning based on initial response. This is separate from prompted CoT and is focused on iterative refinement rather than single-pass CoT reasoning.",
    )

    parser.add_argument(
        "--include-native-cot",
        action="store_true",
        help="Include native CoT from the model if available (e.g. OpenAI's 'reasoning_content'). This is separate from prompted CoT and is only included if the model provides it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Only applicable if --include-native-cot is set. Instructs the model on how much effort to put into its internal reasoning. This is a custom parameter and may not be supported by all models.",
    )
    parser.add_argument(
        "--reasoning-summary",
        type=str,
        default="detailed",
        choices=["auto", "detailed"],
        help="Only applicable if --include-native-cot is set. Controls reasoning summary detail for direct OpenAI responses.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "ollama", "openai", "gemini"],
        help="Provider override. Default uses model name inference.",
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=30,
        help="Seconds between OpenAI batch status polls.",
    )
    parser.add_argument(
        "--openai-mode",
        type=str,        default="auto",
        choices=["auto", "direct", "batch"],
        help="Only applicable for OpenAI provider. Auto-selects direct vs batch mode based on native CoT usage, but can be manually overridden if desired (e.g. to use direct mode without native CoT).",
    )

    args = parser.parse_args()

    runner = build_runner(args)
    runner.run_experiment(file_type=args.file_type, prompt_dir=args.prompt_dir)


if __name__ == "__main__":
    main()
