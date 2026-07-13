# ASONAM26-LLM-Human-Biases

Do LLMs Share Human Biases? Assessing Model Alignment with Human Bias Perception in Partisan News

We systematically investigate five core research questions:

- Q1: How closely do LLM bias judgments align with human annotations
- Q2: Do models align differently with annotators from different political groups?
- Q3: To what extent do different LLMs agree on political bias judgments?
- Q4: How sensitive are model predictions to contextual metadata such as political affiliation, source information, and demographic context?
- Q5: How does reasoning structure affect model predictions and alignment?

## Reproducible Environment Setup

Install requirements packages from repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set API keys only for providers you use:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Execution Pipeline

[!IMPORTANT]

The generation modules use context-relative paths. Commands in the steps below must be executed from within their respective subdirectories.

### 1. Generate Prompts

#### Generate Direct Evaluation Prompts (Q1 - Q4)

Run from `src/prompt_generation` (required because paths are relative in this module):

```bash
cd src/prompt_generation
python prompt_generator.py --clean --fetch --all-prompts --version direct
```

- Outputs saved to: data/prompts/direct

_Optional: Selective Variant Generation_

If you only wish to construct specific metadata families, pass explicit targeting flags:

```bash
python prompt_generator.py --prompts articles_info politics sources source_politics source_pii politics_pii pii_combined_all --version direct
```

#### Generate Chain-of-Thought Prompts (Q5)

To extract a mathematically stable 10% stratified subset across demographic intersections, find high-conflict edge cases, and isolate them into high-reasoning prompt chains:

```bash
python sample_row_selection.py
python generate_prompts_for_indexes.py
```

- Outputs saved to: data/prompts/cot/

### 2. Run Model Inference

The runner scripts live in [examples/model_runner](examples/model_runner). Each script resolves the repository root internally, so you can launch them either from the repository root or by first changing into that folder. The Python command they call runs [src/model_runner/main.py](src/model_runner/main.py) from the repository root, reads prompts from `data/prompts`, and writes outputs to `results/`.

#### OpenAI

Use the OpenAI script when you want to run `gpt-*` models or other OpenAI-compatible model names:

```bash
cd examples/model_runner
bash openai_batch_runner.sh [file_type] [version1 version2 ...]
```

Examples:

```bash
bash openai_batch_runner.sh
bash openai_batch_runner.sh all direct
bash openai_batch_runner.sh politics direct cot
```

#### Gemini

Use the Gemini script for Gemini models:

```bash
cd examples/model_runner
bash gemini_runner.sh [file_type] [version1 version2 ...]
```

#### Ollama

Use the Ollama script for local Ollama models:

```bash
cd examples/model_runner
bash ollama_runner.sh [file_type] [version1 version2 ...]
```

#### Parameters You Can Change

The first argument is `file_type`. Use `all` to run every prompt family, or pass a specific prompt family name supported by the runner. The remaining arguments are prompt `version` values. If you pass more than one version, the script runs them one after another. For example, you want to run the experiment 2 times, you can have `direct-1` and `direct-2`.

To change the model, temperature, worker count, checkpoint size, context length, or provider-specific settings, edit the variables near the top of the bash script. The most important ones are:

- `MODEL`
- `FILE_TYPE`
- `TEMP`
- `WORKERS`
- `CHECKPOINT`
- `CTX_LEN`
- `PORT` for Ollama
- `POLL_INTERVAL` for OpenAI batch runs

If you want full control without editing the shell script, call [src/model_runner/main.py](src/model_runner/main.py) directly and pass CLI flags such as:

```bash
python src/model_runner/main.py \
	--provider openai \
	--file-type all \
	--model gpt-5.4-mini \
	--version direct \
	--prompt-dir data/prompts \
	--output-dir results \
	--workers 8 \
	--checkpoint-size 100 \
	--model-temperature 0.5
```

Available flags include `--provider`, `--file-type`, `--model`, `--version`, `--output-dir`, `--workers`, `--checkpoint-size`, `--model-temperature`, `--context-length`, `--include-explanation`, `--include-prompted-cot`, `--include-native-cot`, `--include-chained-prompts`, `--reasoning-effort`, `--reasoning-summary`, `--openai-mode`, and `--ollama-server-port`.

### 3. Run Analysis Scripts (Q1 - Q5)

Each analysis file in [src/analysis](src/analysis) is designed to answer one research question. These scripts use relative paths (for example, `../../results` and `../../data`), so run them from inside [src/analysis](src/analysis).

Run from repository root:

```bash
cd src/analysis
```

Question-to-script mapping:

- Q1 (model vs human alignment): [src/analysis/most_aligned_models.py](src/analysis/most_aligned_models.py)
- Q2 (alignment by annotator politics): [src/analysis/models_align_with_politics.py](src/analysis/models_align_with_politics.py)
- Q3 (inter-model agreement): [src/analysis/inter_model.py](src/analysis/inter_model.py)
- Q4 (metadata sensitivity): [src/analysis/metadata_analysis.py](src/analysis/metadata_analysis.py)
- Q5 (reasoning structure effects): [src/analysis/cot_analysis.py](src/analysis/cot_analysis.py)
- Human-human reliability baseline used in Q1 context: [src/analysis/data_analysis.py](src/analysis/data_analysis.py)

Run all question scripts:

```bash
python most_aligned_models.py
python models_align_with_politics.py
python inter_model.py
python metadata_analysis.py
python cot_analysis.py
python data_analysis.py
```

Outputs:

- CSV outputs are written under [analysis_reports](analysis_reports) in question-specific folders, each with a `csv` subfolder.
- Figures are written in the corresponding `figures` subfolders.
- The scripts create output folders automatically if they do not exist.


