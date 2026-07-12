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
