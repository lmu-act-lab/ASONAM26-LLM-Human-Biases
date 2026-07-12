#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
PYTHON_BIN="$REPO_ROOT/venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded environment from $ENV_FILE"
else
    echo "No .env found at $ENV_FILE (continuing with current shell env vars)"
fi

if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Check .env file or set it in your environment."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

MODEL="gpt-5.4-mini"
FILE_TYPE="${1:-all}"
DEFAULT_VERSIONS=("direct")

if [[ $# -ge 2 ]]; then
    VERSIONS=("${@:2}")
    if [[ ${#VERSIONS[@]} -eq 1 && "${VERSIONS[0]}" == *,* ]]; then
        IFS=',' read -r -a VERSIONS <<< "${VERSIONS[0]}"
    fi
else
    VERSIONS=("${DEFAULT_VERSIONS[@]}")
fi
TEMP=0.0
CHECKPOINT=100
POLL_INTERVAL=60

for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting OpenAI Batch model=$MODEL version=$VERSION file_type=$FILE_TYPE"
    echo "------------------------------------------"

    "$PYTHON_BIN" "$REPO_ROOT/src/model_runner/main.py" \
        --provider openai \
        --openai-mode direct \
        --file-type "$FILE_TYPE" \
        --model "$MODEL" \
        --version "$VERSION" \
        --prompt-dir "$REPO_ROOT/data/prompts" \
        --output-dir "$REPO_ROOT/results" \
        --model-temperature $TEMP \
        --checkpoint-size $CHECKPOINT \
        --batch-poll-interval $POLL_INTERVAL \
        --include-chained-prompts

    echo "Completed OpenAI Batch model=$MODEL version=$VERSION file_type=$FILE_TYPE"
done

echo "All OpenAI batch jobs submitted/completed successfully for file_type=$FILE_TYPE."
