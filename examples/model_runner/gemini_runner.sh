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

if [[ -z "$GEMINI_API_KEY" ]] && [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is not set. Check .env file or set it in your environment."
    exit 1
fi

MODEL="gemini-2.5-flash-lite"
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
WORKERS=1
CHECKPOINT=50
CTX_LEN=16000

for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting Gemini model=$MODEL version=$VERSION file_type=$FILE_TYPE"
    echo "------------------------------------------"

    "$PYTHON_BIN" "$REPO_ROOT/src/model_runner/main.py" \
        --provider gemini \
        --file-type "$FILE_TYPE" \
        --model "$MODEL" \
        --version "$VERSION" \
        --prompt-dir "$REPO_ROOT/data/prompts" \
        --output-dir "$REPO_ROOT/results" \
        --workers $WORKERS \
        --model-temperature $TEMP \
        --checkpoint-size $CHECKPOINT \
        --context-length $CTX_LEN \
        --include-chained-prompts

    echo "Completed Gemini model=$MODEL version=$VERSION file_type=$FILE_TYPE"
done

echo "All Gemini runs completed successfully for file_type=$FILE_TYPE."
