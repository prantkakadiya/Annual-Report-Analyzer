#!/usr/bin/env bash

LLAMA_SERVER="${LLAMA_SERVER_PATH:-}"
if [ -z "$LLAMA_SERVER" ]; then
    if command -v llama-server &> /dev/null; then
        LLAMA_SERVER="llama-server"
    elif [ -x "$HOME/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
    else
        echo "Error: llama-server not found."
        exit 1
    fi
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"

MODEL="${LLM_MODEL_PATH:-$PROJECT_ROOT/llm/Qwen3.5-9B-Q4_K_M.gguf}"
MMPROJ="${MMPROJ_PATH:-$PROJECT_ROOT/llm/mmproj-F16.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi
if [ ! -f "$MMPROJ" ]; then
    echo "Error: mmproj file not found at $MMPROJ"
    exit 1
fi

echo "Starting Vision LLM server with model: $MODEL and mmproj: $MMPROJ"
exec "$LLAMA_SERVER" -m "$MODEL" --mmproj "$MMPROJ" -ngl 20 -fa on -ctk q8_0 -ctv q8_0 --no-cache-prompt --port 8080
