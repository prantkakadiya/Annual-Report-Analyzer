#!/usr/bin/env bash

# Find llama-server
LLAMA_SERVER="${LLAMA_SERVER_PATH:-}"

if [ -z "$LLAMA_SERVER" ]; then
    if command -v llama-server &> /dev/null; then
        LLAMA_SERVER="llama-server"
    elif [ -x "$HOME/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
    else
        echo "Error: llama-server not found."
        echo "Please set LLAMA_SERVER_PATH or install llama.cpp"
        exit 1
    fi
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"

# Find model
MODEL="${LLM_MODEL_PATH:-$PROJECT_ROOT/llm/Qwen3.5-9B-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

echo "Starting LLM server with model: $MODEL"
echo "Using llama-server: $LLAMA_SERVER"

exec "$LLAMA_SERVER" -m "$MODEL" -ngl 20 -fa on -ctk q8_0 -ctv q8_0 --port 8080
