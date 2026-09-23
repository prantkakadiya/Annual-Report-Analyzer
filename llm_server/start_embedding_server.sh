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

# Find embedding model
MODEL="${EMBEDDING_MODEL_PATH:-$PROJECT_ROOT/Embedding model/Qwen3-Embedding-0.6B-Q8_0.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

echo "Starting Embedding server with model: $MODEL"
echo "Using llama-server: $LLAMA_SERVER"

# -np 1                : single slot — only one request at a time
# -c 8192              : 8K context (safely fits 16,000 char chunks which can hit ~4200 tokens)
# --no-cache-prompt    : disable prompt cache to prevent monotonic RAM growth across batches
exec "$LLAMA_SERVER" -m "$MODEL" -ngl 999 --embedding -c 8192 -b 512 -np 1 --no-cache-prompt -fa on --port 8081
