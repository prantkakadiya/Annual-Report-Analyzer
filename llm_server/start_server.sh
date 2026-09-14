#!/usr/bin/env bash

# Find llama-server
LLAMA_SERVER="${LLAMA_SERVER_PATH:-}"

if [ -z "$LLAMA_SERVER" ]; then
    if command -v llama-server &> /dev/null; then
        LLAMA_SERVER="llama-server"
    elif [ -x "/home/prant/llama.cpp/build/bin/llama-server" ]; then
        LLAMA_SERVER="/home/prant/llama.cpp/build/bin/llama-server"
    else
        echo "Error: llama-server not found."
        echo "Please set LLAMA_SERVER_PATH or install llama.cpp"
        exit 1
    fi
fi

# Find model
MODEL="${LLM_MODEL_PATH:-../llm/Qwen3.5-9B-Q4_K_M.gguf}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

echo "Starting LLM server with model: $MODEL"
echo "Using llama-server: $LLAMA_SERVER"

$LLAMA_SERVER -m "$MODEL" -ngl 999 -c 8192 -fa on -ctk q8_0 -ctv q8_0 --port 8080
