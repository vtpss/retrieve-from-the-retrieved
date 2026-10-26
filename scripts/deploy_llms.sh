echo "Use a separate runtime to deploy backbone models in a local server (vLLM) for evaluation"

vllm serve google/gemma-4-E4B-it \
    --served-model-name gemma-4-E4B-it \
    --seed 0 \
    --port 8000 \
    --language-model-only \
    --max-model-len 131072

vllm serve openai/gpt-oss-20b \
    --served-model-name gpt-oss-20b \
    --seed 0 \
    --port 8001 \
    --max-model-len 131072
