# Assuming a `gemma-4-E4B-it` model is deployed at localhost:8000, and a `gpt-oss-20b` model is deployed at localhost:8001, run the following command to evaluate the benchmarks

export EVALUATION_PHASE="final" # ["benchmark_probing", "final"]
export DEBUG=0  # 1 for verbose logging, 0 for mininal logging

# export AGENT_WORKFLOW="zeroshot"                              # `Zeroshot` baseline
# export AGENT_WORKFLOW="react-1iter"; export MAX_ITERATIONS=1  # `Minimal` baseline
# export AGENT_WORKFLOW="rlm"                                   # `RLM` baseline
# export AGENT_WORKFLOW="rlm+rr"                                # `RLM+RR` baseline
# export AGENT_WORKFLOW="react"                                 # `ReAct` baseline
# export AGENT_WORKFLOW="react-slim"                            # `SLIM` baseline
# export AGENT_WORKFLOW="ours"                                  # `ReAct+RR` baseline
export AGENT_WORKFLOW="ours-abl-filter-sparse-window"; export CONTEXT_FILTERING_WITH="sparse" # `ReAct+RR (sparse search + neighborhood expansion) ablation`


BENCHMARKS=(
    "livenewsbench"
    "sealqa"
    "deepsearchqa"
)


for benchmark in "${BENCHMARKS[@]}"; do
    mkdir -p "results/benchmarks/${benchmark}"
    echo "Running evaluation ${AGENT_WORKFLOW} ${MODEL_FOLDER} on benchmark: ${benchmark}"
    python benchmarks/${benchmark}_eval_script.py \
        --agent_workflow ${AGENT_WORKFLOW} \
        --model "hosted_vllm/gemma-4-E4B-it" --base_url "http://localhost:8000/v1" --api_key "dummy" \
        --sub_model "hosted_vllm/gpt-oss-20b" --sub_base_url "http://localhost:8001/v1" --sub_api_key "dummy" \
        --use_separated_judge
done
