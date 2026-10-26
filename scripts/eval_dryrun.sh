export EVALUATION_PHASE="final" # ["benchmark_probing", "final"]
export DEBUG=1  # 1 for verbose logging, 0 for mininal logging

AGENT_WORKFLOWS=(
  "zeroshot"
  "rlm"
  "rlm-rr"
  "react"
  "react-slim"
  "ours"
)
# export AGENT_WORKFLOW="ours-abl-filter-sparse-window"; export CONTEXT_FILTERING_WITH="sparse" # `ReAct+RR (sparse search + neighborhood expansion) ablation`

BENCHMARKS=(
    "livenewsbench"
    # "sealqa"
    # "deepsearchqa"
)

for benchmark in "${BENCHMARKS[@]}"; do
    mkdir -p "results/benchmarks/${benchmark}"
    for AGENT_WORKFLOW in "${AGENT_WORKFLOWS[@]}"; do
        echo "Running evaluation ${AGENT_WORKFLOW} on benchmark: ${benchmark}"
        python benchmarks/${benchmark}_eval_script.py \
            --agent_workflow ${AGENT_WORKFLOW} --model $LITELLM_MODEL --base_url $LITELLM_BASE_URL --api_key $LITELLM_API_KEY --debug
    done
done
