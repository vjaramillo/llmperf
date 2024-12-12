#!/bin/bash
model=${1:-<MODEL_TO_BENCHMARK >}
vu=${2:-1}
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:8000/v1"
export TOKENIZERS_PARALLELISM=true
max_requests=$(expr ${vu} \* 8 )
date_str=$(date '+%Y-%m-%d-%H-%M-%S')
/home/ubuntu/aws_llmperf/bin/python ./token_benchmark_ray.py \
       --model ${model} \
       --mean-input-tokens 100 \
       --stddev-input-tokens 10 \
       --mean-output-tokens 100 \
       --stddev-output-tokens 10 \
       --max-num-completed-requests ${max_requests} \
       --timeout 7200 \
       --num-concurrent-requests ${vu} \
       --results-dir "vllm_bench_results/${date_str}" \
       --llm-api openai \
       --additional-sampling-params '{}' \
       --max-seq-len 1800 \
       --custom-prompts-location "/home/ubuntu/prompts.json"