#!/bin/bash
model=${1:-<MODEL_TO_BENCHMARK >}
vu=${2:-1}
export TOKENIZERS_PARALLELISM=true
max_requests=$(expr ${vu} \* 8 )
date_str=$(date '+%Y-%m-%d-%H-%M-%S')
/Users/vicvel/Documents/code/llmperf-venv/bin/python ./token_benchmark_ray.py \
       --model mistral \
       --max-num-completed-requests ${max_requests} \
       --timeout 7200 \
       --num-concurrent-requests ${vu} \
       --results-dir "lmi_bench_results/${date_str}" \
       --llm-api lmiclient \
       --additional-sampling-params '{"do_sample": true, "max_new_tokens": 128, "stream": true, "output_formatter": "jsonlines"}' \
       --max-seq-len 1800 \
       --custom-prompts-location "/Users/vicvel/Documents/code/llmperf/prompts.json"