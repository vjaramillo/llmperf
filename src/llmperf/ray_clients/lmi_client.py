import os
import json
import time
from typing import Any, Dict

import requests
import ray

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class lmiClient(LLMClient):
    """Client for LMI Container."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        prompt = f"""<system_message> {prompt["system_message"]}. </system_message>
                     <context> {prompt["context"]}. </context>
                     <question>: {prompt["question"]} </question>"""

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]

        payload = {
            "inputs": prompt,
            "parameters": {
                **sampling_params
            }
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}
        url = os.environ.get("API_URL")
        headers = {'Content-type': 'application/json'}
        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                
                for line in response.iter_lines(chunk_size=None):
                    if line:
                        data = json.loads(line.decode('utf-8'))
                    
                    # Based on the streamed data, we can see that for LMI for each 
                    # line a token is received. Thus we can increase the tokens receive +1 for each line.
                    tokens_received += 1

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(data["error"]["message"])
                        
                    delta = data["token"]["text"]
                    if delta:
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                
                generated_text = data["generated_text"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = time_to_next_token
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
