### nvidia-Qwen3.8-27B-NVFP4  (20261003_100333)

- server: `vllm serve nvidia/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --trust-remote-code --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --gpu-memory-utilization 0.5 --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-s`
- env: torch 2.11.0+cu130, vllm 0.25.1 (harness venv), gpu NVIDIA GB10
- status: completed 

| prompt tokens | TTFT median s | TTFT p95 s | prefill tok/s (avg) |
|---|---|---|---|
| 32 | 0.123 | 0.264 | 634 |
| 128 | 0.139 | 0.360 | 1184 |
| 512 | 0.250 | 0.385 | 2150 |
| 2048 | 0.809 | 0.947 | 2533 |
| 8192 | 3.904 | 4.997 | 2040 |
| 16384 | 8.146 | 8.265 | 2010 |

| context tokens | TTFT median s | TTFT p95 s | prefill tok/s |
|---|---|---|---|
| 32768 | 18.027 | 18.551 | 1802 |
| 65536 | 43.315 | 43.333 | 1515 |

| output tokens | decode tok/s avg | median | peak | TTFT s |
|---|---|---|---|---|
| 512 | 12.4 | 12.4 | 12.7 | 0.146 |
| 1024 | 12.3 | 12.4 | 12.7 | 0.139 |
| 2048 | 12.3 | 12.4 | 12.8 | 0.163 |

| spec_enabled: output tokens | decode tok/s avg | median | peak |
|---|---|---|---|
| 512 | 24.8 | 9.2 | 9.4 |
| 1024 | 22.2 | 9.3 | 9.4 |
| 2048 | 21.6 | 9.3 | 10.3 |

| spec_disabled: output tokens | decode tok/s avg | median | peak |
|---|---|---|---|
| 512 | 12.2 | 12.2 | 13.6 |
| 1024 | 12.2 | 12.2 | 13.8 |
| 2048 | 12.2 | 12.2 | 13.9 |

spec comparison: {"config": {"output_lengths": [512, 1024, 2048]}, "spec_enabled": {"512": {"requested_output_tokens": 512, "actual_output_tokens": 512, "output_tokens_exact": true, "output_truncated": false, "ttft_s": 1.922, "decode_time_s": 20.658, "tok_per_sec_avg": 24.78, "tok_per_sec_peak": 9.45, "tok_per_sec_min": 8.26, "tok_per_sec_median": 9.19, "output_text_preview": "We need to respond to user: \"You are a creative writer. Write a detailed, multi-paragraph story about the discovery of an ancient underwater civilization. Describe the ocean environment, the architecture of their cities, their technolog

| concurrency | aggregate tok/s | per-request tok/s | mean latency s | failed |
|---|---|---|---|---|
| 1 | 12.2 | 12.2 | - | 0 |
| 2 | 23.3 | 11.7 | - | 0 |
| 3 | 30.7 | 10.2 | - | 0 |
| 4 | 45.0 | 11.2 | - | 0 |


### nvidia-Qwen3.8-27B-NVFP4-sglang  (20261003_104445)

- server: `~/Projects/sglang/.venv/bin/python -m sglang.launch_server --model-path nvidia/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port 8000 --trust-remote-code --context-length 131072 --max-running-requests 4 --chunked-prefill-size 8192 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --fp`
- env: torch 2.11.0+cu130, vllm 0.25.1 (harness venv), gpu NVIDIA GB10
- status: completed 

| prompt tokens | TTFT median s | TTFT p95 s | prefill tok/s (avg) |
|---|---|---|---|
| 32 | - | - | - |
| 128 | - | - | - |
| 512 | - | - | - |
| 2048 | - | - | - |
| 8192 | - | - | - |
| 16384 | - | - | - |

| context tokens | TTFT median s | TTFT p95 s | prefill tok/s |
|---|---|---|---|
| 32768 | - | - | - |
| 65536 | - | - | - |

| output tokens | decode tok/s avg | median | peak | TTFT s |
|---|---|---|---|---|
| 512 | 12.3 | - | - | - |
| 1024 | 12.3 | - | - | - |
| 2048 | 12.3 | - | - | - |

| concurrency | aggregate tok/s | per-request tok/s | mean latency s | failed |
|---|---|---|---|---|
| 1 | 12.2 | 12.2 | - | 0 |
| 2 | 23.3 | 11.7 | - | 0 |
| 3 | 30.6 | 10.2 | - | 0 |
| 4 | 44.6 | 11.2 | - | 0 |


### nvidia-Qwen3.8-27B-NVFP4-sglang_mtp  (20261003_111305)

- server: `~/Projects/sglang/.venv/bin/python -m sglang.launch_server --model-path nvidia/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port 8000 --trust-remote-code --context-length 131072 --max-running-requests 4 --chunked-prefill-size 8192 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --fp`
- env: torch 2.11.0+cu130, vllm 0.25.1 (harness venv), gpu NVIDIA GB10
- status: completed 

| prompt tokens | TTFT median s | TTFT p95 s | prefill tok/s (avg) |
|---|---|---|---|
| 32 | - | - | - |
| 128 | - | - | - |
| 512 | - | - | - |
| 2048 | - | - | - |
| 8192 | - | - | - |
| 16384 | - | - | - |

| context tokens | TTFT median s | TTFT p95 s | prefill tok/s |
|---|---|---|---|
| 32768 | - | - | - |
| 65536 | - | - | - |

| output tokens | decode tok/s avg | median | peak | TTFT s |
|---|---|---|---|---|
| 512 | 24.8 | - | - | - |
| 1024 | 22.2 | - | - | - |
| 2048 | 22.3 | - | - | - |

| concurrency | aggregate tok/s | per-request tok/s | mean latency s | failed |
|---|---|---|---|---|
| 1 | 34.1 | 34.1 | - | 0 |
| 2 | 44.1 | 22.1 | - | 0 |
| 3 | 70.5 | 23.5 | - | 0 |
| 4 | 76.3 | 19.1 | - | 0 |


