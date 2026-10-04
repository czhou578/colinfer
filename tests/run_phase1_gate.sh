#!/usr/bin/env bash
# Phase 1 exit gate (PLAN.md 6): BF16 parity vs HF on 30 prompts, NVFP4 path vs vLLM, WikiText
# perplexity per checkpoint. Sequential: each step owns the GPU. Logs in tests/parity_out/logs/.
set -u
cd "$(dirname "$0")/.."
O=tests/parity_out; L=$O/logs
step() { local name=$1; shift; echo "=== $(date '+%H:%M:%S') START $name"; "$@" > "$L/$name.log" 2>&1; local rc=$?; echo "=== $(date '+%H:%M:%S') END $name rc=$rc"; tail -n 3 "$L/$name.log" | sed 's/^/    /'; }
PY="uv run --no-sync python"
step ours_bf16   $PY tests/parity_hf.py ours --ckpt Qwen/Qwen3.8-27B --out $O/ours_bf16.pt
step ref_bf16    $PY tests/parity_hf.py ref  --ckpt Qwen/Qwen3.8-27B --out $O/ref_bf16.pt
step cmp_bf16    $PY tests/parity_hf.py compare $O/ref_bf16.pt $O/ours_bf16.pt
step ours_nvfp4  $PY tests/parity_hf.py ours --ckpt nvidia/Qwen3.8-27B-NVFP4 --out $O/ours_nvfp4.pt
step vllm_nvfp4  env VLLM_USE_FASTOKENS=0 CUTE_DSL_ARCH=sm_121a $HOME/Projects/model-benchmarks/.venv/bin/python tests/vllm_reference.py --ckpt nvidia/Qwen3.8-27B-NVFP4 --out $O/vllm_nvfp4.json
step cmp_nvfp4   $PY tests/parity_vllm.py $O/vllm_nvfp4.json $O/ours_nvfp4.pt
step ppl_bf16    $PY tests/perplexity.py --ckpt Qwen/Qwen3.8-27B --json $O/ppl_bf16.json
step ppl_fp8     $PY tests/perplexity.py --ckpt Qwen/Qwen3.8-27B-FP8 --json $O/ppl_fp8.json
step ppl_nvfp4   $PY tests/perplexity.py --ckpt nvidia/Qwen3.8-27B-NVFP4 --json $O/ppl_nvfp4.json
echo "PHASE 1 GATE DONE"
