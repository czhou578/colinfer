# Phase 4: speculative decoding (2026-10-03)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4`, single slot, FP8 KV, 8k max context, 256 new tokens (fewer when the
model stops), `tests/spec_check.py`. Plain decode (same prefill, same CUDA graph, no drafting): 11.0-12.8 tok/s.

## Results

| Prompt | Plain | n-gram (k=3) | MTP k=3, T=0 | acceptance | tok / cycle | MTP k=3, T=0.7 (top-p .95, top-k 20) |
|---|---|---|---|---|---|---|
| code edit | 11.0 | 20.6 | **39.7** | 0.95 | 3.90 | 39.6 |
| JSON from a list | 12.7 | 22.6 | **40.6** | 0.98 | 3.90 | 40.4 |
| code generation | 12.7 | 14.1 | **35.3** | 0.79 | 3.37 | 34.3 |
| prose story | 12.7 | 12.6 | **24.6** | 0.44 | 2.33 | 24.1 |
| mean of the four | | 15.6 | **32.4** | | | **31.8** |

MTP configuration: drafter weights FP8 (per-row scales), draft lm_head over the 64k most frequent tokens
(`tools/draft_vocab.py`: WikiText prose + Python + JSON) plus the request's prompt tokens.

| Exit criterion (PLAN.md Phase 4) | Result |
|---|---|
| Greedy outputs identical with and without spec | **Met**: all prompts token-identical, n-gram and MTP |
| Output distribution unchanged at T > 0 | **Met**: rejection sampling with a deterministic proposal; `tests/test_spec_accept.py` checks the emitted marginal over 30k trials (TV < 0.02) |
| >= 35 tok/s on a code/chat mix at T = 0 | Code and structured output 35.3-40.6: met. Mean including open prose 32.4: not met |
| >= 30 tok/s at T = 0.7 | **Met**: 31.8 mean (prose 24.1) |

## How it works

- **Verify** (`FastQwen35.verify`): k+1 rows per slot through the decode kernels; attention writes KV for all
  rows and is causal among them; GDN runs `gdn_conv_multi` / `gdn_delta_multi` in verify mode (outputs for
  every row, state untouched).
- **Accept + commit** on the GPU: greedy (argmax == next draft) or speculative sampling; `commit` re-runs the
  GDN recurrence from the untouched state for exactly the accepted rows and writes it (KV beyond the accepted
  length is overwritten later: KV truncation is free). Verify + commit are bit-exact with sequential decode
  steps (`tests/test_spec_gdn.py`).
- **MTP cycle** (`engine/spec/mtp.py`, one CUDA graph): verify, accept, commit, MTP catch-up rows for the newly
  committed positions using the target's true hidden states, k-1 chained MTP steps; the next cycle's input is
  written in place. ~98 ms per cycle: ~85 ms verify (4-row GEMV), ~13 ms drafting.
- **n-gram drafter** (`engine/spec/ngram.py`): O(1) suffix index; strong on edits that copy input text.

## Bit-exactness bug found and fixed

The first MTP runs diverged from plain decode after ~77 tokens on prose. Bisecting layer by layer from a
cloned state showed an FP8 GEMV row giving different results at M=1 and M=4 for K=6144 (not for K=5120):
nvcc contracted the epilogue's `sum * scale + residual` into an FMA in one template instance but not the
other. All CUDA code is now compiled with `--fmad=false` (only explicit `fmaf` calls fuse); GEMV rows are
bit-identical for every M on every model shape, decode speed unchanged.

## Not done / next

- NVFP4 skinny GEMM on `mma.sync` for M = 5-16: the CUDA-core GEMV costs +8% at M=5, +21% at M=6, +45% at
  M=8, so k > 3 and tree drafts are not yet worth it. With it, adaptive k (k = 5-7 when acceptance > 0.9)
  would lift code / structured output toward ~45-50 tok/s.
- Multi-slot speculation (the engine decodes multiple slots without speculation today).
- Seeded reproducibility for speculative sampling (uniforms come from the global CUDA generator).
- Better drafters for prose (EAGLE-3 / DFlash-style), Phase 6.
