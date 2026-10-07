# Phase 4: speculative decoding (2026-10-03)

Checkpoint `nvidia/Qwen3.8-27B-NVFP4`, single slot, FP8 KV, 8k max context, 256 new tokens (fewer when the model
stops), `tests/spec_check.py`. Plain decode (same prefill, same CUDA graph, no drafts): 11.0-12.8 tok/s.

## Results

| Prompt | Plain | n-gram (k=3) | MTP k=3, T=0 | acceptance | tok / cycle | MTP k=3, T=0.7 (top-p .95, top-k 20) |
|---|---|---|---|---|---|---|
| code edit | 11.0 | 20.6 | **39.7** | 0.95 | 3.90 | 39.6 |
| JSON from a list | 12.7 | 22.6 | **40.6** | 0.98 | 3.90 | 40.4 |
| code generation | 12.7 | 14.1 | **35.3** | 0.79 | 3.37 | 34.3 |
| prose story | 12.7 | 12.6 | **24.6** | 0.44 | 2.33 | 24.1 |
| mean of the four | | 15.6 | **32.4** | | | **31.8** |

The MTP configuration: FP8 drafter weights (per-row scales) and a draft lm_head over the 64k most frequent tokens plus
the prompt tokens of the request. `tools/draft_vocab.py` chose the 64k tokens from WikiText prose, Python and JSON.

| Exit criterion (PLAN.md Phase 4) | Result |
|---|---|
| Greedy outputs identical with and without spec | **Met**: all prompts token-identical, n-gram and MTP |
| Output distribution unchanged at T > 0 | **Met**: rejection sampling with a deterministic proposal. `tests/test_spec_accept.py` checks the emitted marginal over 30k trials (TV < 0.02). |
| >= 35 tok/s on a code/chat mix at T = 0 | Code and structured output 35.3-40.6: met. The mean with open prose, 32.4: not met. |
| >= 30 tok/s at T = 0.7 | **Met**: 31.8 mean (prose 24.1) |

## How it works

- **Verify** (`FastQwen35.verify`) runs k+1 rows per slot through the decode kernels. Attention writes KV for all rows
  and is causal among them. GDN runs `gdn_conv_multi` / `gdn_delta_multi` in verify mode: outputs for each row, and no
  change to the state.
- **Accept + commit** run on the GPU, with greedy acceptance (argmax == next draft) or speculative sampling. `commit`
  runs the GDN recurrence again from the unchanged state, for the accepted rows only, and writes the result. Later steps
  overwrite the KV past the accepted length, so KV truncation costs nothing. Verify + commit are bit-exact with
  sequential decode steps (`tests/test_spec_gdn.py`).
- **The MTP cycle** (`engine/spec/mtp.py`) is one CUDA graph. It runs the verify, the accept and the commit. Then it runs
  MTP catch-up rows for the newly committed positions with the true hidden states of the target, and k-1 chained MTP
  steps. It writes the input of the next cycle in place. A cycle takes ~98 ms: ~85 ms verify (4-row GEMV) and ~13 ms
  drafts.
- **The n-gram drafter** (`engine/spec/ngram.py`) uses an O(1) suffix index. It is strong on edits that copy the input
  text.

## Bit-exactness bug found and fixed

The first MTP runs went away from plain decode after ~77 tokens on prose. We bisected layer by layer from a cloned
state. An FP8 GEMV row gave different results at M=1 and M=4 for K=6144, but not for K=5120. The cause: nvcc
contracted the epilogue `sum * scale + residual` into an FMA in one template instance, but not in the other.

Now all CUDA code compiles with `--fmad=false`, and only explicit `fmaf` calls fuse. GEMV rows are bit-identical for all
M on all model shapes, and the decode speed did not change.

## Not done / next

- An NVFP4 skinny GEMM on `mma.sync` for M = 5-16. The CUDA-core GEMV costs +8% at M=5, +21% at M=6 and +45% at M=8.
  Thus k > 3 and tree drafts do not help yet. With this GEMM, an adaptive k (k = 5-7 when the acceptance is > 0.9)
  would move code and structured output toward ~45-50 tok/s.
- Multi-slot speculation. Today the engine decodes multiple slots without speculation.
- Seeded reproducibility for speculative sampling. The uniforms come from the global CUDA generator.
- Better drafters for prose (EAGLE-3 / DFlash-style), in Phase 6.
