# Project history

Dated logs and raw measurements, kept as written. They describe the engine as it was at each date, including paths and
switches that have since been removed (the alternatives each measurement rejected were deleted on 2026-10-06); the
current design is in `docs/architecture.md`.

| File | Date | Contents |
|---|---|---|
| `baseline.md`, `baselines_2026-10-03.md` | 2026-10-02/03 | Phase 0: the chip's bandwidth and GEMM ceilings, vLLM / SGLang baselines, the frozen targets |
| `gemm_peak_2026-10-02.txt`, `gemv_bench_2026-10-03.txt` | 2026-10-02/03 | raw GEMM-ceiling and GEMV benchmark output |
| `bf16_inventory.txt`, `fp8_inventory.txt`, `nvfp4_inventory.txt` | 2026-10-03 | tensor inventories of the three checkpoints (`tools/tensor_inventory.py`) |
| `phase1_results.md` | 2026-10-03 | the PyTorch reference model: parity with transformers and vLLM, perplexity per checkpoint |
| `phase2_progress.md`, `decode_8k_2026-10-03_trace_summary.txt` | 2026-10-03 | the first decode kernels and CUDA graphs |
| `phase3_progress.md` | 2026-10-03 | the tensor-core prefill path |
| `phase4_results.md` | 2026-10-03 | speculative decoding (n-gram, then the MTP head) |
| `results.md` | 2026-10-04 | Phase 5: the server as a daily driver, against vLLM and SGLang |
| `phase6_progress.md` | 2026-10-04 to 06 | Phase 6, sections 1-19: skinny GEMM, INT decode copies, decode attention, drafter work, GDN kernels, early exit, low-rank draft head |
