# Project history

These files are the dated logs and raw measurements of the project. They describe the engine as it was at each date,
with paths and switches that the project removed later. On 2026-10-06, the project deleted the alternatives that each
measurement rejected. On 2026-10-07, we edited the language of the Markdown logs to ASD-STE100 style. Their facts did not change.
`docs/architecture.md` describes the current design. In the raw outputs, home-directory paths show as `~`. Some logs
refer to model-benchmarks, a separate benchmark harness that is not part of this repository.

| File | Date | Contents |
|---|---|---|
| `baseline.md`, `baselines_2026-10-03.md` | 2026-10-02/03 | Phase 0: the bandwidth and GEMM ceilings of the chip, vLLM / SGLang baselines, the frozen targets |
| `gemm_peak_2026-10-02.txt`, `gemv_bench_2026-10-03.txt` | 2026-10-02/03 | raw GEMM-ceiling and GEMV benchmark output |
| `bf16_inventory.txt`, `fp8_inventory.txt`, `nvfp4_inventory.txt` | 2026-10-03 | tensor inventories of the three checkpoints (`tools/tensor_inventory.py`) |
| `phase1_results.md` | 2026-10-03 | the PyTorch reference model: parity with transformers and vLLM, perplexity per checkpoint |
| `phase2_progress.md`, `decode_8k_2026-10-03_trace_summary.txt` | 2026-10-03 | the first decode kernels and CUDA graphs |
| `phase3_progress.md` | 2026-10-03 | the tensor-core prefill path |
| `phase4_results.md` | 2026-10-03 | speculative decoding (n-gram, then the MTP head) |
| `results.md` | 2026-10-04 | Phase 5: the server as a daily driver, against vLLM and SGLang |
| `phase6_progress.md` | 2026-10-04 to 07 | Phase 6, sections 1-20: skinny GEMM, INT decode copies, decode attention, drafter work, GDN kernels, early exit, low-rank draft head, the sliding-window drafter (not kept) |
