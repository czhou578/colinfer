# Qwen3.6-35B-A3B (qwen3_5_moe) on colinfer: the plan

**Status:** 2026-11-14: Phase 0, step 1 is done (§5.1): a process with no model takes about 2.5 GB. The window of step 2 is open · **Author:** colin-spark · **Date:** 2026-11-14
**Related:** `qwen4exp_plan.md` (the MoE family that this plan reuses), `../architecture.md` (the 27B),
`~/Projects/dsv4-specialize/plans/PLAN_SECOND_MODEL_FIT.md` (§5.5: the model under vLLM) and `PLAN_QWEN_PRUNE.md`
(§7.4: the overhead of vLLM, and K 208)
**Scope:** a third model family in this repo, its tests and its docs; one memory probe; about five short test windows
with the DeepSeek server stopped. No change to the DeepSeek server. The port of DeepSeek is not in this plan (§10).

| Step | Status |
|---|---|
| §5 Phase 0: the memory floor of the engine (one window of about 15 min) | 🔨 2026-11-14: step 1, the process with no model: about 2.5 GB (§5.1). Step 2, the window with the 27B, is open |
| §6 A: the reference model and the weights (no window) | ⬜ |
| §6 K1, K2: the kernel changes (no window) | ⬜ |
| §6 M1: golden decode on the real checkpoint | ⬜ |
| §6 M2: prefill and long context | ⬜ |
| §6 M3: the drafter and exact speculation | ⬜ |
| §6 M4: the family, the server and the memory trim | ⬜ |
| §7 M5: the measure window beside the vLLM numbers | ⬜ |
| §10 The decision on DeepSeek | ⬜ |

---

## 1. The goal

Serve `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` (22.0 GiB on disk) from colinfer, as the fast model beside the DeepSeek
server. The guarantees are those of the other families: exact speculation, one request at a time, and a perplexity
gate of 0.5% for each quantization beyond the checkpoint's own. The 27B and Qwen3.8-Flash-Next stay bit for bit as
they are.

Three results decide the value of the work:

1. **Memory.** Under vLLM the full model with its draft uses 29.6 GiB at a 128k context, and about 29 GiB is available
   beside DeepSeek. The goal is at most 26.5 GiB, the use of the pruned K 208 model, with no loss of quality.
2. **Speed.** vLLM decodes 98 to 101 tok/s with the draft and 58 tok/s without it. The goal is at least these rates.
3. **A real number for the engine's fixed memory.** It decides the DeepSeek question of §10.

## 2. The checkpoint (read from the files on 2026-11-14)

`model_type` is `qwen3_5_moe`. The backbone is the 27B's: Gated DeltaNet layers with full attention in every fourth
layer, the same tokenizer (248,320 tokens), the same chat template family and one MTP layer. A MoE replaces the MLP.

| Value | 35B-A3B | 27B |
|---|---|---|
| Layers (GDN / attention) | 40 (30 / 10) | 64 (48 / 16) |
| Hidden size | 2,048 | 5,120 |
| GDN heads (key / value), head sizes | 16 / 32, 128 / 128 | 16 / 48, 128 / 128 |
| Attention heads (query / KV), head size | 16 / 2, 256 | 24 / 4, 256 |
| MLP | 256 experts, top-8, width 512, one shared expert of width 512 | dense, width 17,408 |
| KV cache per token (fp8) | 10.2 KB | 32 KB |

Bytes in GB (1e9), as in the other pages of this repo:

| Part | GB | Format |
|---|---|---|
| Routed experts (40 × 256, three matrices each) | 18.12 | NVFP4: `weight_packed` u8, `weight_scale` e4m3 per 16 weights, `weight_global_scale` f32, `input_global_scale` f32 |
| Shared experts | 0.07 | NVFP4, the same four tensors |
| GDN in_proj_qkv / in_proj_z / out_proj | 1.01 | FP8 e4m3 with a `weight_scale` per row (bf16 [N, 1]) |
| Attention q / k / v / o | 0.27 | FP8, a scale per row |
| lm_head | 0.51 | FP8, a scale per row |
| Embedding | 1.02 | BF16 |
| Routers (`mlp.gate`), `shared_expert_gate`, GDN a / b, conv, norms | 0.05 | BF16 |
| MTP layer (attention, `fc`, 256 stacked experts) | 1.69 | BF16 |
| Vision tower (27 blocks, output 2,048) | 0.89 | BF16 |

The target weights without the vision tower and the MTP layer are 21.05 GB.

**What differs from the two checkpoints that the engine reads today:**

- **The tensor names** are those of llm-compressor: `weight_packed`, `weight_global_scale`, `input_global_scale`. The
  nvidia checkpoints use `weight`, `weight_scale_2`, `input_scale`. The global scales are large (7,000 to 61,000), so
  they are probably divisors. A reads the format definition and proves the rule.
- **The FP8 layers have a scale per row**, and the activations use a dynamic scale per token
  (`quantization_config`: weights `channel`, inputs `dynamic`, `token`). The 27B has one scale per tensor and a static
  `input_scale`. `Fp8Linear` takes one scalar today.
- **The lm_head is FP8**, not NVFP4.
- **The input scale of an expert.** For gate and up it is one value per layer, as in Flash-Next. For down it differs
  per expert (142 to 166 distinct values in a layer). The prefill must quantize the down input per (token, expert)
  pair.
- **One tensor per expert:** 10,240 experts × 12 tensors. The loader must stack them.
- **The MTP experts are BF16 and stacked** (`gate_up_proj` [256, 1024, 2048], `down_proj` [256, 2048, 512]).

The router is that of Flash-Next: a softmax over all experts in fp32, the top k, a renormalization, and a sigmoid
gate on the shared expert (`transformers` 5.18, `Qwen3_5MoeTopKRouter`, `Qwen3_5MoeSparseMoeBlock`). The MTP wiring is
the 27B's (`fc` over the normed embedding and hidden state, one decoder layer, a final norm), with the MoE as its MLP.

## 3. What the engine has, and what it needs

| Part | State | Work |
|---|---|---|
| GDN decode and prefill kernels | The head sizes are compile-time values, 128 / 128: they fit | None expected; a test with 32 value heads |
| Decode attention (`attn_decode.cu`) | Builds for 6 and 12 query heads per KV head | Add the build for 8 |
| FP8 skinny GEMM, FP8 prefill GEMM | One scalar weight scale, a static input scale | A scale per row; in prefill, a dynamic scale per token |
| MoE decode kernels (`moe.cu`) | The sizes fit: K 2,048, I 512, E 256. `TOPK` is the constant 10 | `TOPK` becomes a parameter: 8 and 10 |
| MoE prefill (`moe_prefill.cu`) | `TOPK` 10; one down input scale per layer | `TOPK` 8; a down input scale per pair |
| Shared expert | BF16 path in Flash-Next; NVFP4 `SwiGLUMLP` in the 27B | Use the 27B's module |
| Reference model | `qwen35.py` (dense), `qwen4exp.py` | `qwen35moe.py`: the dense reference with a MoE block |
| Loader | nvidia names | The llm-compressor names; stack the experts |
| Drafter | `mtp.py` (dense MLP), `mtp4.py` (Flash-Next) | `mtp.py` with a MoE MLP; an NVFP4 copy of the MTP experts |
| Family dispatch | An unknown `model_type` gets the dense family and fails in the config reader | A `qwen3_5_moe` entry in `family.py` |
| Vision | The 27B's tower with a 2,560-wide output for Flash-Next | The same class with a 2,048-wide output |
| Scheduler, server, grammar, session cache | Family hooks exist | The memory constants, the defaults and a cycle cost model |

No new kernel class is necessary. That is the difference from Flash-Next, which needed five new mechanisms.

## 4. Memory and speed (estimates)

**Memory**, in GB (1e9), for one slot of 131,072 tokens with the drafter:

| Item | GB |
|---|---|
| Target weights | 21.05 |
| Drafter: an NVFP4 copy of the MTP experts, the rest BF16 | about 0.55 (1.69 in BF16) |
| KV cache of the slot (10.2 KB per token), and the drafter's (1.0 KB) | 1.48 |
| GDN state of the slot | 0.06 |
| Checkpoint ring, 8 snapshots of 0.063 | 0.50 |
| The engine's fixed memory X: CUDA context, Python and torch, graph pool, prefill workspace, pinned buffers | 5.3 to 6.6 today (§5) |
| **Sum** | **28.9 to 30.2 (26.9 to 28.2 GiB)** |

- The goal of 26.5 GiB (28.5 GB) needs X at or below about 4.8 GB.
- A margin of 4 GiB beside DeepSeek with no change there needs X at or below about 3.0 to 3.7 GB.
- With the DeepSeek context cap (D1 of the fit plan, 1.1 GiB) the limit is about 4.2 to 4.9 GB.
- A slot of 262,144 tokens adds 1.48 GB.
- The vision tower adds 0.89 GB with the first image. The INT decode copies would add about 1 GB; this plan does not
  make them.

So the fixed memory decides the result. Phase 0 measures it before any other work.

**Speed.** One token reads about 2.47 GB: 1.28 GB of GDN and attention projections, 0.57 GB of routed experts, 0.51 GB
of lm_head, 0.07 GB of shared experts and 0.04 GB of routers. At 210 to 238 GB/s the ceiling of plain decode is 85
to 96 tok/s. vLLM measures 58 tok/s on the bot prompts. The projections and the head are 72% of the bytes, so the FP8
path matters more than the MoE path. A verify of M rows reads the union of the rows' experts, as in Flash-Next, so
the scheduler needs the cost model with the distinct-expert count.

## 5. Phase 0: the memory floor of the engine

The docs give the fixed memory only as a remainder: 5.3 GB (the plan of Flash-Next), 5.8 GB (its budget table) and up
to 6.6 GB (the measured constant less the weights and the ring). No measurement splits it.

1. Write `tools/mem_floor.py` (done, §5.1). It records `MemAvailable`, the resident memory of the process and
   `torch.cuda.memory_reserved` after each step:
   - the start of Python and the import of torch;
   - the CUDA context;
   - the load of the kernel extension (`engine/kernels`);
   - the imports of the server (the tokenizer, the HTTP layer, llguidance);
   - one small CUDA graph.
   This part needs about 3 GB and no model. It can run beside the DeepSeek server.
2. In one window, start the 27B at its smallest settings:
   `--slots 1 --max-seq-len 131072 --spec none --decode-weights checkpoint --session-cache-gb 0 --no-vision`. Record
   the same values at the ready line and after one prompt of 100,000 tokens. Then start it once more with the
   defaults of the drafter and the session cache, to get the cost of each.
3. Report a table of the fixed memory by part, and the parts that a flag or a small change can remove.

The window of step 2 is `colinfer_floor.sh` in `~/Projects/dsv4-specialize` (its pull request #64), on the run
library of the other test windows. It has four points, all with 1 slot of 131,072 tokens, no INT copies and no vision
tower: f1 with no drafter and no prefix caching, f2 with the checkpoint ring, f3 with the session cache (boot only, so
that no test conversation goes into `~/.cache/colinfer/sessions`), and f4 with the MTP drafter. The state of the slot
is 4.45 GB with no drafter and 4.72 GB with it (`startup_bytes` on the meta device).

CAUTION: Step 2 needs about 35 GB. The DeepSeek server must be stopped, or the machine can power off. You start the
window.

**Gate.** The table exists, and it names the changes that bring X to 4.8 GB or less. If no set of changes reaches
6.0 GB, colinfer saves less than 1 GiB against vLLM. Then only the speed goal remains, and you decide whether the
plan goes on.

### 5.1 Result of step 1 (2026-11-14, 11:00 to 11:10, beside the DeepSeek server)

`tools/mem_floor.py` ran five times, with about 31 GB of `MemAvailable`. A parent with no torch starts a child, and
measures after each step of the child. The noise of `MemAvailable` before a run was at most 0.03 GB, and its drift
over a run was −0.06 to +0.24 GB. A first run, 3 minutes after the restore of the DeepSeek server, had a drift of
0.8 GB and is not in the table.

| Step | Growth of the fall of `MemAvailable` (GB, range of the runs) |
|---|---|
| The interpreter and `import torch` | 0.29 to 0.33 |
| The CUDA context and one tensor | 0.23 to 0.31 |
| The kernel extension (`engine/kernels`) | 0.00 |
| The imports of the server (`engine.server.api`, the drafter) | 0.02 to 0.05 |
| The tokenizer of the 27B | 0.20 to 0.34 |
| One bf16 matmul (cuBLAS), 2 runs | 0.18 to 0.24 |
| The self-test of each matmul path (`engine/selftest.py`), after cuBLAS | 1.29 to 1.31 |
| One small CUDA graph in a graph pool | 0.00 |
| **Sum at the end, 5 runs** | **2.32 to 2.63 (median 2.48)** |

At the end, the child holds 1.94 GB of anonymous host memory, 0.51 GB of file-backed pages (the libraries) and 0.44 GB
in its CUDA context. The torch allocator holds 0.15 GB of that. A release of the allocator cache frees nothing more.

What the result says:

- **The process floor is about 2.5 GB.** The docs give 5.3 to 6.6 GB for the fixed memory. So 2.8 to 4.1 GB of it
  is not the process: it is the graph pool of a real model, the prefill workspace, the buffers of the session cache
  (0.74 GB by `docs/architecture.md`), the drafter and the slack of the allocator. Step 2 measures these parts.
- **The goal of §4 is possible.** X at or below 4.8 GB leaves about 2.3 GB for these parts. The stricter limit of
  3.0 to 3.7 GB leaves 0.5 to 1.2 GB, so it needs a trim of each part.
- **The self-test is the largest step: 1.3 GB**, 1.12 GB of it anonymous host memory. It is the first use of the
  other matmul paths: the engine's kernels, cuBLASLt, CUTLASS and FlashInfer. A split of the self-test by path can
  show whether one library takes most of it. That is a candidate for a trim, with step 2.
- **Against vLLM:** its two processes hold 4.76 GB of host memory with a model loaded (1.72 GB for the API server,
  3.04 GB for the engine core; read from `/proc` on 2026-11-14). colinfer has one process.

## 6. Milestones

The hours use the scale of `qwen4exp_plan.md` §6, where Flash-Next was 500 hours.

| Milestone | Work | Proves | Hours |
|---|---|---|---|
| 0. Floor | §5 | The fixed memory by part | 6 |
| A. Reference and weights | `engine/model/qwen35moe.py`: the config and the reference (the layers of `qwen35.py`, a MoE block). `engine/weights/qwen35moe_loader.py`: the llm-compressor names, the scale rules, the stacked experts. `tools/verify_qwen35moe.py`: the headers and the scales without the GPU | The tiny model is bit-identical to `Qwen3_5MoeForConditionalGeneration` of transformers in fp32: one-shot, split, stepwise, with a cache, with an image. The loader reads a synthetic checkpoint in the unsloth layout | 20 |
| K1. MoE kernels | `TOPK` as a parameter in `moe.cu`, `moe_prefill.cu`, `moe_bind.cpp` and `moe_kernels.py`. A down input scale per pair in the prefill path | `test_moe.py` and `test_moe_prefill.py` pass for 8 and for 10. Each output row of Flash-Next is bit-identical to the row before the change | 25 |
| K2. FP8 and attention | A weight scale per row in the FP8 skinny GEMM and in `Fp8Linear`. A dynamic input scale per token in the FP8 prefill GEMM. The attention build for 8 query heads per KV head | The kernels equal their torch references. The 27B's golden tokens do not change | 25 |
| M1. Golden decode | `engine/model/fast_qwen35moe.py`: the decode model on the kernels, one slot, no speculation | The greedy tokens of three prompts equal the reference's on the real checkpoint | 25 |
| M2. Prefill | The prefill path: the dense layer functions, the grouped MoE GEMM, the chunk loop | The golden tokens with the prompts through the prefill. Perplexity within 0.5% of the reference (8 WikiText windows). Passkey retrieval at 16k, 64k and 120k | 30 |
| M3. Speculation | `mtp.py` with a MoE MLP in the MTP layer. An NVFP4 copy of the MTP experts, kept in the drafter cache. The cycle cost model | The tokens with speculation equal the tokens of plain decode, greedy and sampled. The acceptance and the best draft lengths | 35 |
| M4. Family and server | The `qwen3_5_moe` entry in `family.py`: the load, the state, the defaults, the memory constants. The vision tower. The trims that Phase 0 names. `scheduler_check` for this family | The server answers on both APIs, with tool calls and a JSON schema. The memory check asks for what the engine takes | 20 |
| M5. Measure and docs | §7. `README.md`, `docs/architecture.md`, `docs/server.md`, `docs/checkpoints.md` | The three results of §1 | 15 |

The sum is about 200 hours. A, K1 and K2 are independent and can run in three git worktrees, as for Flash-Next. M1
needs all three.

**Rules for every milestone:**

- `tests/check.sh --quick` passes before each merge. The full `tests/check.sh` passes at M1, M3 and M4: the 27B's
  golden requests stay bit for bit.
- A change in `csrc/` or in a file of `.claude/rules/sanitizers.md` gets its sanitizer check.
- Each pull request changes at most 400 lines. A milestone that is larger becomes several pull requests, and each one
  passes its tests on its own.

**The defaults of the family** (proposals, see §9): 1 slot of 131,072 tokens, 8 checkpoints, `--spec mtp`, vision on
the first image, no decode copies, a `--mem-cap-gb` from the measured peak.

## 7. M5: the measure window

One window with the DeepSeek server stopped, about 30 minutes. It runs the tests of the vLLM windows, so the numbers
compare with `PLAN_QWEN_PRUNE.md` §7.4 (point o0):

1. The smoke test: a short request and a tool call.
2. One prompt of about 120,000 tokens: the time, and the lowest `MemAvailable`.
3. The bot decode test (`model_eval.speed`, 5 sessions), with the drafter and without it.
4. The memory at the ready line and at the peak, and the resident memory of the process.
5. The tool-call suite of the gate window (85 cases).

| Result | vLLM (measured) | Goal for colinfer |
|---|---|---|
| Use at the peak, 128k, with the draft | 29.64 GiB | At most 26.5 GiB |
| Decode with the draft | 98 to 102 tok/s | At least 98 tok/s |
| Decode with no draft | 58 tok/s | At least 58 tok/s |
| Prompt of 119,745 tokens | 46 s | At most 46 s |
| Tool-call suite | 50 to 52 of 85 | Within the noise of the suite |

The window needs a point for a colinfer server in `qwen_fit.sh` of `dsv4-specialize`. That change is one pull
request in that repo.

## 8. Risks

| Risk | Effect | Guard |
|---|---|---|
| The fixed memory of the engine stays near 6 GB | No memory gain against vLLM | Phase 0 comes first and costs 6 hours and one short window |
| The scale rule of the llm-compressor format is read wrong | Garbage text, or a small silent error | A reads the format definition. M2 also compares the perplexity of 8 windows with the `prompt_logprobs` of vLLM 0.33 on the same checkpoint, recorded once in a window |
| A down input scale per expert makes the prefill slower | A slower prefill than vLLM's 2,600 tok/s | K1 measures the prefill of one layer before M2 |
| The dynamic input scale per token of the FP8 layers has no fast path | A slower prefill | K2 measures it. The fallback is a BF16 input with the FP8 weights, as decode does |
| Top-k near-ties differ from transformers on real text | A golden token differs at a near-tie | The rule of Flash-Next: probability descending, index ascending, the same rule in every path |
| `TOPK` as a parameter changes the bits or the speed of Flash-Next | A regression of a served model | K1 proves bit identity per row, and `bench/moe_bench.py` runs before and after |
| The NVFP4 copy of the MTP experts lowers the acceptance | Less speed, never another output | M3 measures the acceptance of both drafters; keep BF16 if the copy loses more than it saves |
| A test load over-commits the memory | The machine powers off | Each real-checkpoint run needs about 30 GB: run it only in a window, or with at least 46 GB of `MemAvailable` |

## 9. Open questions for you

1. The default slot: 131,072 tokens (the proposal), or 262,144 for 1.48 GB more?
2. The drafter with an NVFP4 copy of its experts (about 1.1 GB less), if M3 shows that the speed holds?
3. No INT decode copies in this plan: they add about 1 GB. A later phase can add them under the perplexity gate.
4. The memory goal: 26.5 GiB (the use of K 208), or the stricter 25 GiB (a margin of 4 GiB with no change to DeepSeek)?
5. May the work read the Qwen3.5 MoE code of vLLM and transformers, to confirm the MTP wiring?

## 10. After M5: the decision on DeepSeek

This plan does not port DeepSeek. M5 gives the two numbers that the decision needs: the fixed memory of a colinfer
process, and the speed of colinfer against vLLM on a MoE model with a draft.

| Option | Memory against two vLLM servers (estimate) | Work |
|---|---|---|
| Stop here: Qwen in colinfer, DeepSeek in vLLM | The gain of this plan: 0.4 to 2.7 GiB before the trims of Phase 0 | None |
| DeepSeek in its own colinfer process | 3.5 to 5 GiB more. Most of it is the memory of lever D4 of the fit plan | The port below |
| Both models in one colinfer process | About 9 to 11 GiB in all: the process pays its fixed memory once | The port, and two loaded families in one server |

The port needs, at least:

- EXL3 kernels for decode and prefill (trellis codes with sign vectors);
- FP8 with a scale per 128 × 128 block, which the engine cannot serve today;
- the attention of DeepSeek V4: low-rank projections, the compressor, the indexer and its 432-byte KV records;
- the hash routing of layers 0 to 2, and a different expert count in each layer;
- the DSpark drafter with its Markov head, and the DSML tool-call format;
- `prompt_logprobs` in the API, for the replay gate of `dsv4-specialize`.

It is about 1.5 to 2 times the plan of Flash-Next, and its speed against the present kernels is not known. If you
choose it, it gets its own plan. A separate engine for DeepSeek saves no more memory than a second colinfer process,
and it pays the fixed memory a second time.

## 11. Status log

- **2026-11-14, Phase 0 step 1.** `tools/mem_floor.py` and its test are written, and the tool ran five times beside
  the DeepSeek server (§5.1). A process with no model takes about 2.5 GB. Step 2 needs a window.
- **2026-11-14.** The plan is written. The checkpoint is on disk (5 shards, 22.0 GiB). Its config, tensor list and
  scales are read (§2). Nothing is built.
