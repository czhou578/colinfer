# Common brief: Qwen3.8-Flash-Next (qwen4_exp) support in colinfer

Date: 2026-10-24. Branch: moe. Repo: /home/colin-spark/Projects/colin-inference-engine (read README.md and docs/architecture.md first,
then the files of your area). Python: `uv run ...` from the repo root (torch 2.14.1+cu130, transformers 5.18.0, CUTLASS 4.8.0 submodule
in csrc/third_party/cutlass, FlashInfer 0.7). Tests: `uv run pytest tests/<file> -q`. Lint: ruff, line length 150.

## The machine and the engine today
- One DGX Spark: GB10, sm_121a, 48 SMs, 128 GB unified memory shared by CPU and GPU, ~238 GB/s. An over-commit of the unified memory
  POWERS OFF the machine (no OOM error). MemAvailable is 122 GB now; the server is stopped. Never allocate near the limit.
- The engine serves nvidia/Qwen3.8-27B-NVFP4 (dense, 64 layers = 48 Gated DeltaNet + 16 gated attention, hidden 5120). Decode is one
  CUDA graph per step / speculative cycle (fixed shapes, no host sync inside), all decode linears run on the tensor-core skinny GEMM
  (csrc/skinny.cu, 1..16 rows per weight pass, each output row bit-identical for any row count: this is what makes speculation exact).
  Prefill runs 2048-token chunks on CUTLASS SM120 NVFP4 GEMMs (csrc/gemm_nvfp4.cu) and cuBLASLt FP8. One request at a time, 3 slots
  with contiguous KV, a ring of GDN-state checkpoints, a conversation cache on NVMe (engine/runtime/sessions.py), MTP speculation
  (engine/spec/mtp.py). Any quantization beyond the checkpoint's own must pass a perplexity gate of <= 0.5% (tests/perplexity.py).
- Code map: engine/model/qwen35.py (plain-PyTorch reference, op for op like transformers), engine/model/fast.py (decode on kernels),
  engine/model/prefill.py, engine/weights/{loader,direct,quantize}.py, engine/runtime/{build,scheduler,sessions,decode,prepared}.py,
  engine/spec/mtp.py, engine/selftest.py, csrc/*.cu + bindings.cpp + shapes.h (HEAD_DIM 256, GDN_DK 128, GDN_DV 128), tests/.
- Writing rule for docs and comments: short sentences, active voice, present tense (see /home/colin-spark/.claude/CLAUDE.md).

## The target: Qwen/Qwen3.8-Flash-Next, served from nvidia/Qwen3.8-Flash-Next-NVFP4
- model_type qwen4_exp, class Qwen4ExpForConditionalGeneration. transformers 5.18.0 has it:
  .venv/lib/python3.12/site-packages/transformers/models/qwen4_exp/{modeling_qwen4_exp.py (2609 lines), modular_qwen4_exp.py (1151,
  readable), configuration_qwen4_exp.py}. The indexer key cache is in transformers/cache_utils.py (~lines 340-420 and 650-680).
- Configs: the nvidia config.json and the tensor summary are in docs/history/qwen4exp/.
  Tensor names / dtypes / shapes of the NVFP4 checkpoint: docs/history/qwen4exp/nvidia_tensors.txt (index patterns, then the safetensors
  headers of 4 files). hf_quant_config.json: docs/history/qwen4exp/nvidia.hf_quant_config.json. <scratchpad> =
  /tmp/claude-1000/-home-colin-spark-Projects-colin-inference-engine/e772124d-8725-4782-a45d-65987527a4c2/scratchpad
- The checkpoint (132.7 GB, 10 shards + model-fp8-mtp-ple.safetensors 53.7 GB) is downloading now into ~/.cache/huggingface/hub
  (models--nvidia--Qwen3.8-Flash-Next-NVFP4). Do not start another download. It is not usable until the download ends.
- Architecture (125B total, 6B active, + 51B n-gram table + 4B MTP):
  - hidden 2560, 48 layers, layer_types: 12 x (3 linear_attention, 1 full_attention). vocab 248320, untied lm_head (BF16 in the
    checkpoint), native context 262144, rope theta 1e7, partial_rotary_factor 0.25 (rotary dim 64), mrope_section [11,11,10]
    interleaved (images). NO per-layer input/post-attention RMSNorm and NO final norm: the hyper-connection norms replace them.
  - Hyper-connections (hc_count 4, hc_lowrank 320): the residual stream is 4 x 2560 = 10240 wide. Before each sublayer a
    GatedResidual: grouped zero-centered RMSNorm per stream (hc_norm, weight [10240]), mix = sigmoid(up(silu(down(normed) / 4)))
    with down [320, 10240], up [10240, 320]; sublayer input = mean over the 4 streams of (mix * normed); injection weights =
    2 * sigmoid(block_inject_weight [4, 10240] . normed / 4); after the sublayer: stream += out * inj[stream]. The model ends with
    hyper_connection_mixer (same, use_combine False) -> [2560] -> lm_head. The token embedding is repeated 4 times at the start.
  - Gated DeltaNet: 48 value heads, 16 key heads, dims 128/128 (same as the 27B: the GDN kernels can be reused), conv kernel 4,
    in_proj_qkv [10240, 2560], in_proj_z [6144, 2560], in_proj_b / in_proj_a [48, 2560], out_proj [2560, 6144]. DIFFERENCE: the gated
    output norm uses SIGMOID(z) (output_gate_type "sigmoid"), not silu(z).
  - Attention (12 layers): 24 q heads over 2 KV heads (GQA 24/2; the 27B is 24/4), head_dim 256, q_proj [12288, 2560] = (q | gate)
    per head, k/v_proj [512, 2560], o_proj [2560, 6144], q/k zero-centered RMSNorm, output gate sigmoid. Plus Qwen Sparse Attention
    (QSA): an indexer index_qk_proj [640, 2560] = 4 query heads + 1 key head of 128 dims, q_layernorm / k_layernorm [128], budget
    2048 tokens, compress ratio 4: raw keys of every token are cached; complete 4-token blocks are mean-pooled in fp32, normed,
    roped at the block's first position; score(block) = sum over the 4 heads of relu(q_h . k_block) / sqrt(128); top-512 blocks
    plus the tail (the < 4 tokens of the incomplete block) are the tokens the query attends to. With <= 2048 + 3 visible tokens
    every token is selected (dense).
  - MoE (every layer): router gate [512, 2560] bf16 -> softmax (fp32) -> top-10 -> renormalize to sum 1 -> bf16 weights; 512
    experts of SwiGLU with intermediate 640 (gate_proj / up_proj [640, 2560], down_proj [2560, 640]); plus one shared expert
    (intermediate 640) scaled by sigmoid(shared_expert_gate [1, 2560] . x). out = sum_k w_k expert_k(x) + gate * shared(x).
  - PLE (Per-Layer Embedding) at layer index 1 (ple_layer_ids [2], one-indexed) only, applied to the 10240-wide stream before the
    layer: 16 hashed n-gram heads (8 bigram + 8 trigram) over the token and its 1-2 predecessors, each head a prime-sized vocab
    (~20M rows) with 160 dims -> concatenated to 2560; key_proj [10240, 2560], value_proj [2560, 2560], norms [10240], a gate
    sign(g) sqrt(|g|) with g = (key_normed . query_normed) / sqrt(2560) per stream, sigmoid, a dilated depthwise conv (kernel 4,
    dilation 3 -> a 9-token state) with silu, added. The n-gram context (the 2 previous token ids) resets at eos (248044)
    segment starts. The hashing uses splitmix64 multipliers and int64 wraparound (modeling_qwen4_exp.py lines 1040-1180).
  - MTP: mtp.* tensors = fc_embedding, fc_hidden, pre_fc_norm_embedding [2560], pre_fc_norm_hidden [10240], one full decoder layer
    (QSA attention + MoE with FP8 block-scaled experts + hyper-connections) and a hyper_connection_mixer. transformers ignores it;
    the engine's own drafter (engine/spec/mtp.py) would use it.
  - Vision tower model.visual.* (27 blocks, out 2560), BF16; images are a later phase.
- Checkpoint storage (nvidia_tensors.txt): routed experts NVFP4 per expert: weight U8 [N, K/2], weight_scale e4m3 [N, K/16],
  weight_scale_2 f32 scalar, input_scale f32 scalar (one per expert per projection). Everything else BF16 (attention, GDN, shared
  experts, routers, hyper-connections, PLE projections, embeddings, lm_head, vision). The n-gram table: 128 shards
  layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight F8_E4M3 [2500012, 160] with ONE bf16 weight_scale, all in
  model-fp8-mtp-ple.safetensors (51.2 GB). MTP experts: F8_E4M3 with weight_scale_inv bf16 [5, 20] / [20, 5] (128 x 128 blocks).
- Bytes: routed experts 67.8 GB in NVFP4 (1.41 GB per layer, 2.76 MB per expert); the BF16 rest about 10 GB (GDN 4.2, attention 1.2,
  hyper-connections 1.3, shared experts + routers 0.6, embeddings 1.27, lm_head 1.27, PLE projections 0.07). KV cache: 12 layers x
  2 heads x 256 x 2 (k, v) x 1 byte fp8 = 12 KB per token (the 27B: 32 KB), plus the raw indexer keys. GDN state per slot: 36
  layers x 48 x 128 x 128 x 4 B = 113 MB. A plain decode token streams about 1.33 GB of experts + the non-expert weights it reads
  (about 8.6 GB in BF16; much less if requantized like the 27B's INT6 / INT5 copies) + 16 n-gram rows.
- Public reference point: a vLLM build of this checkpoint on one DGX Spark decodes 16.9 tok/s single-stream, prefill 12.3 s at 25k
  tokens (MagneticLab/Qwen3.8-Flash-Next-NVFP4 model card).

## Design rules that your plan must respect
1. Exactness: the decode path, the verify path (k + 1 rows) and prefill must give the same tokens for the same input (bit-identical
   rows across row counts for every decode kernel). Expert routing must therefore be computed per row and not depend on the other
   rows in the launch.
2. CUDA graphs: a decode step / speculative cycle has fixed shapes and no host synchronization. Data-dependent work (which experts,
   which blocks) is scheduled on the device.
3. Memory: budget every tensor; the whole engine (weights, decode copies, KV of all slots, states, drafter, graph pool, page cache
   for the n-gram table) must stay well under 128 GB with margin. Prefer designs that need fewer bytes resident.
4. Quantization beyond the checkpoint's own (e.g. INT6 copies of BF16 projections, 4-bit n-gram rows) is an option only with the
   perplexity gate (<= 0.5%); the exact path comes first, the faster copy second (as the 27B's prepared directory does).
5. Reuse before new code: GDN kernels, skinny GEMM, norms, prologue, sessions, scheduler. New kernels follow the style of csrc/ (PDL,
   shape checks in bindings.cpp, a test file per kernel, memcheck / racecheck per .claude/rules/sanitizers.md).
6. Proposed module names (change them only with a reason): engine/model/qwen4exp.py (reference model: Qwen4ExpConfig,
   Qwen4ExpState, Qwen4ExpForCausalLM), engine/model/fast_qwen4exp.py (decode on kernels), engine/model/prefill_qwen4exp.py,
   engine/weights/ple.py (n-gram table rows from disk), csrc/moe.cu (routing + expert skinny GEMMs), csrc/moe_prefill.cu (grouped
   GEMM, gather / scatter), csrc/qsa.cu (indexer scoring, top-k, sparse decode attention), csrc/hc.cu (hyper-connection kernels),
   tests/test_qwen4exp_small.py, tests/test_moe.py, tests/test_moe_prefill.py, tests/test_qsa.py, tests/test_ple.py,
   tests/test_hc.py.

## Rules for this planning task
- Read-only on the repository. Write only under <scratchpad>. Do not download anything. Do not load a model on the GPU. Small GPU
  probes (a few hundred MB, a few seconds) are fine, e.g. to time a kernel idea or check a CUTLASS instantiation compiles.
- Be concrete: file names, function signatures, tensor shapes and dtypes, kernel grid design, the test that proves each piece,
  estimated hours. Say what you verified by reading code and what is a guess.
- End with: decisions (with your recommendation and the alternative), risks, open questions for the user (only the ones that change
  the work materially).
