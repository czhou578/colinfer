# Plan 3: hyper-connections, QSA, PLE, GDN delta, MTP shape, serving integration (planner 3, Opus, 2026-10-24) — saved by the lead

## Verified facts
- The chat template is byte-identical to the 27B's (same md5): chat.py and --reasoning-effort need no change. eos 248044 is
  <|endoftext|>, not <|im_end|> (248046): the PLE n-gram context resets only on <|endoftext|>, which chats rarely contain.
- torch.topk on CUDA breaks ties toward the lowest index (probed at sizes 600 to 65536). Ties are common: blocks whose 4 dot
  products are all <= 0 score exactly 0. Rule: (score descending, index ascending); -0.0 is first turned into +0.0.
- numpy int64 wraparound, XOR and % equal torch's on CPU and CUDA (100k random triples): the host can compute n-gram ids.
- A row at position p sees all tokens iff p + 1 <= 2051. HF's rope uses bf16 cos/sin. Image positions hash as <|image_pad|> (248056).

## Key decisions
1. Hyper-connections: kernels hc_norm (norm.cu with a weight that cycles per stream); down + inject as one stacked skinny pass over
   [328, 10240]; hc_mix = the up product (K = 320, too small for the skinny GEMM), sigmoid and 4-stream mean in one kernel; hc_post
   as an epilogue mode of the skinny GEMM: x4[:, s] = bf16(x4 + bf16(bf16(acc) * inj_s)); the MoE combine needs the same epilogue.
   Bytes: 1.28 GB per token in BF16 (as much as the experts). Whole BF16 decode stream ~9.9 GB -> ~22 tok/s at best; with INT6/INT5
   copies of GDN, attention, HC and shared experts ~4.5 GB -> ~50 tok/s. Copies needed for speed, under the perplexity gate.
   Speculation requires row-invariant kernels for every reduction (torch reductions may change bits with the row count).
2. QSA: per slot and layer keep raw indexer keys bf16 [Lmax, 128] (3 KB/token over 12 layers) and a derived block-key cache
   [Lmax/4, 128] (0.75 KB/token). Block keys are computed with the same device function at prefill, decode and verify when a block's
   4th token is processed. Sessions save only raw keys; block keys are rebuilt after a restore (~3 ms at 128k). Decode and verify
   ALWAYS use one sparse kernel (qsa_attn), with all positions in the list for short rows (a per-launch dense choice would make a
   row's bits depend on its neighbours). Cost ~2.1 MB of KV per token row per layer at any context; T = 8 verify: 17 MB per layer,
   200 MB per cycle. Prefill staged: M1 FA2 causal (prompt <= 2051); M2a FA2 with a custom mask from the shared indexer (exact, dense
   cost); M2b a custom FP8 kernel over the union of the selected blocks of each 64-row tile. FlashInfer's block-sparse wrapper
   (R = 1, C = 4) rejected: no K/V reuse across rows (up to 8.6 GB per layer-chunk), a host-side plan per layer, a bf16 copy of the
   cache.
3. PLE: the host computes the ids and reads 16 rows of 160 B per token through a C++ pread thread pool (page cache, readahead off).
   Decode: after each cycle the host reads [y', drafts] (the graph already wrote them), gathers 16 (k+1) rows, copies them into the
   graph's fixed buffer before the replay: 0.05-0.5 ms per cycle, worst ~1 ms when all rows are cold. Prefill: the next chunk's
   rows are gathered while the GPU runs the current chunk. The n-gram context needs no stored state (a function of the slot
   history). The PLE conv state [10240, 9] joins the ring and the checkpoint files.
4. GDN: a runtime gate_act (0 silu, 1 sigmoid) in gdn_delta and gated_rmsnorm. Everything else identical to the 27B.
5. MTP: the shapes of fc_embedding / fc_hidden are missing (shard 00009's header is not in the tensor list). pre_fc_norm_hidden
   [10240] implies the drafter takes the target's 4-stream x4 before the final mixer. No mtp.norm: the MTP mixer output goes
   straight to lm_head. ~50 h.

## Risks
- Cold rows of the 51 GB n-gram table slow a cycle (hit rate unmeasured). MTP semantics may need reverse engineering (a wrong guess
  costs acceptance only). MoE verify cost grows with distinct experts. HF agreement on top-k boundary ties: measure later. HC prefill
  elementwise passes ~150 ms per chunk until fused.

## Questions for the user
- Web access to read vLLM PR #55513 / SGLang's qwen4_exp MTP code? Default context 262144 (no YaRN)? INT copies of the HC matrices
  under the gate?

## Interfaces needed
- Planner 1: skinny_bf16 (row-invariant, HC epilogue mode); stacked BF16 linears q|k|v|index = 13952 rows and GDN qkv|z; reference
  modules (GatedResidual, a QSA indexer with torch.topk semantics, PLE taking a row source); PleTable row -> (shard r // 2500012,
  row r % 2500012) (the contiguous split is unverified); memory lines for the state in F.
- Planner 2: moe(x[M,2560]) at decode (row-invariant, M <= 16) and prefill, with an optional hc = (x4, inj) epilogue; a cycle-cost
  model for k; the MTP FP8-block experts as NVFP4 copies.

---------------------------------------------------------------------------------------------------------------------------------
## A. Hyper-connections (csrc/hc.cu, tests/test_hc.py)
Dtype at each op (HF rounding; weights bf16; every bf16 op computes in fp32 and rounds once):
1. n = bf16(x_s * rsqrt(mean_s(x^2) + eps) * (1 + w)), fp32 per 2560-wide stream (same formula as k_rmsnorm).
2. d = bf16(n W_down^T) [M, 320]; e = bf16(n W_inj^T) [M, 4], fp32 accumulation.
3. a = bf16(silu(bf16(d / 4))) (precise expf).
4. u = bf16(a W_up^T) [M, 10240].
5. mix = bf16(sigmoid(u)).
6. p_s = bf16(mix_s * n_s).
7. y = bf16((((p0 + p1) + p2) + p3) / 4).
8. inj_s = 2 * bf16(sigmoid(e_s / 4)).
9. After the sublayer: o = bf16(acc); x4_s = bf16(x4_s + bf16(o * inj_s)).
The final mixer is steps 1-7, then lm_head. No norm.
Decode kernels (M <= 16, row-invariant):
- K1 hc_norm: launch_rmsnorm with a w_rows argument; block r uses weight row r % 4 of the [4, 2560] weight; input viewed [M*4, 2560].
- K2 down + inject: W_di = [W_down; W_inj; 4 zero rows] = [328, 10240] on the skinny GEMM (BF16 first, INT6 later; K % 512 ok).
  bf16_gemv does not fit (M <= 8: a 16-row verify would stream twice; for W_up 216 of 256 threads idle).
- K3 hc_mix: 160 blocks x 128 threads; block c0 owns columns c0..c0+15 of all 4 streams (4 contiguous runs of 16 x 640 B);
  recomputes a [M, 320] into smem (10 KB); warp s handles stream s; 8 lanes per W_up row load five 16-byte chunks; a fixed
  xor-shuffle tree reduces to u; then mix, p into smem, sync, y [M, 2560]; block 0 writes inj [M, 4]. Reads 6.5 MB, ~30 us.
- K4 hc_post as a skinny epilogue: pass hc_x4 and hc_inj instead of residual; the final reduction item does v = bf16(acc * scale);
  for s in 0..3: x4[m, s N + n] = bf16(x4[m, s N + n] + bf16(v * inj[m, s])). Updates x4 in place, writes no y. o_proj and out_proj
  use it; the MoE combine needs the same. Fallback: a standalone hc_post kernel (~10 us).
Bytes: per GatedResidual 6.55 + 6.55 + 0.08 + 0.02 = 13.2 MB; over 96 + the mixer: 1.28 GB per token (~5.6 ms at 228 GB/s).
Whole decode stream in BF16: experts 1.33, GDN 4.15, attention 1.23, HC 1.28, shared + routers 0.6, lm_head 1.27, PLE 0.07: ~9.9 GB,
~44 ms (22 tok/s). With copies: GDN INT5 1.43, attention INT6 0.5, HC INT6 ~0.55, shared INT6 0.2, routers BF16 0.12, lm_head NVFP4
or INT6 0.36-0.52: ~4.5 GB, ~20 ms. HC copies save ~0.73 GB per token (~16% of a step); W_up (K = 320) needs a low-bit format inside
hc_mix (INT8 per row or INT6 planes); padding K to 512 for skinny_int saves only 35%. The perplexity gate covers all copies together.
Prefill: M1/M2 torch ops on [T, 10240] (cuBLAS bf16 matmuls ~27 GFLOP per HC, ~37 ms per chunk; each elementwise pass moves 42 MB,
~34 GB per chunk, ~150 ms). Later fused: hc_norm_prefill, hc_mix_prefill, hc_post_prefill. Chunk scratch 3 x 42 MB.
Tests: vs the reference GatedResidual (equal bits where the rounding points allow, else <= 1 ulp); rows bit-identical for M = 1..16;
epilogue vs torch; memcheck + racecheck.

## B. QSA (csrc/qsa.cu, tests/test_qsa.py)
Reference semantics (lines 672-902): qk = index_qk_proj(x), 640 rows = 4 q heads of 128 then 1 key of 128; q: q_layernorm (zero-
centered) then rope on the first 64 dims (halves of 32) with the same mrope cos/sin as attention; raw keys cached WITHOUT norm or
rope; blocks aligned at absolute positions 4j..4j+3; pooled = bf16(fp32 mean), then k_layernorm, then rope at the block's first
token; score = sum_h relu(q_h . k_b) / sqrt(128) in fp32; selection top-min(512, nb) plus the tail (p + 1) % 4 tokens (a row's own
token is attended only if in the tail or its block wins).
State per slot and layer: KV fp8 [2, Lmax, 256]; idx_raw bf16 [Lmax, 128]; idx_blk bf16 [Lmax/4, 128] (the decode indexer reads
16.8 MB per layer at 262k vs 4x from raw keys). Block j is (re)written whenever a row at position 4j + 3 is processed; rejected
verify rows leave stale blocks only at j >= committed/4, rewritten before any row reads them.
Decode / verify kernels (T <= 16 per slot):
- Stacked projection q 12288 | k 512 | v 512 | index 640 = 13952 rows in one weight pass (wanted).
- attn_prologue works unchanged for 24/2 (grid y = Hq + 2 Hkv).
- qsa_prologue: grid (B T, 5) x 128 threads; four q heads get norm + rope into q_idx [B, T, 4, 128]; the key is written raw to
  idx_raw[b, pos_t + t] when the slot is active; reuse the rope helper with sec_h / sec_w (move to common.cuh).
- qsa_block_keys: for blocks with 4j + 3 in [pos, pos + T): sequential fp32 sum, /4, bf16, norm, rope at the rope position of token
  4j (pos + rope_delta, or an explicit position array) into idx_blk. The same function serves prefill and the restore rebuild.
- qsa_score: grid (B, Lmax/4/256), blocks past nb exit; A = q_idx rows (T*4 <= 64, bf16), B = 256 block keys via cp.async; mma
  m16n8k16 fp32; score[t, j] = (((r0 + r1) + r2) + r3) / sqrt(128), r = relu canonicalized to +0, -inf for j >= (p_t + 1) // 4;
  writes scores [B, 16, Lmax/4] fp32 (4 MB per slot). Rows of the mma are independent: bits do not depend on T.
- qsa_topk: one 1024-thread CTA per (b, t); nb <= 512 -> all blocks; else a 4-pass 8-bit radix select on ordered-uint keys gives thr
  and count_gt; an ordered compaction in index order keeps a block if key > thr, or key == thr and its rank among equals is
  < 512 - count_gt; output sel int32 [B, 16, 512] ascending + nsel; ~30 us per layer.
- qsa_attn (adapted from attn_decode.cu k_tc): grid (B T, Hkv = 2, NBS = 24), 128 threads; rows = the G = 12 q heads of one KV head
  for one token (MT = 1, 4 tile rows unused); logical key i -> position 4 sel[i/4] + i % 4 for i < 4 nsel, else the tail; tiles of
  32 keys (8 groups of 4 contiguous rows, 1 KB runs); stripe z owns tiles z, z + 24, ...; then QK^T / online softmax / PV as today;
  k_combine with splits = 24 and the output gate. Today's GQA dependence: the G template (only G = 6 compiled, line 409), MAXROWS 48
  = G x T, NB = 12 sized for 4 KV heads (with 2 KV heads NB must be 24 to fill 48 SMs at T = 1). Bits identical between decode and
  verify (the list is a function of the row alone). Cost 2051 x 2 x 256 x 2 B ~ 2.1 MB per row per layer; T = 8: 17 MB per layer,
  ~80 us; 200 MB per cycle.
Prefill (chunk 2048 at prefix P): qsa_prologue + qsa_block_keys; score in groups of 256 rows (scores [256, nb], 64 MB at full
context) then qsa_topk (same score code as decode -> prefill selects exactly what decode would); cost at 262k ~1 GB of score
traffic per layer-chunk (4.5 ms), ~54 ms per chunk over 12 layers (a fused top-k later). Attention: (iii) now dense FA2: plain
causal if P + T <= 2051, else single_prefill_with_kv_cache(custom_mask = packed selection bits) (exact at any length, dense cost,
~80 ms per layer at 128k); (i) FlashInfer BSR rejected (see above); (ii) M2b custom kernel: per 64-row tile the union of the
selected blocks (bitmap over nb in smem, scan), FP8 QK^T as attn_prefill.cu over the union tiles, per-row membership from a merge-
walk pointer into each row's sorted list; cost scales with the union size (measure). MTP: the drafter's attention layer has its own
indexer -> the same caches and kernels.
Tests: score + selection vs torch (same fp32 scores give the same set as torch.topk, ties included); T = 1 vs T = 16 bit identity;
qsa_attn vs SDPA with the explicit mask; a dense list vs attn_decode within tolerance; block keys vs the reference pooling; sanitizers.

## C. PLE (engine/weights/ple.py, the PLE layer in fast_qwen4exp.py, tests/test_ple.py)
Host: ctx_s(p) = tok[p - s] if p - s >= 0 and tok[p - s + 1 .. p - 1] holds no eos, else eos (248044) (= HF's chunked
_shift_right_ignore_eos). Bigram ids (t0 m0 ^ t1 m1) % size_h + off_h for heads 0..7; trigram adds ^ t2 m2 for heads 8..15; int64
wraparound; image tokens use 248056. Reader: PleTable.gather(ids int64 [n, 16], out pinned u8 [n, 16, 160]) through a C++ pread
pool in bindings, fadvise RANDOM; no O_DIRECT (the page cache is the hot-row LRU). Prefill: gather the first chunk at admission;
gather each next chunk where _prefill_chunk already overlaps host work (scheduler.py ~542-543). Decode: after the replay and event
sync in _decode (~590-597), read self.tok[b, :kmax + 1], hash, gather, copy non-blocking into the graph's buffer [kmax + 1, 16, 160]
before the next replay; plain decode the same with nxt; hook model.feed(slot, prev2, rows). Later: the graph starts layer 0 while
the host gathers, then waits on a host flag; track a cold-row metric.
Device math: emb = bf16(fp8 * scale) (exact product = torch's bf16 multiply); kp | vp: one stacked BF16 pass over [12800, 2560]
(65.5 MB); kn = grouped norm(kp); qn = grouped norm(x4); g = bf16(sum bf16(kn * qn)); g = bf16(g / sqrt(2560));
g = bf16(sqrt(bf16(max(|g|, bf16(1e-6))))) * sign(g); gv = bf16(bf16(sigmoid(g)) * vp) [4, 2560]; u = norm_conv(gv);
c[t] = bf16(sum_{j=0..3} w_j u[t - 3 (3 - j)]) over the 9-row state followed by the new rows; act = bf16(silu(c));
x4 = bf16(x4 + bf16(gv + act)). M1-M2 torch ops; M3 needs kernels for the reductions (ple_gate, ple_conv / ple_conv_commit, the
same verify/commit split as gdn_conv; commit inside the graph: state = gather(cat(state, u), n + arange(9))).
State: the conv state [B, 10240, 9] bf16 in the ring and checkpoint files; the n-gram context derived from the slot's history.

## D. GDN
Runtime int gate_act in launch_gdn_delta (gdn_step.cu ~190) and launch_gated_rmsnorm (prefill_ops.cu ~125) + bindings, default 0
(silu), 1 sigmoid. The rest identical: 48/16/128, conv 4, A_log / dt_bias, L2 norms, repeat_interleave, the norm weight used
without (1 + w), eps 1e-6. KernelGDN.forward takes the already-normed x and a residual; here x is the HC output [M, 2560] and
out_proj gets the HC epilogue. Needs StackedBf16Linear for qkv | z. ~3 h.

## E. MTP block
Missing: shapes of fc_embedding / fc_hidden. pre_fc_norm_embedding [2560] on embed(x_{i+1}); pre_fc_norm_hidden [10240] on the
target's x4 before hyper_connection_mixer. Output: attention + indexer + MoE (FP8 128x128 block experts) + 2 HCs, then its own
mixer, straight to lm_head (no mtp.norm). Hypotheses: fc_* output 10240 directly (fc_hidden [10240, 10240]) or 2560 then repeated 4x;
decide by the header, then by acceptance rate. Implications: the hidden saved for the drafter (h_last, ring_h, sessions h_dim) becomes
10240 wide; the drafter gets its own KV + idx_raw + idx_blk, NVFP4 expert copies (1.4 GB), an NVFP4 or low-rank draft head (lm_head
is BF16); drafter block keys rebuilt on restore (the one-row shift breaks the 4-alignment). ~50 h (M3).

## F. Serving integration: what assumes the dense model
- engine/model/fast.py: FastState.view (256-263) and ModelState.reset (qwen35.py 199-205) handle only k/v/conv/rec;
  KernelAttention.forward (288-297); FastDecoderLayer (380-389) norms + residual; FastQwen35.forward/verify/commit (400-428);
  to_fast (431-451); attach_decode_copies (464-494); load_fast_model refuses plain fp8 (201-202): the PLE shards must be skipped
  (loader.SKIP_PREFIXES, loader.py 28).
- csrc/attn_decode.cu 409: only G == 6.
- scheduler.py: the ring from state.rec/conv (~241-246) and ring_h of hidden_size; _snapshot (284-300), _copy_kv (317-324),
  _restore (337-348), zeroing in _admit (461-463); _pick_k: ctx_s from state.k, ATTN_ROWS, q_per_kv (86, 230-237, 254-271) (QSA
  attention is bounded at 2051 keys; the indexer grows with L/4); the PLE hook in _prefill_chunk and _decode.
- sessions.py: shapes + fingerprint (119), row bytes (128), gdn_bytes (129), h_dim (130), own_bufs (136), _kv (241), _gdn (253);
  the fingerprint must cover the PLE file.
- build.py: startup_bytes builds Qwen35Config / FastState (121-127); ENGINE_FIXED_BYTES 40e9 is the 27B's; load_model (YaRN,
  to_fast, INT copies).
- decode.py: the PLE hook only. selftest.py: add skinny_bf16, hc_mix, qsa_score/topk, qsa_attn. vision.py 116-134 Qwen3_5Vision*
  (later). worker.py 39: max_seq_len 524288 (YaRN) -> default 262144 here. chat.py: no change. gateway matches "qwen": no change.
Smallest generalization (M4): FastState gets extra: dict[str, dict[int, Tensor]] and slot_tensors(b) (conv, rec, ple_conv) /
seq_views(b, r0, r1) (k, v, idx_raw: one row per position; idx_blk rebuilt or copied as [r0/4, r1/4)); ring, _copy_kv and sessions
iterate over these; startup_bytes asks the model family for its state shapes; the model exposes feed() (PLE host step),
drafter_hidden_dim and ctx_cost(L, rows).
Memory (3 slots x 262144): KV 9.66 GB; idx_raw 2.42; idx_blk 0.60; GDN + PLE state 0.35; ring 32 x 115.6 MB = 3.7; drafter caches
~1.0; scratch ~0.2; the n-gram page cache reclaimable, not counted.

## G. Work breakdown (hours)
M1 (~62 h): golden greedy decode on 3 prompts of <= 2051 tokens, 1 slot, no speculation, vs planner 1's reference (HF cannot load
125B here): GDN gate flag 3; G = 12 instantiation + raw-key write + qsa_prologue 8; HC in torch ops 4; PLE host side 10 (hash parity
with HF NGramEmbedding, eos, chunked = whole); PLE device side in torch ops 6; FastQwen4Exp decode model + state 12; prefill pieces
(HC, PLE, attention, GDN) 9 (prefill = decode tokens); integration + debugging 10 (golden_qwen4exp).
M2 (~118 h): qsa_score + qsa_topk 16; qsa_attn 20; qsa_block_keys 8; prefill indexer + FA2 with mask 8; union-sparse prefill kernel
24; HC decode kernels + skinny epilogue 14; HC prefill fusion 10; C++ gather pool + chunk overlap 8; 3 slots 4; passkey / long-context
perplexity 6.
M3 (~74 h): row-invariant verify/commit (HC, PLE kernels, conv commit) 12; per-cycle PLE host step 8; MTP drafter 50; k cost hook 4.
Proof: golden with speculation on and off is token-identical.
M4 (~26 h): state generalization + ring 8; sessions.py + stub tests 12; scheduler_check on this model 6.
M5 (~36 h): build.py family dispatch, selftest, flags/defaults, docs 20; INT copies of HC/GDN/attention/shared experts + perplexity
gate 16.

## H. Decisions table
| Topic | Recommended | Alternative |
|---|---|---|
| Block keys | Derived cache + raw keys | Raw keys only (4x indexer bytes) |
| Sessions | Save raw keys, rebuild block keys | Also save block keys |
| Decode attention | Always sparse | Dense kernel for short rows (breaks bit identity) |
| Prefill attention | FA2 + mask, then the union kernel | FlashInfer BSR |
| W_up | Custom hc_mix | Pad K to 512 |
| PLE | Host gather with a page-cache LRU | A resident table (51 GB does not fit) |
| HC/PLE in M1 | torch ops, kernels by M3 | Kernels from the start |
Critical files: csrc/attn_decode.cu, csrc/skinny.cu, engine/runtime/scheduler.py, engine/runtime/sessions.py, modeling_qwen4_exp.py.
