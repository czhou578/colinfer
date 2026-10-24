# Plan 1: reference model, loader, memory (planner 1, Opus, 2026-10-24) — saved by the lead

## Key decisions
1. New module engine/model/qwen4exp.py. Qwen4ExpConfig subclasses Qwen35Config (rotary, rope_inv_freq, mrope_axes, l2norm, both delta
   rules, apply_rotary, RMSNorm reused). GatedDeltaNet and Attention are subclassed (sigmoid gated norm; the indexer). Copy only the
   grouped RMSNorm, GatedResidual, MoE, PLE and QSA code. qwen35.py unchanged. "indexed_attention" maps to "full_attention".
2. Expert storage per layer and projection: [E,N,K] dense in the tiny test, or NVFP4 w [E,N,K/2], sf [E,N,K/16], gs [E], in_scale [E];
   get(e) dequantizes one expert. The forward is the HF eager loop (ascending expert id, bf16 index_add_). The tiny test maps HF
   gate_up_proj = cat(gate, up). At full size the same reference runs NVFP4 W4A16 in 78 GB.
3. Loader: stack at load time. Add read_tensors(..., into={name: preallocated view}) to direct.py (~10 lines): each expert's bytes
   land in their stacked slot: no 67.8 GB of copies, ~8.6 s for 77.7 GB at 9 GB/s, 384 MB pinned staging. The prepared directory
   keeps only the decode copies.
4. N-gram table stays on disk. NgramTable reads rows with O_DIRECT through a preadv thread pool (no liburing header; io_uring out).
   O_DIRECT keeps the page cache empty (buffered reads would cache 4 KB per 160 B row, 25x). A bounded device row cache later,
   sized by tools/ple_hotset.py (runs on CPU today). The host hashes each row from the slot's token history: the fast state holds no
   n-gram context; the PLE's only device state is the zero-initialised conv window.
5. Memory default: 2 slots x 262144, 16 checkpoints. Decode milestone ~94 GB + 6 GB reserve; final (copies, drafter, 2 GB row
   cache) ~109 GB.
6. Family dispatch by config.json model_type. Two duck-typed state hooks, kv_tensors() and snap_tensors(), replace the direct
   .k/.v/.conv/.rec loops in the scheduler and sessions; the 27B's order and fingerprint stay byte-identical.

## Facts checked
- No final norm (code and tensor list). Padded n-gram vocab = 320,001,536 = 128 x 2,500,012 exactly (primes 20000003..20000171,
  total 320,001,446). Multipliers for vocab 248320 / seed 1234: [23703573157769, 20109073645365, 8052911324071] (< 2^45.07):
  token x mult < 2^63, no wraparound, XOR stays non-negative; torch.remainder == Python %.
- transformers concatenates the shards in natural numeric order. A tiny HF model runs on CPU in 0.27 s; split prefill matches
  one-shot to 6e-8. HF keeps the GDN conv state as K = 4 columns (math identical). Cache: conv_states {0: [1,128,4], 1: [1,256,9],
  2: [1,2] int64}; indexer_keys [1,70,16]; position_ids stored on the cache as [3,1,70].
- NVMe random 4 KB O_DIRECT reads (Python threads, during the download): batch 1 = 0.3 ms, 16 = 1.1 ms, 64 = 2.4 ms; 32768 reads
  = 0.57 s (~57k IOPS). No liburing header; io_uring enabled in the kernel.
- Guesses: MTP fc_* shapes, hot-set size, MoE kernel efficiency, graph pool size.

## Risks
- QSA top-k near-ties can flip between split and one-shot runs (tests assert the masks; fixed seeds). Python row reads too slow for
  prefill (32k rows per 2k chunk ~0.5 s): C++ reads + a cache. A second swizzled scale copy would cost +7.5 GB. Plain decode: 10.2 GB
  per token on the exact BF16 path (~22 tok/s); 4.7 GB with INT5/INT6/NVFP4 copies (~49 tok/s); lm_head (1.27 GB BF16) is the
  largest single item. Image tokens hash as image_token_id 248056 (scheduler _ids() already does this).

## Open questions
1. 2 x 262144 or 3 x 131072? 2. Exact BF16 path as the first milestone, copies in phase B? 3. Disk n-gram table + small row cache,
no resident 4-bit table (25.6 GB)? 4. Vision and MTP deferred?

## Interfaces needed
- Planner 2: the expert layout (separate gate/up or fused [E,2N,K/2]; scales swizzled or not); the reduction-order contract;
  FastMoE.adopt(ref).
- Planner 3: QSA state raw keys [B,1,L,128] bf16 + block-key cache [B,1,L/4,128] + per-row rope positions; a before_replay(tokens)
  hook that fills the n-gram rows [B,T,16,160] u8; the PLE conv window [B,10240,9] bf16 in the state; the family prefill function.

---------------------------------------------------------------------------------------------------------------------------------
## 1. Reference model: engine/model/qwen4exp.py
### 1.1 Qwen4ExpConfig(Qwen35Config)
Extra fields (defaults = nvidia checkpoint): moe_intermediate_size 640, shared_expert_intermediate_size 640, num_experts 512,
num_experts_per_tok 10, norm_topk_prob True, output_gate_type "sigmoid", hidden_act "silu", hc_count 4, hc_lowrank 320,
ple_layer_ids [2] (one-indexed), ple_embed_dim 2560, ple_conv_kernel_size 4, ngram_size 3, heads_per_ngram 8,
ngram_vocab_size_base 20_000_000, make_ngram_vocab_size_divisible_by 128, seed 1234, split_ngram_parts 128, eos_token_id 248044
(first of a list), image_token_id 248056, indexer_n_heads 4, indexer_kv_heads 1, indexer_head_dim 128, indexer_budget 2048,
indexer_compress_ratio 4, mtp_num_hidden_layers 1, mtp_layer_types ["full_attention"], intermediate_size 0 (unused).
Properties: ple_layers = {i - 1}, block_topk = budget // ratio (512), ngram_heads 16, ngram_dim 160.
from_checkpoint: reads text_config; maps "indexed_attention" -> "full_attention"; builds layer_types from full_attention_interval
when missing. Refuses: rope_type != default; indexer_kv_heads != 1; budget % ratio != 0; rotary_dim > indexer_head_dim; a PLE layer
that is not linear_attention; ple_embed_dim % ngram_heads != 0; output_gate_type outside {sigmoid, silu}; any QSA field missing.
max_seq_len > max_position_embeddings is refused for this family for now (YaRN would also scale the indexer's rope and norms).
### 1.2 Qwen4ExpState
pos, max_seq_len, cfg; conv[i] [B, conv_dim, K-1], rec[i] [B, 48, 128, 128] fp32 per GDN layer (as ModelState); k[i], v[i]
[B, 2, L, 256] per attention layer; ik[i] [B, 1, L, 128] model dtype = raw indexer keys (not normed, not roped), time on dim 2;
rope_pos [B, L, 3] int64 (the (t, h, w) position of every cached row; block keys rope at their block start); ple_conv[i]
[B, 10240, 9] model dtype, zero-initialised (the last 9 conv inputs gated_value_normed); ple_ctx[i] [B, 2] int64 init eos (the fast
state drops it: the host derives it). reset(kv=True): pos 0; conv, rec, ple_conv zeroed; ple_ctx = eos.
### 1.3 Modules
GroupRMSNorm(dim, group, eps): x.float(), reshape [..., dim/group, group], x * rsqrt(mean(x^2) + eps), flatten, * (1 + w.float()),
type_as(x); group None = qwen35.RMSNorm (q_norm, k_norm, q_layernorm, k_layernorm). RMSNormGatedSig: qwen35.RMSNormGated with
sigmoid(gate.float()). GatedDeltaNet4(qwen35.GatedDeltaNet): swaps self.norm when output_gate_type == "sigmoid"; forward inherited.
GatedResidual(cfg, combine=True): hc_norm GroupRMSNorm(4H, H), input_mix_weight_down 4H->320, input_mix_weight_up 320->4H,
block_inject_weight 4H->4 or None; forward(h) -> (x [.., H], inj [.., 4]) or x. Router(cfg): weight [E, H]. ExpertWeights: weight
[E,N,K] dense or w/sf/gs/in_scale NVFP4, get(e, dtype) -> [N,K] (dequant_nvfp4(w[e], sf[e], gs[e])). Experts(cfg): gate_proj,
up_proj, down_proj ExpertWeights; forward(x2, idx, w) -> [N,H]. SwiGLU(H, I) (shared expert). SparseMoe(cfg): gate, experts,
shared_expert, shared_expert_gate (H->1). NgramHash(cfg, ple_index): persistent buffers layer_multipliers, ngram_heads_vocab_sizes,
ngram_heads_offsets (HF / checkpoint names); rows(tokens [B,T], ctx [B,2]) -> [B,T,16] int64. NgramEmbedding(cfg, ple_index,
table=None): ngram_embedding.weight [padded, d] in memory (tiny; HF name) or an external NgramTable; forward(tokens, state, li) ->
[B,T,16 d]. PLELayer(cfg, li, ple_index): ple_embedding, key_proj 2560->4H, value_proj 2560->H, norm_key / norm_query / norm_conv
GroupRMSNorm(4H, H), conv1d depthwise kernel 4 dilation 3 no bias; forward(h, ple_ids, state, li) -> [B,T,4H]. QSAIndexer(cfg):
index_qk_proj H->(nh+1) 128, q_layernorm, k_layernorm; select(x, cos, sin, state, li, pos0) -> bool [B,1,T,L].
IndexedAttention(qwen35.Attention): adds indexer; forward computes the mask first, then the base attention through SDPA with that
mask and enable_gqa. DecoderLayer4(cfg, i): linear_attn | self_attn, mlp (SparseMoe), ple (or None), attn_hyper_connection,
mlp_hyper_connection; block_type as qwen35. Qwen4ExpForCausalLM(cfg, ngram_table=None): embed_tokens, layers, hyper_connection_mixer
(GatedResidual(combine=False)), lm_head. NO norm. new_state(B, L); forward(input_ids [B,T], state=None, last_only=False,
positions=None ([T] or [T,3]), embeds=None, ple_ids=None, return_hyper=False) -> fp32 logits [B,T|1,V] (+ the pre-mixer stream
[B,T,4H] with return_hyper; MTP needs it). state=None builds a temporary state of length T (= the HF no-cache path). ple_ids
defaults to input_ids; image rows carry image_token_id.
### 1.4 Op order (model dtype bf16; fp32 in tests; every op rounds to the model dtype as HF)
1. x = embed(ids) (or embeds); h = x.repeat(1,1,4); stream s = h[..., sH:(s+1)H].
2. state.rope_pos[:, pos0:pos0+T] = positions (1D expands to 3 axes); cos, sin = rotary(cfg, positions).
3. Per layer: a. if ple: h = h + ple(h, ple_ids, state, li). b. xa, inj = attn_hc(h): hc_norm grouped per stream; mix =
   sigmoid(up(silu(down(n) / 4))); xa = (mix.view(..,4,H) * n.view(..,4,H)).mean(-2); inj = 2 sigmoid(block_inject(n) / 4).
   c. y = linear_attn(xa, state, li) or self_attn(xa, cos, sin, state, li). d. h = h + (y[..., None, :] * inj[..., :, None]).flatten(-2)
   (HF adds to hyper_input = h before the norm). e. xm, inj = mlp_hc(h); y = moe(xm); h = h + (y[...,None,:] * inj[...,:,None]).flatten(-2).
4. x = hyper_connection_mixer(h) (no inject); logits = lm_head(x) (bf16 in HF); return .float(); state.pos += T.
### 1.5 MoE math (HF eager)
x2 = x.view(-1,H); shared = down(silu(gate(x2)) * up(x2)); logits = F.linear(x2, gate.weight); p = softmax(logits, -1, float32);
w, idx = topk(p, 10); w = w / w.sum(-1, keepdim); w = w.to(logits.dtype); out = zeros_like(x2); mask = one_hot(idx, E+1).permute(2,1,0);
for e in nonzero(mask.sum((-1,-2)) > 0) ascending, skip e == E: kpos, tok = where(mask[e]); g, u = linear(x2[tok], gate_e),
linear(x2[tok], up_e) (HF: one fused gate_up linear chunked; same per-element math; tolerance); hcur = linear(silu(g) * u, down_e)
* w[tok, kpos, None]; out.index_add_(0, tok, hcur). return out + sigmoid(shared_expert_gate(x2)) * shared.
Reduction contract proposed: per token sum in ascending expert id, bf16 terms into a bf16 accumulator, shared term last.
[LEAD'S DECISION: the reference keeps HF semantics; the kernels use fp32 partials in k-slot order with one rounding (plan 2); kernel
tests use a tolerance. Bit identity is required only between the kernel's own paths (decode == verify).]
### 1.6 QSA indexer (vectorized per query; C = ratio, D = 128, nh = 4, L = pos0 + T)
1. qk = index_qk_proj(x); q = q_layernorm(qk[..., :nh D].view(B,T,nh,D)); rope the first R = 64 dims of q with this call's cos/sin.
2. state.ik[li][:, 0, pos0:L] = qk[..., nh D:]; keys = state.ik[li][:, 0, :L].
3. nb = L // C; kb = keys[:, :nb C].view(B,nb,C,D).float().mean(2).to(dtype), then k_layernorm, then rope (bf16) with
   rotary(cfg, state.rope_pos[b, ::C][:nb]).
4. Query t at p = pos0 + t has nbq = (p+1) // C complete visible blocks. s = matmul(q.float(), kb.float().T) -> [B,T,nh,nb] ->
   transpose to [.., nb, nh] -> relu -> sum over nh -> / sqrt(D); s[j >= nbq] = -inf.
5. kk = min(block_topk, nb); sel = topk(s, kk).indices; keep sel < nbq; allowed_block[sel] = True; token mask =
   allowed_block.repeat_interleave(C) over [0, nb C) OR the tail [nbq C, p]. Causality implied.
6. SDPA with the bool mask [B,1,T,L]. Below p+1 <= budget + C - 1 the mask is all true (dense).
At full size process queries in sub-chunks of 256 ([T,nh,nb] fp32 reaches 2 GB at 262k). select also returns the mask for tests.
### 1.7 PLE exact
Primes: head hi = ple_index 16 + h: size_h = _find_nth_prime_after(base - 1, hi + 1); offsets running sums; padded = ceil(total/128)
128; copy HF's _is_prime. Multipliers: _build_layer_multipliers(vocab, 3, ple_index, seed) with gamma 0x9E3779B97F4A7C15,
M1 0xBF58476D1CE4E5B9, M2 0x94D049BB133111EB, PRIME_1 10007, half = ((2^63-1) // vocab) // 2, m_i = 2 (splitmix(base_seed +
gamma (i+1) mod 2^64) % half) + 1. Ids (int64): hist = cat(ctx, tokens); cur = tokens; p1 = hist[:, 1:-1]; p2 = where(p1 == eos,
eos, hist[:, :-2]) (= _shift_right_ignore_eos: shift k valid iff no eos in [p-k, p-1]; an invalid shift-1 equals eos anyway; holds
with a cached context since only the 2-token window matters; at sequence start the context is (eos, eos)). a2 = cur m0 ^ p1 m1;
a3 = a2 ^ p2 m2; heads 0-7 = a2 % size_h + off_h; heads 8-15 = a3 % size_h + off_h; head-major concat. New ctx = hist[:, -2:].
Assert every id < vocab; image placeholders map to image_token_id before hashing. Embedding rows -> [B,T,16,d] -> [B,T,2560]; on
disk bf16(e4m3.float() * scale.float()) (exact product in fp32). Layer: kn = norm_key(key_proj(e)).view(..,4,H); v = value_proj(e);
qn = norm_query(h).view(..,4,H); g = (kn * qn).sum(-1, keepdim) / sqrt(H); g = g.abs().clamp_min(1e-6).sqrt() * g.sign();
gv = sigmoid(g) * v[..,None,:]; gvn = norm_conv(gv.flatten(-2)); gv = gv.flatten(-2); conv: xin = cat(state.ple_conv[li],
gvn.transpose(1,2)) [B,4H,9+T]; state.ple_conv[li] = xin[..., -9:]; out = gv + silu(F.conv1d(xin, w, groups=4H, dilation=3))
.transpose(1,2) (= HF's update_conv_state with K = 9 plus the zero pad).

## 2. tests/test_qwen4exp_small.py (CPU, fp32)
Tiny config: hidden 64, layers 4 [lin, lin, lin, full], heads 4/2, head_dim 32, partial 0.25 (R = 8, mrope [1,1,2]), GDN 2/4 heads
x 16, conv 4, vocab 101, eos 3; MoE E = 4 top-2, moe_I 32, shared_I 32; hc 4, lowrank 16, ple_layer_ids [2], ple_embed_dim 64
(d = 4), base 1009 (16896 rows); indexer nh 2, D 16, budget 16, ratio 4 (sparse from p >= 19 in a 70-token test); output_gate
sigmoid. Init normal(0, 0.08); norms normal(0, 0.1); A_log uniform(-1, 1); PLE conv and the n-gram table random (HF zero-inits the
PLE conv: overwrite). ids randint(0, 101, (1, 70)) with eos at 19, 20 and 45, plus an image-pad run for the image test.
hf_tiny: Qwen4ExpTextConfig(..., experts_implementation="eager") (default grouped_mm has another accumulation order);
_attn_implementation = "sdpa"; Qwen4ExpForCausalLM(cfg).float().eval(); extend tests/hf_fallback.py to unwrap the four functions of
modeling_qwen4_exp too. State dict: ours X -> model.X except lm_head.weight; layers.i.mlp.experts.{gate,up}_proj.weight ->
model.layers.i.mlp.experts.gate_up_proj = cat([gate, up], 1) [E, 2I, H]; down_proj -> down_proj [E, H, I]; assert no unexpected keys
and no missing ones apart from rotary buffers.
Tests: 1 test_config_from_checkpoint (fixture tests/data/qwen4exp_config.json + refusals); 2 test_ngram_params_match_transformers
(real config: sizes, offsets, multipliers == HF's; padded == 128 x 2,500,012); 3 test_ngram_ids_match_transformers (HF NGramEmbedding
with ple_embed_dim 16 (d = 1) and weight = arange so the output equals the row ids; one-shot and through a DynamicCache split at every
position, eos at the split boundary and twice in a row; numpy-vs-torch hash test engine/weights/ple.ngram_rows);
4 test_stepwise_decode_matches_prefill; 5 test_split_prefill_matches_one_shot splits [1, 3, 19, 20, 21, 40, 69]; 6 test_matches_hf_tiny
(no cache); 7 test_matches_hf_tiny_with_cache (HF split at 33 then 6 single steps); 8 test_sparse_selection_happens (the mask at p = 60
has 16 + (61 % 4) tokens; a budget-1e6 copy gives different logits); 9 test_matches_hf_tiny_with_an_image (position_ids =
cat([arange(T)[None], pos.T])[:, None] shape [4,1,T], inputs_embeds, ple_input_ids=ids: without it HF calls reverse_embedding, which
fails on random embeds); 10 test_nvfp4_experts_equal_dense_dequant (exact).
HF quirks: position_ids [4,B,T] (row 0 = text positions) or [3,B,T] or 2D; HF concatenates continuation positions onto
cache.position_ids; conv_mask None without padding; the GDN conv state has K columns; the n-gram context sits in conv_states[2].

## 3. Loader
### 3.1 direct.py: read_tensors(path, names=None, device="cuda", into=None): into[name] a preallocated contiguous destination with the
same dtype and numel on any device; dest() uses it after a shape/byte check; those names are left out of the result. Test in
tests/test_direct_read.py.
### 3.2 engine/weights/qwen4exp_loader.py: load_qwen4exp(path, device="cuda", experts="nvfp4") -> Qwen4ExpForCausalLM (the reference
tree; to_fast adopts it later): cfg from_checkpoint; model on meta; preallocate per layer and projection w [512, N, K/2] u8 and sf
[512, N, K/16] e4m3 on the GPU, gs and in_scale CPU f32 [48, 3, 512] moved in one copy; build into for all 147k tensors (a layer's
experts may span two files: layers 8, 14, 21, 27, 33, 40, 46); per model-0000x file read_tensors with names excluding
model.visual.*, mtp.*, *.ngram_embedding.*; BF16 tensors as plain; check layer_multipliers, ngram_heads_vocab_sizes,
ngram_heads_offsets against the computed buffers (raise on mismatch); NgramTable(path) attached to layers[1].ple. quant_kind would
call the shards "plain fp8" and raise: exclude by name. Cost ~77.7 GB at 9 GB/s ~8.6 s + ~147k small async copies (~1 s); 384 MB
pinned; no transient GPU bytes. Rejected: offline stacking (68 GB more disk, a mandatory prepare step, no gain). MTP skipped
(later via quant_kind fp8_block + dequant_fp8_block; scale_inv bf16 [5,20]/[20,5]). Vision skipped.
### 3.3 engine/weights/ple.py: NgramTable(path, prefix="model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
threads=32): parses the header of model-fp8-mtp-ple.safetensors with direct.header; rows_per_shard 2500012, n_shards 128, dim 160,
scale (bf16 tensor + float), base (= 8 + header length); asserts n_shards x rows == padded. offsets(r) = base +
data_offsets[f"{prefix}.shard_{r // 2500012}.weight"][0] + (r % 2500012) * 160 (use the header's own offsets: the shards are not
stored in index order). rows(ids int64[n]) -> uint8[n,160]: dedupe, sort by offset, O_DIRECT preadv of the 4 KB page (8 KB when a
row straddles) on a thread pool into aligned buffers, copy out 160 B per row. rows_into(ids, out_pinned); dequant(raw, dtype) =
(raw.view(e4m3).float() * scale).to(dtype). ngram_rows(tokens, ctx, cfg) -> int64[T,16] (numpy). Reuse direct._open and header.
Thread pool because liburing is absent and preadv releases the GIL; 57k IOPS: enough for decode, not prefill (32k rows per chunk):
a C++ pool ops().ple_gather(fd, offsets, out) + a device row cache are follow-ups.

## 4. Memory budget (GB = 1e9)
Routed experts NVFP4 + scales 67.95 (2,764,800 B x 24,576); GDN projections + conv + norms 4.17; attention q/k/v/o + indexer 1.235;
hyper-connections 1.27; shared experts + routers + gates 0.60; embedding 1.27; lm_head 1.27; PLE projections + norms 0.07; BF16 rest
total 9.88. Decode copies (phase B, gate) +3.10 (GDN INT5 1.43, attention INT6 0.49, HC INT6 0.51, shared NVFP4 0.13, lm_head NVFP4
0.36, routers/indexer/PLE kept 0.18). Per slot-token: KV fp8 12,288 B + raw indexer keys bf16 3,072 B (fp8 would change the
selection: no) + block-key cache 768 B + rope positions 12 B = ~16.1 KB: 1 slot x 262144 = 4.2 GB; 2 = 8.45; 3 = 12.7. GDN + PLE
state per slot 115.6 MB (conv 2.2 + rec 113.2 + PLE conv 0.18); ring 16 x 115.6 MB = 1.85 GB (32 = 3.7); MTP drafter later 1.6-2.7 +
KV 0.7-1.06; graph pool ~0.5 (guess); CUDA context, allocator, pinned ~5.3 (from the 27B's ENGINE_FIXED_BYTES); n-gram page cache 0
(O_DIRECT); n-gram row cache later 1-2; vision later 0.92; growth reserve 6.
Decode milestone (2 x 262144, 16 checkpoints, no MTP, no copies): ~94.2 + 6 = 100 GB of 122. Final (copies + MTP + 2 GB row cache,
2 slots) ~109 GB; 3 slots ~113 GB (too tight). Optional: FP8 W8A8 prefill copies in place of the BF16 originals, -4.9 GB (gate).
Hot set: tools/ple_hotset.py tokenises WikiText-103, TinyStories (HF datasets cache), site-packages Python and the golden prompts;
hashes with the real config; reports distinct rows per million tokens, an LRU hit-rate curve for 1e5..3.2e7 rows (16 MB..5 GB) by 1%
shard sampling, bigram vs trigram splits, cross-corpus coverage of a static top-N preload. Needs no checkpoint.
Plain decode bytes per token at 32k: (a) exact 1.33 + 8.61 + 0.23 GDN state + 0.05 KV/indexer = 10.2 GB, 44 ms, ~22.5 tok/s;
(b) copies 1.33 + 3.10 + 0.28 = 4.7 GB, 20.4 ms, ~49 tok/s ideal. At 262k the indexer adds 0.2 GB (cached block keys) or 0.8 GB.

## 5. Integration surface (grep)
build.py: load_model, load_drafter, startup_bytes (Qwen35Config, FastState, MtpState), apply_yarn, attach_decode_copies.
scheduler.py: prefill/prepare_prefill imports; the ring and restore over conv/rec (238-242, 295-297, 343-345); slot reset (459-461);
_copy_kv (319); ctx_s; q_per_kv; hidden_size for ring_h. decode.py: prepare_prefill and model(tok, state). sessions.py: 117-129,
244-254. mtp.py; loader.load_model (perplexity); tests/scheduler_check.py (block_type); the tools.
Proposal: engine/model/family.py with family_of(path) keyed by model_type (qwen3_5 vs qwen4_exp) -> Family with load(path,
decode_weights, log, prep, max_seq_len), state_bytes(path, slots, L, spec), fixed_bytes, prefill, prepare_prefill, supports (mtp,
vision, yarn). States get kv_tensors() ([B,H,L,D] tensors, time on dim 2: FastState [k[i], v[i] for sorted i]; the new state
appends ik and the block cache) and snap_tensors() (FastState [conv...] + [rec...]; new adds ple_conv), all zero-reset. Scheduler
and sessions call these. Scheduler and DecodeGraph take prefill_fn / prepare_fn from the family; a family-specific decode graph adds
before_replay(tokens). The 27B's tensor order and sessions fingerprint shapes stay identical. build_engine refuses spec=mtp, vision
and YaRN for qwen4_exp until they exist.

## 6. Phase A work (~40 h; A1-A10 before the download ends)
A1 Config + fixture (1.5 h); A2 norms, GatedResidual, GDN4, MoE both storages (3); A3 NgramHash/PLE torch + numpy (3); A4 QSA +
IndexedAttention + state (3); A5 model forward/new_state (2); A6 HF parity tests + hf_fallback (4 + 4 debugging); A7 direct.py into
(1.5); A8 NgramTable + tests/test_ple.py (synthetic sharded file, shards out of order, straddling rows, last row) (3); A9 loader +
synthetic nvidia-layout checkpoint writer tests/qwen4exp_ckpt.py (tiny reference model, experts quantized with quantize.nvfp4_global_scale
/ quantize, per-expert names, a layer split across files, a ple file with 4 shards, mtp/visual decoys, index.json) +
tests/test_qwen4exp_loader.py (CUDA, tiny): stacked == per-expert; logits == dequantized reference exactly; decoys not loaded;
corrupted multiplier raises (5); A10 hot-set tool (3); A11 once downloaded: tools/verify_qwen4exp.py header/buffer checks + timed load
+ a short GPU reference forward, sane WikiText perplexity (3); A12 (decode milestone) family dispatch + hooks (4).
