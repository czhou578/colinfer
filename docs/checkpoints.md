# Model checkpoints on this machine

Written 2026-10-03, updated 2026-10-24. Two Qwen3.8-27B checkpoints and the Qwen3.8-Flash-Next MoE checkpoint are in the default Hugging Face cache (`~/.cache/huggingface/hub`). The
FP8 checkpoint `Qwen/Qwen3.8-27B-FP8` was deleted on 2026-10-09: the engine cannot serve its block-scaled FP8, and
only the FP8 perplexity baseline used it.
We downloaded them with `hf download <repo>` from the project venv. Thus vLLM, SGLang and `transformers` find them by
repo id with no extra configuration. Our loader finds the same path with
`huggingface_hub.snapshot_download(repo, local_files_only=True)`.

| Repo id | Format | Hub commit | Size | Shards | Status |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | `1d4bf0f2ff60` | 55.6 GB | 18 | downloaded, verified |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 (mixed: MLP + lm_head NVFP4, attention/GDN FP8, MTP BF16) | `482ca0f38322` | 21.9 GB | 3 | downloaded, verified |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | NVFP4 routed experts, the rest BF16; the n-gram table and the MTP experts FP8 in one 53.7 GB file | `fc694b54fb01` | 132.7 GB | 10 + 1 | downloaded and verified 2026-10-24 (`docs/history/qwen4exp_plan.md`) |

## Local paths (snapshot directories, one per Hub commit)

- `Qwen/Qwen3.8-27B` (the source of the INT6 / INT5 decode copies, `tools/int6_requant.py`, and the BF16 perplexity
  baseline)
  `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `nvidia/Qwen3.8-27B-NVFP4` (the serving checkpoint)
  `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80`
- `nvidia/Qwen3.8-Flash-Next-NVFP4` (the MoE target of branch `moe`; the engine does not serve it yet)
  `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47`

## Verification

`uv run python tools/verify_checkpoint.py <repo id>` checks each shard. For the Qwen repos, it uses the `crc32.txt` of
the repo. For the nvidia repo, it uses the LFS sha256 from the Hub. All 87 shards passed on 2026-10-03, right after the
download. The download times on the Wi-Fi of this machine were: NVFP4 5.1 min, FP8 7.4 min, BF16 11.4 min (30 to
140 MB/s).

`uv run python tools/verify_qwen4exp.py` checks `nvidia/Qwen3.8-Flash-Next-NVFP4` further, without the GPU: the
config, the index and the headers (the dtype and shape of each tensor of the language model), the n-gram hash buffers
against `engine/weights/ple.py`, the shards and rows of the n-gram table, and the global and input scales of the
experts.

## Where the bytes are

huggingface_hub 1.x keeps large files in a shared content-addressed store,
`~/.cache/huggingface/hub/blobs/<first 2 hex>/<sha256>`. The `models--*/snapshots/<commit>/` directory of each model
holds symlinks into this store. Small files are in the `blobs/` directory of the model. Thus
`du -sh models--Qwen--Qwen3.8-27B` shows only ~10 MB. For the real size, use `du -sh --apparent-size -L <snapshot dir>`.
For the total, use `du -sh ~/.cache/huggingface/hub` (102 GB with the three Qwen3.8-27B checkpoints, 73 GB on
2026-10-09 with these two and a DeepSeek GGUF).

## Notes

- `nvidia/Qwen3.8-27B-NVFP4` has mixed precision. `docs/history/baseline.md` section 3 and
  `docs/history/nvfp4_inventory.txt` show which modules are NVFP4, FP8 or BF16.
- To list the tensors of a checkpoint, run `uv run python tools/tensor_inventory.py <repo id or path>`.
- To download again or update, run `uv run hf download <repo>`. To delete, run `uv run hf cache delete` (interactive).
  Removing only the `models--*` directory leaves its large files in `blobs/`. A blob can belong to several models
  (the NVFP4 and BF16 checkpoints share their tokenizer), so delete only the blobs that no other snapshot links to.
- Each checkpoint contains a vision tower. The engine serves images with the vision tower of
  `nvidia/Qwen3.8-27B-NVFP4` (0.92 GB BF16, loaded on the first image; `engine/model/vision.py`). The loader of
  `nvidia/Qwen3.8-Flash-Next-NVFP4` skips its vision tower: images for that model are a later milestone.
- The n-gram table of `nvidia/Qwen3.8-Flash-Next-NVFP4` (51.2 GB, in `model-fp8-mtp-ple.safetensors`) stays on disk.
  The loader does not load the table: it attaches a row reader to the file (`NgramTable`, `engine/weights/ple.py`).
