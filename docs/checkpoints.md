# Model checkpoints on this machine

Written 2026-10-03. All three Qwen3.8-27B variants are in the default Hugging Face cache (`~/.cache/huggingface/hub`).
We downloaded them with `hf download <repo>` from the project venv. Thus vLLM, SGLang and `transformers` find them by
repo id with no extra configuration. Our loader finds the same path with
`huggingface_hub.snapshot_download(repo, local_files_only=True)`.

| Repo id | Format | Hub commit | Size | Shards | Status |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | `1d4bf0f2ff60` | 55.6 GB | 18 | downloaded, verified |
| `Qwen/Qwen3.8-27B-FP8` | FP8 | `017b9c7af6b5` | 30.9 GB | 66 | downloaded, verified |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 (mixed: MLP + lm_head NVFP4, attention/GDN FP8, MTP BF16) | `482ca0f38322` | 21.9 GB | 3 | downloaded, verified |

## Local paths (snapshot directories, one per Hub commit)

- `Qwen/Qwen3.8-27B` (the reference for the Phase 1 parity harness)
  `/home/colin-spark/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `Qwen/Qwen3.8-27B-FP8` (the FP8 fallback path and the FP8-vs-NVFP4 comparison)
  `/home/colin-spark/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- `nvidia/Qwen3.8-27B-NVFP4` (the serving checkpoint)
  `/home/colin-spark/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80`

## Verification

`uv run python tools/verify_checkpoint.py <repo id>` checks each shard. For the Qwen repos, it uses the `crc32.txt` of
the repo. For the nvidia repo, it uses the LFS sha256 from the Hub. All 87 shards passed on 2026-10-03, right after the
download. The download times on the Wi-Fi of this machine were: NVFP4 5.1 min, FP8 7.4 min, BF16 11.4 min (30 to
140 MB/s).

## Where the bytes are

huggingface_hub 1.x keeps large files in a shared content-addressed store,
`~/.cache/huggingface/hub/blobs/<first 2 hex>/<sha256>`. The `models--*/snapshots/<commit>/` directory of each model
holds symlinks into this store. Small files are in the `blobs/` directory of the model. Thus
`du -sh models--Qwen--Qwen3.8-27B` shows only ~10 MB. For the real size, use `du -sh --apparent-size -L <snapshot dir>`.
For the total, use `du -sh ~/.cache/huggingface/hub` (102 GB with these three).

## Notes

- `nvidia/Qwen3.8-27B-NVFP4` has mixed precision. `docs/history/baseline.md` section 3 and
  `docs/history/nvfp4_inventory.txt` show which modules are NVFP4, FP8 or BF16.
- To list the tensors of a checkpoint, run `uv run python tools/tensor_inventory.py <repo id or path>`.
- To download again or update, run `uv run hf download <repo>`. To delete, run `uv run hf cache delete` (interactive)
  or remove the `models--*` directory.
- Each checkpoint contains a vision tower (~0.9 GB BF16). The engine is text-only and does not use it.
