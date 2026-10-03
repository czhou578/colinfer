# Model checkpoints on this machine

Written 2026-10-03. All three Qwen3.8-27B variants live in the default Hugging Face cache
(`~/.cache/huggingface/hub`), downloaded with `hf download <repo>` from the project venv, so
vLLM, SGLang and `transformers` resolve them by repo id with no extra configuration, and our
loader resolves the same path with `huggingface_hub.snapshot_download(repo, local_files_only=True)`.

| Repo id | Format | Hub commit | Size | Shards | Status |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | `1d4bf0f2ff60` | 55.6 GB | 18 | downloaded, verified |
| `Qwen/Qwen3.8-27B-FP8` | FP8 | `017b9c7af6b5` | 30.9 GB | 66 | downloaded, verified |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 (mixed: MLP + lm_head NVFP4, attention/GDN FP8, MTP BF16) | `482ca0f38322` | 21.9 GB | 3 | downloaded, verified |

## Local paths (snapshot directories, one per Hub commit)

- `Qwen/Qwen3.8-27B` (reference for the Phase 1 parity harness)
  `/home/colin-spark/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `Qwen/Qwen3.8-27B-FP8` (FP8 fallback path and FP8-vs-NVFP4 comparison)
  `/home/colin-spark/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- `nvidia/Qwen3.8-27B-NVFP4` (the serving checkpoint)
  `/home/colin-spark/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80`

## Verification

`uv run python tools/verify_checkpoint.py <repo id>` checks every shard against the repo's
`crc32.txt` (Qwen repos) or the Hub's LFS sha256 (nvidia repo). All 87 shards passed on
2026-10-03 right after download. Download wall times on this box's Wi-Fi: NVFP4 5.1 min,
FP8 7.4 min, BF16 11.4 min (30 to 140 MB/s).

## Where the bytes actually are

huggingface_hub 1.x keeps large files in a shared content-addressed store,
`~/.cache/huggingface/hub/blobs/<first 2 hex>/<sha256>`, and the per-model
`models--*/snapshots/<commit>/` directories hold symlinks into it (small files sit in the
model's own `blobs/`). So `du -sh models--Qwen--Qwen3.8-27B` reports ~10 MB; use
`du -sh --apparent-size -L <snapshot dir>` for the real size, or `du -sh ~/.cache/huggingface/hub`
for the total (102 GB with these three).

## Notes

- `nvidia/Qwen3.8-27B-NVFP4` is mixed precision; see `docs/baseline.md` section 3 and
  `docs/nvfp4_inventory.txt` for exactly which modules are NVFP4 vs FP8 vs BF16.
- Inventory of any checkpoint: `uv run python tools/tensor_inventory.py <repo id or path>`.
- Re-download or update: `uv run hf download <repo>`. Delete: `uv run hf cache delete` (interactive)
  or remove the `models--*` directory.
- The vision tower inside each checkpoint (~0.9 GB BF16) is not used by the text-only engine.
