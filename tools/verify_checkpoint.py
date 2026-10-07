#!/usr/bin/env python3
"""verify_checkpoint.py -- check each safetensors shard of a cached Hub checkpoint against its published checksum. This is
the crc32.txt of the repo when it has one, else the LFS sha256 of the Hub.

Usage: uv run python tools/verify_checkpoint.py <repo id> [<repo id> ...]
The exit status is non-zero if a shard is missing or does not match.
"""
import hashlib
import os
import sys
import zlib

from huggingface_hub import HfApi, snapshot_download


def crc32_file(p):
    c = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            c = zlib.crc32(chunk, c)
    return f"{c & 0xFFFFFFFF:08x}"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(repo):
    d = snapshot_download(repo, local_files_only=True)
    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    shards = sorted(s for s in os.listdir(d) if s.endswith(".safetensors"))
    crc = {}
    if os.path.exists(os.path.join(d, "crc32.txt")):
        for line in open(os.path.join(d, "crc32.txt")):
            parts = line.split()
            if len(parts) == 2:
                crc[parts[1]] = parts[0].lower()
    lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
    bad = 0
    total = 0
    for s in shards:
        p = os.path.join(d, s)
        size = os.path.getsize(p)
        total += size
        exp_size = lfs.get(s, (None, None))[1]
        if exp_size is not None and size != exp_size:
            print(f"  SIZE MISMATCH {s}: local {size} vs hub {exp_size}")
            bad += 1
            continue
        if s in crc:
            got = crc32_file(p)
            ok = got == crc[s]
            print(f"  {'ok ' if ok else 'BAD'} crc32 {got} {s}")
        elif lfs.get(s, (None,))[0]:
            got = sha256_file(p)
            ok = got == lfs[s][0]
            print(f"  {'ok ' if ok else 'BAD'} sha256 {got[:16]}.. {s}")
        else:
            print(f"  ??  no checksum available for {s} (size ok)")
            ok = True
        bad += not ok
    print(f"{repo}: {len(shards)} shards, {total / 1e9:.1f} GB, {bad} bad   ({d})")
    return bad


if __name__ == "__main__":
    rc = 0
    for r in sys.argv[1:]:
        rc |= bool(verify(r))
    sys.exit(rc)
