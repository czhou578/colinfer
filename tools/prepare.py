#!/usr/bin/env python3
"""Prepare: gather the offline artifacts that make decode faster into one directory with a manifest
(engine/runtime/prepared.py). The engine then loads that one layout, or runs from the checkpoint alone.

   uv run python tools/prepare.py                 # ~/.cache/colinfer/prepared/<snapshot>/: INT copies, drafter, low-rank head, manifest.json
   uv run python tools/prepare.py --skip-int      # no INT6 / INT5 copies (no BF16 checkpoint): decode streams the FP8 weights
   uv run python tools/prepare.py --skip-lowrank  # no low-rank head: the full draft head

For each artifact, the tool keeps a file that is in the prepared directory already. Else it takes the file from where its
tool writes it (a hard link; a copy on another file system). Else it makes the file:
  * The INT6 / INT5 decode copies: tools/int6_requant.py --bits 6 --filter self_attn and --bits 5 --filter linear_attn.
    They need the BF16 checkpoint Qwen/Qwen3.8-27B (55.6 GB) in the Hugging Face cache, and a few minutes on the GPU.
  * The fine-tuned MTP drafter: taken from ~/.cache/colinfer/drafter/mtp_ft.safetensors when tools/train_drafter.py made
    one, or from --drafter. The tool never trains one: that needs a running server for its data, and hours.
  * The low-rank draft head: tools/lowrank_draft_head.py, on the drafter above. It loads the model (65 GB of memory,
    a few minutes on the GPU).
Then it writes manifest.json. A new drafter replaces the old one, and the tool makes the low-rank head again, because
the basis depends on the drafter's outputs.

A qwen4_exp checkpoint (Qwen3.8-Flash-Next; config.json model_type) has other artifacts: the decode copies of its BF16
matrices, one file per class (tools/requant_qwen4exp.py, which reads the checkpoint itself: about 30 s on the GPU).
The tool keeps the files that exist, makes the missing ones, and writes the manifest with "family": "qwen4_exp". The
27B's steps (the INT6 / INT5 copies from Qwen/Qwen3.8-27B, the drafter, the low-rank head) do not apply to it.

   uv run python tools/prepare.py --model nvidia/Qwen3.8-Flash-Next-NVFP4 [--formats gdn_out=int6] [--skip-int]

CAUTION: an over-commit of the unified memory powers the machine off. Stop the server before you run the tool without
--skip-int and --skip-lowrank: the steps that make files load the model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from engine.runtime.prepared import DECODE4_FORMATS, DECODE_FILES, DRAFTER_FILE, LOWRANK_FILE, Prepared, decode4_file, prepared_dir
from engine.runtime.prepared import read as read_manifest
from engine.runtime.prepared import write as write_manifest

TOOLS = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.expanduser("~/.cache/colinfer/drafter")  # where tools/train_drafter.py and tools/lowrank_draft_head.py write
INT_ARGS = (("--bits", "6", "--filter", "self_attn"), ("--bits", "5", "--filter", "linear_attn"))  # one per DECODE_FILES entry


def same(a: str, b: str) -> bool:
    """The same file (a hard link), or a copy with the same size and modification time."""
    try:
        if os.path.samefile(a, b):
            return True
        sa, sb = os.stat(a), os.stat(b)
        return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)
    except OSError:
        return False


def take(src: str, dst: str) -> str:
    """dst becomes src: a hard link when the file systems allow it, else a copy that keeps the modification time."""
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def model_type(path: str) -> str | None:
    """The model_type of the checkpoint directory's config.json (None without one)."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f).get("model_type")
    except (OSError, ValueError):
        return None


def prepare4(path: str, out: str, *, skip_int: bool = False, formats: dict[str, str] | None = None, run=None, log=print) -> Prepared:
    """The prepared directory of a qwen4_exp checkpoint: the decode copies of tools/requant_qwen4exp.py (one file per
    class with a format other than bf16; formats: DECODE4_FORMATS by default). It keeps the files that exist and makes the
    missing ones in one run of the tool, unless skip_int. Then it writes the manifest (family qwen4_exp)."""
    run = run or (lambda argv: subprocess.run(argv, check=True))
    formats = dict(formats or DECODE4_FORMATS)
    os.makedirs(out, exist_ok=True)
    cp = os.path.basename(os.path.normpath(path))
    want = {c: decode4_file(c, f) for c, f in formats.items() if f != "bf16"}
    missing = [c for c, name in want.items() if not os.path.exists(os.path.join(out, name))]
    for c, name in want.items():
        log(f"  {name}: " + ("kept" if c not in missing else "skipped (--skip-int): decode streams the BF16 weights of this class" if skip_int
                             else "made by tools/requant_qwen4exp.py (reads the checkpoint's BF16 tensors: about 30 s on the GPU)"))
    for c, f in formats.items():
        if f == "bf16":
            log(f"  {c}: bf16 (no copy): decode streams the checkpoint's weights of this class")
    if missing and not skip_int:
        spec = ",".join(f"{c}={formats[c]}" for c in missing)
        run([sys.executable, os.path.join(TOOLS, "requant_qwen4exp.py"), "--model", path, "--out", out, "--formats", spec,
             "--classes", ",".join(missing)])
    decode = [name for name in want.values() if os.path.exists(os.path.join(out, name))]
    log(f"  {DRAFTER_FILE}, {LOWRANK_FILE}: not for this family (the drafter is the checkpoint's MTP block, engine/spec/mtp4.py)")
    mp = write_manifest(out, cp, decode, None, None, family="qwen4_exp")
    log(f"manifest: {mp}")
    return read_manifest(path, out)


def prepare(path: str, out: str, *, bf16: str = "Qwen/Qwen3.8-27B", skip_int: bool = False, skip_lowrank: bool = False,
            drafter: str | None = None, legacy_decode=(), legacy_drafter: str | None = None, legacy_lowrank: str | None = None,
            run=None, log=print, formats: dict[str, str] | None = None) -> Prepared:
    """Fills the prepared directory `out` of the checkpoint directory `path` and writes its manifest. legacy_*: where the
    tools write the artifacts today (taken from there when present). run: runs a tool's argv (subprocess by default).
    A qwen4_exp checkpoint goes to prepare4 (with skip_int and formats); the other arguments are the 27B's."""
    if model_type(path) == "qwen4_exp":
        return prepare4(path, out, skip_int=skip_int, formats=formats, run=run, log=log)
    run = run or (lambda argv: subprocess.run(argv, check=True))
    os.makedirs(out, exist_ok=True)
    cp = os.path.basename(os.path.normpath(path))
    py = sys.executable
    # 1. the INT6 / INT5 decode copies
    decode = []
    for name, legacy, args in zip(DECODE_FILES, tuple(legacy_decode) + (None, None), INT_ARGS):
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            log(f"  {name}: kept")
        elif legacy and os.path.exists(legacy):
            log(f"  {name}: {take(legacy, dst)} from {legacy}")
        elif skip_int:
            log(f"  {name}: skipped (--skip-int): decode streams the FP8 weights of these projections")
            continue
        else:
            log(f"  {name}: made by tools/int6_requant.py {' '.join(args)} (needs the BF16 checkpoint {bf16})")
            run([py, os.path.join(TOOLS, "int6_requant.py"), "--nvfp4", path, "--bf16", bf16, *args, "--out", dst])
        decode.append(name)
    # 2. the fine-tuned drafter: taken, never made
    dst = os.path.join(out, DRAFTER_FILE)
    src = drafter or legacy_drafter
    new_drafter, drafter_name = False, None
    if src and os.path.exists(src):
        if os.path.exists(dst) and same(src, dst):
            log(f"  {DRAFTER_FILE}: kept (the same as {src})")
        else:
            new_drafter = os.path.exists(dst)
            log(f"  {DRAFTER_FILE}: {take(src, dst)} from {src}" + (" (a new drafter: the low-rank head is made again)" if new_drafter else ""))
        drafter_name = DRAFTER_FILE
    elif os.path.exists(dst):
        log(f"  {DRAFTER_FILE}: kept")
        drafter_name = DRAFTER_FILE
    else:
        log(f"  {DRAFTER_FILE}: none (tools/train_drafter.py makes one): the checkpoint's MTP head")
    # 3. the low-rank draft head, on that drafter. The manifest goes out first, so the tool's model has the INT copies.
    write_manifest(out, cp, decode, drafter_name, None)
    dst = os.path.join(out, LOWRANK_FILE)
    lowrank_name = None
    if new_drafter and os.path.exists(dst):
        os.remove(dst)
    if os.path.exists(dst):
        log(f"  {LOWRANK_FILE}: kept")
        lowrank_name = LOWRANK_FILE
    elif not new_drafter and legacy_lowrank and os.path.exists(legacy_lowrank):
        log(f"  {LOWRANK_FILE}: {take(legacy_lowrank, dst)} from {legacy_lowrank}")
        lowrank_name = LOWRANK_FILE
    elif skip_lowrank:
        log(f"  {LOWRANK_FILE}: skipped (--skip-lowrank): the full draft head")
    else:
        weights = os.path.join(out, drafter_name) if drafter_name else "none"
        log(f"  {LOWRANK_FILE}: made by tools/lowrank_draft_head.py --weights {weights} (loads the model: a few minutes)")
        run([py, os.path.join(TOOLS, "lowrank_draft_head.py"), "--weights", weights, "--out", dst])
        lowrank_name = LOWRANK_FILE
    mp = write_manifest(out, cp, decode, drafter_name, lowrank_name)
    log(f"manifest: {mp}")
    return read_manifest(path, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gather the offline artifacts that make decode faster into one prepared directory")
    ap.add_argument("--model", default=None, help="the NVFP4 checkpoint: an HF repo id in the local cache, or a directory (default: the engine's)")
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B", help="the BF16 checkpoint that the INT copies are made from")
    ap.add_argument("--out", default=None, help="the prepared directory (default: ~/.cache/colinfer/prepared/<snapshot>)")
    ap.add_argument("--drafter", default=None,
                    help="a fine-tuned MTP head to take (default: ~/.cache/colinfer/drafter/mtp_ft.safetensors when it exists)")
    ap.add_argument("--skip-int", action="store_true", help="no INT6 / INT5 decode copies when they have to be made")
    ap.add_argument("--skip-lowrank", action="store_true", help="no low-rank draft head when it has to be made")
    ap.add_argument("--formats", default=None, help="qwen4_exp: class=format overrides of the decode copies (tools/requant_qwen4exp.py)")
    a = ap.parse_args(argv)
    from engine.model.fast import decode_copies_paths
    from engine.weights.loader import MODEL, resolve
    path = resolve(a.model or MODEL)
    out = os.path.expanduser(a.out) if a.out else prepared_dir(path)
    print(f"prepared directory {out} for {os.path.basename(path)}:")
    if model_type(path) == "qwen4_exp":
        from tools.requant_qwen4exp import parse_formats
        try:
            prep = prepare4(path, out, skip_int=a.skip_int, formats=parse_formats(a.formats))
        except subprocess.CalledProcessError as e:
            print(f"tools/requant_qwen4exp.py failed (exit {e.returncode}); the manifest is not written", file=sys.stderr)
            return 1
        print(f"decode: {len(prep.decode)} files of copies" if prep.decode else "decode: the BF16 weights (no copies)")
        return 0
    try:
        prep = prepare(path, out, bf16=a.bf16, skip_int=a.skip_int, skip_lowrank=a.skip_lowrank,
                       drafter=os.path.expanduser(a.drafter) if a.drafter else None, legacy_decode=decode_copies_paths(path),
                       legacy_drafter=os.path.join(DRAFT_DIR, DRAFTER_FILE), legacy_lowrank=os.path.join(DRAFT_DIR, LOWRANK_FILE))
    except subprocess.CalledProcessError as e:
        print(f"a tool failed (exit {e.returncode}): {' '.join(os.path.basename(x) for x in e.cmd[1:3])}; the manifest holds what is done so far",
              file=sys.stderr)
        return 1
    decode = "INT6 / INT5 copies" if prep.decode else "the FP8 weights"
    drafter = "fine-tuned" if prep.drafter else "the checkpoint head"
    print(f"decode: {decode}; drafter: {drafter}; draft head: {'low-rank' if prep.lowrank else 'full'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
