"""The prepared directory: the offline artifacts that make decode faster, in one place with a manifest.

tools/prepare.py makes it at ~/.cache/colinfer/prepared/<snapshot>/ (PREPARED_DIR): the INT6 / INT5 decode copies of
the attention / GDN projections, the fine-tuned MTP drafter when one exists, and the low-rank draft head. manifest.json
names the files. The engine reads the manifest at startup (engine/runtime/build.py). Without one, the engine runs from
the checkpoint alone: FP8 projections at decode, the checkpoint's MTP head and the full draft head. Thus the engine has
two configurations, prepared or plain, instead of one per optional file. A manifest that names a missing file is an
error, not a fallback: run tools/prepare.py again.

The files keep the names that their tools give them, so a hard link into the prepared directory keeps the size and the
modification time. The disk cache of conversations fingerprints the files by those (engine/runtime/build.py _file_id),
so the cache of a server that read the files from their old places stays valid.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

PREPARED_DIR = os.path.expanduser("~/.cache/colinfer/prepared")
MANIFEST = "manifest.json"
VERSION = 1
DECODE_FILES = ("attn_gdn_int6_self_attn.safetensors", "attn_gdn_int5_linear_attn.safetensors")  # as tools/int6_requant.py names them
DRAFTER_FILE = "mtp_ft.safetensors"      # as tools/train_drafter.py names it
LOWRANK_FILE = "draft_head_pca.safetensors"  # as tools/lowrank_draft_head.py names it


@dataclasses.dataclass(frozen=True)
class Prepared:
    dir: str
    checkpoint: str              # the basename of the checkpoint's snapshot directory
    decode: tuple[str, ...]      # the INT6 / INT5 decode copies (absolute paths), or empty: decode streams the FP8 weights
    drafter: str | None          # the fine-tuned MTP head, or None: the checkpoint's
    lowrank: str | None          # the low-rank draft head, or None: the full head
    made: str = ""


def prepared_dir(path: str) -> str:
    """The prepared directory of a checkpoint directory."""
    return os.path.join(PREPARED_DIR, os.path.basename(os.path.normpath(path)))


def read(path: str, dir: str | None = None) -> Prepared | None:
    """The manifest of a checkpoint's prepared directory (dir: another directory), or None when there is none. A
    manifest of another version or checkpoint, or one that names a missing file, raises ValueError."""
    d = dir or prepared_dir(path)
    mp = os.path.join(d, MANIFEST)
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        m = json.load(f)
    if not isinstance(m, dict) or m.get("version") != VERSION:
        raise ValueError(f"{mp}: not a prepared manifest of version {VERSION}; run tools/prepare.py again")
    cp = os.path.basename(os.path.normpath(path))
    if m.get("checkpoint") != cp:
        raise ValueError(f"{mp}: prepared for {m.get('checkpoint')!r}, not for {cp!r}")

    def one(name):
        if name is None:
            return None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{mp}: a file name must be a string, not {name!r}")
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise ValueError(f"{mp} names {name}, which is missing; run tools/prepare.py again")
        return p
    decode = m.get("decode") or []
    if not isinstance(decode, list):
        raise ValueError(f"{mp}: decode must be a list of file names")
    return Prepared(d, cp, tuple(one(n) for n in decode), one(m.get("drafter")), one(m.get("lowrank")), str(m.get("made") or ""))


def write(d: str, checkpoint: str, decode, drafter: str | None, lowrank: str | None) -> str:
    """Writes the manifest of d (file names relative to d) and returns its path."""
    m = {"version": VERSION, "checkpoint": checkpoint, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "decode": list(decode),
         "drafter": drafter, "lowrank": lowrank}
    mp = os.path.join(d, MANIFEST)
    tmp = mp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
        f.write("\n")
    os.replace(tmp, mp)
    return mp
