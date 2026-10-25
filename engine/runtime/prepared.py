"""The prepared directory: the offline artifacts that make decode faster, in one place with a manifest.

tools/prepare.py makes it at ~/.cache/colinfer/prepared/<snapshot>/ (PREPARED_DIR): the decode copies, and for the 27B
also the fine-tuned MTP drafter when one exists and the low-rank draft head. manifest.json names the files.

The manifest's "decode" list names the decode copies of the checkpoint's family (the directory belongs to one checkpoint):
for the 27B (qwen3_5) the INT6 / INT5 copies of its FP8 projections (DECODE_FILES, engine/model/fast.py
attach_decode_copies); for Qwen3.8-Flash-Next (qwen4_exp) the copies of its BF16 matrices, one file per class
(decode4_file, tools/requant_qwen4exp.py, engine/model/fast_qwen4exp.py attach_decode_copies4). The optional "family" key
holds the checkpoint's model_type (tools/prepare.py writes it; a manifest without it is the 27B's). The version stays 1:
the 27B's manifests read as before. The engine reads the manifest at startup (engine/runtime/build.py). Without one,
the engine runs from the checkpoint alone: the checkpoint's weights at decode (the 27B: FP8 projections; qwen4_exp:
BF16), the checkpoint's MTP head and the full draft head. Thus the engine has two configurations, prepared or plain,
instead of one per optional file. A manifest that names a missing file is an
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
# qwen4_exp: the classes of tools/requant_qwen4exp.py and the format of each (the decision and its measurements are in that
# tool's docstring: the classes in bf16 have no copy, because their copies break the perplexity gate)
DECODE4_CLASSES = ("gdn_in", "gdn_out", "attn", "lm_head", "hc", "hc_up", "shared")
DECODE4_FORMATS = {"gdn_in": "bf16", "gdn_out": "bf16", "attn": "int6", "lm_head": "int6", "hc": "bf16", "hc_up": "int8", "shared": "bf16"}
LOWRANK_FILE = "draft_head_pca.safetensors"  # as tools/lowrank_draft_head.py names it


@dataclasses.dataclass(frozen=True)
class Prepared:
    dir: str
    checkpoint: str              # the basename of the checkpoint's snapshot directory
    decode: tuple[str, ...]      # the INT6 / INT5 decode copies (absolute paths), or empty: decode streams the FP8 weights
    drafter: str | None          # the fine-tuned MTP head, or None: the checkpoint's
    lowrank: str | None          # the low-rank draft head, or None: the full head
    made: str = ""
    family: str = ""              # the checkpoint's model_type, when the manifest names it


def decode4_file(cls: str, fmt: str) -> str:
    """The file of one class of the qwen4_exp decode copies (tools/requant_qwen4exp.py)."""
    return f"decode4_{cls}_{fmt}.safetensors"


def decode4_files(formats: dict[str, str] | None = None) -> list[str]:
    """The files of the classes with a copy (a class in "bf16" has none), in DECODE4_CLASSES order."""
    f = DECODE4_FORMATS if formats is None else formats
    return [decode4_file(c, f[c]) for c in DECODE4_CLASSES if f.get(c, "bf16") != "bf16"]


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
    return Prepared(d, cp, tuple(one(n) for n in decode), one(m.get("drafter")), one(m.get("lowrank")), str(m.get("made") or ""),
                    str(m.get("family") or ""))


def write(d: str, checkpoint: str, decode, drafter: str | None, lowrank: str | None, family: str | None = None) -> str:
    """Writes the manifest of d (file names relative to d) and returns its path. family: the checkpoint's model_type
    (left out when None, as the 27B's manifests)."""
    m = {"version": VERSION, "checkpoint": checkpoint, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "decode": list(decode),
         "drafter": drafter, "lowrank": lowrank}
    if family is not None:
        m["family"] = family
    mp = os.path.join(d, MANIFEST)
    tmp = mp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
        f.write("\n")
    os.replace(tmp, mp)
    return mp
