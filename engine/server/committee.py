"""Committee mode: a request answered by several members of the model, then reconciled into one reply.

A request whose model name ends in "-committee" (or is "committee") runs one member request per role: the same
conversation, with the role appended to its last message (an implementer, a skeptic and a reviewer by default). The
members run one after another through the single lane of the engine (engine/server/api.py run_committee). Then a
reconcile request, the conversation with the members' answers appended, writes the final answer. That is the reply the
client gets, in the format of the endpoint it asked. The members' answers go to the log, and a chat completion without
a stream returns them in a `committee` field.

Cost: about 4x the time of a single answer for answers of the usual length (three members and the reconcile pass, in
turn). The members' prompts share the conversation's prefix, so the prefix cache leaves them the role text to prefill,
and the reconcile request prefills the answers. The mode is for one question at a time, such as a review or a design
question. In an agentic session every tool-calling turn would pay the 4x, so a client switches the model name for the
question and back (docs/claude_code.md).

The roles and the reconcile instruction come from --committee-roles (a JSON file {"roles": [...], "reconcile": "..."}),
else the defaults below. --committee-size picks how many members: the roles in order, then EXTRA_ROLE.
"""
from __future__ import annotations

import dataclasses
import json
import os

SUFFIX = "-committee"
DEFAULT_SIZE = 3
ROLE_TAG = "[Committee role]"
DEFAULT_ROLES = (
    "You are the implementer of this committee. Give the most concrete and complete answer you can, with the specifics "
    "(names, lines, commands, numbers) that make it actionable.",
    "You are the skeptic of this committee. Look for what is wrong, missing or risky in the obvious answer: wrong assumptions, "
    "edge cases, failure modes, things that need checking. Say what you would verify and how.",
    "You are the reviewer of this committee. Weigh the trade-offs and the alternatives, say which option you would choose and "
    "why, and what you would not do.",
)
EXTRA_ROLE = "You are a member of this committee. Answer the request independently and carefully."
RECONCILE = ("The {n} members of a committee answered the request above independently. Their answers follow, in no particular "
             "order. Write the single best final answer to the request: keep what the members agree on, settle their "
             "disagreements by checking against the request and the context, keep the valid concerns, and drop errors and "
             "repetition. Write it as your own single answer: do not mention the committee, its members, their answers, "
             "drafts or reviews, and do not refer to what \"the reviews\" or \"the answers\" say.")


def is_committee(model) -> bool:
    """Whether a request's model name asks for committee mode: "<name>-committee", or "committee"."""
    return isinstance(model, str) and (model == "committee" or (model.endswith(SUFFIX) and len(model) > len(SUFFIX)))


@dataclasses.dataclass(frozen=True)
class Committee:
    roles: tuple[str, ...] = DEFAULT_ROLES
    reconcile: str = RECONCILE  # {n}: the number of answers

    @classmethod
    def from_config(cls, size: int | None, path: str | None) -> Committee | None:
        """The committee of --committee-size and --committee-roles; None when the mode is off (size 0). Without a size,
        the file's roles, else DEFAULT_SIZE members. More members than roles get EXTRA_ROLE."""
        roles, reconcile = list(DEFAULT_ROLES), RECONCILE
        if path:
            with open(os.path.expanduser(path)) as f:
                d = json.load(f)
            ok = isinstance(d, dict) and isinstance(d.get("roles"), list) and d["roles"] and all(isinstance(r, str) and r.strip() for r in d["roles"])
            if not ok:
                raise ValueError(f'{path}: "roles" must be a non-empty list of strings')
            rc = d.get("reconcile", RECONCILE)
            if not isinstance(rc, str) or not rc.strip():
                raise ValueError(f'{path}: "reconcile" must be a string')
            roles, reconcile = [r.strip() for r in d["roles"]], rc.strip()
        if size is None:
            size = len(roles) if path else DEFAULT_SIZE
        if size <= 0:
            return None
        return cls(tuple((roles + [EXTRA_ROLE] * size)[:size]), reconcile)


def _append(msgs: list[dict], text: str) -> list[dict]:
    """The messages with `text` at the end of the last user message (a paragraph of its text, or a text part after its
    images), or as a new user message when the last message is another role (a tool result, a system reminder). The
    conversation before stays the same, so the prefix cache serves it."""
    out = [dict(m) for m in msgs]
    if out and out[-1].get("role") == "user":
        c = out[-1].get("content")
        if isinstance(c, list):
            out[-1]["content"] = list(c) + [{"type": "text", "text": text}]
        else:
            out[-1]["content"] = f"{c}\n\n{text}" if isinstance(c, str) and c else text
        return out
    out.append({"role": "user", "content": text})
    return out


def member_messages(msgs: list[dict], role: str) -> list[dict]:
    """The conversation of one member: the request with its role appended."""
    return _append(msgs, f"{ROLE_TAG} {role}")


def reconcile_messages(msgs: list[dict], answers: list[str], reconcile: str = RECONCILE) -> list[dict]:
    """The conversation of the reconcile pass: the request with the instruction and the members' answers appended."""
    n = len(answers)
    parts = [reconcile.replace("{n}", str(n))]
    parts += [f"--- Answer {i} of {n} ---\n{a.strip()}" for i, a in enumerate(answers, 1)]
    parts.append("--- End of the answers ---")
    return _append(msgs, "\n\n".join(parts))


def answer_text(msg: dict) -> str:
    """A member's answer as text for the reconcile pass: its content, and each tool call as a line (the reasoning is
    left out)."""
    parts = [(msg.get("content") or "").strip()]
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        parts.append(f"[tool call] {fn.get('name')}({fn.get('arguments') or ''})")
    return "\n\n".join(p for p in parts if p) or "(no answer)"
