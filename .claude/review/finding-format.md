# Finding format

Every finder (slop, security, memrace) reports in this format. The `reviewing-features` skill reads it, and
`tools/review.py add` writes it into the ledger.

## The output

Your final message is one fenced `json` block and nothing else. The block holds an array with one object per finding.
An empty array means that you found nothing. Say so with `[]`. Do not invent a finding to fill the array.

The example below is invented: the file and the defect do not exist.

```json
[
  {
    "lens": "security",
    "title": "A negative widget count reaches the engine thread unchecked",
    "file": "engine/server/example.py",
    "line": 120,
    "symbol": "parse_widgets",
    "bug_class": "unchecked-input-to-engine",
    "severity": "critical",
    "feasibility": "high",
    "introduced": "diff",
    "evidence": "example.py:120 parse_widgets copies body['widgets'] without a range check -> example.py:310 torch.empty(widgets) on the engine thread -> RuntimeError ends the engine loop",
    "failure_scenario": "POST /v1/chat/completions with widgets=-1: every request in flight fails and the server stops serving",
    "fix": "reject widgets < 0 with a 400 in parse_widgets, next to the other range checks",
    "other_instances": ["engine/server/example.py:131 gadgets"],
    "verify_with": ""
  }
]
```

| Field | Content |
|---|---|
| `lens` | `slop`, `security` or `memrace`. |
| `title` | The claim, at most 80 characters. |
| `file`, `line` | The line where the defect is. Use the line number in the working tree, not in the patch. |
| `symbol` | The function or class that holds the line. The ledger uses it to recognize the same finding again. |
| `bug_class` | A kebab-case name for the class of defect. Use a name from the list below, or from the `known` list in your prompt, when one fits. |
| `severity` | `critical`, `high`, `medium` or `low`, as REVIEW.md defines them. |
| `feasibility` | `high`, `medium` or `low`: how cheap and safe the fix is, as REVIEW.md defines it. |
| `introduced` | `diff` when the diff adds the defect or makes it reachable. `pre-existing` otherwise. |
| `evidence` | The path from an entry point to the line, with `file:line` for each step. |
| `failure_scenario` | A concrete input, or a concrete interleaving of threads, and the wrong result that it gives. |
| `fix` | The smallest fix. |
| `other_instances` | The other places with the same defect, as `file:line`. The fix step fixes the class, not only one line. |
| `verify_with` | Memrace only: the tool and the repro idea (`.claude/agents/memrace-verifier.md` has the tools). |

## Rules

- Report one finding per root cause. Put the other places with the same cause in `other_instances`.
- Give a `file:line` for each claim. Do not make a claim from a name alone: read the code.
- Do not report a finding that the `known` list in your prompt already has, also when the line number changed.
- Report at most three findings of severity `low`.
- Do not report what ruff already checks: import order, unused names, line length.

## Bug classes

Use these names when they fit. Add a new name only for a new class.

| Lens | Classes |
|---|---|
| security | `unchecked-input-to-engine`, `exception-on-engine-thread`, `unbounded-request-cost`, `credential-leak`, `log-leak`, `bind-or-auth-widened`, `unsafe-deserialization`, `path-from-input`, `shell-injection`, `permission-widened`, `dependency-risk` |
| memrace | `missing-wait-stream`, `pinned-buffer-reuse`, `capture-unsafe-op`, `smem-race`, `kernel-oob`, `uninitialized-read`, `lock-missing`, `lock-order`, `lost-wakeup`, `check-then-act`, `loop-blocked`, `cross-thread-asyncio`, `leak-per-request`, `unbounded-cache`, `leak-on-error-path`, `thread-leak`, `fd-leak`, `disk-budget` |
| slop | the smell names of `.claude/review/slop-detector.md` in kebab case, such as `unnecessary-abstraction`, `duplicated-logic`, `inconsistent-convention`, `excessive-defensive`, `catch-all-except`, `junk-drawer`, `implementation-tests`, `fake-coverage`, `narrating-comment`, `meaningless-name`, `gratuitous-async`, `resource-lifecycle`, `magic-number`, `excessive-config`, `mini-framework`, `scattered-ownership`, `change-propagation`, `future-proofing` |
