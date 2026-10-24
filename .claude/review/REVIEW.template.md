# Review rules

The reviewers of this repository read this file: the finders and the refuters of the `reviewing-features` skill
(`.claude/agents/`). It tells a reviewer what failure looks like in this repository. Keep it short. This file is complete in itself: no section sends
the reader to a further file. Replace each paragraph in *italics* with the facts of this project, and delete the
italics.

## Contents

- [What this project is](#what-this-project-is)
- [What failure looks like here](#what-failure-looks-like-here): the failures, in order of cost
- [Concurrency map](#concurrency-map): the threads, the tasks and the streams, and what they share
- [Trust boundaries](#trust-boundaries)
- [Conventions](#conventions): what the slop lens checks a patch against
- [Security checklist](#security-checklist): what the security lens traces first
- [Repro](#repro): how the memrace verifier runs a test here
- [Not findings](#not-findings): the intentional choices
- [Severity](#severity), [Feasibility](#feasibility), [Triage](#triage)
- [Evidence](#evidence): what a finding must cite
- [Skip](#skip): the paths that no reviewer reads
- [Checks](#checks): the lint and test commands, for the gate and the Stop hook

## What this project is

*One paragraph: what the software does, for whom, on what machine, and its main promise (exact output, no data loss,
a latency bound).*

## What failure looks like here

*A numbered list, in order of cost. Each item names the failure, the component where it happens, and the test or the
check that shows it. Example: "The engine thread ends. Any exception that reaches `Worker.loop` ends the process, and
every request in flight fails. Thus the HTTP layer must check each client value before it reaches the engine thread."*

## Concurrency map

*A table with one row per thread, task or stream: what it shares with the others, and how they synchronize (the lock,
the queue, the `wait_stream`). The memrace lens starts from this table.*

| Thread, task or stream | Shares with | Synchronization |
|---|---|---|
| | | |

## Trust boundaries

- **Untrusted:** *the inputs that an attacker controls: request bodies and headers, files from users, messages from a
  broker.*
- **Trusted:** *the inputs that the operator controls: configuration, command-line flags, checkpoints.*
- **The project's own files:** *caches and state that the project writes. A damaged file must fail its own read, not
  the process.*

## Conventions

*The conventions that the project made on purpose. The slop lens reports a patch that breaks one: a second copy of a
single definition, a different error convention, a test that only checks a mock.*

- *One definition of X lives in `path`; a second copy is a finding.*
- *Client errors are 400s. Server bugs are 500s with a traceback.*
- *A comment tells why, not what.*
- *The tests run the real code on stubs (`tests/...`). A test that only checks that a mock was called is a finding.*

## Security checklist

*The checks that matter most here, in order, each with its files. The security lens traces these first, then its
general classes (unchecked input, unbounded cost, deserialization, paths, shells, credentials, exposure,
dependencies).*

1. *...*

## Repro

*How the memrace verifier runs a test in this repository: the interpreter (so that a sanitizer wraps it and not a
launcher), the stubs and fixtures to reuse, what it must never run, and the limits of the machine. Start a caution
with the consequence, then the action that avoids it.*

- The interpreter is `...`: `... -m pytest <file> -q -x`.
- Reuse the stubs: *...*
- Never run: *...*

## Not findings

These choices are intentional. Do not report them.

- *...*

## Severity

Each lens uses these four levels:

- **critical:** one of the failures above that ends the process, corrupts state, or breaks a security boundary.
- **high:** a leak with each request; a race that fails one request; an error that is silently lost; a cost without a
  limit from one request; a large duplication of logic; a bypass of a type or a check with a real risk at run time.
- **medium:** an abstraction without a present purpose, a poor module boundary, a weak test, a setting that never
  changes, repeated defensive code.
- **low:** a name, a comment that repeats the code, a small duplication, a local difference of style.

Do not raise a severity because code is ugly.

## Feasibility

Feasibility tells how cheap and safe the fix is.

- **high:** a local change, in one function or at the call sites of one pattern. The existing tests, or one new test,
  show that the fix works. The output, the file layout on disk and the API do not change.
- **medium:** a change in several modules, a new test fixture, or a measured performance cost.
- **low:** a design change, a rewrite of a component, or a change of the disk format or of the API.

## Triage

The `reviewing-features` skill fixes a finding only when all of these are true:

- A refutation failed: two of the refuters could not disprove it, or a repro fails on the current code.
- Its severity is critical or high.
- Its feasibility is high.
- The diff introduced it.

A confirmed critical or high finding with medium or low feasibility goes to you for a decision. A finding that the diff
did not introduce goes to the backlog.

## Evidence

- Give a `file:line` for each step from an entry point to the defect. An entry point is *a route, a command-line flag,
  the main loop, or the run function of a thread*.
- Read the code. Do not judge code from its name.
- For a race, name the two threads, the two lines, and the order of events that fails.
- For a leak, name what grows, with what it grows, and what should free it.
- A run confirms a memrace finding, not a vote: a repro test that fails on the current code because of the defect.
- Report at most three low findings per lens. Do not report what the linter checks.

## Skip

`tools/review.py` reads the paths in this section, one `path` per bullet (a directory, a file, or a glob). No diff of
the review harness shows them.

- `*.lock`

## Checks

`.claude/hooks/quick-check.sh`, the gate of a review and the Stop hook read this section, one `key` per bullet with its values
in backticks. Without this section, the Stop hook does nothing and a review runs no check. Write the bullets as in the
example, outside the code block:

    - `code`: `src`, `tests`, `pyproject.toml`
    - `lint`: `ruff check src tests`
    - `test`: `pytest tests/ -q -x`
    - `full`: `make check`
    - `full-for`: `src/core`, `*.cu`
    - `min-gb`: `16`

| Key | Content |
|---|---|
| `code` | The paths whose change makes the Stop hook run. Their hash is the fingerprint of a pass. |
| `lint`, `test` | The quick check, in this order. Each one runs through a shell at the root of the repository. |
| `full`, `full-for` | The full check, and the paths of the scope that need it (a directory, a file, or a glob). Say above what the full check needs. |
| `min-gb` | The tests do not run with less memory available (`MemAvailable`, in GB). Omit it to always run them. |
