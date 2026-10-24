# Review rules

The reviewers of colinfer read this file: the finders and the refuters of the `reviewing-features` skill
(`.claude/agents/`, see `docs/review.md`), and later the review of pull requests on GitHub. It tells a reviewer what failure looks like in
this repository. Keep it short. This file is complete in itself: no section sends you to a further file.

## Contents

- [What colinfer is](#what-colinfer-is)
- [What failure looks like here](#what-failure-looks-like-here): the six failures, in order of cost
- [Concurrency map](#concurrency-map): the threads and streams, and what they share
- [Trust boundaries](#trust-boundaries)
- [Conventions](#conventions): what the slop lens checks a patch against
- [Security checklist](#security-checklist): what the security lens traces first
- [Repro](#repro): how the memrace verifier runs a test on this machine
- [Not findings](#not-findings): the intentional choices
- [Severity](#severity), [Feasibility](#feasibility), [Triage](#triage)
- [Evidence](#evidence): what a finding must cite
- [Skip](#skip): the paths that no reviewer reads
- [Checks](#checks): the check commands, for the gate of a review and the Stop hook

## What colinfer is

colinfer is an inference engine for one model (Qwen3.8-27B) on one machine (a DGX Spark, GB10) for one person. The
server runs one request at a time. Its main promise is exact output: the tokens with speculation are the tokens
without speculation, bit for bit.

## What failure looks like here

1. **The output changes.** A change must not change the tokens of a greedy or seeded request, with speculation on or
   off (`tests/golden.py`, `tests/scheduler_check.py`). A change that is faster but not exact is critical.
2. **The engine thread ends.** One engine thread owns the GPU (`engine/server/worker.py`). Any exception that reaches
   `Worker._loop` ends the process with `os._exit(1)`, and every request in flight fails. A device-side assert does
   the same, through the poisoned CUDA context. Thus the HTTP layer must check each client value with a 400 before the
   value reaches the engine thread. `Stream.feed` and `Stream.finish` contain their own exceptions: keep it so.
3. **The machine powers off.** The GPU and the host share 128 GB of unified memory, and the server takes about 88 GB.
   An over-commit powers off the machine without an out-of-memory error. An allocation whose size a request sets, or
   that grows with each request, is high or critical.
4. **Two threads or two streams race.** The map below lists where. A race that corrupts the KV cache, the GDN state or
   a cached conversation is critical.
5. **The login token leaks.** The gateway (`engine/server/gateway.py`) forwards the claude.ai login token of Claude
   Code to api.anthropic.com only. A change that sends the token to the local server, writes it to a log, or follows
   a redirect with it is critical.
6. **A cached conversation restores wrong.** The disk cache (`engine/runtime/sessions.py`) must give back the exact
   bytes of the slot. A damaged or missing file must make the read fail, and the request then prefills.

## Concurrency map

| Thread or stream | Shares with | Synchronization |
|---|---|---|
| The asyncio loop of uvicorn (`engine/server/api.py`) | the engine thread; the threads of `asyncio.to_thread` (template render, tokenize, grammar compile) | `Worker.inbox` (`queue.Queue`); `loop.call_soon_threadsafe` into the asyncio queue of each request; the lock of `ChatFormat` |
| The engine thread (`Worker`) | everything on the GPU | it is the only thread that runs the scheduler |
| The session writer (`SessionStore._run`) | the slots' KV rows, the checkpoint ring, pinned buffers, its own CUDA stream | `SessionStore.lock` and `cv`; `wait()` before a slot rewrites rows that a save reads; `pinned()` |
| The direct readers (`engine/weights/direct.py`) | pinned chunk buffers | the `free` and `filled` queues |
| Side CUDA streams (`engine/model/fast.py`, `engine/spec/mtp.py`) | the tensors of the main stream | `wait_stream` in both directions |
| CUDA graphs (`engine/model/fast.py`) | a memory pool | no allocation, no host sync and no new stream inside a capture |
| Non-blocking copies from pinned host buffers (scheduler masks, sessions, drafter) | the host code that reuses the buffer | the copy must finish before the host writes the buffer again |

## Trust boundaries

- **Untrusted:** the bodies and headers of HTTP requests to the server and to the gateway: messages, base64 images,
  tool schemas, JSON schemas, stop lists, sampling values, token ids.
- **Trusted:** the checkpoints in the Hugging Face cache, the command-line flags, the night-shift task file.
- **The engine's own files:** the session cache and the prepared directory under `~/.cache/colinfer`. The engine
  wrote them, but a damaged file must fail its read, not the server.

## Conventions

The slop lens checks a patch against these conventions. Several commits removed duplication on purpose: one definition
of the kernel shapes (`csrc/shapes.h`), one `build_engine()` for the server, the checks and the benchmarks
(`engine/runtime/build.py`), one `MODEL`, one timing loop, the server's defaults in `ServerConfig`
(`engine/server/worker.py`). A patch that adds a second copy of one of these is a finding.

- Client errors are 400s. Server bugs are 500s with a traceback.
- A comment tells why, not what. Module docstrings and docs use ASD-STE100 style (short sentences, active voice).
- The tests run the real code on stubs (`tests/test_scheduler.py` `StubModel`, `tests/test_api.py` `FakeWorker`). A
  test that only checks that a mock was called is a finding.
- A flag that never changes on this machine is a finding (a commit removed ten of them).

## Security checklist

The security lens traces these first, in this order.

1. **Client values that reach the engine thread.** A value from a request body that reaches the engine thread or the
   GPU without a check in the HTTP layer (`engine/server/api.py`, `openai.py`, `anthropic.py`): token ids, sampling
   values, sizes, stop lists, tool schemas, JSON schemas, and the `int()` and `float()` of JSON values (`Infinity`
   parses).
2. **A cost without a limit from one request.** Memory first (failure 3), then time: the grammar compile of a JSON
   schema, the template render, the decode of an image, the prefill of a long prompt. The limits are `max_queue`, `max_body_bytes`, the prompt text (`max_text_bytes`),
   `max_output_tokens`, `max_request_seconds`, `image_max_pixels` and the context length. Look for a path around them.
3. **Images.** The decode of client bytes with PIL: a decompression bomb, a size check after the decode instead of
   before, a format that PIL should not open.
4. **The gateway.** The `authorization` and `x-api-key` headers go to api.anthropic.com only, never to the local
   server, never into a log. Routing by model name, the hop-by-hop headers, `follow_redirects=False`, the bind address.
5. **Authentication and exposure.** The `--api-key` check (constant-time, which paths it covers), the bind address,
   CORS, and what `/health`, `/metrics` and `/v1/status` show.
6. **Files.** A path made from request data. The temporary file and rename of the session cache. `torch.load` without
   `weights_only=True`, `np.load(allow_pickle=True)`, `pickle`.
7. **Subprocesses.** `tools/night_shift.py` builds argument lists without a shell. Look for a shell, for task text in
   a place where it becomes an option, for permission rules that widen, for `--dangerously-skip-permissions`, and for
   keys in an environment or a log.
8. **Logs.** A request body, a header or a key in a log line.

## Repro

The memrace verifier writes a repro test for one leak or race, and runs it. These are its limits on this machine.

> **CAUTION:** The GPU and the host share 128 GB of unified memory. An over-commit powers off the machine without an
> out-of-memory error. Before each command that uses the GPU, check the memory:
> `awk '/MemAvailable/ {if ($2 < 16 * 1024 * 1024) {print "memory: " int($2 / 1048576) " GB available"; exit 1}}' /proc/meminfo && <command>`.
> Never load the real model: no `build_engine`, no `tests/golden.py`, no `tests/scheduler_check.py`, no
> `tests/check.sh` without `--quick`. Run one verifier at a time, and none while `tests/check.sh` runs.

- The interpreter is `.venv/bin/python`: `.venv/bin/python -m pytest <file> -q -x`, so that a sanitizer wraps Python
  and not `uv`.
- Reuse the stubs: `StubModel`, `stub_prefill`, `Checked` and `prompt` of `tests/test_scheduler.py`, the `make`
  fixture pattern of `tests/test_sessions.py`, `FakeWorker` and `client` of `tests/test_api.py`. The package `tests`
  imports from any directory.
- To hold a thread at a critical point, gate it with a `threading.Event` through `monkeypatch`, as
  `test_a_ring_buffer_is_not_reused_while_its_save_is_pending` in `tests/test_sessions.py` does.
- Each tool of the verifier's table was tested on this machine (GB10, CUDA 13.0, torch 2.14) on 2026-10-16: each one
  caught a planted bug and passed the fixed version.

## Not findings

These choices are intentional. Do not report them.

- The server has no authentication by default and listens on 127.0.0.1 (README, "Quick start"). With `--api-key`, the
  `/v1/` endpoints require the key.
- The server runs one request at a time. The three slots are a cache of conversations, not parallel requests.
- The engine loop ends the process on any exception, and systemd restarts it. Report the input that reaches the loop,
  not the exit.
- `engine/model/qwen35.py` repeats the math of the fast path in plain PyTorch. It is the reference of the kernel tests.
- The kernels compile for sm_121a only, with `--fmad=false`.
- The server fetches no URLs. Images arrive as base64.
- The session writer and the direct readers are daemon threads.
- A number with a measurement in `docs/history/`, in a commit message or in a comment is not a magic number.

## Severity

Each lens uses these four levels:

- **critical:** one of the failures 1, 2, 3, 5 or 6 above; a race that corrupts state; a security boundary that fails.
- **high:** a leak with each request; a race that fails one request; an error that is silently lost; a cost without a
  limit from one request (time, not memory); a large duplication of logic; a bypass of a type or a check with a real
  risk at run time.
- **medium:** an abstraction without a present purpose, a poor module boundary, a weak test, a setting that never
  changes, repeated defensive code.
- **low:** a name, a comment that repeats the code, a small duplication, a local difference of style.

Do not raise a severity because code is ugly.

## Feasibility

Feasibility tells how cheap and safe the fix is.

- **high:** a local change, in one function or at the call sites of one pattern. The existing tests, or one new test,
  show that the fix works. The output, the file layout on disk and the API do not change.
- **medium:** a change in several modules, a new test fixture, or a measured performance cost.
- **low:** a design change, a kernel rewrite, or a change of the disk format or of the API.

## Triage

The `reviewing-features` skill fixes a finding only when all of these are true:

- A refutation failed: two of the refuters could not disprove it, or a repro fails on the current code.
- Its severity is critical or high.
- Its feasibility is high.
- The diff introduced it.

A confirmed critical or high finding with medium or low feasibility goes to you for a decision. A finding that the diff
did not introduce goes to the backlog.

## Evidence

- Give a `file:line` for each step from an entry point to the defect. An entry point is an HTTP route, a command-line
  flag, the engine loop, or the run function of a thread.
- Read the code. Do not judge code from its name.
- For a race, name the two threads or streams, the two lines, and the order of events that fails.
- For a leak, name what grows, with what it grows, and what should free it.
- A run confirms a memrace finding, not a vote: a repro test that fails on the current code because of the defect.
- Report at most three low findings per lens. Do not report what ruff checks.

## Skip

`tools/review.py` reads the paths in this section. No diff of the review harness shows them.

- `csrc/third_party`
- `uv.lock`
- `docs/history`
- `tests/golden`
- `*.npy`

## Checks

`.claude/hooks/quick-check.sh`, the gate of a review and the Stop hook read this section. The full check loads the model and
needs about 105 GB of free memory: stop the server first.

- `code`: `engine`, `tests`, `tools`, `bench`, `csrc`, `pyproject.toml`
- `lint`: `uv run ruff check engine tests tools bench`
- `test`: `uv run pytest tests/ -q -x`
- `full`: `tests/check.sh`
- `full-for`: `engine/model`, `engine/spec`, `engine/runtime`, `engine/weights`, `csrc`
- `min-gb`: `16`
