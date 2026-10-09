"""Minimal Prometheus metrics (PLAN.md 4.6: `/metrics` with step-time histograms), with no dependency: counters,
gauges and fixed-bucket histograms in the Prometheus text format. The engine thread writes them, and the HTTP thread
reads them. Single attribute updates under the GIL are consistent enough for monitoring."""
from __future__ import annotations

import bisect
import math

STEP_BUCKETS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0)
TTFT_BUCKETS = (0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0, 300.0)
COUNT_BUCKETS = (1, 2, 3, 4, 5, 6, 8)


def _labels(lab: dict) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(lab.items())) + "}" if lab else ""


class Counter:
    def __init__(self, name, help_):
        self.name, self.help, self.v = name, help_, {}

    def inc(self, n=1.0, **lab):
        key = tuple(sorted(lab.items()))
        self.v[key] = self.v.get(key, 0.0) + n

    def get(self, **lab):
        return self.v.get(tuple(sorted(lab.items())), 0.0)

    def clear(self):
        self.v = {}

    def render(self):
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, v in sorted(self.v.items()):
            out.append(f"{self.name}{_labels(dict(key))} {v:g}")
        return out


class Gauge(Counter):
    def set(self, v, **lab):
        self.v[tuple(sorted(lab.items()))] = float(v)

    def render(self):
        out = super().render()
        out[1] = f"# TYPE {self.name} gauge"
        return out


class Histogram:
    def __init__(self, name, help_, buckets):
        self.name, self.help, self.buckets = name, help_, tuple(buckets)
        self.h = {}  # labels -> [bucket counts..., +Inf], sum

    def observe(self, x, **lab):
        key = tuple(sorted(lab.items()))
        e = self.h.setdefault(key, [[0] * (len(self.buckets) + 1), 0.0])
        e[0][bisect.bisect_left(self.buckets, x)] += 1
        e[1] += x

    def count(self, **lab):
        e = self.h.get(tuple(sorted(lab.items())))
        return sum(e[0]) if e else 0

    def clear(self):
        self.h = {}

    def render(self):
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, (counts, total) in sorted(self.h.items()):
            lab, acc = dict(key), 0
            for b, c in zip(self.buckets + (math.inf,), counts):
                acc += c
                out.append(f"{self.name}_bucket{_labels({**lab, 'le': '+Inf' if b == math.inf else f'{b:g}'})} {acc}")
            out.append(f"{self.name}_sum{_labels(lab)} {total:g}")
            out.append(f"{self.name}_count{_labels(lab)} {acc}")
        return out


class Metrics:
    def __init__(self):
        self.step_seconds = Histogram("colinfer_step_seconds", "engine step time by kind (prefill chunk, decode step / spec cycle) and batch width",
                                      STEP_BUCKETS)
        self.ttft_seconds = Histogram("colinfer_ttft_seconds", "submit to first token, per request", TTFT_BUCKETS)
        self.queue_seconds = Histogram("colinfer_queue_seconds", "submit to slot admission, per request", TTFT_BUCKETS)
        self.tokens_per_cycle = Histogram("colinfer_tokens_per_cycle", "tokens emitted per speculative cycle per slot", COUNT_BUCKETS)
        self.requests = Counter("colinfer_requests_total", "finished requests by finish reason")
        self.prompt_tokens = Counter("colinfer_prompt_tokens_total", "prompt tokens of admitted requests")
        self.cached_tokens = Counter("colinfer_cached_prompt_tokens_total", "prompt tokens served from a prefix checkpoint")
        self.generated_tokens = Counter("colinfer_generation_tokens_total", "generated tokens")
        self.drafted = Counter("colinfer_spec_draft_tokens_total", "draft tokens proposed")
        self.accepted = Counter("colinfer_spec_accepted_tokens_total", "draft tokens accepted")
        self.queue_depth = Gauge("colinfer_queue_depth", "requests waiting for a slot")
        self.slots_busy = Gauge("colinfer_slots_busy", "slots holding a request, by phase")
        self.memory = Gauge("colinfer_gpu_memory_bytes", "torch allocator: allocated / reserved / cap")

    def reset(self):
        for m in self.__dict__.values():
            m.clear()

    def render(self) -> str:
        lines = []
        for m in self.__dict__.values():
            lines += m.render()
        return "\n".join(lines) + "\n"
