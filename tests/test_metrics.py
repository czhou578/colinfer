"""engine/runtime/metrics.py renders the Prometheus text format itself (no dependency): the lines must be what a scraper
expects, with cumulative buckets that include their bound, a +Inf bucket, and _sum / _count."""
from engine.runtime.metrics import Counter, Gauge, Histogram, Metrics


def test_histogram_renders_cumulative_buckets_sum_and_count():
    h = Histogram("t_seconds", "a time", (0.1, 1.0))
    for x in (0.05, 0.1, 0.5, 7.0):  # 0.1 sits on a bound: le is inclusive
        h.observe(x, kind="a")
    assert h.render() == [
        "# HELP t_seconds a time",
        "# TYPE t_seconds histogram",
        't_seconds_bucket{kind="a",le="0.1"} 2',
        't_seconds_bucket{kind="a",le="1"} 3',
        't_seconds_bucket{kind="a",le="+Inf"} 4',
        't_seconds_sum{kind="a"} 7.65',
        't_seconds_count{kind="a"} 4',
    ]
    assert h.count(kind="a") == 4 and h.count(kind="b") == 0


def test_counter_and_gauge_render_with_sorted_labels():
    c = Counter("req_total", "requests")
    c.inc(reason="stop")
    c.inc(2, reason="stop")
    c.inc()
    assert c.render() == ["# HELP req_total requests", "# TYPE req_total counter", "req_total 1", 'req_total{reason="stop"} 3']
    assert c.get(reason="stop") == 3 and c.get() == 1
    g = Gauge("busy", "slots")
    g.set(2, phase="decode", kind="x")
    g.set(0, phase="decode", kind="x")  # a gauge is set, not summed
    assert g.render() == ["# HELP busy slots", "# TYPE busy gauge", 'busy{kind="x",phase="decode"} 0']


def test_metrics_render_every_metric_once_and_reset_clears_them():
    m = Metrics()
    m.requests.inc(reason="stop")
    m.step_seconds.observe(0.08, kind="decode")
    text = m.render()
    assert text.endswith("\n") and text.count("# TYPE colinfer_requests_total counter") == 1
    assert 'colinfer_step_seconds_bucket{kind="decode",le="0.08"} 1' in text.splitlines()
    m.reset()
    assert m.requests.get(reason="stop") == 0 and m.step_seconds.count(kind="decode") == 0
