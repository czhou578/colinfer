"""The parsers and the table of tools/mem_floor.py, on recorded text. The measurement itself needs the GPU and runs
by hand (docs/history/PLAN_QWEN35_MOE.md section 5)."""
import pytest

from tools import mem_floor as mf

MEMINFO = "MemTotal:       127600812 kB\nMemFree:        44823552 kB\nMemAvailable:   89494532 kB\nCached:         34138532 kB\n"
SMAPS = ("aaaab0000000-ffffe0000000 ---p 00000000 00:00 0                        [rollup]\n"
         "Rss:             3260420 kB\nPss:             3036534 kB\nPss_Dirty:       2436000 kB\nPss_Anon:        2436772 kB\n"
         "Pss_File:         535818 kB\nPss_Shmem:         63944 kB\nAnonymous:       2436772 kB\nSwap:                  0 kB\n")


def test_meminfo_reads_a_field_in_bytes():
    assert mf.meminfo(MEMINFO, "MemAvailable") == 89494532 * 1024
    assert mf.meminfo(MEMINFO, "MemTotal") == 127600812 * 1024
    with pytest.raises(KeyError):
        mf.meminfo(MEMINFO, "MemMissing")  # also not a prefix match: "Mem" is no field


def test_smaps_splits_the_resident_memory():
    s = mf.smaps(SMAPS)
    assert s == {"rss": 3036534 * 1024, "anon": 2436772 * 1024, "file": (535818 + 63944) * 1024}
    assert s["anon"] + s["file"] == s["rss"]
    assert mf.smaps("") == {"rss": 0, "anon": 0, "file": 0}


def test_gpu_bytes_finds_the_process():
    smi = "3383498, 25827\n4100, 312\n"
    assert mf.gpu_bytes(smi, 3383498) == 25827 * 2**20
    assert mf.gpu_bytes(smi, 4100) == 312 * 2**20
    assert mf.gpu_bytes(smi, 410) == 0                   # no match on a part of a pid
    assert mf.gpu_bytes("4100, [N/A]\n", 4100) == 0      # the driver gives no value
    assert mf.gpu_bytes("", 4100) == 0


def test_table_gives_the_growth_of_each_step():
    row = {"what": "w", "peak": 0, "rss": 0, "anon": 0, "file": 0, "gpu": 0, "reserved": 0, "allocated": 0}
    rows = [{**row, "step": "a", "avail": 2 * mf.GB, "peak": 3 * mf.GB}, {**row, "step": "b", "avail": 15 * mf.GB // 10, "gpu": mf.GB // 2}]
    lines = mf.table(rows).splitlines()
    assert len(lines) == 4 and lines[0].count("|") == lines[1].count("|") == lines[2].count("|") == lines[3].count("|")
    assert lines[2].startswith("| a | w | 2.00 | +2.00 | 3.00 |")
    assert lines[3].startswith("| b | w | 1.50 | -0.50 | 0.00 |") and "| 0.50 |" in lines[3]


def test_steps_are_distinct_and_start_with_the_interpreter():
    names = [n for n, _ in mf.STEPS]
    assert names[0] == "python" and len(set(names)) == len(names)
