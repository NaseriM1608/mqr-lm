import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from mqr import sampling


def test_shard_selection_replays_independent_of_listing_order():
    files = [{"path": f"{i}.parquet"} for i in range(20)]
    chosen = sampling.choose_shards(files, 8, 4, "en")
    assert chosen == sampling.choose_shards(files[::-1], 8, 4, "en")
    assert len({item["path"] for item in chosen}) == 8
    assert chosen != sampling.choose_shards(files, 8, 5, "en")


def test_shards_not_restricted_to_prefix_and_uniform_across_seeds():
    files = [{"path": f"{i:02}.parquet"} for i in range(10)]
    counts = Counter(item["path"] for seed in range(2000)
                     for item in sampling.choose_shards(files, 2, seed, "en"))
    assert set(counts) == {item["path"] for item in files}
    assert all(300 < n < 500 for n in counts.values())


@pytest.mark.parametrize("counts", [[1], [10, 20], [999] * 11])
def test_window_bounds_strata_and_global_offsets(counts):
    windows = sampling.choose_windows(counts, 128, 7, "fa", "a.parquet")
    assert windows == sampling.choose_windows(counts, 128, 7, "fa", "a.parquet")
    groups = []
    for window in windows:
        group = window["row_group"]
        groups.append(group)
        assert window["row_group_start"] <= group < window["row_group_stop_exclusive"]
        assert 0 <= window["window_start"] <= counts[group] - window["window_rows"]
        assert window["row_group_first_row"] == sum(counts[:group])
    assert len(groups) == len(set(groups))


def test_window_group_and_interior_start_selection_are_uniform():
    counts = Counter()
    starts = Counter()
    for seed in range(4000):
        head = sampling.choose_windows([20] * 12, 4, seed, "en", "x")[0]
        counts[head["row_group"]] += 1
        starts[head["window_start"]] += 1
    assert set(counts) == {0, 1, 2, 3}
    assert all(850 < n < 1150 for n in counts.values())
    assert set(starts) == set(range(17))
    assert all(160 < n < 310 for n in starts.values())


@pytest.mark.parametrize("n,width", [(5, 1), (5, 3), (5, 5), (2, 2)])
def test_exact_row_probabilities_match_exhaustive_windows(n, width):
    for row in range(n):
        probability = sum(start <= row < start + width for start in range(n - width + 1)) / (n - width + 1)
        assert sampling.row_inclusion_probability(n, width, row) == probability
    assert sum(sampling.row_inclusion_probability(n, width, row) for row in range(n)) == pytest.approx(width)


def test_budget_counts_failed_requests_and_never_calls_over_budget():
    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise OSError("read failure")
    reservations = []
    reader = sampling.BudgetReader(Broken(b"abc"), 8, used=2, reserve=reservations.append)
    with pytest.raises(OSError):
        reader.read(4)
    assert reader.bytes_requested == 6
    assert reservations == [4]
    with pytest.raises(sampling.ReadBudgetExceeded):
        reader.read(3)
    with pytest.raises(sampling.ReadBudgetExceeded):
        reader.read()
    assert reservations == [4]


def test_inventory_records_publisher_metadata_without_credentials():
    class API:
        def list_repo_tree(self, repo, **kwargs):
            assert kwargs["revision"] == "a" * 40
            assert kwargs["path_in_repo"] == "data"
            return [SimpleNamespace(path="data/b.parquet", size=5, lfs=SimpleNamespace(sha256="f" * 64), blob_id="b"),
                    SimpleNamespace(path="data/README.md")]
    source = {"repo": "test/repo", "revision": "a" * 40, "prefix": "data"}
    inventory = sampling.enumerate_inventory(source, api=API())
    assert inventory["files"] == [{"path": "data/b.parquet", "size": 5, "lfs_sha256": "f" * 64, "blob_id": "b"}]
    with pytest.raises(ValueError, match="immutable"):
        sampling.enumerate_inventory({**source, "revision": "main"}, api=API())


def test_real_parquet_acquisition_raw_unicode_provenance_resume_and_tamper(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import huggingface_hub
    texts = [f"بخش\u200cهای یك فارسی {i}\n  unchanged  " for i in range(90)]
    local = tmp_path / "input.parquet"
    pq.write_table(pa.table({"text": texts, "url": ["https://example.test/" + str(i) for i in range(90)],
                             "id": [str(i) for i in range(90)]}), local, row_group_size=10)
    opens = []
    class FS:
        def __init__(self, **kwargs):
            assert kwargs == {"token": False}
        def open(self, remote, mode, **kwargs):
            assert kwargs == {"block_size": 1_048_576, "cache_type": "none"}
            opens.append(remote)
            return local.open(mode)
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FS)
    source = {"key": "fa__native", "language": "fa", "kind": "native", "repo": "fixture/native",
              "revision": "a" * 40, "prefix": "data", "shards": 1, "window_rows": 4,
              "max_read_bytes_per_shard": 1024 * 1024}
    monkeypatch.setattr(sampling, "enumerate_inventory", lambda source: {"source": source, "files": [{"path": "data/a.parquet", "size": local.stat().st_size}]})
    root = tmp_path / "audit"
    report = sampling.acquire([source], root, workers=1)
    assert report["status"] == "complete"
    raw = root / report["raw_files"][0]["file"]
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 12
    for row in records:
        assert row["text"] == texts[row["source_row"]]
        assert row["source_row"] // 10 == row["source_row_group"]
        assert "\u200c" in row["text"]
        assert row["sampling"]["row_group_selection_probability"] == 1 / 3
    assert sampling.acquire([source], root, workers=1) == report
    assert len(opens) == 1
    stratum = root / report["shards"][0]["strata"][0]["file"]
    stratum.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        sampling.acquire([source], root, workers=1)


def test_inventory_and_request_are_immutable(tmp_path):
    path = tmp_path / "data.json"
    sampling._write_immutable(path, {"b": 1, "a": "ی"})
    sampling._write_immutable(path, {"a": "ی", "b": 1})
    with pytest.raises(ValueError, match="differs"):
        sampling._write_immutable(path, {"a": "ی", "b": 2})


def test_concurrent_writer_lock_fails_closed(tmp_path):
    (tmp_path / "acquisition.lock").write_text('{"pid":123}\n')
    with pytest.raises(ValueError, match="writer lock"):
        sampling.acquire([{"key": "x"}], tmp_path)
    assert (tmp_path / "acquisition.lock").read_text() == '{"pid":123}\n'


def test_partial_stratum_failure_resumes_same_windows_and_charges_reopens(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import huggingface_hub
    local = tmp_path / "input.parquet"
    pq.write_table(pa.table({"text": [f"original {i}" for i in range(90)]}), local, row_group_size=10)
    class FS:
        def __init__(self, **kwargs):
            pass
        def open(self, remote, mode, **kwargs):
            return local.open(mode)
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FS)
    source = {"key": "en__web", "language": "en", "kind": "web", "repo": "fixture/web",
              "revision": "a" * 40, "prefix": "data", "shards": 1, "window_rows": 3,
              "max_read_bytes_per_shard": 1024 * 1024}
    monkeypatch.setattr(sampling, "enumerate_inventory", lambda source: {"source": source, "files": [{"path": "data/a.parquet", "size": local.stat().st_size}]})
    original = pq.ParquetFile
    class FailSecondRead:
        def __init__(self, *args, **kwargs):
            self.actual = original(*args, **kwargs)
            self.reads = 0
        def __getattr__(self, name):
            return getattr(self.actual, name)
        def read_row_group(self, *args, **kwargs):
            self.reads += 1
            if self.reads == 2:
                raise OSError("synthetic network failure")
            return self.actual.read_row_group(*args, **kwargs)
    monkeypatch.setattr(pq, "ParquetFile", FailSecondRead)
    root = tmp_path / "audit"
    first = sampling.acquire([source], root, workers=1)
    assert first["status"] == "incomplete"
    assert not (root / "manifest.json").exists()
    assert not (root / "raw").exists()
    complete = list((root / "strata").glob("*.complete.json"))
    assert len(complete) == 1
    before = complete[0].read_bytes()
    windows_before = (root / "windows/en__web__00.json").read_bytes()
    monkeypatch.setattr(pq, "ParquetFile", original)
    final = sampling.acquire([source], root, workers=1)
    assert final["status"] == "complete"
    assert complete[0].read_bytes() == before
    assert (root / "windows/en__web__00.json").read_bytes() == windows_before
    assert final["parquet_logical_bytes_requested"] > first["parquet_logical_bytes_requested"]
    ledger = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(x["event"] == "shard_failed" and x["substitution"] is False for x in ledger)


@pytest.mark.parametrize("rows,width", [([], 1), ([0], 1), ([1], 0)])
def test_invalid_windows_rejected(rows, width):
    with pytest.raises(ValueError):
        sampling.choose_windows(rows, width, 0, "x", "y")
