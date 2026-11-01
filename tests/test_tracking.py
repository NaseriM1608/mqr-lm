import json

import pytest

from mqr.tracking import prepare_metrics_append, read_authoritative_metrics


def write_events(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def metric(kind, step, session, value=1):
    return {"event": kind, "step": step, "session_id": session, "full_nll": value}


def metric_events(path):
    return [event for event in read_authoritative_metrics(path) if event["event"] in ("train", "validation")]


def test_crash_replay_discards_uncheckpointed_tail(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_events(path, [
        {"event": "start", "step": 0, "session_id": "old"},
        metric("train", 1, "old"), metric("train", 2, "old"),
        metric("validation", 2, "old"), metric("train", 3, "old"),
        metric("train", 4, "old"), metric("validation", 4, "old"),
        {"event": "resume", "step": 2, "session_id": "new"},
        metric("train", 3, "new", 0.8),
    ])
    events = metric_events(path)
    assert [(event["event"], event["step"]) for event in events] == [
        ("train", 1), ("train", 2), ("validation", 2), ("train", 3),
    ]
    assert events[-1]["session_id"] == "new"
    assert any(event["event"] == "resume" for event in read_authoritative_metrics(path))


def test_intentional_older_checkpoint_produces_one_trajectory(tmp_path):
    path = tmp_path / "metrics.jsonl"
    events = [metric("train", step, "first") for step in range(1, 7)]
    events += [{"event": "resume", "step": 3, "session_id": "second"}]
    events += [metric("train", step, "second") for step in range(4, 8)]
    events += [{"event": "resume", "step": 2, "session_id": "third"}, metric("train", 3, "third")]
    write_events(path, events)
    actual = metric_events(path)
    assert [event["step"] for event in actual] == [1, 2, 3]
    assert [event["session_id"] for event in actual] == ["first", "first", "third"]


def test_same_boundary_train_and_validation_are_preserved_and_duplicates_replace(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_events(path, [
        metric("train", 2, "first"), metric("validation", 2, "first", 1.2),
        {"event": "resume", "step": 2, "session_id": "second"},
        metric("validation", 2, "second", 1.1), metric("train", 3, "second"),
    ])
    actual = metric_events(path)
    assert [(event["event"], event["step"]) for event in actual] == [
        ("train", 2), ("validation", 2), ("train", 3),
    ]
    assert actual[0]["session_id"] == "first"
    assert actual[1]["full_nll"] == 1.1


def test_partial_final_line_is_archived_before_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    complete = json.dumps(metric("train", 1, "first")).encode() + b"\n"
    tail = b'{"event":"train","step":2,"full_nll":'
    path.write_bytes(complete + tail)
    assert len(metric_events(path)) == 1
    prepare_metrics_append(path)
    assert path.read_bytes() == complete
    archives = list(tmp_path.glob("metrics.interrupted-tail.*.bin"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == tail
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "resume", "step": 1}) + "\n")
    assert len(read_authoritative_metrics(path)) == 2
    prepare_metrics_append(path)
    assert len(list(tmp_path.glob("metrics.interrupted-tail.*.bin"))) == 1


def test_valid_record_without_newline_receives_newline(tmp_path):
    path = tmp_path / "metrics.jsonl"
    content = json.dumps(metric("train", 1, "first")).encode()
    path.write_bytes(content)
    prepare_metrics_append(path)
    assert path.read_bytes() == content + b"\n"
    assert len(metric_events(path)) == 1


def test_invalid_interior_record_is_not_modified(tmp_path):
    path = tmp_path / "metrics.jsonl"
    content = b'{"event":"train",\n' + json.dumps(metric("train", 2, "first")).encode() + b"\n"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="interior"):
        prepare_metrics_append(path)
    with pytest.raises(ValueError, match="interior"):
        read_authoritative_metrics(path)
    assert path.read_bytes() == content
    assert not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("event", [
    {"event": "resume", "step": -1}, {"event": "train", "step": True},
    {"event": "validation", "step": 1.0}, {"step": 1}, [],
])
def test_invalid_schema_is_rejected(tmp_path, event):
    path = tmp_path / "metrics.jsonl"
    write_events(path, [event])
    with pytest.raises(ValueError):
        read_authoritative_metrics(path)
    with pytest.raises(ValueError):
        prepare_metrics_append(path)


def test_missing_append_log_needs_no_repair(tmp_path):
    path = tmp_path / "metrics.jsonl"
    prepare_metrics_append(path)
    assert not path.exists()
