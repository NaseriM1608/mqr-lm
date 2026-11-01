"""Read one authoritative training trajectory across checkpoint restarts.

Raw JSONL logs retain all sessions. A resume marker abandons earlier train and
validation records strictly after its checkpoint step. Diagnostic records remain
available, and train/validation records at the checkpoint boundary remain valid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

_METRIC_EVENTS = frozenset(("train", "validation"))


def _validate_record(value: Any, path: Path, line_number: int) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(value, dict) or not isinstance(value.get("event"), str):
        raise ValueError(f"Metrics record must be an object with an event name at {location}")
    if value["event"] in _METRIC_EVENTS or value["event"] == "resume":
        if type(value.get("step")) is not int or value["step"] < 0:
            raise ValueError(f"Metrics step must be a nonnegative integer at {location}")
    return value


def prepare_metrics_append(path: str | Path) -> None:
    """Repair a crash-interrupted final line before appending another session.

    Complete interior records are validated first. A valid final JSON record
    lacking a newline receives one. An invalid final tail is saved verbatim to
    ``<stem>.interrupted-tail.<uuid>.bin`` before truncating only those bytes.
    A missing log needs no repair. This helper assumes one writer per log.
    """
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    body, tail = raw[:boundary], raw[boundary:]
    for line_number, line in enumerate(body.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid interior metrics JSON at {path}:{line_number}") from exc
        _validate_record(value, path, line_number)
    if not tail:
        return
    if tail.strip():
        try:
            value = json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            archive = path.with_name(f"{path.stem}.interrupted-tail.{uuid4().hex}.bin")
            # Make the original bytes durable before removing them from the log.
            with archive.open("xb") as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
            with path.open("r+b") as handle:
                handle.truncate(boundary)
            return
        _validate_record(value, path, body.count(b"\n") + 1)
    with path.open("ab") as handle:
        handle.write(b"\n")


def read_authoritative_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Return surviving metric records plus retained diagnostic/session markers.

    Replayed train/validation records replace the same event at the same step;
    train and validation at one step remain separate. Resume markers remove all
    earlier metric records above their checkpoint step, including validation
    points with no replacement yet. A malformed unfinished final line is ignored
    to permit reading an active or interrupted log; malformed interior lines and
    invalid record schemas raise ``ValueError``.
    """
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    trajectory: list[dict[str, Any] | None] = []
    positions: dict[tuple[str, int], int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if line_number == len(lines) and not raw.endswith(b"\n"):
                break
            raise ValueError(f"Invalid interior metrics JSON at {path}:{line_number}") from exc
        event = _validate_record(value, path, line_number)
        kind = event["event"]
        if kind == "resume":
            for key, position in list(positions.items()):
                if key[1] > event["step"]:
                    trajectory[position] = None
                    del positions[key]
        elif kind in _METRIC_EVENTS:
            key = (kind, event["step"])
            if key in positions:
                trajectory[positions[key]] = None
            positions[key] = len(trajectory)
        trajectory.append(event)
    return [event for event in trajectory if event is not None]
