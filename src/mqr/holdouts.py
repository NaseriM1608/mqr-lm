"""Persistent document/URL/content split identities for growing corpora.

Identity-connected documents share a split. A new connection between conflicting
frozen splits is an error, never a silent reassignment. Near-duplicate protection
still requires running corpus screening over the complete supplied history.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable_assignments(records: list[dict], *, seed: int, train_fraction: float,
                       validation_fraction: float, previous: str | Path | None = None):
    policy = {"seed": seed, "train_fraction": train_fraction,
              "validation_fraction": validation_fraction,
              "identity": "connected document IDs, canonical URLs and normalized content hashes"}
    historical = []
    if previous is not None:
        descriptor = json.loads(Path(previous).read_text(encoding="utf-8"))
        if descriptor.get("schema") != "mqr-stable-holdouts-v1" or descriptor.get("policy") != policy:
            raise ValueError("Frozen split ledger policy differs")
        historical = descriptor["records"]
    by_id, parent = {}, {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(keys):
        roots = [find(key) for key in keys]
        anchor = min(roots)
        for root in roots:
            parent[root] = anchor

    def identities(row):
        values = ["id:" + row["id"], "text:" + row["normalized_sha256"]] + (
            ["url:" + row["canonical_url"]] if row.get("canonical_url") else [])
        aliases = row.get("identity_keys", [])
        if not isinstance(aliases, list) or any(not isinstance(key, str) or not key for key in aliases):
            raise ValueError("Invalid frozen identity aliases")
        return sorted(set(values + aliases))

    for row in [*historical, *records]:
        required = ("id", "language", "content_sha256", "normalized_sha256")
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError("Split ledger identity record is incomplete")
        if row.get("split") not in (None, "train", "validation", "test"):
            raise ValueError("Invalid frozen split")
        old = by_id.get(row["id"])
        if old and any(old[key] != row[key] for key in required):
            raise ValueError("Frozen document identity changed: " + row["id"])
        combined = {**row, **old} if old else dict(row)
        combined["identity_keys"] = sorted(set(identities(row) + (identities(old) if old else [])))
        by_id[row["id"]] = combined
        union(identities(row))
    prior_splits = {}
    for row in historical:
        if row.get("split") not in ("train", "validation", "test"):
            raise ValueError("Historical ledger record lacks a frozen split")
        prior_splits.setdefault(find(identities(row)[0]), set()).add(row["split"])
    if any(len(splits) != 1 for splits in prior_splits.values()):
        raise ValueError("New duplicate identity connects conflicting frozen splits; quarantine the connection")
    anchors = {}
    for key in parent:
        root = find(key)
        anchors[root] = min(anchors.get(root, key), key)
    assignment = {}
    for document_id, row in by_id.items():
        root = find(identities(row)[0])
        if root in prior_splits:
            split = next(iter(prior_splits[root]))
        else:
            raw = json.dumps([seed, anchors[root]], separators=(",", ":"), ensure_ascii=False)
            value = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) / 2**256
            split = "train" if value < train_fraction else "validation" if value < train_fraction + validation_fraction else "test"
        assignment[document_id] = split
    ledger = {"schema": "mqr-stable-holdouts-v1", "policy": policy,
              "records": [{**by_id[key], "split": assignment[key]} for key in sorted(by_id)],
              "limits": "Identity protection only. Keep prior raw history in global near-duplicate screening; this ledger does not detect paraphrases."}
    return {row["id"]: assignment[row["id"]] for row in records}, ledger
