import hashlib
import json

import pytest

from mqr.holdouts import stable_assignments


def row(key, text=None, url=None):
    digest = hashlib.sha256((text or key).encode()).hexdigest()
    return {"id": key, "language": "en", "content_sha256": digest,
            "normalized_sha256": digest, "canonical_url": url}


def assign(rows, previous=None, seed=13):
    return stable_assignments(rows, seed=seed, train_fraction=.8,
                              validation_fraction=.1, previous=previous)


def save(tmp_path, ledger):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


def test_order_independence_and_append_preserves_every_existing_split(tmp_path):
    rows = [row(str(index)) for index in range(200)]
    original, ledger = assign(rows)
    shuffled, second = assign(list(reversed(rows)))
    assert original == shuffled
    assert ledger == second
    expanded, _ = assign(rows + [row("new-" + str(i)) for i in range(100)], save(tmp_path, ledger))
    assert all(expanded[key] == value for key, value in original.items())
    assert set(original.values()) == {"train", "validation", "test"}


def test_url_content_and_transitive_duplicate_connections_share_a_split():
    rows = [row("one", "alpha", "https://example/a"),
            row("two", "beta", "https://example/a"), row("three", "beta", "https://example/b")]
    result, _ = assign(rows)
    assert len(set(result.values())) == 1


def test_frozen_identity_changes_and_policy_changes_fail(tmp_path):
    _, ledger = assign([row("one")])
    path = save(tmp_path, ledger)
    with pytest.raises(ValueError, match="identity changed"):
        assign([row("one", "changed")], path)
    with pytest.raises(ValueError, match="policy differs"):
        assign([row("one")], path, seed=14)


def test_new_bridge_between_conflicting_frozen_splits_is_quarantined(tmp_path):
    first, second = row("one", "alpha", "https://example/a"), row("two", "beta", "https://example/b")
    _, ledger = assign([first, second])
    ledger["records"][0]["split"] = "train"
    ledger["records"][1]["split"] = "test"
    path = save(tmp_path, ledger)
    with pytest.raises(ValueError, match="conflicting frozen splits"):
        assign([row("bridge", "alpha", "https://example/b")], path)


def test_url_alias_is_remembered_across_incremental_ledgers(tmp_path):
    first = row("one", "alpha", "https://example/a")
    _, ledger = assign([first])
    ledger["records"][0]["split"] = "test"
    path = save(tmp_path, ledger)
    _, updated = assign([row("one", "alpha", "https://example/alias")], path)
    path = save(tmp_path, updated)
    result, _ = assign([row("new", "changed", "https://example/alias")], path)
    assert result["new"] == "test"
