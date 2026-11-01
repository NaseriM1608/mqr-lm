import pytest
import torch

from mqr.heads import DirectHead
from mqr.metrics import coverage_metrics
from mqr.retrieval import FaissIPIndex, retrieve_candidates, table_fingerprint


def two_query_head():
    head = DirectHead(2, 3, 2, 2)
    with torch.no_grad():
        head.query_projection.weight.zero_()
        head.query_projection.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))
        head.output.weight.copy_(torch.tensor([[10.0, -100.0], [-100.0, 10.0], [9.5, 9.5]]))
    return head


def test_exact_per_query_top1_can_omit_aggregate_winner():
    head = two_query_head()
    features = torch.zeros(1, 2)
    exact = head(features)
    candidates = retrieve_candidates(head, features, k=1)
    assert exact.argmax(-1).item() == 2
    assert candidates.ids.tolist() == [[0, 1]]
    assert candidates.counts.tolist() == [2]
    assert not coverage_metrics(exact, candidates.ids)["top1_agreement"].item()
    expanded = retrieve_candidates(head, features, k=2)
    assert coverage_metrics(exact, expanded.ids)["top1_agreement"].item()


def test_union_deduplicates_pads_and_scores_every_query():
    head = DirectHead(2, 5, 2, 2)
    with torch.no_grad():
        # Context 0 gives identical queries; context 1 gives opposing queries.
        head.query_projection.weight.copy_(torch.tensor([[1., 0.], [0., 0.], [0., 1.], [0., 0.]]))
        head.query_projection.bias.zero_()
        head.output.weight.copy_(torch.tensor([[3., 0.], [-3., 0.], [2., 0.], [-2., 0.], [0., 1.]]))
    features = torch.tensor([[1., 1.], [1., -1.]])
    candidates = retrieve_candidates(head, features, 1, always_include=[4, 4])
    assert candidates.ids.tolist() == [[0, 4, -1], [0, 1, 4]]
    assert candidates.counts.tolist() == [2, 3]
    full = head(features)
    valid = candidates.ids >= 0
    expected = full.gather(-1, candidates.ids.clamp_min(0))
    torch.testing.assert_close(candidates.scores[valid], expected[valid])
    assert torch.isneginf(candidates.scores[~valid]).all()
    assert not candidates.scores.requires_grad


def test_k_larger_than_vocabulary_and_scalar_context():
    head = two_query_head()
    candidates = retrieve_candidates(head, torch.zeros(2), 100)
    assert candidates.ids.tolist() == [0, 1, 2]
    assert candidates.counts.shape == ()
    assert candidates.counts.item() == 3


@pytest.mark.parametrize("kwargs", [dict(k=0), dict(k=True), dict(k=1, always_include=[-1]),
                                   dict(k=1, always_include=[3]), dict(k=1, method="faiss")])
def test_retrieval_validation(kwargs):
    with pytest.raises(ValueError):
        retrieve_candidates(two_query_head(), torch.zeros(1, 2), **kwargs)


def test_table_fingerprint_detects_effective_geometry_change():
    head = two_query_head()
    original = table_fingerprint(head)
    with torch.no_grad():
        head.output.weight[0, 0].add_(0.5)
    assert table_fingerprint(head) != original
    head.normalize_output_rows = True
    assert table_fingerprint(head) != original


@pytest.mark.parametrize("method", ["flat", "hnsw"])
def test_faiss_index_and_stale_table(method):
    pytest.importorskip("faiss")
    head = two_query_head().eval()
    index = FaissIPIndex(head, method=method)
    features = torch.zeros(1, 2)
    indexed = retrieve_candidates(head, features, 2, index=index)
    flat = retrieve_candidates(head, features, 2)
    assert indexed.ids.tolist() == flat.ids.tolist()
    torch.testing.assert_close(indexed.scores, flat.scores)
    index.assert_current(head, full=True)
    with torch.no_grad():
        head.output.weight[0].add_(1.)
    with pytest.raises(RuntimeError, match="Stale"):
        retrieve_candidates(head, features, 1, index=index)


def test_index_checks_versions_without_hashing_each_search(monkeypatch):
    pytest.importorskip("faiss")
    head = two_query_head().eval()
    index = FaissIPIndex(head)
    monkeypatch.setattr("mqr.retrieval.table_fingerprint", lambda _: pytest.fail("Per-query table hashing"))
    retrieve_candidates(head, torch.zeros(1, 2), 1, index=index)


def test_explicit_fingerprint_catches_version_bypassing_mutation():
    pytest.importorskip("faiss")
    head = two_query_head().eval()
    index = FaissIPIndex(head)
    head.output.weight.data[0, 0].add_(1.)
    with pytest.raises(RuntimeError, match="Stale"):
        index.assert_current(head, full=True)
