from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mqr.heads import DenseHead, DirectHead, MQRHead, build_head


def config(kind="mqr", **overrides):
    fields = dict(kind=kind, d_e=5, d_c=6, num_queries=3, activation="gelu", temperature=1.0,
                  normalize_queries=False, normalize_output_rows=False)
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("kind", ["dense", "direct", "mqr"])
def test_shapes_selected_scores_and_padding(kind):
    torch.manual_seed(14)
    head = build_head(config(kind), d_model=8, vocab_size=11)
    features = torch.randn(2, 4, 8)
    logits = head(features)
    assert logits.shape == (2, 4, 11)
    assert head.make_queries(features).shape == (2, 4, head.num_queries, head.d_e)
    shared = torch.tensor([7, 1, -1, 7])
    chosen = head.score_candidates(features, shared)
    torch.testing.assert_close(chosen[..., :2], logits[..., [7, 1]])
    torch.testing.assert_close(chosen[..., 3], logits[..., 7])
    assert torch.isneginf(chosen[..., 2]).all()
    batched = torch.randint(11, (2, 4, 5))
    torch.testing.assert_close(head.score_candidates(features, batched), logits.gather(-1, batched))


@pytest.mark.parametrize("head", [DenseHead(4, 7), DirectHead(4, 7, 3, 1), MQRHead(4, 7, 3, 5, 1)])
def test_single_query_is_linear_score_in_query_space(head):
    features = torch.randn(2, 4)
    expected = head.make_queries(features).squeeze(-2) @ head.output_vectors().T
    torch.testing.assert_close(head(features), expected)


def test_dense_is_bias_free_standard_projection():
    head = DenseHead(4, 9)
    features = torch.randn(3, 4)
    torch.testing.assert_close(head(features), F.linear(features, head.output.weight))
    assert list(dict(head.named_parameters())) == ["output.weight"]


@pytest.mark.parametrize("temperature", [1.0, 0.3])
def test_dense_loss_and_feature_weight_gradients_match_linear_reference(temperature):
    torch.manual_seed(81)
    head = DenseHead(8, 17, temperature=temperature).double()
    features = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([2, 11, 6])
    loss = F.cross_entropy(head(features), targets)
    actual = torch.autograd.grad(loss, (features, head.output.weight))
    reference = F.cross_entropy(F.linear(features, head.output.weight) / temperature, targets)
    expected = torch.autograd.grad(reference, (features, head.output.weight))
    torch.testing.assert_close(loss, reference)
    for computed, linear in zip(actual, expected):
        torch.testing.assert_close(computed, linear, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_query_half_scores_keep_float32_dtype_and_candidate_equivalence(dtype):
    head = DirectHead(4, 9, 3, 1).to(dtype)
    features = torch.randn(2, 4, dtype=dtype)
    result = head(features)
    candidates = head.score_candidates(features, torch.tensor([1, 7, -1]))
    assert result.dtype == candidates.dtype == torch.float32
    expected = (head.make_queries(features).squeeze(-2) @ head.output.weight.T).float()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(candidates[..., :2], result[..., [1, 7]])
    assert torch.isneginf(candidates[..., -1]).all()


@pytest.mark.parametrize("kind", ["direct", "mqr"])
def test_query_permutation_invariance(kind):
    head = build_head(config(kind), 8, 11)
    features = torch.randn(2, 8)
    expected = head(features)
    projection = head.query_projection
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        weight = projection.weight.reshape(3, 5, -1).clone()
        bias = projection.bias.reshape(3, 5).clone()
        projection.weight.copy_(weight[order].reshape_as(projection.weight))
        projection.bias.copy_(bias[order].reshape_as(projection.bias))
    torch.testing.assert_close(head(features), expected)


def test_queries_are_initialized_independently():
    head = MQRHead(4, 9, 3, 5, 4)
    weights = head.query_projection.weight.reshape(4, 3, 5)
    assert not torch.equal(weights[0], weights[1])


@pytest.mark.parametrize("kind", ["dense", "direct", "mqr"])
def test_exact_ce_propagates_finite_nonzero_gradients(kind):
    head = build_head(config(kind), 8, 11)
    features = torch.randn(2, 4, 8, requires_grad=True)
    gold = torch.randint(11, (2, 4))
    F.cross_entropy(head(features).flatten(0, 1), gold.flatten()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_normalized_candidate_scores_match_full_and_do_not_scan_output_vectors(monkeypatch):
    head = DirectHead(4, 9, 3, 2, normalize_queries=True, normalize_output_rows=True, temperature=0.3)
    features = torch.randn(2, 4)
    expected = head(features)[:, [2, 6]]
    # Candidate scoring must gather before normalization to preserve ANN complexity.
    monkeypatch.setattr(head, "output_vectors", lambda: pytest.fail("Full table read during candidate scoring"))
    torch.testing.assert_close(head.score_candidates(features, torch.tensor([2, 6])), expected)


def test_bfloat16_aggregation_is_float32_and_finite():
    head = DirectHead(4, 9, 3, 4).to(torch.bfloat16)
    features = torch.randn(2, 4, dtype=torch.bfloat16)
    result = head(features)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_score_candidates_has_matching_gradient():
    head = MQRHead(4, 9, 3, 5, 2).double()
    features = torch.randn(2, 4, dtype=torch.float64)
    ids = torch.tensor([[1, 3, 7], [2, 5, 6]])
    selected = head.score_candidates(features, ids).sum()
    params = tuple(head.parameters())
    selected_grads = torch.autograd.grad(selected, params)
    full_grads = torch.autograd.grad(head(features).gather(-1, ids).sum(), params)
    for actual, expected in zip(selected_grads, full_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("ids", [torch.tensor([0.0]), torch.tensor([-2]), torch.tensor([9]), torch.tensor([[0], [1], [2]])])
def test_invalid_candidate_ids_fail(ids):
    with pytest.raises((ValueError, TypeError)):
        DirectHead(4, 9, 3, 2).score_candidates(torch.randn(2, 4), ids)


def test_scalar_context_and_empty_candidate_set():
    head = DirectHead(4, 9, 3, 2)
    features = torch.randn(4)
    torch.testing.assert_close(head.score_candidates(features, torch.tensor([2, 3])), head(features)[[2, 3]])
    assert head.score_candidates(features, torch.empty(0, dtype=torch.long)).shape == (0,)
