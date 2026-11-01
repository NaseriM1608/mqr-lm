import pytest
import torch

from mqr.heads import DirectHead
from mqr.metrics import coverage_metrics, query_diagnostics


def test_mass_uses_full_distribution_and_deduplicates():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    result = coverage_metrics(probabilities.log(), torch.tensor([[1, 1, 2, -1]]), torch.tensor([2]), top_k=2)
    assert result["candidate_count"].item() == 2
    assert result["retained_mass"].item() == pytest.approx(0.45)
    assert result["full_topk_mass"].item() == pytest.approx(0.8)
    assert result["retained_topk_mass"].item() == pytest.approx(0.3)
    assert result["topk_mass_recall"].item() == pytest.approx(0.375)
    assert result["topk_recall"].item() == pytest.approx(0.5)
    assert not result["top1_agreement"].item()
    assert result["gold_included"].item()


def test_shared_candidates_arbitrary_context_shape_and_full_coverage():
    logits = torch.randn(2, 3, 5)
    result = coverage_metrics(logits, torch.arange(5), top_k=50)
    torch.testing.assert_close(result["retained_mass"], torch.ones(2, 3))
    torch.testing.assert_close(result["topk_mass_recall"], torch.ones(2, 3))
    assert result["top1_agreement"].all()
    assert result["candidate_count"].shape == (2, 3)


def test_empty_candidates_and_negative_infinity_support():
    logits = torch.tensor([[-torch.inf, 0., 1.], [-torch.inf, 2., 1.]])
    result = coverage_metrics(logits, torch.empty((2, 0), dtype=torch.long))
    assert result["retained_mass"].eq(0).all()
    assert not result["top1_agreement"].any()
    result = coverage_metrics(logits, torch.tensor([0]))
    assert result["retained_mass"].eq(0).all()
    assert not result["top1_agreement"].any()


def test_scalar_context_and_ties_use_full_argmax():
    result = coverage_metrics(torch.tensor([1., 1., 0.]), torch.tensor([1, 0]), torch.tensor(0), top_k=2)
    assert result["top1_agreement"].shape == ()
    assert result["top1_agreement"].item()
    assert result["gold_included"].item()


@pytest.mark.parametrize("logits,ids", [(torch.tensor([float("nan"), 0.]), torch.tensor([0])),
                                       (torch.tensor([-torch.inf, -torch.inf]), torch.tensor([0])),
                                       (torch.tensor([1., 2.]), torch.tensor([2])),
                                       (torch.tensor([1., 2.]), torch.tensor([-2]))])
def test_invalid_metrics_inputs_fail(logits, ids):
    with pytest.raises(ValueError):
        coverage_metrics(logits, ids)


def test_branch_mass_matches_expected_posterior_responsibility():
    head = DirectHead(2, 5, 2, 3)
    features = torch.randn(2, 2)
    scores = head.component_scores(features)
    exact_probabilities = head(features).softmax(-1)
    posterior = scores.softmax(dim=-2)
    expected = (posterior * exact_probabilities.unsqueeze(-2)).sum(-1)
    result = query_diagnostics(head, features)
    torch.testing.assert_close(result["branch_mass"], expected)
    torch.testing.assert_close(result["branch_mass"].sum(-1), torch.ones(2))


def test_identical_queries_report_collapse():
    head = DirectHead(2, 5, 2, 3)
    with torch.no_grad():
        head.query_projection.weight.zero_()
        head.query_projection.bias.copy_(torch.tensor([1., 0., 1., 0., 1., 0.]))
    result = query_diagnostics(head, torch.zeros(2, 2))
    torch.testing.assert_close(result["mean_pair_cosine"], torch.ones(2))
    assert result["unique_top1"].eq(1).all()
    torch.testing.assert_close(result["branch_mass"], torch.full((2, 3), 1 / 3))
    torch.testing.assert_close(result["normalized_branch_entropy"], torch.ones(2))


def test_single_query_diagnostics_are_defined():
    head = DirectHead(2, 5, 2, 1)
    result = query_diagnostics(head, torch.randn(2, 2))
    assert result["mean_pair_cosine"].eq(0).all()
    assert result["unique_top1"].eq(1).all()
    assert result["normalized_branch_entropy"].eq(0).all()
