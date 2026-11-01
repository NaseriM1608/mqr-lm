import copy
import math

import pytest
import torch

from mqr.config import ExperimentConfig, HeadConfig, ModelConfig
from mqr.head_diagnostics import (calibrate_initial_query_gain, calibrated_projection_gain, centered_logit_std,
                                  diagnose_head, diagnose_initialization, gelu_gaussian_moments,
                                  initialization_fingerprint, initialization_variant, variance_prediction)
from mqr.heads import MQRHead


def test_gelu_quadrature_matches_known_unit_gaussian_moments():
    unit = gelu_gaussian_moments(1.0)
    assert unit["mean"] == pytest.approx(1 / (2 * math.sqrt(math.pi)), rel=1e-10)
    assert unit["second_moment"] == pytest.approx(1 / 3 + 1 / (2 * math.pi * math.sqrt(3)), rel=1e-10)
    zero = gelu_gaussian_moments(0.0)
    assert zero["second_moment"] == 0
    assert zero["derivative_second_moment"] == pytest.approx(0.25)


def test_variance_chain_predicts_attenuation_and_query_averaging():
    dense = variance_prediction("dense", 512, 128, 256, 1)
    direct1 = variance_prediction("direct", 512, 128, 256, 1)
    direct4 = variance_prediction("direct", 512, 128, 256, 4)
    mqr = variance_prediction("mqr", 512, 128, 256, 4)
    assert dense["component_logit_std"] == pytest.approx(math.sqrt(512) * 0.02)
    assert direct1["component_logit_std"] == pytest.approx(0.1024)
    assert direct4["small_score_aggregate_variance"] * 4 == pytest.approx(direct1["small_score_aggregate_variance"])
    assert 0.008 < mqr["small_score_aggregate_std"] < 0.011
    assert 0.018 < mqr["small_score_feature_gradient_rms_ratio_vs_dense"] < 0.025


def test_variants_and_label_free_calibration_leave_canonical_state_unchanged():
    torch.manual_seed(18)
    original = MQRHead(16, 257, 8, 12, 4)
    state = copy.deepcopy(original.state_dict())
    features = torch.randn(4, 16)
    fan_in, metadata = initialization_variant(original, "fan_in")
    torch.testing.assert_close(fan_in.output.weight, original.output.weight)
    torch.testing.assert_close(fan_in.concept.weight, original.concept.weight * ((16 ** -0.5) / 0.02))
    assert metadata["temperature"] == 1
    calibrated, calibration = calibrated_projection_gain(fan_in, features, target_std=0.25)
    assert centered_logit_std(calibrated(features)) == pytest.approx(0.25, rel=1e-5)
    assert calibration["query_projection_gain"] > 1
    assert not calibrated.normalize_queries and not calibrated.normalize_output_rows
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in original.parameters())


def test_full_probe_has_finite_gradients_shared_backbone_and_no_parameter_updates():
    config = ExperimentConfig(model=ModelConfig(vocab_size=257, layers=1, d_model=16,
                                               attention_heads=2, ffn_dim=32, context_length=8),
                              head=HeadConfig(kind="mqr", d_c=12, d_e=8, num_queries=4))
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, 5]])
    rng = torch.get_rng_state().clone()
    report = diagnose_initialization(config, x, y)
    assert torch.equal(torch.get_rng_state(), rng)
    assert report["canonical_parameters_unchanged"] and report["optimizer_updates"] == 0
    assert len(report["reports"]) == 13
    dense_std = report["reports"]["dense/canonical"]["aggregate_logits"]["centered_std"]
    for name, result in report["reports"].items():
        assert math.isfinite(result["full_ce"])
        assert result["feature_gradient"]["finite"]
        assert result["feature_gradient"]["rms"] > 0
        assert all(group["finite"] for group in result["gradient_groups"].values())
        assert result["gradient_groups"]["backbone_all"]["gradient_l2"] > 0
        if name.endswith("logit_matched"):
            assert result["aggregate_logits"]["centered_std"] == pytest.approx(dense_std, rel=1e-5)
    assert report["reports"]["mqr/canonical"]["feature_gradient"]["rms"] < report["reports"]["dense/canonical"]["feature_gradient"]["rms"]


def test_public_pilot_probe_and_inplace_gain_have_explicit_parameter_scope():
    head = MQRHead(16, 257, 8, 12, 4)
    features = torch.randn(6, 16)
    targets = torch.randint(257, (6,))
    before = initialization_fingerprint(head)
    with torch.no_grad():
        report = diagnose_head(head, features, targets)
    assert report["feature_gradient"]["finite"]
    assert initialization_fingerprint(head) == before
    assert all(parameter.grad is None for parameter in head.parameters())
    output_before = head.output.weight.detach().clone()
    concept_before = head.concept.weight.detach().clone()
    metadata = calibrate_initial_query_gain(head, features, target_logit_std=0.2, chunk_positions=2)
    assert metadata["baseline_centered_logit_std"] < 0.2
    assert metadata["achieved_centered_logit_std"] == pytest.approx(0.2, rel=1e-5)
    assert centered_logit_std(head(features)) == pytest.approx(0.2, rel=1e-5)
    assert metadata["changed_parameters"] == ["query_projection.weight"]
    assert metadata["projection_sha256_before"] != metadata["projection_sha256_after"]
    torch.testing.assert_close(head.output.weight, output_before, rtol=0, atol=0)
    torch.testing.assert_close(head.concept.weight, concept_before, rtol=0, atol=0)
