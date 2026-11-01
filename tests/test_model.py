import pytest
import torch
from torch.nn import functional as F

from mqr.config import ExperimentConfig, HeadConfig, ModelConfig
from mqr.model import LanguageModel


def small_config(kind="mqr"):
    return ExperimentConfig(
        model=ModelConfig(vocab_size=29, layers=2, d_model=24, attention_heads=3, ffn_dim=48, context_length=16),
        head=HeadConfig(kind=kind, d_e=12, d_c=16, num_queries=3),
    )


@pytest.mark.parametrize("kind", ["dense", "direct", "mqr"])
def test_shape_and_backward(kind):
    model = LanguageModel(small_config(kind))
    tokens = torch.randint(0, 29, (2, 7))
    logits = model(tokens)
    assert logits.shape == (2, 7, 29)
    F.cross_entropy(logits.reshape(-1, 29), tokens.reshape(-1)).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_causal_prefix_invariance():
    model = LanguageModel(small_config()).eval()
    prefix = torch.randint(0, 29, (2, 5))
    extended = torch.cat((prefix, torch.randint(0, 29, (2, 4))), dim=1)
    torch.testing.assert_close(model(prefix), model(extended)[:, :5], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("split", [1, 3, 6])
def test_cached_backbone_matches_full_sequence(split):
    model = LanguageModel(small_config()).eval()
    tokens = torch.randint(0, 29, (2, 8))
    full, no_cache = model.backbone(tokens)
    assert no_cache is None
    prefix, cache = model.backbone(tokens[:, :split], use_cache=True)
    suffix, cache = model.backbone(tokens[:, split:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), full, rtol=1e-5, atol=1e-6)
    assert len(cache) == 2
    assert cache[0][0].shape == (2, 3, 8, 8)


def test_token_by_token_cache_matches_full_sequence():
    model = LanguageModel(small_config()).eval()
    tokens = torch.randint(0, 29, (2, 8))
    full, _ = model.backbone(tokens)
    cache, pieces = None, []
    for position in range(tokens.size(1)):
        features, cache = model.backbone(tokens[:, position:position + 1], past_key_values=cache, use_cache=True)
        pieces.append(features)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)


def test_context_and_cache_validation():
    model = LanguageModel(small_config()).eval()
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros((1, 17), dtype=torch.long))
    _, cache = model.backbone(torch.zeros((1, 15), dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError, match="context_length"):
        model.backbone(torch.zeros((1, 2), dtype=torch.long), cache)
    with pytest.raises(ValueError, match="per decoder layer"):
        model.backbone(torch.zeros((1, 1), dtype=torch.long), cache[:1])
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty((1, 0), dtype=torch.long))


def test_backbone_is_identically_initialized_across_heads():
    torch.manual_seed(19)
    dense = LanguageModel(small_config("dense"))
    torch.manual_seed(19)
    mqr = LanguageModel(small_config("mqr"))
    for name, value in dense.backbone.state_dict().items():
        torch.testing.assert_close(value, mqr.backbone.state_dict()[name], rtol=0, atol=0)
    assert dense.backbone.embedding.weight.data_ptr() != dense.head.output_vectors().data_ptr()


def test_greedy_generation_matches_full_recomputation_and_restores_training_mode():
    model = LanguageModel(small_config())
    tokens = torch.randint(0, 29, (2, 4))
    model.eval()
    expected = tokens
    for _ in range(4):
        next_token = model(expected)[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, next_token), dim=1)
    model.train()
    actual = model.generate(tokens, max_new_tokens=4, temperature=0)
    assert model.training
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="context_length"):
        model.generate(tokens, max_new_tokens=20)


def test_generation_stops_at_eos_and_top_one_matches_greedy():
    model = LanguageModel(small_config("dense"))
    tokens = torch.ones((2, 4), dtype=torch.long)
    torch.testing.assert_close(
        model.generate(tokens, 3, top_k=1),
        model.generate(tokens, 3, temperature=0),
    )
    with torch.no_grad():
        model.head.output.weight.zero_()
    result = model.generate(tokens, 4, temperature=0, eos_token_id=0)
    assert result.shape == (2, 5)
    assert (result[:, -1] == 0).all()
