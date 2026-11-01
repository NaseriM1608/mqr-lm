"""Exact vocabulary scorers for the dense and multi-query controls.

MQR combines *unnormalized* component scores before the vocabulary softmax.
It does not implement a mixture of independently normalized softmaxes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .config import HeadConfig


def _stable_lse(scores: Tensor, dim: int) -> Tensor:
    # Reductions in half precision can overflow even when each score is finite.
    values = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    if values.size(dim) == 1:
        return values.squeeze(dim)
    return torch.logsumexp(values, dim=dim)


def _validate_ids(ids: Tensor, leading: torch.Size, vocab_size: int) -> None:
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Candidate IDs must be int32 or int64 tensors.")
    if ids.ndim != 1 and (ids.ndim != len(leading) + 1 or ids.shape[:-1] != leading):
        raise ValueError("Candidate IDs must be shared [K] or have features' leading dimensions plus [K].")
    if torch.any((ids < -1) | (ids >= vocab_size)):
        raise ValueError("Candidate IDs must be -1 padding or valid vocabulary indices.")


class VocabularyHead(nn.Module):
    """Common interface; subclasses define how hidden features become queries."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        d_e: int,
        num_queries: int,
        *,
        temperature: float = 1.0,
        normalize_queries: bool = False,
        normalize_output_rows: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("vocab_size", vocab_size), ("d_e", d_e), ("num_queries", num_queries)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_e = d_e
        self.num_queries = num_queries
        self.temperature = float(temperature)
        self.normalize_queries = normalize_queries
        self.normalize_output_rows = normalize_output_rows
        self.output = nn.Embedding(vocab_size, d_e)
        nn.init.normal_(self.output.weight, std=0.02)

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim < 1 or features.shape[-1] != self.d_model:
            raise ValueError(f"Features must end in d_model={self.d_model}.")
        if not features.is_floating_point():
            raise TypeError("Features must be floating point.")

    def _finish_queries(self, queries: Tensor) -> Tensor:
        return F.normalize(queries, dim=-1) if self.normalize_queries else queries

    def make_queries(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def output_vectors(self) -> Tensor:
        rows = self.output.weight
        return F.normalize(rows, dim=-1) if self.normalize_output_rows else rows

    def component_scores(self, features: Tensor) -> Tensor:
        """Return [..., M, V] inner products, including the shared temperature."""
        queries = self.make_queries(features)
        scores = torch.matmul(queries, self.output_vectors().transpose(0, 1))
        return scores if self.temperature == 1.0 else scores / self.temperature

    def _aggregate_scores(self, scores: Tensor) -> Tensor:
        aggregate = _stable_lse(scores, dim=-2)
        return aggregate if self.num_queries == 1 else aggregate - math.log(self.num_queries)

    def score_all(self, features: Tensor) -> Tensor:
        """Exact aggregate logits [..., V], suitable for cross entropy."""
        return self._aggregate_scores(self.component_scores(features))

    def score_candidates(self, features: Tensor, ids: Tensor) -> Tensor:
        """Score every candidate with every query; -1 padding receives -inf.

        The scores use the same law as ``score_all``. A subsequent softmax over
        this smaller set is a truncated inference distribution, not exact NLL.
        """
        self._validate_features(features)
        _validate_ids(ids, features.shape[:-1], self.vocab_size)
        if ids.device != features.device:
            raise ValueError("Candidate IDs and features must be on the same device.")
        queries = self.make_queries(features)
        # Gather first so normalized retrieval does not scan the full vocabulary.
        rows = self.output.weight[ids.long().clamp_min(0)]
        if self.normalize_output_rows:
            rows = F.normalize(rows, dim=-1)
        scores = torch.matmul(queries, rows.transpose(-1, -2))
        if self.temperature != 1.0:
            scores = scores / self.temperature
        aggregate = self._aggregate_scores(scores)
        return aggregate.masked_fill(ids == -1, -torch.inf)

    def forward(self, features: Tensor) -> Tensor:
        return self.score_all(features)


class DenseHead(VocabularyHead):
    """Bias-free vanilla vocabulary projection with no query bottleneck."""

    def __init__(self, d_model: int, vocab_size: int, **kwargs: object) -> None:
        super().__init__(d_model, vocab_size, d_model, 1, **kwargs)

    def make_queries(self, features: Tensor) -> Tensor:
        self._validate_features(features)
        return self._finish_queries(features.unsqueeze(-2))


class DirectHead(VocabularyHead):
    """Independent parallel linear queries directly from decoder features."""

    def __init__(self, d_model: int, vocab_size: int, d_e: int, num_queries: int, **kwargs: object) -> None:
        super().__init__(d_model, vocab_size, d_e, num_queries, **kwargs)
        self.query_projection = nn.Linear(d_model, num_queries * d_e)
        nn.init.normal_(self.query_projection.weight, std=0.02)
        nn.init.zeros_(self.query_projection.bias)

    def make_queries(self, features: Tensor) -> Tensor:
        self._validate_features(features)
        shape = (*features.shape[:-1], self.num_queries, self.d_e)
        return self._finish_queries(self.query_projection(features).reshape(shape))


class MQRHead(VocabularyHead):
    """GELU concept map followed by independent parallel query projections."""

    def __init__(
        self, d_model: int, vocab_size: int, d_e: int, d_c: int, num_queries: int,
        *, activation: str = "gelu", **kwargs: object,
    ) -> None:
        super().__init__(d_model, vocab_size, d_e, num_queries, **kwargs)
        if isinstance(d_c, bool) or not isinstance(d_c, int) or d_c < 1:
            raise ValueError("d_c must be a positive integer.")
        if activation != "gelu":
            raise ValueError("MQR-v1 uses activation='gelu'.")
        self.d_c = d_c
        self.concept = nn.Linear(d_model, d_c)
        self.query_projection = nn.Linear(d_c, num_queries * d_e)
        # Each query occupies independent rows, with no shared or copied initialization.
        for projection in (self.concept, self.query_projection):
            nn.init.normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)

    def make_queries(self, features: Tensor) -> Tensor:
        self._validate_features(features)
        concept = F.gelu(self.concept(features))
        shape = (*features.shape[:-1], self.num_queries, self.d_e)
        return self._finish_queries(self.query_projection(concept).reshape(shape))


def build_head(config: HeadConfig, d_model: int, vocab_size: int) -> VocabularyHead:
    common = dict(
        temperature=config.temperature,
        normalize_queries=config.normalize_queries,
        normalize_output_rows=config.normalize_output_rows,
    )
    if config.kind == "dense":
        return DenseHead(d_model, vocab_size, **common)
    if config.kind == "direct":
        return DirectHead(d_model, vocab_size, config.d_e, config.num_queries, **common)
    if config.kind == "mqr":
        return MQRHead(
            d_model, vocab_size, config.d_e, config.d_c, config.num_queries,
            activation=config.activation, **common,
        )
    raise ValueError(f"Unknown head kind: {config.kind!r}")
