"""Coverage against the full distribution and query-collapse diagnostics."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .heads import VocabularyHead, _validate_ids


@torch.no_grad()
def coverage_metrics(
    exact_logits: Tensor, candidate_ids: Tensor, gold: Tensor | None = None, top_k: int = 50,
) -> dict[str, Tensor]:
    """Return per-context metrics, without averaging contexts or languages.

    ``retained_mass`` sums full-vocabulary probabilities on the deduplicated
    candidate set. ``retained_topk_mass`` sums those same probabilities on its
    intersection with the exact top-k set; it never renormalizes candidates.
    ``topk_mass_recall`` divides this intersection mass by ``full_topk_mass``.
    ``topk_recall`` is the unweighted fraction of exact top-k IDs retrieved.
    ``top1_agreement`` compares exact aggregate argmax to the aggregate argmax
    restricted to candidates. Empty candidate sets always disagree.
    """
    if exact_logits.ndim < 1 or exact_logits.shape[-1] < 1 or not exact_logits.is_floating_point():
        raise ValueError("exact_logits must be a floating tensor ending in a nonempty vocabulary.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if torch.isnan(exact_logits).any() or torch.isposinf(exact_logits).any() or not torch.isfinite(exact_logits).any(dim=-1).all():
        raise ValueError("Each context needs finite probability support, with no NaN or +inf logits.")
    vocab_size = exact_logits.shape[-1]
    leading = exact_logits.shape[:-1]
    _validate_ids(candidate_ids, leading, vocab_size)
    if candidate_ids.device != exact_logits.device:
        raise ValueError("Candidate IDs and logits must be on the same device.")
    ids = candidate_ids.expand(*leading, candidate_ids.shape[-1]) if candidate_ids.ndim == 1 else candidate_ids
    # scatter-add avoids counting duplicate retrieved IDs more than once.
    counts = torch.zeros_like(exact_logits, dtype=torch.long)
    counts.scatter_add_(-1, ids.long().clamp_min(0), (ids >= 0).long())
    present = counts > 0
    values = exact_logits.float() if exact_logits.dtype in (torch.float16, torch.bfloat16) else exact_logits
    probabilities = values.softmax(dim=-1)
    actual_k = min(top_k, vocab_size)
    top_ids = values.topk(actual_k, dim=-1).indices
    top_mass = probabilities.gather(-1, top_ids)
    covered_top = present.gather(-1, top_ids)
    full_topk_mass = top_mass.sum(dim=-1)
    retained_topk_mass = (top_mass * covered_top).sum(dim=-1)
    candidate_count = present.sum(dim=-1)
    full_top1 = values.argmax(dim=-1)
    candidate_top1 = values.masked_fill(~present, -torch.inf).argmax(dim=-1)
    # All retrieved tokens can have exact probability zero when -inf is allowed.
    has_support = (present & torch.isfinite(values)).any(dim=-1)
    result = {
        "candidate_count": candidate_count,
        "retained_mass": (probabilities * present).sum(dim=-1),
        "full_topk_mass": full_topk_mass,
        "retained_topk_mass": retained_topk_mass,
        "topk_mass_recall": retained_topk_mass / full_topk_mass,
        "topk_recall": covered_top.float().mean(dim=-1),
        "top1_agreement": has_support & (full_top1 == candidate_top1),
    }
    if gold is not None:
        if gold.shape != leading or gold.dtype not in (torch.int32, torch.int64):
            raise ValueError("Gold IDs must be integer tensors with logits' leading dimensions.")
        if gold.device != exact_logits.device or torch.any((gold < 0) | (gold >= vocab_size)):
            raise ValueError("Gold IDs must be valid vocabulary IDs on the logits' device.")
        result["gold_included"] = present.gather(-1, gold.long().unsqueeze(-1)).squeeze(-1)
    return result


@torch.no_grad()
def query_diagnostics(head: VocabularyHead, features: Tensor) -> dict[str, Tensor]:
    """Measure angular redundancy, winner diversity and exact branch mass.

    Branch mass is softmax_j(logsumexp_v a_jv), the effective mixture weight
    under globally aggregated LSE. Equal LSE coefficients do not enforce equal
    weights. These diagnostics require exact vocabulary scores and are intended
    for evaluation subsets rather than every timed training step.
    """
    queries = head.make_queries(features).float()
    scores = head.component_scores(features).float()
    unit = F.normalize(queries, dim=-1)
    similarity = torch.matmul(unit, unit.transpose(-1, -2))
    if head.num_queries == 1:
        mean_cosine = similarity.new_zeros(similarity.shape[:-2])
    else:
        mask = ~torch.eye(head.num_queries, device=features.device, dtype=torch.bool)
        mean_cosine = similarity[..., mask].mean(dim=-1)
    winners = scores.argmax(dim=-1)
    # Count each winning ID once while retaining arbitrary leading dimensions.
    duplicate = winners.unsqueeze(-1) == winners.unsqueeze(-2)
    earlier = torch.ones((head.num_queries, head.num_queries), dtype=torch.bool, device=features.device).tril(-1)
    unique_top1 = (~(duplicate & earlier).any(dim=-1)).sum(dim=-1)
    branch_mass = torch.logsumexp(scores, dim=-1).softmax(dim=-1)
    branch_entropy = -(branch_mass * branch_mass.clamp_min(torch.finfo(branch_mass.dtype).tiny).log()).sum(dim=-1)
    if head.num_queries > 1:
        branch_entropy = branch_entropy / math.log(head.num_queries)
    return {
        "query_norms": torch.linalg.vector_norm(queries, dim=-1),
        "mean_pair_cosine": mean_cosine,
        "unique_top1": unique_top1,
        "branch_mass": branch_mass,
        "normalized_branch_entropy": branch_entropy,
    }
