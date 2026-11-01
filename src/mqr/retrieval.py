"""Inference-only candidate union and exact aggregate reranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import torch
from torch import Tensor

from .heads import VocabularyHead


@dataclass(frozen=True)
class CandidateBatch:
    ids: Tensor
    scores: Tensor
    counts: Tensor


def table_fingerprint(head: VocabularyHead) -> str:
    """Fingerprint effective float32 rows, exactly the geometry stored in Faiss."""
    rows = head.output_vectors().detach().float().cpu().contiguous()
    payload = str(tuple(rows.shape)).encode() + rows.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _table_state(head: VocabularyHead) -> tuple[object, ...]:
    rows = head.output.weight
    return (
        id(rows), rows.data_ptr(), rows._version, tuple(rows.shape), rows.dtype,
        rows.device, head.normalize_output_rows,
    )


class FaissIPIndex:
    """A frozen flat or HNSW inner-product index.

    Faiss is optional and imported only here. Search checks parameter identity,
    storage, mutation version and geometry in constant time. A full fingerprint
    is computed at build and can be rechecked explicitly using ``assert_current``.
    Publish an immutable checkpoint; unsupported mutations through ``weight.data``
    bypass PyTorch's version counter and require explicit full validation.
    """

    def __init__(
        self, head: VocabularyHead, method: str = "flat", *, hnsw_m: int = 32,
        ef_search: int = 64, ef_construction: int = 200,
    ) -> None:
        if method not in ("flat", "hnsw"):
            raise ValueError("Faiss method must be 'flat' or 'hnsw'.")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (hnsw_m, ef_search, ef_construction)):
            raise ValueError("HNSW parameters must be positive integers.")
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install the optional faiss-cpu dependency to use FaissIPIndex.") from exc
        rows = head.output_vectors().detach().float().cpu().contiguous().numpy()
        if not torch.isfinite(torch.from_numpy(rows)).all():
            raise ValueError("Cannot index non-finite output rows.")
        self.method = method
        self.d_e = head.d_e
        self.vocab_size = head.vocab_size
        self.fingerprint = table_fingerprint(head)
        self._table_state = _table_state(head)
        if method == "flat":
            self._index = faiss.IndexFlatIP(head.d_e)
        else:
            self._index = faiss.IndexHNSWFlat(head.d_e, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = ef_search
            self._index.hnsw.efConstruction = ef_construction
        self._index.add(rows)

    def assert_current(self, head: VocabularyHead, *, full: bool = False) -> None:
        stale = self.d_e != head.d_e or self.vocab_size != head.vocab_size or self._table_state != _table_state(head)
        if full and not stale:
            stale = self.fingerprint != table_fingerprint(head)
        if stale:
            raise RuntimeError("Stale output index: rebuild it from the frozen checkpoint's current output table.")

    def search(self, head: VocabularyHead, queries: Tensor, k: int) -> Tensor:
        self.assert_current(head)
        if queries.shape[-1] != self.d_e:
            raise ValueError("Query width does not match index width.")
        if not torch.isfinite(queries).all():
            raise ValueError("Index queries must be finite.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        width = min(k, self.vocab_size)
        if queries.numel() == 0:
            return torch.empty((*queries.shape[:-1], width), device=queries.device, dtype=torch.long)
        query_array = queries.detach().float().reshape(-1, self.d_e).cpu().contiguous().numpy()
        _, ids = self._index.search(query_array, width)
        return torch.as_tensor(ids, device=queries.device, dtype=torch.long).reshape(*queries.shape[:-1], width)


@torch.no_grad()
def retrieve_candidates(
    head: VocabularyHead,
    features: Tensor,
    k: int,
    method: str = "flat",
    index: FaissIPIndex | None = None,
    always_include: Iterable[int] = (),
) -> CandidateBatch:
    """Union per-query top-k IDs, sort/deduplicate, then rerank with all queries.

    Sorted IDs make ties reproducible across query order. The returned scores are
    exact aggregate logits on the union. The union can omit the aggregate winner,
    even for exact per-query search; measure coverage against ``head.score_all``.
    ``always_include`` is for predeclared EOS/task IDs, never observed gold tokens.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if method not in ("flat", "faiss", "hnsw"):
        raise ValueError("method must be 'flat', 'faiss' or 'hnsw'.")
    extras = tuple(always_include)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= head.vocab_size for token in extras):
        raise ValueError("always_include must contain valid integer vocabulary IDs.")
    if method in ("faiss", "hnsw") and index is None:
        raise ValueError("Faiss retrieval requires an explicitly constructed frozen index.")
    if method == "hnsw" and index is not None and index.method != "hnsw":
        raise ValueError("method='hnsw' requires an HNSW index.")
    if index is None:
        component = head.component_scores(features)
        if not torch.isfinite(component).all():
            raise ValueError("Retrieval component scores must be finite.")
        per_query = component.topk(min(k, head.vocab_size), dim=-1).indices
    else:
        queries = head.make_queries(features)
        per_query = index.search(head, queries, k)
    leading = features.shape[:-1]
    n_contexts = features.numel() // head.d_model
    if n_contexts == 0:
        ids = torch.empty((*leading, 0), dtype=torch.long, device=features.device)
        return CandidateBatch(ids, features.new_empty((*leading, 0), dtype=torch.float32), torch.zeros(leading, dtype=torch.long, device=features.device))
    per_context = per_query.reshape(n_contexts, -1)
    extra_ids = torch.tensor(extras, dtype=torch.long, device=features.device)
    unions = [torch.unique(torch.cat((row[row >= 0], extra_ids)), sorted=True) for row in per_context]
    counts = torch.tensor([row.numel() for row in unions], dtype=torch.long, device=features.device)
    width = int(counts.max().item())
    ids = torch.full((n_contexts, width), -1, dtype=torch.long, device=features.device)
    for position, row in enumerate(unions):
        ids[position, :row.numel()] = row
    ids = ids.reshape(*leading, width)
    return CandidateBatch(ids, head.score_candidates(features, ids), counts.reshape(leading))
