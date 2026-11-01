# Architecture and probability calculations

MQR changes the vocabulary readout, while retaining a conventional causal decoder. The main experiment uses a 12-layer, 512-wide decoder and a 65,536-entry vocabulary. Input/output embeddings are untied.

## The four head types

| Head | Query construction | Query count / width | Output-table shape |
|---|---|---|---|
| Dense | Final decoder feature | 1 / 512 | 65,536 × 512 |
| Direct1 | Affine projection of the feature | 1 / 128 | 65,536 × 128 |
| Direct4 | Affine projection, then reshape | 4 / 128 | 65,536 × 128 |
| MQR | Affine concept map + GELU, then affine query projection | 4 / 128 | 65,536 × 128 |

The 256-wide concept representation is specific to MQR. Direct4 removes that pathway while preserving query count, table width and score aggregation, differing by only 256 parameters. Query count here is unrelated to attention-head count.

For row-vector decoder feature $F$, MQR forms $C=\mathrm{GELU}(FW_c+b_c)$ and $q_j=CV_j+b_j$. Every query scores every shared output-token row, $a_{jv}=q_j^\top E_v$. The final logits are $z_v=\log\left(\sum_{j=1}^{M}\exp(a_{jv})\right)-\log M$, followed by a single softmax across vocabulary tokens. Training minimizes the mean exact negative log probability of observed next tokens.

Equal coefficients before global aggregation do not mean equal effective branch mass: the corresponding branch-normalized mixture weights depend on each branch's vocabulary partition function. There is no independently learned mixture gate in the tested architecture.

## Initialization

The adjusted arms calibrate the scale of existing final query-projection weights and bias using an unlabeled training-feature bank before training. They add no parameter and no inference operation. These are separately trained checkpoints; the serving results are not produced by rescaling a trained checkpoint after evaluation.

## Retrieval

Each query retrieves token IDs, the IDs are unioned and deduplicated, normally scored EOS is included, and every candidate is reranked with all original queries. Gold tokens are never injected. Candidate-only normalization truncates the original distribution.

Exhaustive per-query retrieval does not guarantee exhaustive aggregate retrieval. With score pairs (10, -100), (-100, 10), and (9.5, 9.5), the last token can have the highest aggregate score while missing both per-query top-one sets. Coverage therefore separates exhaustive-union omissions from extra approximate-index error.

## Source map

- [`heads.py`](../src/mqr/heads.py): head construction, aggregation and all-query candidate scoring.
- [`model.py`](../src/mqr/model.py): shared decoder, exact chunked loss and KV cache.
- [`retrieval.py`](../src/mqr/retrieval.py): exhaustive, Faiss and HNSW retrieval.
- [`metrics.py`](../src/mqr/metrics.py): exact-reference probability and coverage measures.
- [`training.py`](../src/mqr/training.py): training, checkpointing and resume.
- [Paper equations](../research/paper/manuscript.md): formal presentation and limitations.
