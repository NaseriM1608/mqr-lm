# Published research evidence

This directory preserves selected numerical records from the completed MQR-LM
investigation. It is sufficient to inspect the reported comparisons and recompute
the aggregate arithmetic described below. It does not contain a complete corpus,
trained checkpoints, or every remote execution record.

## Recompute the published summaries

Python 3.11 or newer is sufficient. The summarizer uses only the standard library;
it does not import PyTorch, load a model, run search, or contact a remote service.

```bash
python research/summarize.py
```

To save the same JSON in addition to printing it:

```bash
python research/summarize.py --output .local/published-summary.json
```

The script resolves inputs relative to its own location, so it can be invoked from
another working directory. It checks every copy against `provenance.json` before
calculating results.

## Files and units

| Location | Contents | Interpretation |
|---|---|---|
| `results/learning/` | All recorded learning evaluations, endpoint paired-gap export, analysis summary, and passing audit | NLL is in natural-log units per predicted token. Macro means weight English, German, Arabic and Persian equally. |
| `results/cpu/` | Initial five-arm coverage, branch diagnostics, raw timing repeats, resources, summary, passing corrected audit, and correction disclosure | This is the 25 October seed-0 diagnostic. Approximate timing was measured at B128; coverage also uses B32/B512. |
| `results/search/` | Three-arm tuning/confirmation coverage, aggregate timings, summary, audit, and original figure manifest | This is the separate 27-28 October follow-up. The tuning grid contains 36 arm/configuration cells. |
| `protocols/` | Frozen learning design and the two CPU protocols | These are historical experiment documents, not instructions to launch another run. |
| `provenance.json` | Original relative paths, original SHA-256 identities, public paths and public copy hashes | Every selected evidence/protocol copy is byte-for-byte unchanged. Original source paths describe the local execution archive. |

CSV probability columns such as `top1_agreement`, `topk_mass_recall` and
`retained_mass` are **fractions from 0 to 1**, not already percentages. NLL is
lower-is-better. `ms_per_prediction` is a latency, so lower is faster.

`topk_mass_recall` is the per-context fraction of **exact top-50 probability mass**
covered by candidates, then reduced within language and equally across languages.
`retained_mass` instead uses the **full vocabulary distribution** as its
denominator. Perfect greedy agreement does not imply preservation of either
distribution measure. Candidate-normalized likelihood is not exact full-vocabulary
NLL.

The passing fidelity screen requires both greedy agreement at least 0.995 and
top-50 probability-mass recall at least 0.99. The original HNSW study passes zero
of 15 arm/budget cells. The follow-up passes zero of 36 tuning cells. Its selected
settings are diagnostic fallbacks; favorable confirmation alone could not promote
an ineligible tuning fallback.

## What the public script independently checks

- Every published evidence/protocol file's bytes, length and hash against the
  export manifest, plus the recorded audit statuses.
- Five three-seed endpoint macro NLLs from all 60 seed/language/arm endpoint rows;
  paired dense-gap means and sample standard deviations; all 12 MQR-versus-Direct4
  paired endpoint comparisons; and initialization gap closure.
- The original complete 15-cell HNSW screen and exact/HNSW cached-decode timing
  from the published raw repeats: median of three observations per request cell,
  then mean of 16 cells, divided by 16 predictions per request.
- The follow-up's complete 36-cell grid, CSV-to-summary joins, diagnostic selection
  status, confirmation coverage and eligibility, and selected/exact latency
  changes joined to the aggregate timing CSV.

The initial MQR NLL deficit against dense falls from about 0.595 to 0.149 nats.
Closing 74.9% of that **gap** is different from reducing total NLL by 74.9%; the
total original-MQR NLL reduction is about 6.27%. The script reports both quantities.

## Boundaries of these checks

The source audits cover a larger collection of frozen evidence than is published
here. Their passing statuses are recorded evidence, not a new local checkpoint
audit. These exports cannot independently replay candidate membership, reconstruct
all full-vocabulary probabilities, repeat training, or certify the original model
bodies. The follow-up timing CSV already contains reductions, so checking its
arithmetic is not a replay of its raw timing observations.

Audit and summary JSON files retain original machine paths and private-resource
identifiers for provenance. They are not portable input locations, secrets, or a
claim that a reader has access to those resources. The summarizer reads only the
public copies listed above. Upstream text is not redistributed through these
exports; raw corpus and tokenizer-fitting records are excluded.

The retrospective first CPU checker correction is preserved in
[`results/cpu/checker_correction.md`](results/cpu/checker_correction.md). Only
branch-marginal probability sanity tolerances changed from 1e-6 to 1e-4; raw
observations, fidelity thresholds, scoring and frozen experimental code did not
change. The corrected audit is named `results/cpu/audit.json` in this public layout;
its original filename and full hash remain in `provenance.json`.

Three trained seeds, one short learning endpoint, one trained serving seed, one
shared CPU host, correlated validation prefixes and untested deployment baselines
limit generalization. Language cells are not independent training replications.
The search confirmation documents are disjoint from tuning by ID/content hash but
remain in the same bounded development validation distribution. The negative
Direct-only search follow-up does not test another MQR activation or add a new
concept-path learning comparison.
