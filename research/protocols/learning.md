# MQR-v1 broader-data learning confirmation

23 October 2026. Authorized by the user's instruction to proceed after the next-step recommendation. This protocol is a new experiment; the readiness pilots and immutable packages remain historical controls. Dataset and execution identities must be filled and frozen before GPU learning starts.

## Decision

Determine whether initialization-adjusted MQR offers competitive exact language-model likelihood and a useful quality/latency/memory tradeoff. Initial weight scaling changes no architecture. A faster equivalent loss is an execution candidate, not a new modeling objective.

The primary learning comparison is paired macro validation NLL, adjusted MQR minus dense, at the same successful training-token exposure. Also report original MQR, adjusted Direct1 and adjusted Direct4, per-language scores, every seed, actual GPU-hours and elapsed time, overflows and peak allocated/reserved memory. No test-set, translation or NLI score may select settings. No claim follows automatically from a short endpoint.

## Data gates

- Acquire an immutable, seeded multi-shard/row-group/interior-row sample from the already pinned English FineWeb and German/Arabic/Persian FineWeb2 inventories. Audit native Persian Wikipedia separately. Preserve original Unicode and full acquisition provenance; budget failures never substitute a shard prefix.
- Audit domain/source shares by documents and bytes, language predictions and disagreements, repetitions, encoding and Persian orthography. Review seeded ordinary and risk examples. An agent review is labeled as such, not a human language-quality certification. Source weights and any domain filtering must be decided from this audit before model results.
- Keep public benchmark exclusions and all prior readiness validation/test documents out of new training. Preserve a content/URL/ID split ledger and raw-to-selected provenance. Existing frozen held-outs never become training data through append/reselection.
- Target at least 20 million balanced training-token positions; require at least 16,384,000 before the first 1,000-update comparison. Record unique usable positions separately from sampled training exposure. Add versioned, independently seeded sampling rounds if the audited source sample is too small; do not multiply the tiny readiness bundle.
- Use a shared 65,536-entry byte BPE fitted on this snapshot's training documents only, balanced by language. Record actual fitting bytes and document identities. Up to 100 MB/language is a cap, not a claim of data already acquired. Validate bytes, special markers, round trips and fertility, then freeze the tokenizer and prepared hashes.
- Freeze document-isolated validation/test partitions. The validation bank contains 256 disjoint 513-token blocks per language, predicting 131,072 tokens/language (524,288 total). Require at least 131,328 tokens/language in each held-out split. The test stream stays locked.

The source audit is complete before learning. The frozen development filter retains 55,486 of 64,512 candidates (234,895,750 original UTF-8 text bytes), before global duplicate/benchmark screening. English uses FineWeb; German and Arabic use FineWeb2. Both acquired Persian FineWeb2 rounds contain zero original ZWNJ and are excluded from these learning inputs. Persian instead uses naturally joiner-preserving mC4 web text, with native Wikipedia contributing 4.53% of accepted Persian text bytes before subsequent selection/tokenization. No joiners or letter substitutions are inserted. Final balanced token shares and volume must still pass the independent prepared-data audit.

Sampling scope is explicit: two seeded multi-shard Parquet rounds for English/German/Arabic and native Persian Wikipedia, plus 16 seeded mC4 shards with three interior windows within each shard's first8,192 rows. This bounded gzip-prefix design is not a representative sample of all mC4 rows. The common prose/language filter, all-row byte reconciliation and 400 bounded agent excerpt reviews are frozen in `artifacts/data_confirmation_v1/quality`. Residual web noise and language mixing are documented in `SOURCE_FINDINGS.md`; this is development data, not the final main-training corpus. All64,512 raw candidate identities were assigned before filtering, with971 original readiness held-outs reserved as test through the persistent identity ledger.

## Models and schedule

Use the shared 12-layer width-512 decoder, 8 attention heads, FFN 2048, context512 and exact full-vocabulary likelihood. Keep MQR at M4, embedding128, concept256, raw dots, temperature1, no gate/output bias or compulsory normalization.

Run seeds0,1,2 with five arms: dense, original MQR, adjusted MQR, adjusted Direct1 and adjusted Direct4. All arms for a seed start from the same backbone and receive the same successful global windows. Fit each adjusted arm's final query-projection gain against that seed's dense aggregate-logit scale using the same frozen training-only feature bank. Do not reuse observed gains40–43 as constants.

Global batch32 at context512 gives16,384 predictions/update. Initial endpoint1,000 updates gives16,384,000 predictions/arm,245,760,000 across15 runs. Microbatch8/accumulation4 is the starting execution policy. Freeze the exact loss implementation/chunk policy after the separately bounded algebraic-loss GPU audit; offer comparable tuning opportunities to dense and report any per-head execution differences. No architecture-capacity adjustment is an execution fix.

To permit a scientifically consistent continuation, the common LR schedule is fixed before the first update: 50 warmup updates, peak3e-4, cosine horizon4,000 updates, floor0.1 of peak. **The first job stops at1,000**, rather than finishing the cosine schedule. This resolves the earlier proposed1,000-update cosine endpoint: changing an already run schedule to extend it would change the experiment. Any continuation uses the original4,000-update schedule, data, tokenizer, sampler and checkpoints. A different schedule/data/tokenizer requires fresh matched runs.

Evaluate exact logits at steps0,50,125,250,500,750,1,000. Preserve all raw per-window scores and attempt records. Time setup, initialization/probing, train attempts (including retries), validation, checkpoint I/O and whole run separately. Parallel-device runs are not isolated throughput benchmarks; record their device mapping and shared-host contention limits. Paired window/seed variation is reported, not replaced by a claim of statistical certainty from three seeds.

Before the long queue, run all five seed0 arms for a two-update GPU integration prefix, including exact validation and CPU checkpoint reload. These are the first two updates of the same fixed schedule and successful batch stream, not a separately tuned schedule. Preserve the extra step2 validation events; after all five prefixes pass, continue their checkpoints to1,000. Seeds1/2 start fresh under the same plan. A preflight failure prevents the long queue. Resume requires the same recorded hardware/software environment and all source/data/config identities. All raw session costs, including work abandoned after a rollback, remain visible; interrupted sessions have a stated wall-time lower bound.

Execution freeze after the joint-loss audit: retain the reference exact CE and common chunk512 for every learning arm. Joint MQR at512 gained5.50% throughput while increasing allocated peak by0.248GiB; dense slowed18.36%. This modest per-head tradeoff is retained as a separate result. The confirmation uses one common reference loss path and no automatic backend/default promotion.

Require peak reserved allocator memory at most90% of the individual GPU's physical capacity in preflight and learning. Report the post-setup worker peak, which can include validation allocations; driver/other-process memory is outside this allocator measure. A memory failure does not trigger an unreported microbatch/model change.

Retain portable model/optimizer/scaler/sampler/global-RNG checkpoints and initialization sidecars. Verify CPU reloads and config/data/schedule identities. Resume must reproduce the successful batch stream; AMP failures retry the same global batch. Failed/incomplete arms cannot be called completed evidence.

## Bounded execution candidate and decisions

The artifact-only joint loss must pass frozen GPU loss/gradient/optimizer-trajectory gates before its cost measurements. One registered test covers both dense/MQR and chunks64/512, with fresh reversed repeats. Preserve reference implementations and report each head's gain and memory independently. Failed parity stops that candidate without relaxing limits. A failed speed test ends this optimization attempt; it does not justify indefinite kernel tuning.

At1,000 updates report the full paired comparison and learning curves. Use the previously declared project tolerances as context (+0.10 nats initial mechanism screen, +0.03 nats paper-quality target), not as thresholds changed after this experiment. Competitive NLL alone is not an architectural or deployed-speed advantage.

If curves/uncertainty leave the comparison unresolved, cost a matched continuation of all arms under the already fixed schedule before interpreting it as failure. When sufficiently trained comparisons show no useful quality/inference/memory tradeoff, redesign or pause the current variant rather than launching the3B-per-model study automatically. Retrieval tests use the retained checkpoints, exact aggregate-logit references, candidate coverage/mass, real cached generation, and index memory/build costs, including strong dense/direct retrieval alternatives. No paper-level or downstream result is claimed by this protocol.
