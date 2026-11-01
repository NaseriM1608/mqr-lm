# External checker correction after the first audit

The complete run and collection succeeded. The original external audit is preserved unchanged at `../audit.json`; it failed with `Invalid finite numeric observation`. Calling the checker directly traced the failure to `check_branches`, where each branch responsibility marginal and its sum were required to be within 1e-6 of the unit-probability bound. The pre-correction validator is retained as `validate_cpu_before.py`.

The frozen core computes `sum_v softmax_j(component_scores) * softmax_v(full_logits)` in FP32 over 65,536 vocabulary entries. Even for one query, this is a floating-point softmax total rather than an exactly represented constant one. The maximum absolute marginal-total deviations in the unchanged raw records are:

| Arm | Maximum absolute sum error |
|---|---:|
| Dense | 0.00004661083221435547 |
| Original MQR | 0.000014792196452617645 |
| Adjusted MQR | 0.000017482787370681763 |
| Direct1 | 0.000033020973205566406 |
| Direct4 | 0.000014856457710266113 |

All ordinary coverage metrics remain within the original [0, 1 + 1e-6] bounds. Recomputing the recorded branch entropy from the recorded marginal values differs by at most 1.28e-7. The failure is therefore in the external marginal sanity tolerance, rather than a missing arm, altered retrieval scorer, failed cache/numerical gate, or exceeded execution budget.

A separate local synthetic check used two threads and Torch 2.14.1+cpu, with a 65,536-entry logit vector containing one zero and 65,535 entries equal to -8. Its FP32 softmax summed to 1.0000195503234863; summing those FP32 probabilities in FP64 gave 1.000019371774215. This reproduces the insufficient 1e-6 bound without accessing any trained checkpoint. Other synthetic logit shapes can have larger errors: the replacement tolerance is not a universal error guarantee. PyTorch documents finite precision and backend-dependent reduction behavior in [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

The correction introduces a separate 1e-4 tolerance solely for the per-branch marginal upper bound and its unit-total sanity check. It leaves nonnegativity/finite checks, entropy reconstruction, all coverage metrics and summary joins, candidate/cache precision gates, and the 99.5%/99% fidelity point screens unchanged. It neither clamps nor renormalizes observations. All five arms receive the same check. This external-checker adjustment is retrospective and is disclosed as such; no frozen package, measured code, protocol, checkpoint or raw evidence was changed or rerun.

Five new regression cases cover the observed one-query and four-query drift while keeping fidelity screens unchanged, plus rejection of negative, nonfinite and materially out-of-bound marginals. The new audit is written to a fresh `../audit_v2.json`; the failed first audit remains available.
