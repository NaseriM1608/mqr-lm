# Historical Kaggle workflow

The notebooks in [`notebooks/`](../notebooks/) preserve the earlier bounded smoke, calibration and tuning workflows. They expect separately prepared source/data packages and the appropriate Kaggle runtime. They do not automatically reproduce the complete learning confirmation or subsequent CPU studies.

Keep the runtime's CUDA-enabled PyTorch for GPU work; the historical Windows CPU lockfile is not a Kaggle installation recipe. These notebooks contain no saved outputs or credentials. Source/data packaging is implemented in [`mqr.package`](../src/mqr/package.py), with explicit provenance checks.

For the completed results, read the [main README](../README.md), [full report](../research/report/report.pdf), and [reproduction boundaries](REPRODUCING.md). The original operational logs and execution-specific instructions are preserved in the local research archive rather than included in the public repository.
