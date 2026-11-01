# Reproduction guide

There are three different tasks: testing the software, recomputing published aggregates, and repeating the original model experiments. The public repository supports the first two directly. The third additionally requires the original data/tokenizer, checkpoints and private execution records.

## Test the implementation

Use Python 3.11+ in a virtual environment. For CPU tests, install CPU PyTorch before the package:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,corpus,ann]"
python -m pytest -q
```

The optional `corpus` and `ann` dependencies enable tokenizer, Parquet/source and Faiss tests. The historical Windows snapshot in `requirements-cpu.lock` records the original local environment, not a universal Linux/GPU installation recipe. The remote learning/inference environments had their own recorded versions. Keep a GPU environment's existing CUDA build rather than installing this CPU snapshot there.

Five historical chunk-gate tests depend on a private frozen readiness archive. They explicitly skip when it is absent, while the three independent gradient/regression/corrupt-archive checks in that test module still run. With authorized access to the original archive, set `MQR_FROZEN_READINESS_ARCHIVE` to its absolute path. It must match the helper's frozen SHA-256; substituting arbitrary data does not reproduce the gate. [`numerics.py`](../research/methods/chunk/numerics.py) is copied unchanged from the original experiment helper.

## Run a small offline demonstration

```bash
python -m mqr inspect --config configs/runs/smoke_mqr.json
python -m mqr train --config configs/runs/smoke_mqr.json --train-text examples/smoke_train.txt --validation-text examples/smoke_validation.txt --run-dir .local/smoke-mqr --stop-after 2
python -m mqr evaluate --checkpoint .local/smoke-mqr/last.pt --text examples/smoke_validation.txt --batches 1
python -m mqr generate --checkpoint .local/smoke-mqr/last.pt --prompt "The next token is" --max-new-tokens 16
python -m mqr benchmark --checkpoint .local/smoke-mqr/last.pt --prompt "The next token is" --method flat --k 16 --new-tokens 4 --repeats 3
```

The smoke configuration uses a 259-entry byte tokenizer and a tiny model. Its training text is an example fixture. These commands check execution, checkpoint reload and cached generation; they do not reproduce the four-language research NLL or establish generation quality. Choose a new run directory when repeating the example.

## Recompute public aggregates

```bash
python research/summarize.py
python research/summarize.py --output .local/research-summary.json
```

This script uses Python's standard library and the included numerical exports. It verifies their publication hashes and recomputes the main learning means, paired endpoint comparison and fidelity-screen counts. The evidence guide specifies which additional observations are joined from recorded summaries. It does not claim that exported aggregates independently reconstruct all raw candidates, all vocabulary probabilities or every source-bound audit check.

## Read or rebuild the paper

The committed [paper](../research/paper/paper.pdf) and [technical report](../research/report/report.pdf) are immediately readable. The [paper renderer](../research/paper/README.md) can rebuild the manuscript using its supplied assets and the `paper` dependency group. The original full-report builder and one-off orchestration/collection code remain in the local research archive.

## Repeat the original studies

The frozen [learning](../research/protocols/learning.md), [CPU](../research/protocols/cpu.json) and [search](../research/protocols/search.json) protocols describe the actual study scopes. Exact replication additionally needs the prepared balanced streams, training-only tokenizer fit identities, successful batch/replay ledger, retained model/optimizer checkpoints, source-bound original packages, and appropriate hardware/data access. Those large/private materials are not distributed in this repository.

The pilot and multilingual JSON files preserve original configurations and planning context; they are not all the frozen, executed confirmation jobs. In particular, the legacy top-level protocol JSONs are not runnable `mqr train` configurations. Retained Kaggle notebooks describe earlier bounded workflows, not a command that recreates every final result automatically.
