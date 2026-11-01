# Data scope and availability

The reported learning comparison used a bounded four-language development snapshot, not the entire upstream corpora. It selected 55,153 documents and fitted one 65,536-token ByteLevel BPE on training text only.

| Split | Positions per language | Total positions |
|---|---:|---:|
| Training | 6,549,926 | 26,199,704 |
| Validation | 775,909 | 3,103,636 |
| Test | 814,053 | 3,256,212 |

The available positions describe the prepared streams. Each learning run consumed 16.384 million predictions, sampled under the frozen successful-update policy. The test stream remained locked for model selection.

## Sources and sampling

English used FineWeb; German and Arabic used FineWeb2; Persian used mC4 with native Wikipedia. Acquired Persian FineWeb2 samples lacked observed original ZWNJ and were excluded from the confirmation inputs. The retained Persian text preserved original joiners rather than inserting synthetic ones. Original mC4 shard/window restrictions, quality filtering, identity-based splitting and benchmark exclusions constrain representativeness and contamination claims.

The older readiness-source configuration files describe the earlier integration/calibration snapshot. They are retained for that workflow and must not be presented as a complete recipe for the later confirmation corpus. The complete report explains the successive preparation stages.

## What is distributed

The repository includes small synthetic/example text fixtures, source configuration metadata, study protocols, numerical CSV/JSON exports, audits and manuscript figures. It does not include raw web text, downloaded benchmark text, processed token streams, the trained tokenizer, or model/optimizer checkpoints. Upstream corpus access and permissions remain separate from this code repository.

The learning evaluation bank predicted 131,072 tokens per language at each evaluation. The CPU serving banks were different document-based workloads: 32 documents for the first diagnostic/tuning, and 32 additional disjoint documents for follow-up confirmation. Their contexts are correlated prefixes from bounded development validation, not new model seeds or locked-test results.

See the [complete report](../research/report/report.pdf), [learning protocol](../research/protocols/learning.md) and [publication provenance](../research/provenance.json).
