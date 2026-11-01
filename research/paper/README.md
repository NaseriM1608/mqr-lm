# Paper draft

The [nine-page paper](paper.pdf) is the 28 October 2026 draft of **Query Diversity Without a Likelihood Advantage: A Controlled Study of Multi-Query Vocabulary Heads**. It includes the learning comparison, CPU inference diagnostic, fixed-graph search follow-up, limitations, references and appendices. It has not been submitted or published.

`manuscript.md` and `assets/` retain its editable text, five scientific figures and ten Computer Modern equations. The PDF is the original, already reviewed artifact. Author names and affiliations remain to be supplied and verified by the researcher.

## Render the source

From the repository root:

```bash
python -m pip install -e ".[paper]"
python research/paper/build_pdf.py
```

This produces ignored `paper-rebuilt.pdf` and `paper-rebuilt.layout.json` beside the source. It uses ReportLab's standard text fonts and the supplied embedded equation assets; it needs no private checkpoints or Windows font directory. The committed `paper.pdf` is not overwritten. PDF metadata and serialization may differ between library versions.

Rendering text and existing assets is distinct from rerunning experiments or regenerating figures from checkpoints. Public aggregate reanalysis uses [`../summarize.py`](../summarize.py). The original manuscript's references to local experiment paths describe the original research workspace; the public export mapping is [`../provenance.json`](../provenance.json).

`original_figure_manifest.json` preserves the original asset hashes and workspace source identities. Its old workspace paths are provenance records, not a promise that private original files are distributed. No new literature search, experiment or mathematical formula was introduced by packaging this draft for GitHub.
