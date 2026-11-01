"""Recompute published MQR-LM result summaries using the standard library.

This is an arithmetic/provenance check of exported records, not a new model
experiment, checkpoint audit, candidate-search replay, or significance test.
Run from any directory: python research/summarize.py [--output summary.json]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ARMS = ("dense", "mqr", "mqr_scale", "direct1_scale", "direct4_scale")
LANGUAGES = ("en", "de", "ar", "fa")
SEEDS = (0, 1, 2)
GREEDY_THRESHOLD = 0.995
MASS_THRESHOLD = 0.99


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(relative: str):
    return json.loads((HERE / relative).read_text(encoding="utf-8-sig"))


def read_csv(relative: str) -> list[dict[str, str]]:
    with (HERE / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def near(left, right, label: str) -> None:
    left, right = float(left), float(right)
    require(math.isfinite(left) and math.isfinite(right), f"Non-finite value: {label}")
    require(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10),
            f"Published records disagree: {label}")


def passes(row: dict) -> bool:
    return (float(row["top1_agreement"]) >= GREEDY_THRESHOLD
            and float(row["topk_mass_recall"]) >= MASS_THRESHOLD)


def verify_provenance() -> dict:
    manifest = read_json("provenance.json")
    require(manifest["schema"] == "mqr-lm-public-evidence-v1", "Unknown provenance schema")
    seen = set()
    for entry in manifest["files"]:
        relative = entry["published_path"]
        require(relative not in seen, f"Duplicate provenance entry: {relative}")
        seen.add(relative)
        path = (HERE / relative).resolve()
        require(path.is_relative_to(HERE), f"Provenance path escapes research/: {relative}")
        require(path.is_file(), f"Missing published record: {relative}")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(len(content) == entry["bytes"], f"Changed byte length: {relative}")
        require(digest == entry["published_sha256"], f"Changed published bytes: {relative}")
        require(entry["original_sha256"] == digest, f"Export is not byte-identical: {relative}")
    for study in ("learning", "cpu", "search"):
        require(read_json(f"results/{study}/audit.json")["status"] == "passed",
                f"Recorded {study} audit did not pass")
    return {"verified_files": len(seen), "unchanged_copies": True}


def learning_results() -> dict:
    records = read_csv("results/learning/scores.csv")
    endpoint = [r for r in records if int(r["step"]) == 1000]
    expected = {(arm, seed, lang) for arm in ARMS for seed in SEEDS for lang in LANGUAGES}
    lookup = {(r["arm"], int(r["seed"]), r["language"]): r for r in endpoint}
    require(len(endpoint) == len(lookup) == len(expected) and set(lookup) == expected,
            "Incomplete or duplicated five-arm endpoint grid")
    per_seed = {}
    for arm in ARMS:
        for seed in SEEDS:
            group = [lookup[arm, seed, lang] for lang in LANGUAGES]
            mean = statistics.mean(float(r["full_nll"]) for r in group)
            for row in group:
                near(mean, row["macro_nll"], f"{arm}/seed{seed} macro NLL")
                require(int(row["successful_training_tokens"]) == 16_384_000,
                        "Learning endpoint exposure changed")
            per_seed[arm, seed] = mean
    means = {arm: statistics.mean(per_seed[arm, seed] for seed in SEEDS) for arm in ARMS}
    summary = read_json("results/learning/summary.json")
    require(summary["status"] == "passed", "Learning analysis did not pass")
    gap_rows = read_csv("results/learning/paired_endpoint_gaps.csv")
    rows = []
    for arm in ARMS:
        gaps = [per_seed[arm, seed] - per_seed["dense", seed] for seed in SEEDS]
        if arm != "dense":
            recorded = next(r for r in gap_rows if r["arm"] == arm and r["language"] == "macro")
            near(statistics.mean(gaps), recorded["mean_gap"], f"{arm} dense gap")
            near(statistics.stdev(gaps), recorded["sample_standard_deviation"], f"{arm} paired gap SD")
            original = next(r for r in summary["endpoint_gaps"] if r["arm"] == arm and r["language"] == "macro")
            near(statistics.mean(gaps), original["mean_gap"], f"{arm} summary dense gap")
        rows.append({"arm": arm, "mean_macro_nll": means[arm],
                     "gap_vs_dense": statistics.mean(gaps),
                     "paired_dense_gap_sample_sd": statistics.stdev(gaps) if arm != "dense" else None,
                     "per_seed_macro_nll": {str(seed): per_seed[arm, seed] for seed in SEEDS}})
    paired = [per_seed["mqr_scale", seed] - per_seed["direct4_scale", seed] for seed in SEEDS]
    cells = []
    for seed in SEEDS:
        for lang in LANGUAGES:
            difference = float(lookup["mqr_scale", seed, lang]["full_nll"]) - float(lookup["direct4_scale", seed, lang]["full_nll"])
            cells.append({"seed": seed, "language": lang, "mqr_minus_direct4_nll": difference})
    original_gap = means["mqr"] - means["dense"]
    adjusted_gap = means["mqr_scale"] - means["dense"]
    return {
        "training_seeds": list(SEEDS), "languages": list(LANGUAGES), "endpoint_updates": 1000,
        "successful_predictions_per_run": 16_384_000, "completed_runs": len(ARMS)*len(SEEDS),
        "arms": rows,
        "mqr_minus_direct4": {"paired_macro_mean": statistics.mean(paired),
                              "paired_macro_sample_sd": statistics.stdev(paired),
                              "direct4_better_endpoint_cells": sum(r["mqr_minus_direct4_nll"] > 0 for r in cells),
                              "endpoint_cells": len(cells), "cells": cells},
        "initialization": {"dense_gap_closure_fraction": 1-adjusted_gap/original_gap,
                           "total_mqr_nll_reduction_fraction": (means["mqr"]-means["mqr_scale"])/means["mqr"]},
        "limit": "Language cells are paired comparisons, not independent trained seeds; training stopped before schedule completion.",
    }


def cpu_results() -> dict:
    coverage = read_csv("results/cpu/fidelity.csv")
    grid = [r for r in coverage if r["method"] == "hnsw" and r["language"] == "macro"]
    require(len(grid) == 15, "Expected 15 original HNSW arm/budget cells")
    require({(r["arm"], int(r["B"])) for r in grid}
            == {(arm, budget) for arm in ARMS for budget in (32, 128, 512)}, "Original HNSW grid changed")
    for row in grid:
        require(passes(row) == (row["point_screen_passed"] == "True"), "Recorded CPU screen flag differs")
    raw = read_csv("results/cpu/raw_timings.csv")
    groups = defaultdict(list)
    for row in raw:
        if row["stage"] == "warm_decode":
            key = (row["arm"], row["method"], row["document_id"], row["language"], int(row["prefix_length"]))
            require(int(row["head_predictions"]) == 16, "CPU prediction count changed")
            groups[key].append((int(row["repeat"]), float(row["request_ms"])))
    reduced = defaultdict(list)
    for key, values in groups.items():
        require(len(values) == 3 and {r for r, _ in values} == {0, 1, 2}, "Incomplete CPU timing repeats")
        reduced[key[:2]].append(statistics.median(v for _, v in values)/16)
    summary = read_json("results/cpu/summary.json")
    paths = []
    for path in summary["paths"]:
        values = reduced[path["arm"], path["method"]]
        require(len(values) == 16, "Incomplete CPU timing cells")
        measured = statistics.mean(values)
        near(measured, path["decode_ms_per_prediction"], "CPU raw-repeat timing reduction")
        paths.append({"arm": path["arm"], "method": path["method"],
                      "decode_ms_per_prediction": measured,
                      "fidelity": path["fidelity"], "point_screen_passed": path["point_screen_passed"]})
    require(len(paths) == 10, "Expected exact/HNSW timing paths for all five arms")
    return {"date": "2026-10-25", "trained_seed": 0, "hnsw_cells": len(grid),
            "hnsw_joint_screen_passes": sum(passes(r) for r in grid), "timing_paths": paths,
            "timing_reduction": "Means of 16 request-cell medians; three repeats; 16 predictions per request",
            "audit_correction": "Only branch-marginal sanity tolerance changed from 1e-6 to 1e-4; see checker_correction.md.",
            "limit": "Only B128 approximate timing was measured in this first study; fidelity at B32/B512 does not imply timing.",}


def search_results() -> dict:
    coverage = read_csv("results/search/coverage.csv")
    timing = read_csv("results/search/timings.csv")
    summary = read_json("results/search/summary.json")
    require(summary["status"] == "complete_audited_fixed_checkpoint_search", "Search study incomplete")
    require(not summary["new_training"] and not summary["gpu_requested"], "Search scope changed")
    audit_hash = hashlib.sha256((HERE/"results/search/audit.json").read_bytes()).hexdigest()
    require(summary["audit_sha256"] == audit_hash, "Search summary/audit binding changed")
    grid = [r for r in coverage if r["phase"] == "tune" and r["method"] == "hnsw" and r["language"] == "macro"]
    expected = {(a, b, e) for a in ("dense", "direct1_scale", "direct4_scale")
                for b in (128, 256, 512) for e in (128, 256, 512, 1024)}
    require(len(grid) == len(expected) and {(r["arm"], int(r["B"]), int(r["ef_search"])) for r in grid} == expected,
            "Incomplete or duplicated search tuning grid")
    time_lookup = {(r["arm"], r["phase"], r["config_id"], r["family"]): r for r in timing}
    arms = []
    for arm in summary["arms"]:
        name, selected = arm["arm"], arm["selected"]
        local_grid = [r for r in grid if r["arm"] == name]
        require(sum(passes(r) for r in local_grid) == arm["tuning_eligible_configurations"], "Search tuning eligibility differs")
        if not any(passes(r) for r in local_grid):
            require(arm["selection_status"] == "diagnostic_only_no_tuning_eligible",
                    "Ineligible tuning fallback was promoted")
        source_grid = {r["config_id"]: r for r in arm["coverage_summaries"] if r["phase"] == "tune" and r["method"] == "hnsw"}
        for row in local_grid:
            for key in ("top1_agreement", "topk_mass_recall", "retained_mass"):
                near(row[key], source_grid[row["config_id"]]["macro"][key], "Search CSV/summary join")
            require(int(time_lookup[name, "tune", row["config_id"], "warm_decode"]["cells"]) == 8,
                    "Search tuning timing cells changed")
        confirmation = next(r for r in coverage if r["arm"] == name and r["phase"] == "confirmation"
                            and r["method"] == "hnsw" and r["language"] == "macro")
        require(confirmation["config_id"] == selected["config_id"], "Confirmation used a different selected configuration")
        for key in ("top1_agreement", "topk_mass_recall", "retained_mass"):
            near(confirmation[key], arm["confirmation"][key], "Search confirmation CSV/summary join")
        require(passes(confirmation) == arm["confirmation_joint_screen_passed"], "Confirmation screen differs")
        eligible = any(passes(r) for r in local_grid) and passes(confirmation)
        require(eligible == arm["eligible_after_both_phases"], "Post-selection eligibility differs")
        exact = time_lookup[name, "confirmation", "exact", "warm_decode"]
        approximate = time_lookup[name, "confirmation", selected["config_id"], "warm_decode"]
        for record in (exact, approximate):
            require(int(record["cells"]) == 16, "Search confirmation timing cells changed")
            near(float(record["mean_cell_median_ms"])/16, record["ms_per_prediction"], "Search timing units")
        exact_ms, approximate_ms = float(exact["ms_per_prediction"]), float(approximate["ms_per_prediction"])
        near(exact_ms, arm["exact_decode_ms_per_prediction"], "Search exact timing")
        near(approximate_ms, arm["selected_decode_ms_per_prediction"], "Search ANN timing")
        reduction = 1-approximate_ms/exact_ms
        near(reduction, arm["observed_latency_reduction_fraction"], "Search latency fraction")
        arms.append({"arm": name, "selected": selected, "selection_status": arm["selection_status"],
                     "tuning_joint_passes": sum(passes(r) for r in local_grid),
                     "confirmation": {key: float(confirmation[key]) for key in ("top1_agreement", "topk_mass_recall", "retained_mass")},
                     "exact_decode_ms_per_prediction": exact_ms, "ann_decode_ms_per_prediction": approximate_ms,
                     "ann_latency_change_fraction": -reduction, "eligible_after_both_phases": eligible})
    d4_flat = next(r for r in coverage if r["arm"] == "direct4_scale" and r["phase"] == "confirmation"
                   and r["method"] == "flat" and int(r["B"]) == 512 and r["language"] == "macro")
    return {"execution_date": "2026-10-27", "analysis_date": "2026-10-28", "trained_seed": 0,
            "hnsw_tuning_cells": len(grid), "hnsw_joint_tuning_passes": sum(passes(r) for r in grid),
            "arms": arms, "direct4_exhaustive_union_confirmation_b512":
                {key: float(d4_flat[key]) for key in ("topk_mass_recall", "retained_mass")},
            "limit": "Tuning and additional disjoint confirmation documents come from the same bounded development distribution. Diagnostic fallbacks cannot be promoted by confirmation alone. Timing CSVs are recorded aggregates, not independent raw-repeat replays.",}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the deterministic JSON summary to this file")
    args = parser.parse_args()
    result = {"status": "passed", "scope": "Arithmetic and identity checks of published records; no model execution",
              "provenance": verify_provenance(), "thresholds": {"greedy_agreement": GREEDY_THRESHOLD, "top50_mass_recall": MASS_THRESHOLD},
              "learning": learning_results(), "cpu": cpu_results(), "search": search_results()}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
