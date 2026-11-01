"""Artifact gate tests; CPU evidence never impersonates the actual CUDA gate."""
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chunk_numerics_artifact", ROOT / "research/methods/chunk/numerics.py")
numerics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(numerics)
ARCHIVE = Path(os.environ.get("MQR_FROZEN_READINESS_ARCHIVE", ROOT / "artifacts/tuning_v1/mqr-kaggle-readiness.zip"))


@pytest.fixture(scope="module", autouse=True)
def bounded_zip_buffers():
    # CPython's 1GiB ZipExtFile request can exhaust this 10GiB Windows host
    # during concurrent data preparation. Smaller reads preserve every byte,
    # CRC/hash check and numerical gate; the frozen experiment is untouched.
    previous = zipfile.ZipExtFile.MAX_N
    zipfile.ZipExtFile.MAX_N = 8 * 1024**2
    try:
        yield
    finally:
        zipfile.ZipExtFile.MAX_N = previous


@pytest.fixture(scope="module")
def frozen():
    if not ARCHIVE.is_file():
        pytest.skip("Frozen readiness archive is private and is not included in the public repository; "
                    "set MQR_FROZEN_READINESS_ARCHIVE to an authorized copy to run these artifact gates.")
    with numerics.verified_frozen_modules(ARCHIVE) as (modules, binding):
        torch.set_num_threads(2)
        yield modules, binding


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, frozen):
    root = tmp_path_factory.mktemp("chunk-real-prepared")
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith("prepared/"):
                target = root / name.removeprefix("prepared/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return root


def test_cpu_double_oracle_and_original_fp32_ce_all_chunks_masks_and_partial(frozen):
    modules, _ = frozen
    report = numerics.cpu_gate(modules)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 16
    assert report["positions"] == 1025
    assert report["fp64_loss_oracle_separate_from_frozen_fp32_ce"]
    for row in report["cases"]:
        assert row["status"] == "passed"
        if row["all_masked"]:
            assert row["oracle_loss"] == 0
            assert all(value["zero_preserved"] for value in row["oracle_groups"].values())


def test_group_metrics_reject_zero_nonfinite_and_wrong_gradient():
    reference = {"features": torch.tensor([1., 2., -3.]), "concept.bias": torch.zeros(2)}
    assert numerics.gradient_gate(numerics.group_errors(reference, reference))
    candidate = {name: value.clone() for name, value in reference.items()}
    candidate["concept.bias"][0] = 1e-30
    zero = numerics.group_errors(reference, candidate)
    assert not zero["head.concept"]["zero_preserved"]
    assert not numerics.gradient_gate(zero)
    candidate["concept.bias"].zero_()
    candidate["features"].mul_(1.02)
    assert not numerics.gradient_gate(numerics.group_errors(reference, candidate))
    candidate["features"][0] = float("nan")
    errors = numerics.group_errors(reference, candidate)
    assert not errors["features"]["finite"]
    assert not numerics.gradient_gate(errors)
    json.dumps(errors, allow_nan=False)


def test_fp32_error_regression_is_independent_required_gate():
    groups = numerics.group_errors({"features": torch.ones(4)}, {"features": torch.ones(4)})
    baseline = {"loss_delta": 1e-5, "groups": groups}
    candidate = {"loss_delta": 0.001, "groups": groups}
    assert not numerics._fp32_regression(candidate, baseline)
    candidate["loss_delta"] = 1e-5
    assert numerics._fp32_regression(candidate, baseline)
    candidate["groups"] = {}
    assert not numerics._fp32_regression(candidate, baseline)


def test_three_update_trajectory_exercises_existing_optimizer_and_partial_chunks(frozen):
    modules, _ = frozen
    config = modules.config.ExperimentConfig(
        modules.config.ModelConfig(vocab_size=37, layers=1, d_model=12, attention_heads=2,
                                   ffn_dim=24, context_length=67, dropout=0.0),
        modules.config.HeadConfig(kind="mqr", d_e=5, d_c=9, num_queries=4),
        modules.config.TrainConfig(device="cpu", precision="fp32", batch_size=2,
                                   gradient_accumulation=2, max_steps=20, warmup_steps=4,
                                   checkpoint_every=0, eval_every=0, num_threads=2))
    generator = torch.Generator().manual_seed(391)
    batches = []
    for _ in range(3):
        x = torch.randint(0, 37, (4, 67), generator=generator)
        y = torch.randint(0, 37, (4, 67), generator=generator)
        y[0, :3] = -100
        y[-1, -1] = -100
        batches.append((x, y))
    report = numerics.trajectory_gate(modules, config, batches)
    assert report["status"] == "passed"
    assert [row["chunk"] for row in report["gates"]] == list(numerics.CHUNKS)
    digests = [[event["window_sha256"] for event in row["updates"]] for row in report["gates"]]
    assert all(digest == digests[0] for digest in digests)
    for row in report["gates"]:
        assert row["optimizer_steps"] == [3]
        assert len(row["preclip_gradient_gates"]) == 3
        assert row["state_gate_passed"]


def test_byte_verified_dry_run_cannot_pass_missing_gpu_gates(prepared, tmp_path):
    output = tmp_path / "dry"
    report = numerics.run_numerics(prepared, output, "cpu", archive_path=ARCHIVE, dry_run=True)
    assert report["status"] == "ready_for_gpu"
    assert report["all_required_gates_passed"] is False
    assert report["archive_sha256"] == numerics.ARCHIVE_SHA256
    assert report["package_manifest_sha256"] == numerics.PACKAGE_MANIFEST_SHA256
    assert report["helper_sha256"] == numerics._sha(numerics.__file__)
    assert len(report["required_gates"]) == 10
    assert sum(row["status"] == "passed" for row in report["required_gates"]) == 1
    assert len(report["training_probe_bank"]["records"]) == 4
    assert report["training_probe_bank"]["split"] == "train"
    assert json.loads((output / "numerics.json").read_text())["status"] == "ready_for_gpu"
    with pytest.raises(FileExistsError):
        numerics.run_numerics(prepared, output, archive_path=ARCHIVE, dry_run=True)


def test_rejects_non_cuda_production_run(prepared, tmp_path):
    report = numerics.run_numerics(prepared, tmp_path / "non-cuda", "cpu", archive_path=ARCHIVE)
    assert report["status"] == "failed"
    assert report["all_required_gates_passed"] is False
    assert "CUDA device" in report["error"]["message"]


def test_frozen_bytes_checked_before_import(tmp_path):
    corrupt = tmp_path / "frozen.zip"
    corrupt.write_bytes(b"untrusted bytes")
    with pytest.raises(ValueError, match="archive SHA"):
        with numerics.verified_frozen_modules(corrupt):
            pytest.fail("Untrusted source must never import")


def test_wrong_prepared_identity_is_not_dry_run_success(prepared, tmp_path):
    output = tmp_path / "wrong-data"
    report = numerics.run_numerics(tmp_path / "absent", output, "cpu", archive_path=ARCHIVE, dry_run=True)
    assert report["status"] == "failed"
    assert report["all_required_gates_passed"] is False
    assert report["head_cases"] == {}
