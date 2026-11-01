import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mqr.config import ExperimentConfig
from mqr.kaggle import calibration_statistics, inspect_device, run_session, session_configs
from mqr.prepare import prepare_jsonl


def test_t4_plan_preserves_full_schedule_and_matches_backbones(tmp_path):
    report = run_session(stage="calibrate", output_root=tmp_path / "not-created", dry_run=True)
    dense, mqr = (ExperimentConfig.from_dict(report["resolved_configs"][key]) for key in ("dense", "mqr"))
    assert dense.model == mqr.model
    assert dense.head.d_e == dense.model.d_model
    assert dense.training == mqr.training
    assert mqr.training.precision == "fp16"
    assert mqr.training.device == "cuda:0"
    assert mqr.training.max_steps == 183106
    assert report["absolute_stop_after"] == 120
    assert not (tmp_path / "not-created").exists()


def test_cpu_smoke_reloads_and_continues_checkpoint(tmp_path):
    result = run_session(device="cpu", cpu_test=True, output_root=tmp_path / "run")
    for kind in ("dense", "mqr"):
        checks = result["results"][kind]["resume_check"]
        assert checks["finite_weights"] and checks["loaded_on_cpu"]
        assert checks["continued_from_step_2_to"] == 4
        assert result["results"][kind]["training"]["completed_steps"] == 4
        assert result["results"][kind]["calibration"]["measured_optimizer_steps"] == 2
        assert result["results"][kind]["calibration"]["weighted_tokens_per_second"] > 0
        assert (tmp_path / "run" / kind / "last.pt").exists()
    assert json.loads((tmp_path / "run" / "report.json").read_text())["data_identity"]["kind"] == "synthetic_calibration"
    with pytest.raises(ValueError, match="fresh output"):
        run_session(device="cpu", cpu_test=True, output_root=tmp_path / "run")


def test_prepared_bundle_is_verified_and_actual_vocab_applies_to_both_heads(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in (
        {"id": "a", "language": "en", "split": "train", "text": "Independent train document."},
        {"id": "b", "language": "de", "split": "validation", "text": "Separate validation document."},
    )), encoding="utf-8")
    bundle = tmp_path / "prepared"
    prepare_jsonl(source, bundle)
    result = run_session(stage="prepared", data_dir=bundle, output_root=tmp_path / "out", dry_run=True)
    assert result["actual_tokenizer_vocab_size"] == 259
    assert result["data_identity"]["kind"] == "prepared"
    assert all(config["model"]["vocab_size"] == 259 for config in result["resolved_configs"].values())
    with pytest.raises(ValueError, match="outside"):
        run_session(stage="prepared", data_dir=bundle, output_root=bundle / "out", dry_run=True)
    np.save(bundle / "train.npy", np.arange(100, dtype=np.uint32))
    with pytest.raises(ValueError, match="mismatch"):
        run_session(stage="prepared", data_dir=bundle, output_root=tmp_path / "out", dry_run=True)


@pytest.mark.parametrize("kwargs", [
    {"steps": 1001}, {"steps": 3}, {"steps": 4, "warmup_steps": 4},
    {"stage": "prepared"}, {"device": "cpu"},
    {"stage": "calibrate", "device": "cpu", "cpu_test": True},
])
def test_bounded_session_rejects_bad_requests(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_session(output_root=tmp_path / "run", **kwargs)


def test_device_selection_requires_number():
    with pytest.raises(ValueError, match="explicit CUDA"):
        inspect_device("cuda")
    with pytest.raises(ValueError, match="explicit CUDA"):
        inspect_device("cpu")


def test_statistics_weight_elapsed_time_and_exclude_warmup(tmp_path):
    path = tmp_path / "metrics.jsonl"
    events = [
        {"event": "train", "step": 1, "step_seconds": 99, "step_tokens_s": 1, "peak_vram_bytes": 5},
        {"event": "train", "step": 2, "step_seconds": 1, "step_tokens_s": 100, "peak_vram_bytes": 10},
        {"event": "train", "step": 3, "step_seconds": 3, "step_tokens_s": 200, "peak_vram_bytes": 20},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = calibration_statistics(path, 1)
    assert result["weighted_tokens_per_second"] == 175
    assert result["peak_allocated_vram_bytes"] == 20
    assert result["quality_claim"] is False
    assert result["timing_usable_for_comparison"] is True


@pytest.mark.parametrize("overflow_steps,steady_count", [([0], 0), ([1], 1), ([0, 1, 2], 2)])
def test_statistics_distinguish_warmup_overflows_from_unmeasured_steady_attempts(tmp_path, overflow_steps, steady_count):
    path = tmp_path / "overflow-metrics.jsonl"
    events = [
        {"event": "train", "step": 1, "step_seconds": 1, "step_tokens_s": 100, "peak_vram_bytes": None},
        {"event": "train", "step": 2, "step_seconds": 2, "step_tokens_s": 100, "peak_vram_bytes": None},
    ]
    events += [{"event": "amp_overflow", "step": step, "scale": 32768} for step in overflow_steps]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = calibration_statistics(path, warmup_steps=1)
    assert result["amp_overflow_events"] == len(overflow_steps)
    assert result["steady_amp_overflow_events"] == steady_count
    assert result["timing_usable_for_comparison"] is (steady_count == 0)
    assert (result["warning"] is not None) is (steady_count > 0)
    if steady_count:
        assert "cannot justify" in result["warning"]


def test_notebook_defaults_to_smoke_then_bounded_real_data_without_duplicate_synthetic_work():
    notebook = json.loads(Path("notebooks/mqr_kaggle.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    joined = "\n".join(sources)
    assert "RUN_CALIBRATION = False" in joined
    assert "RUN_PREPARED = True" in joined
    assert 'result["training"]["completed_steps"] == 4' in joined
    assert 'result["resume_check"]["continued_from_step_2_to"] == 4' in joined
    assert 'steps=120, warmup_steps=20' in joined
    assert 'require_tokenizers()' in joined
    assert "tokenizers>=0.21,<0.23" in joined
    assert "--no-deps" in joined
    assert "requirements-cpu.lock" not in joined
    assert "/kaggle/working" in joined


def test_notebook_exposes_installed_source_to_an_already_running_kernel(tmp_path):
    notebook = json.loads(Path("notebooks/mqr_kaggle.ipynb").read_text(encoding="utf-8"))
    setup = ast.parse("".join(notebook["cells"][1]["source"]))
    install_index = next(index for index, node in enumerate(setup.body)
                         if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and ast.unparse(node.value.func) == "subprocess.check_call")
    hook_index = next(index for index, node in enumerate(setup.body)
                      if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                      and ast.unparse(node.value.func) == "sys.path.insert")
    assert hook_index > install_index
    # An isolated interpreter has not processed a newly installed editable .pth.
    # Execute the actual notebook hook and ensure it exposes a fixture package.
    package = tmp_path / "src" / "kaggle_kernel_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('VALUE = "current-kernel-source"\n')
    code = ("import sys\nfrom pathlib import Path\n"
            + f"PROJECT = Path({str(tmp_path)!r})\n"
            + ast.unparse(setup.body[hook_index]) + "\n"
            + "import kaggle_kernel_fixture\nprint(kaggle_kernel_fixture.VALUE)\n")
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "current-kernel-source"


@pytest.mark.parametrize("initial,should_install", [
    (None, True), ("0.20.3", True), ("0.23.0", True),
    ("0.21.4", False), ("0.22.2", False),
])
def test_notebook_bootstraps_only_the_tokenizer_binary_when_required(initial, should_install):
    notebook = json.loads(Path("notebooks/mqr_kaggle.ipynb").read_text(encoding="utf-8"))
    setup = ast.parse("".join(notebook["cells"][1]["source"]))
    function = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == "require_tokenizers")
    state = {"installed": initial, "commands": [], "cache_invalidations": 0}

    class MissingPackage(Exception):
        pass

    def version(name):
        assert name == "tokenizers"
        if state["installed"] is None:
            raise MissingPackage()
        return state["installed"]

    def install(command):
        state["commands"].append(command)
        state["installed"] = "0.22.2"

    def invalidate():
        state["cache_invalidations"] += 1

    namespace = {
        "sys": SimpleNamespace(executable="/test/python", modules={}),
        "subprocess": SimpleNamespace(check_call=install),
        "importlib": SimpleNamespace(metadata=SimpleNamespace(version=version, PackageNotFoundError=MissingPackage),
                                     invalidate_caches=invalidate),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-tokenizer-bootstrap", "exec"), namespace)
    expected = "0.22.2" if should_install else initial
    assert namespace["require_tokenizers"]() == expected
    commands = [["/test/python", "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "tokenizers==0.22.2"]]
    assert state["commands"] == (commands if should_install else [])
    assert state["cache_invalidations"] == int(should_install)
    # Calling it again is idempotent and does not touch torch or any dependencies.
    assert namespace["require_tokenizers"]() == expected
    assert len(state["commands"]) == int(should_install)
    # An incompatible already-loaded Rust extension must never be replaced in-place.
    state["installed"] = "0.23.0"
    namespace["sys"].modules["tokenizers"] = SimpleNamespace(__version__="0.23.0")
    with pytest.raises(RuntimeError, match="already imported"):
        namespace["require_tokenizers"]()
    assert len(state["commands"]) == int(should_install)
    namespace["sys"].modules["tokenizers"] = SimpleNamespace(__version__="0.22.2")
    with pytest.raises(RuntimeError, match="different tokenizers version"):
        namespace["require_tokenizers"]()
    # A successful pip exit is insufficient if discovery still exposes a stale wheel.
    namespace["sys"].modules.clear()
    namespace["subprocess"].check_call = lambda command: None
    with pytest.raises(RuntimeError, match="did not expose"):
        namespace["require_tokenizers"]()
