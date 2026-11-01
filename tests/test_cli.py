import json

import pytest

from mqr.cli import main


def test_inspect_and_unknown_planning_fields(tmp_path, capsys):
    main(["inspect", "--config", "configs/runs/smoke_mqr.json"])
    result = json.loads(capsys.readouterr().out)
    assert result["parameters"] > result["backbone_parameters"] > 0
    assert result["batch_tokens_per_optimizer_step"] == 256
    with pytest.raises(SystemExit):
        main(["inspect", "--config", "configs/pilot_default.json"])


def test_training_cli_smoke(tmp_path, capsys):
    main(["train", "--config", "configs/runs/smoke_mqr.json",
          "--train-text", "examples/smoke_train.txt", "--validation-text", "examples/smoke_validation.txt",
          "--run-dir", str(tmp_path / "run"), "--stop-after", "2"])
    result = json.loads(capsys.readouterr().out)
    assert result["completed_steps"] == 2
    main(["evaluate", "--checkpoint", result["checkpoint"], "--text", "examples/smoke_validation.txt", "--batches", "1"])
    metrics = json.loads(capsys.readouterr().out)
    assert 0 < metrics["full_nll"] < 10
