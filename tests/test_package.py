import hashlib
import json
import zipfile

import pytest

from mqr.package import MANIFEST_NAME, build_package, main
from mqr.prepare import prepare_jsonl
from mqr.provenance import file_sha256


@pytest.fixture
def package_inputs(tmp_path):
    root = tmp_path / "project"
    (root / "src" / "mqr").mkdir(parents=True)
    (root / "src" / "mqr" / "__init__.py").write_text('"""fixture"""\n')
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    (root / "README.md").write_text("Fixture package.\n")
    (root / "configs" / "runs").mkdir(parents=True)
    (root / "configs" / "runs" / "small.json").write_text('{"seed":0}\n')
    source = tmp_path / "selected.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in (
        {"id": "tr", "language": "en", "split": "train", "text": "One training document."},
        {"id": "va", "language": "fa", "split": "validation", "text": "A distinct held-out document."},
    )), encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare_jsonl(source, prepared)
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "source_selection.json").write_text('{"source":"authored-test","revision":"v1"}\n')
    (provenance / "selected_documents.jsonl").write_bytes(source.read_bytes())
    (provenance / "NOTES.md").write_text("Fixture provenance.\n")
    return root, prepared, provenance


def test_archive_and_sidecar_verify_every_member_and_exclude_unlisted_files(tmp_path, package_inputs):
    root, prepared, provenance = package_inputs
    for relative in (".venv/cached.py", ".cache/raw.json", "results/old.json", "kaggle.json",
                     ".env", "unlisted.txt", "src/mqr/__pycache__/cached.pyc"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DO NOT PACKAGE")
    for relative in ("kaggle.json", "credentials.json", ".env", "download.parquet", "raw.bin", ".cache/private.json"):
        path = provenance / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DO NOT PACKAGE")
    (prepared / "not-in-manifest.json").write_text("DO NOT PACKAGE")
    output = tmp_path / "bundle.zip"
    report = build_package(root=root, prepared=prepared, provenance=provenance, output=output)
    saved_report = json.loads((tmp_path / "bundle.zip.report.json").read_text())
    assert saved_report["archive_sha256"] == file_sha256(output) == report["archive_sha256"]
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(report["zip_members"])
        internal = json.loads(archive.read(MANIFEST_NAME))
        assert set(internal["files"]) == set(archive.namelist()) - {MANIFEST_NAME}
        for name in archive.namelist():
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == saved_report["zip_members"][name]["sha256"]
            assert len(content) == saved_report["zip_members"][name]["bytes"]
            assert b"DO NOT PACKAGE" not in content
            assert archive.getinfo(name).date_time == (1980, 1, 1, 0, 0, 0)
        assert "prepared/tokenizer.json" in archive.namelist()
        assert "provenance/selected_documents.jsonl" in archive.namelist()
        assert "configs/runs/small.json" in archive.namelist()


def test_identical_inputs_make_identical_archives_and_refuse_overwrite(tmp_path, package_inputs):
    root, prepared, provenance = package_inputs
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    first = build_package(root=root, prepared=prepared, provenance=provenance, output=one)
    second = build_package(root=root, prepared=prepared, provenance=provenance, output=two)
    assert first["archive_sha256"] == second["archive_sha256"]
    assert one.read_bytes() == two.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        build_package(root=root, prepared=prepared, provenance=provenance, output=one)


def test_tampered_prepared_bundle_is_rejected_without_output(tmp_path, package_inputs):
    root, prepared, provenance = package_inputs
    with (prepared / "train.npy").open("ab") as handle:
        handle.write(b"tampered")
    output = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="mismatch"):
        build_package(root=root, prepared=prepared, provenance=provenance, output=output)
    assert not output.exists()
    assert not output.with_name(output.name + ".report.json").exists()


def test_sidecar_collision_prevents_archive_creation(tmp_path, package_inputs):
    root, prepared, _ = package_inputs
    output = tmp_path / "collision.zip"
    (tmp_path / "collision.zip.report.json").write_text("keep this")
    with pytest.raises(ValueError, match="already exists"):
        build_package(root=root, prepared=prepared, output=output)
    assert not output.exists()
    assert (tmp_path / "collision.zip.report.json").read_text() == "keep this"


def test_cli_builds_package_and_prints_digest(tmp_path, package_inputs, capsys):
    root, prepared, provenance = package_inputs
    output = tmp_path / "cli.zip"
    assert main(["--root", str(root), "--prepared", str(prepared), "--provenance", str(provenance),
                 "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["archive_sha256"] == file_sha256(output)
