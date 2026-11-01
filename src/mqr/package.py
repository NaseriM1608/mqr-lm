"""Build one reproducible source/data ZIP for manual Kaggle attachment."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .provenance import file_sha256, prepared_identity


SOURCE_GLOBS = ("src/mqr/*.py", "configs/**/*.json", "docs/**/*.md",
                "notebooks/*.ipynb", "tests/*.py", "examples/*.txt",
                "research/methods/**/*.py")
ROOT_FILES = ("pyproject.toml", "README.md", "CITATION.md", "requirements-cpu.lock")
PROVENANCE_SUFFIXES = frozenset({".json", ".jsonl", ".md"})
EXCLUDED_PARTS = frozenset({".venv", ".cache", "__pycache__", ".git", "results",
                            "credentials", "secrets", "node_modules"})
SENSITIVE_NAMES = frozenset({"kaggle.json", "credentials.json", "secrets.json", "auth.json",
                             "token.json", "service-account.json", "service_account.json"})
MANIFEST_NAME = "package_manifest.json"


def _safe_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.startswith(".") or part.lower() in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name.lower() in SENSITIVE_NAMES:
        return False
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlinks are not packaged: {relative.as_posix()}")
        current = current.parent
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Package path escapes its input directory: {relative.as_posix()}")
    return path.is_file()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_package(*, prepared: str | Path, output: str | Path,
                  root: str | Path = ".", provenance: str | Path | None = None) -> dict:
    """Verify data, select known artifact types, and stream a deterministic ZIP.

    The internal manifest hashes every payload file. The external report also
    hashes that manifest, avoiding an impossible manifest self-hash requirement.
    Neither existing archives nor reports are overwritten.
    """
    project = Path(root).resolve()
    bundle = Path(prepared).resolve()
    if Path(output).is_symlink():
        raise ValueError("Package output cannot replace or follow an existing symlink.")
    destination = Path(output).resolve()
    report_path = destination.with_name(destination.name + ".report.json")
    if destination.suffix.lower() != ".zip":
        raise ValueError("Output must end in .zip.")
    if destination.exists() or report_path.exists():
        raise ValueError("Package or sidecar already exists; choose a new output filename.")
    if not (project / "pyproject.toml").is_file():
        raise ValueError("Project root must contain pyproject.toml.")
    identity = prepared_identity(bundle)
    preparation = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    sources: dict[str, Path] = {}
    for name in ROOT_FILES:
        path = project / name
        if _safe_file(path, project):
            sources[name] = path
    for pattern in SOURCE_GLOBS:
        for path in project.glob(pattern):
            if _safe_file(path, project):
                sources[path.relative_to(project).as_posix()] = path
    if "pyproject.toml" not in sources or not any(name.startswith("src/mqr/") for name in sources):
        raise ValueError("Source package must contain pyproject.toml and src/mqr Python sources.")
    for name in sorted(set(preparation["files"]) | {"manifest.json"}):
        path = bundle / name
        if not _safe_file(path, bundle):
            raise ValueError(f"Prepared manifest advertises an excluded artifact: {name}")
        sources["prepared/" + Path(name).as_posix()] = path
    if provenance is not None:
        provenance_root = Path(provenance).resolve()
        if not provenance_root.is_dir():
            raise ValueError("Provenance must be an existing directory.")
        if destination.is_relative_to(provenance_root):
            raise ValueError("Package output must be outside the provenance directory.")
        for path in provenance_root.rglob("*"):
            if path.suffix.lower() in PROVENANCE_SUFFIXES and _safe_file(path, provenance_root):
                sources["provenance/" + path.relative_to(provenance_root).as_posix()] = path
    members = {name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
               for name, path in sorted(sources.items())}
    manifest = {"format_version": 1, "purpose": "manual Kaggle readiness upload; no external publication",
                "prepared_identity": identity, "files": members,
                "self_hash_policy": "The external .zip.report.json hashes package_manifest.json and every other ZIP member.",
                "installation_policy": "Keep preinstalled CUDA PyTorch; never install requirements-cpu.lock on Kaggle."}
    manifest_bytes = _json_bytes(manifest)
    complete_members = {**members, MANIFEST_NAME: {"sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                                 "bytes": len(manifest_bytes)}}
    # Refuse outputs inside any input directory whose content is being traversed.
    # Source globs exclude artifacts/, so normal project artifacts/ outputs are fine.
    if destination == bundle or destination.is_relative_to(bundle):
        raise ValueError("Package output must be outside the prepared bundle.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_archive = False
    created_report = False
    try:
        with destination.open("xb") as archive_handle:
            created_archive = True
            with zipfile.ZipFile(archive_handle, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for name in sorted(complete_members):
                    info = _zip_info(name)
                    if name == MANIFEST_NAME:
                        archive.writestr(info, manifest_bytes)
                        continue
                    digest = hashlib.sha256()
                    count = 0
                    with sources[name].open("rb") as source_handle, archive.open(info, "w", force_zip64=True) as target:
                        for chunk in iter(lambda: source_handle.read(1_048_576), b""):
                            target.write(chunk)
                            digest.update(chunk)
                            count += len(chunk)
                    if digest.hexdigest() != members[name]["sha256"] or count != members[name]["bytes"]:
                        raise ValueError(f"Input changed during packaging: {name}")
        report = {"format_version": 1, "archive": destination.name,
                  "archive_sha256": file_sha256(destination), "archive_bytes": destination.stat().st_size,
                  "zip_members": complete_members, "prepared_identity": identity,
                  "sidecar": report_path.name}
        with report_path.open("xb") as report_handle:
            created_report = True
            report_handle.write(_json_bytes(report))
    except BaseException:
        if created_report:
            report_path.unlink(missing_ok=True)
        if created_archive:
            destination.unlink(missing_ok=True)
        raise
    return {**report, "output": str(destination), "report_path": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", default="artifacts/mqr-kaggle-readiness.zip")
    parser.add_argument("--root", default=".")
    parser.add_argument("--provenance")
    args = parser.parse_args(argv)
    try:
        report = build_package(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Kaggle package: {exc}\n")
    print(json.dumps({key: report[key] for key in ("output", "report_path", "archive_bytes", "archive_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
