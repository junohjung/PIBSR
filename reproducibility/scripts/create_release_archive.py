#!/usr/bin/env python3
"""Create and inspect the deterministic public reproducibility ZIP."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = "PIBSR_reproducibility"
ROOT_FILES = (".gitattributes", ".gitignore", "CITATION.cff", "README.md")
ROOT_DIRECTORIES = (
    "mfa/mfa",
    "mfa_2Dburgers/mfa",
    "mfa_NS/mfa",
    "third_party",
    "reproducibility",
)
EXCLUDED_DIRECTORY_NAMES = {
    ".build",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "work",
}
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FORBIDDEN_CAVITY_PREFIXES = (
    "reproducibility/calculations/lid_driven_cavity/archived_source/",
    "reproducibility/calculations/lid_driven_cavity/generation/",
    "reproducibility/calculations/lid_driven_cavity/protocol/",
)
FORBIDDEN_CAVITY_FILES = {
    "reproducibility/data/lid_driven_cavity/generation_manifest.json",
}
class ArchiveError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(relative_path: Path) -> bool:
    directory_parts = relative_path.parts[:-1]
    return (
        any(
            part in EXCLUDED_DIRECTORY_NAMES
            or part.startswith("build")
            or part.startswith("install")
            for part in directory_parts
        )
        or relative_path.name == ".DS_Store"
        or relative_path.suffix in {".pyc", ".pyo"}
    )


def release_files() -> list[Path]:
    selected = []
    for relative_name in ROOT_FILES:
        path = REPOSITORY_ROOT / relative_name
        if not path.is_file():
            raise ArchiveError("missing required root file: {}".format(relative_name))
        selected.append(path)

    for relative_name in ROOT_DIRECTORIES:
        directory = REPOSITORY_ROOT / relative_name
        if not directory.is_dir():
            raise ArchiveError("missing required directory: {}".format(relative_name))
        for path in directory.rglob("*"):
            relative_path = path.relative_to(REPOSITORY_ROOT)
            if excluded(relative_path):
                continue
            if path.is_symlink():
                raise ArchiveError("symbolic link is not allowed: {}".format(relative_path))
            if path.is_file():
                selected.append(path)

    relative_names = [path.relative_to(REPOSITORY_ROOT).as_posix() for path in selected]
    if len(relative_names) != len(set(relative_names)):
        raise ArchiveError("release selection contains duplicate paths")
    return sorted(selected, key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def validate_data_only_cavity_boundary(files: list[Path]) -> None:
    for path in files:
        relative_name = path.relative_to(REPOSITORY_ROOT).as_posix()
        if (
            relative_name in FORBIDDEN_CAVITY_FILES
            or any(relative_name.startswith(prefix) for prefix in FORBIDDEN_CAVITY_PREFIXES)
            or path.name == "generate_ramped_cavity.py"
            or "firedrake" in path.name.lower()
        ):
            raise ArchiveError(
                "Firedrake generation material is outside the release boundary: "
                + relative_name
            )

        if path.suffix == ".py":
            for line in path.read_text(encoding="utf-8").splitlines():
                statement = line.strip().lower()
                if statement.startswith("import firedrake") or statement.startswith(
                    "from firedrake"
                ):
                    raise ArchiveError(
                        "Firedrake import is outside the release boundary: "
                        + relative_name
                    )

        environment_file = path.suffix.lower() in {
            ".yaml",
            ".yml",
            ".toml",
            ".lock",
        } or (
            path.suffix.lower() in {"", ".in", ".txt"}
            and path.name.lower().startswith(("environment", "requirements"))
        )
        if environment_file and "firedrake" in path.read_text(
            encoding="utf-8", errors="ignore"
        ).lower():
            raise ArchiveError(
                "Firedrake environment material is outside the release boundary: "
                + relative_name
            )


def zip_info(path: Path, archive_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIME)
    info.create_system = 3
    mode = stat.S_IMODE(path.stat().st_mode)
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_archive(destination: Path, files: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for path in files:
                relative_name = path.relative_to(REPOSITORY_ROOT).as_posix()
                archive_name = "{}/{}".format(ARCHIVE_ROOT, relative_name)
                with path.open("rb") as source, archive.open(
                    zip_info(path, archive_name), mode="w", force_zip64=True
                ) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def inspect_archive(path: Path) -> tuple[list[zipfile.ZipInfo], int]:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ArchiveError("CRC verification failed: {}".format(corrupt))
        records = archive.infolist()

    names = [record.filename for record in records]
    if len(names) != len(set(names)):
        raise ArchiveError("archive contains duplicate paths")
    expected_prefix = ARCHIVE_ROOT + "/"
    for name in names:
        normalized = PurePosixPath(name)
        if (
            not name.startswith(expected_prefix)
            or normalized.is_absolute()
            or ".." in normalized.parts
        ):
            raise ArchiveError("archive contains an unsafe path: {}".format(name))
    return records, sum(record.file_size for record in records)


def write_reports(archive_path: Path, records: list[zipfile.ZipInfo], total_size: int) -> None:
    archive_digest = sha256(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    contents = archive_path.with_name(archive_path.stem + ".contents.txt")
    sidecar.write_text(
        "{}  {}\n".format(archive_digest, archive_path.name), encoding="utf-8"
    )
    lines = [
        "archive\t{}".format(archive_path.name),
        "sha256\t{}".format(archive_digest),
        "compressed_bytes\t{}".format(archive_path.stat().st_size),
        "uncompressed_bytes\t{}".format(total_size),
        "file_count\t{}".format(len(records)),
        "",
        "bytes\tpath",
    ]
    lines.extend("{}\t{}".format(record.file_size, record.filename) for record in records)
    contents.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    default_output = REPOSITORY_ROOT.parent / "PIBSR_Zenodo_upload"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--name",
        default="PIBSR_reproducibility_v1.0.0.zip",
        help="output ZIP filename",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if Path(args.name).name != args.name or not args.name.endswith(".zip"):
        raise ArchiveError("--name must be a plain filename ending in .zip")
    output_directory = args.output_dir.resolve()
    try:
        output_directory.relative_to(REPOSITORY_ROOT)
    except ValueError:
        pass
    else:
        raise ArchiveError("--output-dir must be outside the repository")

    files = release_files()
    validate_data_only_cavity_boundary(files)
    archive_path = output_directory / args.name
    build_archive(archive_path, files)
    records, total_size = inspect_archive(archive_path)
    write_reports(archive_path, records, total_size)
    print(
        "Created {} ({} files, {} compressed bytes, {} uncompressed bytes)".format(
            archive_path, len(records), archive_path.stat().st_size, total_size
        )
    )
    print("SHA256: {}".format(sha256(archive_path)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as error:
        print("Archive creation failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
