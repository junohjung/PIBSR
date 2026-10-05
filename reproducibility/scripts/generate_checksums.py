#!/usr/bin/env python3
"""Regenerate the curated-result inventories and package checksum file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUBSET_ROOTS = (
    "results/one_d_convection_diffusion",
    "results/two_d_burgers",
    "results/lid_driven_cavity",
    "results/verification",
)
SUBSET_MANIFEST = "PACKAGE_SUBSET_MANIFEST.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"} or any(
        part
        in {
            ".cache",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "work",
        }
        for part in path.parts
    )


def files_below(root: Path, excluded_names: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names and not ignored(path)
    )


def write_subset_manifest(relative_root: str) -> None:
    root = PACKAGE_ROOT / relative_root
    records = ["relative_path\tsha256\tbytes"]
    for path in files_below(root, {SUBSET_MANIFEST}):
        records.append(
            "{}\t{}\t{}".format(
                path.relative_to(root).as_posix(), sha256(path), path.stat().st_size
            )
        )
    (root / SUBSET_MANIFEST).write_text("\n".join(records) + "\n", encoding="utf-8")


def write_package_manifest() -> None:
    manifest = PACKAGE_ROOT / "SHA256SUMS"
    records = []
    for path in files_below(PACKAGE_ROOT, {manifest.name}):
        records.append(
            "{}  {}".format(sha256(path), path.relative_to(PACKAGE_ROOT).as_posix())
        )
    manifest.write_text("\n".join(records) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    parse_args(argv)
    pinn = PACKAGE_ROOT / "pinn"
    manifest = pinn / "MANIFEST.json"
    if manifest.exists():
        names = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({name: sha256(pinn / name) for name in names}, indent=2) + "\n")
    for relative_root in SUBSET_ROOTS:
        write_subset_manifest(relative_root)
    write_package_manifest()


if __name__ == "__main__":
    main()
