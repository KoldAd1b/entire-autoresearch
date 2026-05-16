#!/usr/bin/env python3
"""Print a JSON inventory of experiment code and result artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".ipynb",
    ".R",
    ".jl",
    ".js",
    ".ts",
    ".sql",
}


def file_entry(path: Path, base: Path) -> dict[str, object]:
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": stat.st_size,
        "modified_utc": mtime,
    }


def sorted_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def inventory_experiment(experiment: Path, base: Path) -> dict[str, object]:
    result_root = experiment / "results"
    code_files = [
        file_entry(path, base)
        for path in sorted_files(experiment)
        if result_root not in path.parents and path.suffix in CODE_SUFFIXES
    ]
    result_artifacts = [file_entry(path, base) for path in sorted_files(result_root)]
    return {
        "name": experiment.name,
        "path": experiment.relative_to(base).as_posix(),
        "code_files": code_files,
        "result_artifacts": result_artifacts,
    }


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    experiments_root = base / "experiments"
    experiments = [
        inventory_experiment(path, base)
        for path in sorted(experiments_root.iterdir())
        if path.is_dir()
    ] if experiments_root.exists() else []
    print(
        json.dumps(
            {
                "experiments_root": experiments_root.relative_to(base).as_posix(),
                "experiments": experiments,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
