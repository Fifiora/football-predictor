"""Versioned model registry on disk with a champion pointer."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel

REQUIRED_METADATA = ("data_snapshot", "code_version", "params", "metrics")


def save_model(
    registry_dir: Path, version: str, model: MetaModel, calibrator: Calibrator, metadata: dict
) -> Path:
    """Store one immutable model version; returns its directory."""
    missing = [k for k in REQUIRED_METADATA if k not in metadata]
    if missing:
        raise ValueError(f"metadata is missing required keys: {missing}")
    target = Path(registry_dir) / version
    if target.exists():
        raise FileExistsError(f"model version {version!r} already exists")
    target.mkdir(parents=True)
    joblib.dump(model, target / "model.joblib")
    joblib.dump(calibrator, target / "calibrator.joblib")
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return target


def set_champion(registry_dir: Path, version: str) -> None:
    registry_dir = Path(registry_dir)
    if not (registry_dir / version).is_dir():
        raise ValueError(f"unknown model version {version!r}")
    tmp = registry_dir / "champion.txt.tmp"
    tmp.write_text(version)
    os.replace(tmp, registry_dir / "champion.txt")


def load_champion(registry_dir: Path) -> tuple[str, MetaModel, Calibrator]:
    registry_dir = Path(registry_dir)
    pointer = registry_dir / "champion.txt"
    if not pointer.exists():
        raise FileNotFoundError("no champion has been set in this registry")
    version = pointer.read_text().strip()
    return (
        version,
        joblib.load(registry_dir / version / "model.joblib"),
        joblib.load(registry_dir / version / "calibrator.joblib"),
    )
