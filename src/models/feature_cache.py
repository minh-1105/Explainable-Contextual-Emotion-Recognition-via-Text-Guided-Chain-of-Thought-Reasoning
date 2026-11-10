"""Per-split caches of frozen image and BERT node features for Notebook 04.

Layout under ``outputs/multimodal_pipeline/<run_id>/features/``::

    <image_backbone>/spec.json, train.npz, val.npz, test.npz
    bert/spec.json, cues_<split>.npz, affect_free_<split>.npz

Every file stores ``sample_ids`` in dataset order. Caches of run
``20261104-071259-h100`` can be imported instead of recomputed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.run_records import check_or_write_json

IMAGE_KEYS = ("person_features", "context_features", "labels")
NODE_KEYS = ("node_features", "node_mask", "node_types")

# Files of the H100 run that hold each cache; rows are matched by sample_id.
H100_IMAGE_FILES = {"train": "vit_features.npz", "val": "vit_features.npz", "test": "test_features.npz"}
H100_NODE_FILES = {
    ("cues", "train"): "bert_graph_features.npz",
    ("cues", "val"): "bert_graph_features.npz",
    ("cues", "test"): "test_features.npz",
    ("affect_free", "train"): "affect_free_features.npz",
    ("affect_free", "val"): "affect_free_features.npz",
    ("affect_free", "test"): "affect_free_features.npz",
}


def image_cache_path(feature_root: Path, backbone: str, split: str, spec: dict) -> Path:
    directory = Path(feature_root) / backbone
    directory.mkdir(parents=True, exist_ok=True)
    check_or_write_json(directory / "spec.json", spec)
    return directory / f"{split}.npz"


def node_cache_path(feature_root: Path, node_set: str, split: str, spec: dict) -> Path:
    directory = Path(feature_root) / "bert"
    directory.mkdir(parents=True, exist_ok=True)
    check_or_write_json(directory / "spec.json", spec)
    return directory / f"{node_set}_{split}.npz"


def save_features(path: Path, sample_ids: Sequence[str], arrays: dict[str, np.ndarray]) -> None:
    for name, values in arrays.items():
        assert len(values) == len(sample_ids), f"{name}: {len(values)} dòng, cần {len(sample_ids)}"
    temporary = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, sample_ids=np.asarray(sample_ids), **arrays)
    temporary.replace(path)


def load_features(path: Path, sample_ids: Sequence[str], keys: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        assert data["sample_ids"].tolist() == list(sample_ids), f"{Path(path).name}: sample_id lệch"
        return {key: data[key] for key in keys}


def import_rows(source: Path, sample_ids: Sequence[str], keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Take the rows of ``sample_ids`` from an H100 cache file."""
    with np.load(source, allow_pickle=False) as data:
        position = {sample_id: row for row, sample_id in enumerate(data["sample_ids"].tolist())}
        missing = [sample_id for sample_id in sample_ids if sample_id not in position]
        assert not missing, f"{Path(source).name} thiếu {len(missing)} người, ví dụ {missing[:3]}"
        rows = np.asarray([position[sample_id] for sample_id in sample_ids])
        return {key: data[key][rows] for key in keys}


__all__ = [
    "H100_IMAGE_FILES", "H100_NODE_FILES", "IMAGE_KEYS", "NODE_KEYS", "image_cache_path",
    "import_rows", "load_features", "node_cache_path", "save_features",
]
