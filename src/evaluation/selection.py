"""Dev-only selection for probability fusion and per-person label budgets."""
import numpy as np
from sklearn.metrics import f1_score

from .emotion_metrics import mean_average_precision


def mix_probabilities(full, resnet, weight):
    """Mix aligned probability matrices using the weight of the full model."""
    full, resnet = np.asarray(full), np.asarray(resnet)
    if full.shape != resnet.shape or full.ndim != 2:
        raise ValueError("Probability matrices must have matching two-dimensional shapes")
    if not 0 <= weight <= 1 or not np.isfinite(full).all() or not np.isfinite(resnet).all():
        raise ValueError("Finite scores and a weight in [0, 1] are required")
    return weight * full + (1 - weight) * resnet


def select_fusion_weight(targets, full, resnet, grid):
    """Choose weight by dev mAP; exact ties prefer the smaller full weight."""
    candidates = [{"weight": float(w), "map": mean_average_precision(
        targets, mix_probabilities(full, resnet, float(w)))} for w in sorted(set(grid))]
    if not candidates or not all(np.isfinite(row["map"]) for row in candidates):
        raise ValueError("A nonempty grid and dev labels with positives are required")
    best = max(candidates, key=lambda row: (row["map"], -row["weight"]))
    return {"weight": best["weight"], "dev_candidates": candidates}


def per_sample_top_k(probabilities, k):
    """Predict k labels per person; tied scores prefer lower label indices."""
    values = np.asarray(probabilities)
    if values.ndim != 2 or not np.isfinite(values).all() or not 1 <= k <= values.shape[1]:
        raise ValueError("Finite 2D scores and k in [1, num_classes] are required")
    chosen = np.argsort(-values, axis=1, kind="stable")[:, :k]
    result = np.zeros(values.shape, dtype=np.int32)
    np.put_along_axis(result, chosen, 1, axis=1)
    return result


def select_sample_top_k(targets, probabilities, grid):
    """Choose k by dev macro-F1, breaking exact ties toward smaller k."""
    candidates = [{"k": int(k), "macro_f1": float(f1_score(
        targets, per_sample_top_k(probabilities, int(k)), average="macro", zero_division=0))}
        for k in sorted(set(grid))]
    if not candidates:
        raise ValueError("A nonempty k grid is required")
    best = max(candidates, key=lambda row: (row["macro_f1"], -row["k"]))
    return {"k": best["k"], "dev_candidates": candidates}


__all__ = ["mix_probabilities", "select_fusion_weight", "per_sample_top_k", "select_sample_top_k"]
