"""Metrics and validation-only protocol selection for multi-label emotions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def _as_binary_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return matrix.astype(np.int32, copy=False)


def _threshold_vector(thresholds: float | Sequence[float], width: int) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(width, float(values), dtype=np.float64)
    if values.shape != (width,):
        raise ValueError(f"thresholds must have shape ({width},), got {values.shape}")
    return values


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    predictions: np.ndarray | None = None,
    thresholds: float | Sequence[float] = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute thresholded metrics and AP when probabilities are available."""
    targets_array = _as_binary_matrix(targets, "targets")
    if probabilities is None and predictions is None:
        raise ValueError("probabilities or predictions is required")

    probabilities_array = None
    if probabilities is not None:
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        if probabilities_array.shape != targets_array.shape:
            raise ValueError("probabilities and targets must have the same shape")

    if predictions is None:
        threshold_array = _threshold_vector(thresholds, targets_array.shape[1])
        predictions_array = (probabilities_array >= threshold_array).astype(np.int32)
    else:
        predictions_array = _as_binary_matrix(predictions, "predictions")
        if predictions_array.shape != targets_array.shape:
            raise ValueError("predictions and targets must have the same shape")

    names = list(class_names or [str(index) for index in range(targets_array.shape[1])])
    if len(names) != targets_array.shape[1]:
        raise ValueError("class_names length does not match target width")

    per_class: list[dict[str, object]] = []
    ap_values: list[float] = []
    for index, name in enumerate(names):
        target = targets_array[:, index]
        prediction = predictions_array[:, index]
        positives = int(target.sum())
        ap = float("nan")
        if probabilities_array is not None and positives:
            ap = float(average_precision_score(target, probabilities_array[:, index]))
        ap_values.append(ap)
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": positives,
                "ap": ap,
                "precision": float(precision_score(target, prediction, zero_division=0)),
                "recall": float(recall_score(target, prediction, zero_division=0)),
                "f1": float(f1_score(target, prediction, zero_division=0)),
                "tp": int(np.logical_and(target == 1, prediction == 1).sum()),
                "fp": int(np.logical_and(target == 0, prediction == 1).sum()),
                "fn": int(np.logical_and(target == 1, prediction == 0).sum()),
                "tn": int(np.logical_and(target == 0, prediction == 0).sum()),
            }
        )

    return {
        "num_samples": int(targets_array.shape[0]),
        "map": float(np.nanmean(ap_values)) if np.isfinite(ap_values).any() else float("nan"),
        "macro_f1": float(f1_score(targets_array, predictions_array, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets_array, predictions_array, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(targets_array, predictions_array, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(targets_array, predictions_array, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(targets_array, predictions_array, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(targets_array, predictions_array, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(targets_array, predictions_array, average="micro", zero_division=0)),
        "mean_true_labels": float(targets_array.sum(axis=1).mean()),
        "mean_predicted_labels": float(predictions_array.sum(axis=1).mean()),
        "per_class": per_class,
    }


def select_per_class_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    grid: Sequence[float] | None = None,
) -> np.ndarray:
    """Select each label threshold on validation F1, preferring values near 0.5."""
    targets_array = _as_binary_matrix(targets, "targets")
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.shape != targets_array.shape:
        raise ValueError("probabilities and targets must have the same shape")
    candidates = np.asarray(grid if grid is not None else np.arange(0.05, 1.0, 0.05))
    thresholds = np.full(targets_array.shape[1], 0.5, dtype=np.float64)
    for index in range(targets_array.shape[1]):
        target = targets_array[:, index]
        if target.sum() == 0:
            continue
        scores = np.asarray(
            [
                f1_score(target, probabilities_array[:, index] >= value, zero_division=0)
                for value in candidates
            ]
        )
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        thresholds[index] = candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]]
    return thresholds


def constant_top_k_predictions(
    num_samples: int,
    label_order: Sequence[int],
    k: int,
    num_classes: int,
) -> np.ndarray:
    """Predict the same k validation-most-prevalent labels for every sample."""
    if not 1 <= k <= num_classes:
        raise ValueError("k must be between 1 and num_classes")
    predictions = np.zeros((num_samples, num_classes), dtype=np.int32)
    predictions[:, np.asarray(label_order, dtype=np.int64)[:k]] = 1
    return predictions


def select_constant_top_k(targets: np.ndarray) -> dict[str, object]:
    """Choose k on validation macro F1, breaking ties with micro F1 then smaller k."""
    targets_array = _as_binary_matrix(targets, "targets")
    prevalence = targets_array.sum(axis=0)
    label_order = np.argsort(-prevalence, kind="stable")
    candidates: list[dict[str, float | int]] = []
    for k in range(1, targets_array.shape[1] + 1):
        predictions = constant_top_k_predictions(
            len(targets_array), label_order, k, targets_array.shape[1]
        )
        candidates.append(
            {
                "k": k,
                "macro_f1": float(f1_score(targets_array, predictions, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(targets_array, predictions, average="micro", zero_division=0)),
            }
        )
    best = max(candidates, key=lambda row: (row["macro_f1"], row["micro_f1"], -row["k"]))
    return {
        "k": int(best["k"]),
        "label_order": label_order.tolist(),
        "validation_candidates": candidates,
    }


def _average_precision(target: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with tied scores grouped, matching scikit-learn."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_target = target[order]
    sorted_scores = scores[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_target)[last_of_tie]
    precision = true_positives / (last_of_tie + 1)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def mean_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    """Mean AP over labels that have at least one positive sample."""
    targets_array = _as_binary_matrix(targets, "targets")
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.shape != targets_array.shape:
        raise ValueError("scores and targets must have the same shape")
    values = [
        _average_precision(targets_array[:, index], scores_array[:, index])
        for index in range(targets_array.shape[1])
        if targets_array[:, index].any()
    ]
    return float(np.mean(values)) if values else float("nan")


def threshold_boundary_report(
    targets: np.ndarray,
    thresholds: Sequence[float],
    grid: Sequence[float],
) -> dict[str, object]:
    """Count labels with positives whose selected threshold sits on a grid edge."""
    targets_array = _as_binary_matrix(targets, "targets")
    threshold_array = _threshold_vector(thresholds, targets_array.shape[1])
    candidates = np.asarray(grid, dtype=np.float64)
    has_positive = targets_array.any(axis=0)
    at_min = has_positive & np.isclose(threshold_array, candidates.min())
    at_max = has_positive & np.isclose(threshold_array, candidates.max())
    return {
        "grid_min": float(candidates.min()),
        "grid_max": float(candidates.max()),
        "labels_with_positives": int(has_positive.sum()),
        "at_min": int(at_min.sum()),
        "at_max": int(at_max.sum()),
        "at_min_indices": np.flatnonzero(at_min).tolist(),
        "at_max_indices": np.flatnonzero(at_max).tolist(),
    }


def paired_image_bootstrap(
    targets: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    image_ids: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    metric: str = "macro_f1",
) -> dict[str, float | int | str]:
    """Bootstrap a paired metric difference by resampling whole images.

    ``metric="macro_f1"`` compares binary predictions; ``metric="map"`` compares
    scores and averages AP over labels with positives in each resample.
    """
    targets_array = _as_binary_matrix(targets, "targets")
    if metric == "macro_f1":
        a_array = _as_binary_matrix(predictions_a, "predictions_a")
        b_array = _as_binary_matrix(predictions_b, "predictions_b")

        def score(target: np.ndarray, values: np.ndarray) -> float:
            return f1_score(target, values, average="macro", zero_division=0)

    elif metric == "map":
        a_array = np.asarray(predictions_a, dtype=np.float64)
        b_array = np.asarray(predictions_b, dtype=np.float64)
        if not (np.isfinite(a_array).all() and np.isfinite(b_array).all()):
            raise ValueError("scores must be finite")
        score = mean_average_precision
    else:
        raise ValueError("metric must be 'macro_f1' or 'map'")
    if not (targets_array.shape == a_array.shape == b_array.shape):
        raise ValueError("targets and prediction arrays must have the same shape")
    image_ids_array = np.asarray(image_ids)
    if image_ids_array.shape != (len(targets_array),):
        raise ValueError("image_ids must contain one value per sample")

    unique_ids = np.unique(image_ids_array)
    grouped = [np.flatnonzero(image_ids_array == image_id) for image_id in unique_ids]

    def difference(indices: np.ndarray) -> float:
        target = targets_array[indices]
        return float(score(target, a_array[indices]) - score(target, b_array[indices]))

    observed = difference(np.arange(len(targets_array)))
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        sampled_groups = rng.integers(0, len(grouped), size=len(grouped))
        indices = np.concatenate([grouped[index] for index in sampled_groups])
        differences[draw] = difference(indices)

    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "metric": metric,
        "difference_a_minus_b": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n_images": int(len(unique_ids)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }
