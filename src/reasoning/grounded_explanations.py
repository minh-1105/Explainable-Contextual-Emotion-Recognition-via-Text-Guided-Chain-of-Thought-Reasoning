"""Modality-ablation evidence packages, grounded explanations and their metrics.

Reproduces Notebook 04 sections 11, 12, 12c and 14.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

import numpy as np

from src.datasets.emotic_dataset import EMOTION_LABELS
from src.reasoning.qwen_cues import parse_json_object, read_jsonl

MODALITIES = ("person", "context", "graph")
LABEL_WORDS = {label.lower() for label in EMOTION_LABELS}
INTERVENTION_NOTE = "Each modality token was replaced by zero; values report the resulting probability."

# The prompt of Notebook 04 section 12 (the one in prompts.json fails .format).
EXPLANATION_PROMPT = """You are given the output and intervention evidence of an emotion-recognition model.
Explain why the model produced its prediction using only the supplied evidence.
Do not add a cue, causal claim or emotion that is absent from the evidence.
State which modality mattered most and cite probability changes when available.
Return JSON only as {{"most_important_modality": "person|context|graph", "explanation": "..."}}.

Evidence package:
{evidence}
"""


def build_evidence_packages(
    sample_ids: Sequence[str],
    probabilities: np.ndarray,
    ablated: dict[str, np.ndarray],
    thresholds: np.ndarray,
    cue_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One package per person: predicted labels (top-1 when none passes) with ablated probabilities."""
    predictions = probabilities >= thresholds
    packages = []
    for position, sample_id in enumerate(sample_ids):
        predicted = np.flatnonzero(predictions[position])
        if len(predicted) == 0:
            predicted = np.asarray([int(probabilities[position].argmax())])
        record = cue_records[sample_id]
        packages.append({
            "sample_id": sample_id,
            "prediction": [
                {
                    "emotion": EMOTION_LABELS[label],
                    "probability": round(float(probabilities[position, label]), 4),
                    **{
                        f"probability_without_{modality}": round(float(ablated[modality][position, label]), 4)
                        for modality in MODALITIES
                    },
                }
                for label in predicted
            ],
            "visible_cues": record["response"] if record["schema_ok"] else None,
            "intervention": INTERVENTION_NOTE,
        })
    return packages


def explanation_record(evidence: dict[str, Any], raw: str, truncated: bool) -> dict[str, Any]:
    payload = parse_json_object(raw)
    return {
        "sample_id": evidence["sample_id"],
        "evidence": evidence,
        "response": payload,
        "parse_ok": isinstance(payload, dict) and isinstance(payload.get("explanation"), str),
        "truncated": truncated,
        "raw": raw,
    }


def generate_explanations(
    evidence_list: Sequence[dict[str, Any]],
    path,
    ask: Callable[[list, list[str]], tuple[list[str], list[bool]]],
    load_image: Callable[[str], Any],
    batch_size: int = 32,
    progress: Callable | None = None,
) -> int:
    """Append explanations to ``path`` for packages whose evidence has none yet.

    A stored record is kept only when its evidence equals the current package, so
    records copied from an older run are reused exactly when the model agrees.
    Returns the number of newly generated explanations.
    """
    done = {record["sample_id"]: record["evidence"] for record in read_jsonl(path)}
    stale = [evidence["sample_id"] for evidence in evidence_list
             if evidence["sample_id"] in done and done[evidence["sample_id"]] != evidence]
    assert not stale, f"{len(stale)} explanation có evidence khác hiện tại, ví dụ {stale[:3]}; dùng file mới."
    todo = [evidence for evidence in evidence_list if evidence["sample_id"] not in done]
    starts = range(0, len(todo), batch_size)
    with path.open("a", encoding="utf-8") as file:
        for start in (progress(starts) if progress else starts):
            batch = todo[start:start + batch_size]
            texts, truncated = ask(
                [load_image(evidence["sample_id"]) for evidence in batch],
                [EXPLANATION_PROMPT.format(evidence=json.dumps(evidence, ensure_ascii=False, indent=2))
                 for evidence in batch],
            )
            for evidence, raw, cut in zip(batch, texts, truncated):
                file.write(json.dumps(explanation_record(evidence, raw, cut), ensure_ascii=False) + "\n")
            file.flush()
    return len(todo)


def actual_top_modality(evidence: dict[str, Any]) -> str:
    drops = {
        modality: np.mean([row["probability"] - row[f"probability_without_{modality}"] for row in evidence["prediction"]])
        for modality in MODALITIES
    }
    return max(drops, key=drops.get)


def allowed_numbers(evidence: dict[str, Any]) -> np.ndarray:
    values = set()
    for row in evidence["prediction"]:
        values.add(row["probability"])
        for modality in MODALITIES:
            without = row[f"probability_without_{modality}"]
            values.update({without, abs(row["probability"] - without)})
    return np.asarray(sorted(values))


def cited_numbers(text: str) -> list[float]:
    numbers = []
    for raw, percent in re.findall(r"(\d*\.?\d+)\s*(%?)", text):
        if percent or "." in raw:
            numbers.append(float(raw) / (100 if percent else 1))
    return numbers


def explanation_scores(
    records: Sequence[dict[str, Any]],
    keep_only: dict[str, np.ndarray],
    position: dict[str, int],
) -> dict[str, Any]:
    """Agreement, cited-number, unpredicted-emotion and modality-level ERASER metrics."""
    usable = [
        record for record in records
        if record["parse_ok"] and record["response"].get("most_important_modality") in MODALITIES
    ]
    assert usable, "Không có explanation parse được; đọc trường raw trong file explanation."
    actual = [actual_top_modality(record["evidence"]) for record in usable]
    stated = [record["response"]["most_important_modality"] for record in usable]
    majority = max(MODALITIES, key=actual.count)

    matched = total = invented_emotion = 0
    for record in usable:
        text = record["response"]["explanation"]
        allowed = allowed_numbers(record["evidence"])
        for value in cited_numbers(text):
            total += 1
            matched += bool(np.isclose(allowed, value, atol=0.006).any())
        named = set(re.findall(r"[a-z]+", text.lower())) & LABEL_WORDS
        predicted = {row["emotion"].lower() for row in record["evidence"]["prediction"]}
        invented_emotion += bool(named - predicted)

    def comprehensiveness(evidence, modality):
        return np.mean([row["probability"] - row[f"probability_without_{modality}"] for row in evidence["prediction"]])

    def sufficiency(evidence, modality):
        row_index = position[evidence["sample_id"]]
        return np.mean([
            row["probability"] - keep_only[modality][row_index, EMOTION_LABELS.index(row["emotion"])]
            for row in evidence["prediction"]
        ])

    def eraser(choose):
        chosen = [(record["evidence"], choose(record)) for record in usable]
        return {
            "comprehensiveness": float(np.mean([comprehensiveness(evidence, modality) for evidence, modality in chosen])),
            "sufficiency": float(np.mean([sufficiency(evidence, modality) for evidence, modality in chosen])),
        }

    by_modality = {modality: eraser(lambda record, m=modality: m) for modality in MODALITIES}
    return {
        "n_generated": len(records),
        "n_usable": len(usable),
        "truncated_rate": float(np.mean([record["truncated"] for record in records])),
        "stated_modality_counts": {modality: stated.count(modality) for modality in MODALITIES},
        "measured_top_modality_counts": {modality: actual.count(modality) for modality in MODALITIES},
        "modality_agreement": float(np.mean([a == b for a, b in zip(actual, stated)])),
        "modality_agreement_always_majority": actual.count(majority) / len(actual),
        "cited_numbers_total": total,
        "cited_numbers_in_evidence": matched / total if total else None,
        "explanations_naming_unpredicted_emotion": invented_emotion / len(usable),
        "eraser": {
            "stated_modality": eraser(lambda record: record["response"]["most_important_modality"]),
            "always_majority_modality": by_modality[majority],
            "random_modality": {
                metric: float(np.mean([by_modality[m][metric] for m in MODALITIES]))
                for metric in ("comprehensiveness", "sufficiency")
            },
            "measured_top_modality": eraser(lambda record: actual_top_modality(record["evidence"])),
        },
    }


__all__ = [
    "EXPLANATION_PROMPT", "INTERVENTION_NOTE", "MODALITIES", "actual_top_modality",
    "build_evidence_packages", "explanation_scores", "generate_explanations",
]
