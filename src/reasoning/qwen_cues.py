"""Qwen3-VL cue extraction: prompt, target marking, parsing and semantic graph nodes.

The prompt text, schema check and node construction reproduce Notebook 04 run
``20261104-071259-h100`` exactly, so cues imported from that run stay valid.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from PIL import Image, ImageDraw

CUE_KEYS = ("person_cues", "context_cues", "relations", "situation")
NODE_TYPE = {"person": 0, "context": 1, "relation": 2, "situation": 3}
NO_CUE_TEXT = "No reliable visible cues were extracted."

# Words removed from every cue for the affect-free variant of Notebook 04 section 13.
AFFECT_WORDS = frozenset({
    "happy", "sad", "angry", "smiling", "smile", "joy", "fear", "anxious", "excited", "confident",
    "worried", "upset", "calm", "relaxed", "bored", "surprised", "tired", "focused", "engaged",
    "peaceful", "pain", "annoyed",
})


def build_cue_prompt(max_person_cues: int, max_context_cues: int, max_relations: int) -> str:
    """Return the cue-extraction prompt; it never lists emotion labels."""
    return f"""The target person is the one inside the red box.
Extract only visible information that may help a separate emotion-recognition model.
Do not name or infer any emotion. Do not use facts that are not visible.
Return one JSON object with exactly these keys in this order:
- "person_cues": a non-empty list of at most {max_person_cues} short visible face, pose, gesture, clothing or action cues.
- "context_cues": a non-empty list of at most {max_context_cues} short visible scene, object or other-person cues.
- "relations": a list of at most {max_relations} short visible relations involving the target, such as "target facing person_2". Use [] if none is visible.
- "situation": one short sentence describing only what is visibly happening.
Every list element must be a plain string."""


def mark_target(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    """Draw a red box around the target person on a copy of the image."""
    marked = image.copy()
    width = max(3, round(max(marked.size) / 150))
    ImageDraw.Draw(marked).rectangle([float(value) for value in bbox], outline=(255, 0, 0), width=width)
    return marked


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Decode the first JSON object in a model response, or return None."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def cue_schema_ok(payload: Any) -> bool:
    """Check keys, key order, string lists and non-empty person/context cues."""
    if not isinstance(payload, dict) or list(payload) != list(CUE_KEYS):
        return False
    for key in CUE_KEYS[:3]:
        if not isinstance(payload[key], list) or not all(
            isinstance(value, str) and value.strip() for value in payload[key]
        ):
            return False
    return (
        bool(payload["person_cues"])
        and bool(payload["context_cues"])
        and isinstance(payload["situation"], str)
        and bool(payload["situation"].strip())
    )


def cue_record(item: dict[str, Any], raw: str, truncated: bool, seconds: float) -> dict[str, Any]:
    """Build one JSONL record in the layout of ``qwen_cues.jsonl``."""
    payload = parse_json_object(raw)
    return {
        "sample_id": item["sample_id"],
        "relative_image_path": item["relative_image_path"],
        "bbox": item["bbox"].tolist(),
        "parse_ok": payload is not None,
        "schema_ok": cue_schema_ok(payload),
        "truncated": truncated,
        "response": payload,
        "raw": raw,
        "seconds": seconds,
    }


def semantic_nodes(
    record: dict[str, Any],
    max_person_cues: int,
    max_context_cues: int,
    max_relations: int,
    max_nodes: int,
) -> list[tuple[str, str]]:
    """Turn one cue record into typed ``(kind, text)`` graph nodes."""
    payload = record["response"] if record["schema_ok"] else None
    if payload is None:
        return [("situation", NO_CUE_TEXT)]
    nodes = [("person", text) for text in payload["person_cues"][:max_person_cues]]
    nodes += [("context", text) for text in payload["context_cues"][:max_context_cues]]
    nodes += [("relation", text) for text in payload["relations"][:max_relations]]
    nodes.append(("situation", payload["situation"]))
    assert len(nodes) <= max_nodes
    return nodes


def strip_affect(text: str) -> str:
    """Delete affect words; an emptied node becomes ``unknown``."""
    kept = re.sub(
        r"[A-Za-z]+", lambda match: "" if match.group(0).lower() in AFFECT_WORDS else match.group(0), text
    )
    return " ".join(kept.split()) or "unknown"


def read_jsonl(path) -> list[dict[str, Any]]:
    """Read a JSONL file; a missing file reads as empty."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cue_stats(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Parse, schema and truncation rates over cue records."""
    count = max(len(records), 1)
    return {
        "people": len(records),
        "parse_rate": sum(record["parse_ok"] for record in records) / count,
        "schema_rate": sum(record["schema_ok"] for record in records) / count,
        "truncated_rate": sum(record["truncated"] for record in records) / count,
    }


__all__ = [
    "AFFECT_WORDS", "CUE_KEYS", "NODE_TYPE", "NO_CUE_TEXT", "build_cue_prompt", "cue_record",
    "cue_schema_ok", "cue_stats", "mark_target", "parse_json_object", "read_jsonl",
    "semantic_nodes", "strip_affect",
]
