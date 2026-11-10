"""Cue statistics over a qwen_cues.jsonl file, written to logs/nb04_cues.json.

Usage (cwd = ECER-paper): python scripts/cue_stats.py <path to qwen_cues.jsonl>
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AFFECT = {"happy", "sad", "angry", "smiling", "smile", "joy", "fear", "anxious", "excited", "confident",
          "worried", "upset", "calm", "relaxed", "bored", "surprised", "tired", "focused", "engaged",
          "peaceful", "pain", "annoyed"}

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
ok = [r["response"] for r in records if r["schema_ok"]]


def words(cues):
    return [w for c in cues["person_cues"] + cues["context_cues"] + [cues["situation"]]
            for w in re.findall(r"[a-z]+", c.lower())]


hits = collections.Counter(w for x in ok for w in words(x) if w in AFFECT)
by_split = collections.Counter(r["sample_id"].split(":")[0] for r in records)
stats = {
    "people": len(records),
    "by_split": dict(by_split),
    "parse_rate": sum(r["parse_ok"] for r in records) / len(records),
    "schema_rate": len(ok) / len(records),
    "truncated_rate": sum(bool(r["truncated"]) for r in records) / len(records),
    "person_cues_mean": sum(len(x["person_cues"]) for x in ok) / len(ok),
    "context_cues_mean": sum(len(x["context_cues"]) for x in ok) / len(ok),
    "relations_mean": sum(len(x["relations"]) for x in ok) / len(ok),
    "relations_none_rate": sum(not x["relations"] for x in ok) / len(ok),
    "affect_rate": sum(any(w in AFFECT for w in words(x)) for x in ok) / len(ok),
    "affect_top": hits.most_common(8),
}
(ROOT / "logs" / "nb04_cues.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
