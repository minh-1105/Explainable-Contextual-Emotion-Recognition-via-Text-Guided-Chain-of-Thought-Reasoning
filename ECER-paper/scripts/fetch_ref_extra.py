"""Second metadata pass: find DOIs/venues via Crossref title search for entries that arXiv alone
does not give a venue for. A hit is accepted only if the normalised title is identical and the
year matches. Output: logs/ref_sources_extra.json (including misses, so absence is visible)."""
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "ref_sources_extra.json"

QUERIES = [
    ("lee2019caer", "Context-Aware Emotion Recognition Networks", 2019),
    ("mittal2020emoticon", "EmotiCon: Context-Aware Multimodal Emotion Recognition using Frege's Principle", 2020),
    ("zhang2023emotionclip", "Learning Emotion Representations from Verbal and Nonverbal Communication", 2023),
    ("he2016resnet", "Deep Residual Learning for Image Recognition", 2016),
    ("yang2023captions", "Contextual Emotion Estimation from Image Captions", 2023),
    ("rha2026erv", "Emotion-Coherent Reasoning for Multimodal LLMs via Emotional Rationale Verifier", 2026),
    ("hase2020las", "Leakage-Adjusted Simulatability: Can Models Generate Non-Trivial Explanations of Their Behavior in Natural Language?", 2020),
    ("lin2025eibench", "Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models", 2025),
]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    out = {}
    for key, title, year in QUERIES:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": title, "rows": 5}, timeout=60,
                         headers={"User-Agent": "ecer-paper-ref-check/1.0 (mailto:none@example.com)"})
        hit = None
        for m in r.json()["message"]["items"]:
            t = (m.get("title") or [""])[0]
            y = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            if norm(t) == norm(title) and y == year:
                hit = {
                    "doi": m["DOI"], "title": t, "year": y,
                    "container": (m.get("container-title") or [""])[0],
                    "pages": m.get("page"), "volume": m.get("volume"), "issue": m.get("issue"),
                    "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
                    "type": m.get("type"),
                }
                break
        out[key] = hit or {"miss": True, "query": title}
        time.sleep(0.5)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in out.items():
        print(k, "->", "MISS" if v.get("miss") else f"{v['doi']} | {v['container']} | {v['year']} | {v['pages']}")


if __name__ == "__main__":
    main()
