import json
import sys
from pathlib import Path

import numpy as np

MODALITIES = ("person", "context", "graph")
ROOT = Path(__file__).resolve().parent.parent
run = Path(sys.argv[1])


def drop(evidence, modality):
    return float(np.mean([row["probability"] - row[f"probability_without_{modality}"] for row in evidence["prediction"]]))


def per_person(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = [r for r in records if r["parse_ok"] and r["response"].get("most_important_modality") in MODALITIES]
    drops = [{m: drop(r["evidence"], m) for m in MODALITIES} for r in usable]
    actual = [max(d, key=d.get) for d in drops]
    majority = max(MODALITIES, key=actual.count)
    stated = [r["response"]["most_important_modality"] for r in usable]
    images = [":".join(r["evidence"]["sample_id"].split(":")[:2]) for r in usable]
    return {
        "image": np.unique(images, return_inverse=True)[1],
        "agreement_stated": np.array([a == s for a, s in zip(actual, stated)], dtype=float),
        "agreement_majority": np.array([a == majority for a in actual], dtype=float),
        "comprehensiveness_stated": np.array([d[s] for d, s in zip(drops, stated)]),
        "comprehensiveness_majority": np.array([d[majority] for d in drops]),
        "comprehensiveness_random": np.array([np.mean(list(d.values())) for d in drops]),
    }, majority


def bootstrap(values, a, b, draws=2000, seed=42):
    image = values["image"]
    n_images = image.max() + 1
    diff = values[a] - values[b]
    sums = np.bincount(image, weights=diff, minlength=n_images)
    sizes = np.bincount(image, minlength=n_images)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
        samples.append((counts * sums).sum() / (counts * sizes).sum())
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {"difference_a_minus_b": float(diff.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
            "n_people": int(len(diff)), "n_images": int(n_images), "n_resamples": draws, "seed": seed}


out = {}
for split, name in (("eval", "explanations.jsonl"), ("test", "explanations_test.jsonl")):
    values, majority = per_person(run / name)
    out[split] = {
        "majority_modality": majority,
        "agreement_stated_vs_majority": bootstrap(values, "agreement_stated", "agreement_majority"),
        "comprehensiveness_stated_vs_majority": bootstrap(values, "comprehensiveness_stated", "comprehensiveness_majority"),
        "comprehensiveness_stated_vs_random": bootstrap(values, "comprehensiveness_stated", "comprehensiveness_random"),
    }
target = ROOT / "logs" / "explanation_bootstrap.json"
target.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
