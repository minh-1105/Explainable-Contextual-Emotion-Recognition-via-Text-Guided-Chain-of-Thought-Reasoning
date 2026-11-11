import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from src.datasets.emotic_dataset import EMOTION_LABELS, EmoticDataset


def summarize(dataset, indices):
    samples = [dataset.samples[index] for index in indices]
    targets = np.stack([dataset._category_target(sample).numpy() for sample in samples]).astype(int)
    images = Counter(sample.relative_image_path for sample in samples)
    cardinality = targets.sum(axis=1)
    support = targets.sum(axis=0)
    return {
        "people": len(samples),
        "images": len(images),
        "mean_labels": float(cardinality.mean()),
        "cardinality_counts": {str(k): int((cardinality == k).sum()) for k in range(len(EMOTION_LABELS) + 1)},
        "people_in_multi_person_images_pct": 100 * sum(count for count in images.values() if count > 1) / len(samples),
        "median_category_annotation_entries": float(np.median([len(sample.category_annotations) for sample in samples])),
        "category_annotation_entry_counts": dict(sorted(Counter(str(len(sample.category_annotations)) for sample in samples).items())),
        "label_support": {label: int(count) for label, count in zip(EMOTION_LABELS, support)},
        "label_prevalence_pct": {label: 100 * int(count) / len(samples) for label, count in zip(EMOTION_LABELS, support)},
    }


def build():
    run = REPO / "results/qwen3_multimodal/20261104-071259-h100"
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["category_policy"] != "union" or config["n_train_samples"] is not None or config["n_validation_samples"] is not None:
        raise ValueError("EDA requires the frozen full-data union protocol")
    datasets = {
        split: EmoticDataset(
            REPO / "data/Annotations/Annotations.mat",
            [REPO / "data/pami_emotic/emotic", REPO / "data/cvpr_emotic/cvpr_emotic"],
            split=split, category_policy=config["category_policy"], strict_images=True,
        )
        for split in ("train", "val", "test")
    }
    image_ids = sorted({sample.relative_image_path for sample in datasets["val"].samples})
    random.Random(config["seed"]).shuffle(image_ids)
    dev_end = round(len(image_ids) * config["validation_dev_fraction"])
    dev_images = set(image_ids[:dev_end])
    eval_images = set(image_ids[dev_end:])
    indices = {
        name: [i for i, sample in enumerate(datasets["val"].samples) if sample.relative_image_path in images]
        for name, images in (("dev", dev_images), ("eval", eval_images))
    }
    stats = {"train": summarize(datasets["train"], range(len(datasets["train"])))}
    stats.update({name: summarize(datasets["val"], selected) for name, selected in indices.items()})
    stats["test"] = summarize(datasets["test"], range(len(datasets["test"])))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    test = json.loads((run / "test_results.json").read_text(encoding="utf-8"))
    for name, recorded in (("dev", metrics["dev"]), ("eval", metrics["eval"]), ("test", test["main_checkpoint_test"])):
        if stats[name]["people"] != recorded["num_samples"] or not np.isclose(stats[name]["mean_labels"], recorded["mean_true_labels"], rtol=0, atol=1e-6):
            raise ValueError(f"Local annotation targets disagree with frozen {name} results")
    notebook = json.loads((REPO / "notebooks/04_Qwen3_Multimodal.ipynb").read_text(encoding="utf-8"))
    output = "".join("".join(o.get("text", [])) for o in notebook["cells"][7]["outputs"] if o.get("name") == "stdout")
    for name in ("dev", "eval"):
        match = re.search(rf"{name}: ([\d,]+) người / ([\d,]+) ảnh", output)
        if match is None or stats[name]["images"] != int(match[2].replace(",", "")):
            raise ValueError(f"Local {name} image count disagrees with saved notebook output")
    if stats["test"]["images"] != test["people"]["test_images"]:
        raise ValueError("Local test image count disagrees with frozen results")
    payload = {
        "annotation": "data/Annotations/Annotations.mat",
        "target_builder": "src/datasets/emotic_dataset.py:EmoticDataset._category_target",
        "category_policy": config["category_policy"],
        "validation_seed": config["seed"],
        "validation_dev_fraction": config["validation_dev_fraction"],
        "labels": list(EMOTION_LABELS),
        "splits": stats,
    }
    output = ROOT / "logs/data_eda.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, row in stats.items():
        low = min(row["label_support"], key=row["label_support"].get)
        high = max(row["label_support"], key=row["label_support"].get)
        print(name, row["images"], row["people"], f"labels={row['mean_labels']:.3f}", f"multi-person={row['people_in_multi_person_images_pct']:.2f}%", row["category_annotation_entry_counts"], low, row["label_support"][low], high, row["label_support"][high])
    print("Wrote logs/data_eda.json; local counts/cardinality agree with frozen dev/eval/test results")


if __name__ == "__main__":
    build()
