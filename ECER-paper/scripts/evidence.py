import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
RUN = "results/qwen3_multimodal/20261104-071259-h100"


def collect(repo=REPO):
    values = {}

    def load(path):
        return json.loads((repo / path).read_text(encoding="utf-8"))

    def put(name, value, path, key, scale=1, digits=None):
        text = str(value) if digits is None else f"{float(value) * scale:.{digits}f}"
        if name in values:
            raise ValueError(f"Duplicate evidence key: {name}")
        values[name] = {"text": text, "source": path, "key": key}

    def get(name, data, key, path, scale=1, digits=2):
        value = data
        for part in key.split("."):
            value = value[part]
        put(name, value, path, key, scale, digits)

    tp, mp, cp, xp = [f"{RUN}/{name}.json" for name in ("test_results", "metrics", "config", "explanation_metrics")]
    bp = "ECER-paper/logs/explanation_bootstrap.json"
    t, m, c, x, b = [load(path) for path in (tp, mp, cp, xp, bp)]
    np, nq = "notebooks/03_Evaluate_Visual_Baseline.ipynb", "notebooks/04_Qwen3_Multimodal.ipynb"
    n, q = load(np), load(nq)
    seeds = sorted({row["seed"] for row in t["runs"]})
    put("seeds", f"{seeds[0]}--{seeds[-1]}", tp, "runs[*].seed")
    put("seed-count", len(seeds), tp, "runs[*].seed:unique-count")
    put("seed-main", c["seed"], cp, "seed")
    put("labels", len(m["thresholds"]), mp, "thresholds:length")
    put("qwen", c["qwen_model_id"].split("/")[-1].removesuffix("-Instruct"), cp, "qwen_model_id")
    put("vit", "ViT-B/" + c["vit_backbone"].split("_")[2], cp, "vit_backbone")
    for key in ("epochs", "batch_size", "fusion_dim", "graph_layers", "fusion_layers", "max_person_cues", "max_context_cues", "max_relations", "max_graph_nodes", "bert_max_length", "max_new_tokens_cues", "max_new_tokens_explanation"):
        put(key.replace("_", "-"), c[key], cp, key)
    get("lr", c, "learning_rate", cp, digits=4)
    get("weight-decay", c, "weight_decay", cp, digits=4)

    def stdout(cell):
        return "".join("".join(o.get("text", [])) for o in cell.get("outputs", []) if o.get("output_type") == "stream" and o.get("name") == "stdout")

    split_output = stdout(q["cells"][7])
    put("train-people", int(re.search(r"train: ([\d,]+) người", split_output)[1].replace(",", "")), nq, "cells[7].outputs:train")
    for split in ("dev", "eval"):
        get(f"{split}-people", m, f"{split}.num_samples", mp, digits=0)
        match = re.search(rf"{split}: ([\d,]+) người / ([\d,]+) ảnh", split_output)
        put(f"{split}-images", int(match[2].replace(",", "")), nq, f"cells[7].outputs:{split}")
        get(f"{split}-cardinality", m, f"{split}.mean_true_labels", mp, digits=3)
    for key in ("test", "test_images"):
        put(key.replace("_", "-"), t["people"][key], tp, f"people.{key}")
    get("test-cardinality", t, "main_checkpoint_test.mean_true_labels", tp, digits=3)
    for metric in ("map", "macro_f1", "micro_f1"):
        get(f"main-eval-{metric}", m, f"eval.{metric}", mp, scale=100)
        get(f"main-test-{metric}", t, f"main_checkpoint_test.{metric}", tp, scale=100)
    for variant, splits in t["variants"].items():
        for split, metrics in splits.items():
            for metric, stats in metrics.items():
                for stat in ("mean", "sd"):
                    get(f"{variant}-{split}-{metric}-{stat}", t, f"variants.{variant}.{split}.{metric}.{stat}", tp, scale=1 if metric == "mean_predicted_labels" else 100)
    for group, data, prefix in (("boot", t["bootstrap"], "bootstrap"), ("resboot", t["resnet_comparison"]["bootstrap"], "resnet_comparison.bootstrap")):
        for name in data:
            for field, suffix in (("difference_a_minus_b", "delta"), ("ci95_lower", "lo"), ("ci95_upper", "hi")):
                get(f"{group}-{name}-{suffix}", t, f"{prefix}.{name}.{field}", tp, scale=100)
    first_boot = next(iter(t["bootstrap"].values()))
    put("resamples", first_boot["n_resamples"], tp, "bootstrap.*.n_resamples")
    confidence_key = next(key for key in first_boot if re.fullmatch(r"ci\d+_lower", key))
    put("confidence", re.search(r"\d+", confidence_key)[0], tp, f"bootstrap.*.{confidence_key}")
    put("affect-words", len(t["affect_words"]), tp, "affect_words:length")
    get("affect-changed", t, "affect_changed_rate", tp, scale=100)
    get("constant-k", t, "constant_top_k_test.k", tp, digits=0)
    for metric in ("map", "macro_f1", "micro_f1"):
        get(f"constant-test-{metric}", t, f"constant_top_k_test.{metric}", tp, scale=100)
    for split in ("eval", "test"):
        d = x[split]
        for field in ("n_generated", "n_usable", "cited_numbers_total"):
            put(f"{split}-{field}", d[field], xp, f"{split}.{field}")
        put(f"{split}-unusable", d["n_generated"] - d["n_usable"], xp, f"{split}.n_generated-n_usable")
        for modality, count in d["stated_modality_counts"].items():
            put(f"{split}-stated-{modality}", count, xp, f"{split}.stated_modality_counts.{modality}")
        for field in ("modality_agreement", "modality_agreement_always_majority", "explanations_naming_unpredicted_emotion", "truncated_rate"):
            get(f"{split}-{field}", x, f"{split}.{field}", xp, scale=100)
        get(f"{split}-number-match", x, f"{split}.cited_numbers_in_evidence", xp, scale=100, digits=4)
        for choice, metrics in d["eraser"].items():
            for metric in metrics:
                get(f"{split}-{choice}-{metric}", x, f"{split}.eraser.{choice}.{metric}", xp, digits=4)
        for measure, result in b[split].items():
            if isinstance(result, dict):
                for field, suffix in (("difference_a_minus_b", "delta"), ("ci95_lower", "lo"), ("ci95_upper", "hi")):
                    get(f"exboot-{split}-{measure}-{suffix}", b, f"{split}.{measure}.{field}", bp, scale=100 if measure.startswith("agreement") else 1, digits=3 if measure.startswith("agreement") else 4)
    for field in ("parse_rate", "schema_rate", "truncated_rate"):
        get(f"cue-{field}", m, f"qwen_cues.{field}", mp, scale=100)
        get(f"test-cue-{field}", t, f"test_cues.{field}", tp, scale=100)
    for mode, run in (("person_only", "20261019-091248"), ("context_only", "20261019-093151"), ("person_context", "20261019-094057")):
        hp = f"results/runs/{mode}/{run}/logs/history.csv"
        with (repo / hp).open(encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        best = max(history, key=lambda row: float(row["val_map"]))
        put(f"resnet-{mode}-val-map", best["val_map"], hp, "max(val_map)", 100, 2)
        for metric in ("map", "macro_f1", "micro_f1"):
            get(f"resnet-{mode}-test-{metric}", t, f"resnet_comparison.models.{mode}.union.{metric}", tp, scale=100)
        if mode == "person_only":
            rp = f"results/runs/{mode}/{run}/config.json"
            rc = load(rp)
            put("resnet", rc["backbone"].replace("resnet", "ResNet-"), rp, "backbone")
            put("resnet-epochs", rc["epochs"], rp, "epochs")
            put("resnet-image", rc["image_size"], rp, "image_size")
            put("crop-padding", rc["crop_padding"], rp, "crop_padding", 100, 0)
    for line in stdout(n["cells"][14]).splitlines():
        match = re.fullmatch(r"(\w+) - (\w+): ([\d.-]+) CI95=\[([\d.-]+), ([\d.-]+)\]", line)
        if match:
            for value, suffix in zip(match.groups()[2:], ("delta", "lo", "hi")):
                put(f"respair-{match[1]}-{match[2]}-{suffix}", value, np, "cells[14].outputs", 100, 2)
    for condition, ap, f1 in re.findall(r"(delta_\w+)\s+mAP=([\d.]+) macro-F1=([\d.]+)", stdout(n["cells"][16])):
        put(f"region-{condition}-map", ap, np, "cells[16].outputs", 100, 2)
        put(f"region-{condition}-f1", f1, np, "cells[16].outputs", 100, 2)
    swap_text = stdout(n["cells"][18])
    swap, _ = json.JSONDecoder().raw_decode(swap_text[swap_text.index("{"):])
    for field in ("num_pairs", "num_images", "swap_accuracy", "tie_rate", "mean_target_minus_source_f1"):
        put(f"swap-{field}", swap[field], np, f"cells[18].outputs.{field}", 100 if field in ("swap_accuracy", "tie_rate") else 1, 2 if field in ("swap_accuracy", "tie_rate") else 5 if field.startswith("mean") else None)
    benchmark_path = f"{RUN}/batch_benchmark.json"
    benchmark = load(benchmark_path)
    for batch, stats in benchmark.items():
        if batch != "projection":
            put(f"batch-{batch}", batch, benchmark_path, batch)
            put(f"batch-{batch}-seconds", stats["seconds_per_person"], benchmark_path, f"{batch}.seconds_per_person", 1, 3)
            put(f"batch-{batch}-same", stats["same_as_single"], benchmark_path, f"{batch}.same_as_single", 100, 2)
    ep = "ECER-paper/logs/data_eda.json"
    eda = load(ep)
    for split, row in eda["splits"].items():
        for field in ("people", "images", "mean_labels", "people_in_multi_person_images_pct", "median_category_annotation_entries"):
            get(f"eda-{split}-{field}", eda, f"splits.{split}.{field}", ep, digits=3 if field == "mean_labels" else 2 if field.endswith("_pct") else 0)
        get(f"eda-{split}-empty-targets", eda, f"splits.{split}.cardinality_counts.0", ep, digits=0)
        for label, count in row["label_support"].items():
            slug = label.lower().replace("/", "-")
            put(f"eda-{split}-{slug}-support", count, ep, f"splits.{split}.label_support.{label}")
            get(f"eda-{split}-{slug}-prevalence", eda, f"splits.{split}.label_prevalence_pct.{label}", ep)
    return values


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
