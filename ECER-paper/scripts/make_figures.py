import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
RUN = "results/qwen3_multimodal/20261104-071259-h100"
NAMES = {"full": "Full", "vit_only": "ViT-only", "no_context": "No context\ntoken", "no_person": "No person\ntoken", "full_no_affect": "Word\ndeletion"}


def build(figure_dir=ROOT / "figures", repo=REPO):
    paths = [f"{RUN}/test_results.json", f"{RUN}/explanation_metrics.json"]
    test, explanation = [json.loads((repo / path).read_text(encoding="utf-8")) for path in paths]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 240})
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, split in zip(axes, ("eval", "test")):
        mean = [test["variants"][name][split]["map"]["mean"] * 100 for name in NAMES]
        sd = [test["variants"][name][split]["map"]["sd"] * 100 for name in NAMES]
        ax.bar(range(len(NAMES)), mean, yerr=sd, capsize=3, color=["#235789", "#9ca9b4", "#629677", "#b88b4a", "#8468a3"])
        ax.set_xticks(range(len(NAMES)), NAMES.values(), fontsize=9)
        ax.set_title("Held-out validation" if split == "eval" else "Test")
        ax.set_ylabel("mAP (%)")
        ax.set_ylim(0, 50)
        ax.yaxis.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(figure_dir / "prediction.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    positions = np.arange(2)
    for index, (key, label, color) in enumerate((("modality_agreement", "Stated modality", "#235789"), ("modality_agreement_always_majority", "Most frequent modality", "#b88b4a"))):
        axes[0].bar(positions + (index - 0.5) * 0.32, [100 * explanation[s][key] for s in ("eval", "test")], width=0.32, label=label, color=color)
    axes[0].set_ylim(0, 60)
    axes[0].set_ylabel("Modality agreement (%)")
    axes[0].set_xticks(positions, ("Eval", "Test"))
    axes[0].legend(fontsize=8, frameon=False)
    choices = (("stated_modality", "Stated", "#235789"), ("always_majority_modality", "Most frequent", "#b88b4a"), ("random_modality", "Uniform average", "#9ca9b4"), ("measured_top_modality", "Largest drop", "#629677"))
    for index, (key, label, color) in enumerate(choices):
        axes[1].bar(positions + (index - 1.5) * 0.18, [explanation[s]["eraser"][key]["comprehensiveness"] for s in ("eval", "test")], width=0.18, label=label, color=color)
    axes[1].set_ylabel("Comprehensiveness")
    axes[1].set_ylim(0, 0.22)
    axes[1].set_xticks(positions, ("Eval", "Test"))
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    for ax in axes:
        ax.yaxis.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(figure_dir / "faithfulness.png")
    plt.close(fig)
    eda = json.loads((repo / "ECER-paper/logs/data_eda.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    labels = sorted(eda["labels"], key=eda["splits"]["train"]["label_support"].get, reverse=True)
    y = np.arange(len(labels))
    for index, (split, color, marker) in enumerate((("train", "#235789", "o"), ("dev", "#b88b4a", "s"), ("eval", "#629677", "^"), ("test", "#8468a3", "x"))):
        row = eda["splits"][split]
        axes[0].scatter([row["label_prevalence_pct"][label] for label in labels], y + (index - 1.5) * 0.16, label=split.capitalize(), color=color, marker=marker, s=16)
        cardinalities = [int(k) for k in row["cardinality_counts"]]
        axes[1].plot(cardinalities, [100 * row["cardinality_counts"][str(k)] / row["people"] for k in cardinalities], color=color, marker=marker, markersize=3, linewidth=1.2, label=split.capitalize())
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("People positive for category (%)")
    axes[0].set_title("Union-target prevalence")
    axes[0].xaxis.grid(alpha=0.2)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].set_xticks(range(0, len(eda["labels"]) + 1, 2))
    axes[1].set_xlim(0, len(eda["labels"]))
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Positive categories per person")
    axes[1].set_ylabel("People (%)")
    axes[1].set_title("Target cardinality distribution")
    axes[1].yaxis.grid(alpha=0.2)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(figure_dir / "data_eda.png")
    plt.close(fig)
    print("Generated prediction.png and faithfulness.png from results/; data_eda.png from logs/data_eda.json")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    build(figure_dir=args.output_dir)
