import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
RUN = "results/multimodal_pipeline/20261110-074559"
NAMES = {
    "full": "Full", "vit_only": "ViT-only", "no_person_image": "No person\nimage", "no_context_image": "No context\nimage",
    "no_person_cues": "No person\ncues", "no_context_cues": "No context\ncues", "full_no_affect": "Word\ndeletion",
}
RESNET = {"person_only": "ResNet\nperson", "context_only": "ResNet\ncontext", "person_context": "ResNet\nperson+ctx"}


def build(figure_dir=ROOT / "figures", repo=REPO):
    load = lambda name: json.loads((repo / RUN / name).read_text(encoding="utf-8"))
    train, test, explanation = load("metrics.json"), load("test_results.json"), load("explanation_metrics.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 240})
    figure_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained", width_ratios=(7, 10))
    for ax, split in zip(axes, ("eval", "test")):
        summary = train["summary"] if split == "eval" else test["summary"]
        rows = [summary[name]["extended"]["eval"] if split == "eval" else summary[name]["extended"] for name in NAMES]
        mean = [100 * row["map"]["mean"] for row in rows]
        sd = [100 * row["map"]["sd"] for row in rows]
        labels = list(NAMES.values())
        colors = ["#235789"] + ["#9ca9b4"] * (len(NAMES) - 1)
        if split == "test":
            mean += [100 * test["resnet"][name]["map"] for name in RESNET]
            sd += [0] * len(RESNET)
            labels += list(RESNET.values())
            colors += ["#b88b4a"] * len(RESNET)
        ax.bar(range(len(mean)), mean, yerr=sd, capsize=3, color=colors)
        ax.axhline(mean[0], color="#235789", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(mean)), [label.replace("\n", " ") for label in labels], fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title("Held-out validation (eval)" if split == "eval" else "Test")
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
    axes[0].legend(frameon=False, fontsize=8)
    choices = (("stated_modality", "Stated", "#235789"), ("always_majority_modality", "Most frequent", "#b88b4a"),
               ("random_modality", "Uniform average", "#9ca9b4"), ("measured_top_modality", "Measured top", "#629677"))
    for index, (key, label, color) in enumerate(choices):
        axes[1].bar(positions + (index - 1.5) * 0.2, [explanation[s]["eraser"][key]["comprehensiveness"] for s in ("eval", "test")], width=0.2, label=label, color=color)
    axes[1].set_ylabel("Comprehensiveness")
    axes[1].set_ylim(0, 0.085)
    axes[1].set_xticks(positions, ("Eval", "Test"))
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.yaxis.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(figure_dir / "faithfulness.png")
    plt.close(fig)


if __name__ == "__main__":
    build()
