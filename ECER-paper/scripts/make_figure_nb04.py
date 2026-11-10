import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT.parent / "results" / "qwen3_multimodal" / "20261104-071259-h100"
metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
results = json.loads((RUN / "test_results.json").read_text(encoding="utf-8"))
LABELS = {"full": "Full", "vit_only": "No cue token", "no_context": "No context", "no_person": "No person",
          "full_no_affect": "Cues w/o affect words"}

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1, 1.1, 1.15]})

history = metrics["history"]
epochs = [row["epoch"] for row in history]
ax[0].plot(epochs, [row["train_loss"] for row in history], color="tab:blue", lw=1.2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Train loss", color="tab:blue")
twin = ax[0].twinx()
twin.plot(epochs, [100 * row["dev_map"] for row in history], color="tab:orange", lw=1.2)
twin.set_ylabel("Dev mAP (%)", color="tab:orange")
ax[0].set_title("(a) Main fusion model", fontsize=8)

for name, label in LABELS.items():
    per_epoch = {}
    for row in results["history"]:
        if row["variant"] == name:
            per_epoch.setdefault(row["epoch"], []).append(100 * row["dev_map"])
    xs = sorted(per_epoch)
    ax[1].plot(xs, [np.mean(per_epoch[x]) for x in xs], lw=1.1, label=label)
    ax[1].fill_between(xs, [min(per_epoch[x]) for x in xs], [max(per_epoch[x]) for x in xs], alpha=0.12)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Dev mAP (%)")
ax[1].set_title("(b) Variants, mean of 3 seeds", fontsize=8)
ax[1].legend(fontsize=5.5, frameon=False)

names = list(LABELS)
means = [100 * results["variants"][name]["test"]["map"]["mean"] for name in names]
sds = [100 * results["variants"][name]["test"]["map"]["sd"] for name in names]
ax[2].bar(range(len(names)), means, yerr=sds, capsize=3, color="tab:green")
resnet = results["resnet_comparison"]["models"]
best = max(resnet, key=lambda name: resnet[name]["union"]["map"])
ax[2].axhline(100 * resnet[best]["union"]["map"], color="black", ls=":", lw=1, label="Best ResNet-50")
ax[2].axhline(100 * results["constant_top_k_test"]["map"], color="red", ls="--", lw=1, label="Constant")
ax[2].set_xticks(range(len(names)), [LABELS[name] for name in names], rotation=30, ha="right", fontsize=5.5)
ax[2].set_ylim(100 * results["constant_top_k_test"]["map"] - 2, max(means) + 2)
ax[2].set_ylabel("Test mAP (%)")
ax[2].set_title("(c) Test mAP (7,280 people)", fontsize=8)
ax[2].legend(fontsize=5.5, frameon=False, loc="lower right")

for axis in (*ax, twin):
    axis.tick_params(labelsize=6)
    axis.xaxis.label.set_size(7)
    axis.yaxis.label.set_size(7)
fig.tight_layout(pad=0.4)
fig.savefig(ROOT / "figures" / "fig_fusion.pdf")
fig.savefig(ROOT / "figures" / "fig_fusion.png", dpi=200)
print("wrote figures/fig_fusion.pdf and .png")
