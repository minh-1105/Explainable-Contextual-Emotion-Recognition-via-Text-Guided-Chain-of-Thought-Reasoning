"""Build figures/fig_pilot.{pdf,png} from result files (read-only inputs).

Panel a: baseline val mAP per epoch (history.csv of the three ResNet-50 runs).
Panel b: number of predicted labels, Engagement vs all other labels, pilot D/E
         (parsed from logs/engagement.txt, which is the re-run of
         novelty_evidence/scripts/engagement.py).
Panel c: pilot micro-F1 against union ground truth, with controls.
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results"
(ROOT / "figures").mkdir(exist_ok=True)

# a) baselines
runs = {"person_only": "Person only", "context_only": "Context only",
        "person_context": "Person + context"}
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw={"width_ratios": [1, 0.9, 1.3]})
for run, label in runs.items():
    with open(next((RESULTS / "runs" / run).glob("*/logs/history.csv")), newline="") as f:
        rows = list(csv.DictReader(f))
    ax[0].plot([int(r["epoch"]) + 1 for r in rows], [100 * float(r["val_map"]) for r in rows],
               label=label, lw=1.2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Val mAP (%)")
ax[0].set_title("(a) ResNet-50 baselines", fontsize=8)
ax[0].legend(fontsize=6, frameon=False)

# b) Engagement vs others
txt = (ROOT / "logs" / "engagement.txt").read_text(encoding="utf-8")
union_block = txt.split("--- GT=majority")[0]
pat = re.compile(r"^\s+(\w[\w/]*)\s+D=\((\d+),(\d+)\) E=\((\d+),(\d+)\) support=(\d+)", re.M)
cnt = {"D": {"Engagement": 0, "Other": 0}, "E": {"Engagement": 0, "Other": 0}}
for name, tpd, fpd, tpe, fpe, _ in pat.findall(union_block):
    key = "Engagement" if name == "Engagement" else "Other"
    cnt["D"][key] += int(tpd) + int(fpd)
    cnt["E"][key] += int(tpe) + int(fpe)
xs = [0, 1]
ax[1].bar([x - 0.18 for x in xs], [cnt["D"]["Engagement"], cnt["E"]["Engagement"]], 0.36,
          label="Engagement")
ax[1].bar([x + 0.18 for x in xs], [cnt["D"]["Other"], cnt["E"]["Other"]], 0.36,
          label="All other labels")
ax[1].set_xticks(xs, ["Direct (D)", "Structured (E)"])
ax[1].set_ylabel("Predictions (300 people)")
ax[1].set_title("(b) Where E's extra labels go", fontsize=8)
ax[1].legend(fontsize=6, frameon=False, loc="upper center", ncol=1)
for x, k in ((-0.18, "Engagement"), (0.18, "Other")):
    ax[1].text(0 + x, cnt["D"][k] + 6, str(cnt["D"][k]), ha="center", fontsize=6)
    ax[1].text(1 + x, cnt["E"][k] + 6, str(cnt["E"][k]), ha="center", fontsize=6)
ax[1].set_ylim(0, max(cnt["D"]["Other"], cnt["E"]["Other"]) * 1.45)

# c) micro-F1 with controls
metrics = json.loads((RESULTS / "vlm_pilot" / "20261019-104614" / "metrics.json").read_text(encoding="utf-8"))
m = re.search(r"micro F1 without Engagement: D=([\d.]+) E=([\d.]+)", txt).groups()
always = float(re.search(r"'always Engagement' micro F1=([\d.]+)", txt).group(1))
prior = (ROOT / "logs" / "prior.txt").read_text(encoding="utf-8")
top2 = float(re.search(r"GT=union: top1: .*?top2: macro=[\d.]+ micro=([\d.]+)", prior).group(1))
bars = [("D", 100 * metrics["direct"]["micro_f1"]),
        ("E", 100 * metrics["structured"]["micro_f1"]),
        ("D\nno Eng.", 100 * float(m[0])), ("E\nno Eng.", 100 * float(m[1])),
        ("Always\nEng.", 100 * always), ("Top-2\nconst.", 100 * top2)]
colors = ["C0", "C1", "C0", "C1", "0.5", "0.5"]
ax[2].bar(range(len(bars)), [b[1] for b in bars], color=colors)
ax[2].set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=5.5)
for i, (_, v) in enumerate(bars):
    ax[2].text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=6)
ax[2].set_ylabel("Micro-F1 (%), union GT")
ax[2].set_ylim(0, 52)
ax[2].set_title("(c) Micro-F1 and controls", fontsize=8)

for a in ax:
    a.tick_params(labelsize=5.5)
    a.xaxis.label.set_size(7)
    a.yaxis.label.set_size(7)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(pad=0.6)
fig.savefig(ROOT / "figures" / "fig_pilot.pdf")
fig.savefig(ROOT / "figures" / "fig_pilot.png", dpi=200)
print("engagement/other prediction counts:", cnt)
print("micro-F1 bars:", [(b[0].replace("\n", " "), round(b[1], 1)) for b in bars])
