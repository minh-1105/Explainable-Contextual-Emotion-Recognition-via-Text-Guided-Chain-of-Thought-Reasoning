import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results"
TEX = (ROOT / "main.tex").read_text(encoding="utf-8")
LOG = {n: (ROOT / "logs" / f"{n}.txt").read_text(encoding="utf-8")
       for n in ("boot", "engagement", "prior", "votes", "pairs_check")}

fails, oks = [], 0


def expect(label, s):
    """s must occur verbatim in main.tex."""
    global oks
    if s in TEX:
        oks += 1
    else:
        fails.append(f"{label}: expected {s!r} in main.tex")


def num(x, d=1):
    return f"{x:.{d}f}"


def signed(x, d=1):
    v = round(x, d)
    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.{d}f}"


def delta(point, lo, hi, d=1):
    return f"{signed(point, d)} [{signed(lo, d)}, {signed(hi, d)}]"


m = json.loads((RESULTS / "vlm_pilot" / "20261019-104614" / "metrics.json").read_text(encoding="utf-8"))
D, E = m["direct"], m["structured"]
expect("macro D", num(100 * D["macro_f1"]))
expect("macro E", num(100 * E["macro_f1"]))
expect("micro D", num(100 * D["micro_f1"]))
expect("micro E", num(100 * E["micro_f1"]))
expect("pred labels D", num(D["mean_predicted_labels"], 2))
expect("pred labels E", num(E["mean_predicted_labels"], 2))
expect("true labels", num(D["mean_true_labels"], 2))
expect("parse E", num(100 * E["parse_rate"], 2))
expect("sec D", num(D["mean_seconds"], 1))
expect("sec E", num(E["mean_seconds"], 2))
d = m["delta_macro_f1_structured_minus_direct"]
assert m["n_people"] == 300 and m["n_images"] == 279

boot = LOG["boot"]
g = re.search(r"GT=union\s+macro F1: D=([\d.]+)% E=([\d.]+)% E-D=([+-][\d.]+)% CI95=\[([+-][\d.]+)%, ([+-][\d.]+)%\]", boot)
expect("macro delta", delta(*(float(x) for x in g.groups()[2:])))
for got, want in ((100 * d["point"], float(g.group(3))), (100 * d["ci95"][0], float(g.group(4))),
                  (100 * d["ci95"][1], float(g.group(5)))):
    assert abs(got - want) < 0.15, ("metrics.json vs boot.py macro delta", got, want)
g = re.search(r"GT=union\s+micro F1: D=([\d.]+)% E=([\d.]+)% E-D=([+-][\d.]+)% CI95=\[([+-][\d.]+)%, ([+-][\d.]+)%\]", boot)
expect("micro delta", delta(*(float(x) for x in g.groups()[2:])))

eng = LOG["engagement"]
u, mj = eng.split("--- GT=majority")
g = re.search(r"Engagement\s+D=\((\d+),(\d+)\) E=\((\d+),(\d+)\) support=(\d+)", u)
tpd, fpd, tpe, fpe, sup = map(int, g.groups())
expect("Eng D predicted", f"D predicts it for {tpd + fpd} people")
expect("Eng E predicted", f"E for {tpe + fpe}")
expect("Eng support", f"({sup} of 300 people)")
g = re.search(r"micro F1 without Engagement: D=([\d.]+) E=([\d.]+) E-D=([+-][\d.]+) CI95=\[([+-][\d.]+),([+-][\d.]+)\]", u)
expect("noEng D", f"is {100 * float(g.group(1)):.1f} for D")
expect("noEng E", f"{100 * float(g.group(2)):.1f} for E")
expect("noEng delta", f"{signed(100 * float(g.group(3)))} points (CI {signed(100 * float(g.group(4)))} to {signed(100 * float(g.group(5)))})")

pr = LOG["prior"]
g = re.search(r"GT=union: top1: .*?top2: macro=([\d.]+) micro=([\d.]+)", pr)
expect("top2 micro union", f"reaches micro-F1 {100 * float(g.group(2)):.1f}")

vt = LOG["votes"]
for split in ("val", "test"):
    blk = re.search(rf"\[{split}\] people=(\d+) annotators/person=([\d.]+).*?\n\s+labels/person union=([\d.]+) >=2votes=[\d.]+ majority=([\d.]+)\n\s+share of union labels picked by only 1 annotator: ([\d.]+)%", vt, re.S)
    people, ann, un, mjv, one = blk.groups()
    expect(f"label table {split}", f"{split.capitalize()} & {int(people):,} & {ann} & {un} & {mjv} & {one}\\%")
blk = re.search(r"\[train\] people=(\d+) annotators/person=([\d.]+).*?\n\s+labels/person union=([\d.]+)", vt, re.S)
expect("label table train", f"Train & {int(blk.group(1)):,} & {blk.group(2)} & {blk.group(3)} & {blk.group(3)} & --")

pc = LOG["pairs_check"]
g = re.search(r"\[test/majority\] pairs=(\d+).*?disjoint_nonempty=(\d+) \(([\d.]+)%\)", pc)
expect("disjoint test", f"{g.group(2)} of {int(g.group(1)):,} same-image pairs ({g.group(3)}\\%)")

cfg = json.loads((RESULTS / "vlm_pilot" / "20261019-104614" / "config.json").read_text(encoding="utf-8"))
assert cfg["max_pixels"] == 409600 and cfg["seed"] == 42
assert cfg["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct" and cfg["category_policy"] == "union"
expect("max pixels", "409,600")

# ---- numbers/quotes taken from the Etesam 2024 and Lei 2024 PDFs (pdftotext output in logs/) ----
et = (ROOT / "logs" / "etesam2024.txt").read_bytes().decode("latin-1")
for s in ("36.83", "28.83", "34.47"):
    assert re.search(rf"\b{re.escape(s)}\b", et), s
    expect(f"etesam {s}", s)
assert "could not employ mAP" in re.sub(r"\s+", " ", et)
assert "LLaVA finetuned on the validation set with augmentation" in re.sub(r"\s+", " ", et)
lei = re.sub(r"\s+", " ", (ROOT / "logs" / "lei2024.txt").read_bytes().decode("utf-8", "replace"))
assert "increases precision but decreases recall" in lei
assert "when we add CoT, the model performance decreases compared to not using CoT" in lei

run = RESULTS / "qwen3_multimodal" / "20261104-071259-h100"
cues = json.loads((ROOT / "logs" / "nb04_cues.json").read_text(encoding="utf-8"))
comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
nb04 = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
results = json.loads((run / "test_results.json").read_text(encoding="utf-8"))
explain = json.loads((run / "explanation_metrics.json").read_text(encoding="utf-8"))
bench = json.loads((run / "batch_benchmark.json").read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def unsigned_ci(b):
    return f"CI {pct(b['ci95_lower'])} to {pct(b['ci95_upper'])}"


def signed_ci(b):
    return f"CI {signed(100 * b['ci95_lower'])} to {signed(100 * b['ci95_upper'])}"


assert cues["people"] == 27040 and cues["parse_rate"] == 1.0 and cues["truncated_rate"] == 0.0
expect("cue schema", f"{pct(cues['schema_rate'], 2)}\\%")
expect("cue counts", f"{cues['person_cues_mean']:.2f} / {cues['context_cues_mean']:.2f}")
expect("relations", f"{cues['relations_mean']:.2f} ({pct(cues['relations_none_rate'])}\\%)")
expect("affect", f"{pct(cues['affect_rate'])}\\%")
for word, count in cues["affect_top"][:3]:
    expect(f"affect {word}", f"``{word}'' ({count:,} occurrences)" if word == "smiling" else f"{count}")
for size in ("48", "32", "16"):
    expect(f"batch {size} speed", f"{bench[size]['seconds_per_person']:.2f}")
assert all(bench[s]["schema_rate"] == 1.0 for s in ("16", "32", "48"))
expect("batch agreement", f"output for {round(48 * bench['48']['same_as_single'])}, while all 48")
expect("peak vram", f"{bench['48']['peak_vram_gb']:.1f}~GB")
assert len(results["affect_words"]) == 22
assert results["people"] == {"test": 7280, "test_images": 4389}
test_cues = results["test_cues"]
assert test_cues["truncated_rate"] == 0.0
expect("test cues", f"{pct(test_cues['parse_rate'], 2)}\\% of outputs parse, {pct(test_cues['schema_rate'], 2)}\\% match the schema")
expect("affect changed", f"changes {pct(results['affect_changed_rate'])}\\% of the 34,320 cue sets")

assert comparison["people"] == {"train": 23706, "dev": 1643, "eval": 1691}
labels = {"full": "Full (person, context, cue)", "vit_only": "No cue token (vision only)",
          "no_context": "No context token", "no_person": "No person token",
          "full_no_affect": "Cues without affect words"}
for name, label in labels.items():
    v = results["variants"][name]
    cells = [f"{pct(v[split][m]['mean'])}$\\pm${pct(v[split][m]['sd'])}"
             for split in ("eval", "test") for m in ("map", "macro_f1", "micro_f1")]
    cells.append(f"{v['test']['mean_predicted_labels']['mean']:.2f}")
    expect(f"variant {name}", label + " & " + " & ".join(cells))
earlier = " and ".join([", ".join(pct(comparison["variants"][n]["map"]["mean"]) for n in ("full", "vit_only", "no_context")),
                        pct(comparison["variants"]["no_person"]["map"]["mean"])])
expect("earlier pass", f"gave eval mAP {earlier}")
prevalence = sum(r["support"] for r in nb04["eval"]["per_class"]) / 26 / 1691
constant, constant_test = comparison["constant_top_k"], results["constant_top_k_test"]
assert constant["k"] == constant_test["k"] == 26
expect("constant row", f"Constant (top-26) & {pct(prevalence)}$^\\dagger$ & {pct(constant['macro_f1'])} & "
       f"{pct(constant['micro_f1'])} & {pct(constant_test['map'])}$^\\dagger$ & {pct(constant_test['macro_f1'])} & "
       f"{pct(constant_test['micro_f1'])} & 26.00")
resnet = results["resnet_comparison"]
for name, label, short in (("person_only", "person only", "Person only"), ("context_only", "context only", "Context only"),
                           ("person_context", "person + context", "Person + context")):
    u = resnet["models"][name]["union"]
    cells = f"{pct(u['map'], 2)} & {pct(u['macro_f1'], 2)} & {pct(u['micro_f1'], 2)} & {u['mean_predicted_labels']:.2f}"
    expect(f"resnet {name}", f"ResNet-50, {label} & -- & -- & -- & {cells}")
    expect(f"baseline table {name}", f"{short} & {cells}")
expect("baseline constant", f"Constant (top-26) & {100 * constant_test['map']:.2f}$^\\dagger$ & {pct(constant_test['macro_f1'], 2)} & "
       f"{pct(constant_test['micro_f1'], 2)} & 26.00")
majority = sorted(pct(m["majority_map"]) for m in resnet["models"].values())
expect("resnet majority", f"against {majority[0]} to {majority[-1]} for the ResNets")
expect("pipeline majority", f"test mAP is {pct(resnet['pipeline_majority_labels']['main_checkpoint']['map'])} for the main checkpoint "
       f"and {pct(resnet['pipeline_majority_labels']['full_no_affect']['map_seed42'])} for the affect-free")
assert resnet["pipeline_majority_labels"]["main_checkpoint"]["map"] == resnet["pipeline_majority_labels"]["full"]["map_seed42"]

boot = results["bootstrap"]
expect("full vs constant", f"$+${pct(boot['full_vs_constant_test']['difference_a_minus_b'])} macro-F1, {unsigned_ci(boot['full_vs_constant_test'])}")
expect("full vs vit test", f"{signed(100 * boot['full_vs_vit_only_test']['difference_a_minus_b'])} macro-F1 ({signed_ci(boot['full_vs_vit_only_test'])})")
expect("full vs others test", ", ".join(signed(100 * boot[f"full_vs_{n}_test"]["difference_a_minus_b"]) for n in ("no_context", "no_person"))
       + " and " + signed(100 * boot["full_vs_full_no_affect_test"]["difference_a_minus_b"]))
for key in ("full_vs_no_context_test", "full_vs_no_person_test", "full_vs_full_no_affect_test"):
    assert boot[key]["ci95_upper"] < 0, key
expect("affect-free vs vit test", f"$+${pct(boot['full_no_affect_vs_vit_only_test']['difference_a_minus_b'])} macro-F1 on the seed-42 runs ({unsigned_ci(boot['full_no_affect_vs_vit_only_test'])})")
expect("affect-free vs vit eval", f"by {pct(boot['full_no_affect_vs_vit_only_eval']['difference_a_minus_b'])} macro-F1 points on the seed-42 runs (95\\% {unsigned_ci(boot['full_no_affect_vs_vit_only_eval'])})")
rb = resnet["bootstrap"]
expect("full vs resnet context", f"{signed(100 * rb['full_vs_resnet_context_only']['difference_a_minus_b'])}, {signed_ci(rb['full_vs_resnet_context_only'])}")
expect("full vs resnet person", f"$+${pct(rb['full_vs_resnet_person_only']['difference_a_minus_b'])}, {unsigned_ci(rb['full_vs_resnet_person_only'])}")
expect("full vs resnet person context", f"$+${pct(rb['full_vs_resnet_person_context']['difference_a_minus_b'])}, {unsigned_ci(rb['full_vs_resnet_person_context'])}")
expect("affect-free vs resnet context", f"$+${pct(rb['full_no_affect_vs_resnet_context_only']['difference_a_minus_b'])} ({unsigned_ci(rb['full_no_affect_vs_resnet_context_only'])})")
for key in ("full_no_affect_vs_resnet_person_only", "full_no_affect_vs_resnet_person_context"):
    assert rb[key]["ci95_lower"] > 0, key

full_runs = sorted((r for r in results["runs"] if r["variant"] == "full" and r["split"] == "test"), key=lambda r: r["seed"])
expect("full seeds test", f"({pct(full_runs[0]['map'])} mAP against {pct(full_runs[1]['map'])} and {pct(full_runs[2]['map'])})")
e, t = nb04["eval"], results["main_checkpoint_test"]
assert abs(full_runs[0]["map"] - t["map"]) < 1e-9 and abs(full_runs[0]["macro_f1"] - t["macro_f1"]) < 1e-9
expect("main checkpoint", f"{pct(e['map'])} mAP, {pct(e['macro_f1'])} macro-F1 and {pct(e['micro_f1'])} micro-F1 on the eval half "
       f"and {pct(t['map'])}, {pct(t['macro_f1'])} and {pct(t['micro_f1'])} on test")
expect("main precision", f"{pct(t['micro_precision'])}\\% micro-precision and {pct(t['micro_recall'])}\\% micro-recall")
ap = {r["class_name"]: pct(r["ap"]) for r in t["per_class"]}
expect("test AP", f"from {ap['Engagement']} for Engagement and {ap['Happiness']} for Happiness to {ap['Fear']} for Fear "
       f"and {ap['Embarrassment']} for Embarrassment")


def counts(c):
    return f"{c['person']:,} / {c['context']:,} / {c['graph']:,}"


for split, column in (("eval", 1), ("test", 2)):
    x = explain[split]
    cells = {
        "generated": f"{x['n_generated']:,} / {x['n_usable']:,}",
        "truncated": f"{pct(x['truncated_rate'], 2)}\\%",
        "stated": counts(x["stated_modality_counts"]),
        "measured": counts(x["measured_top_modality_counts"]),
        "agreement": f"{pct(x['modality_agreement'])}\\%",
        "majority": f"{pct(x['modality_agreement_always_majority'])}\\%",
        "cited": f"{x['cited_numbers_total']:,}",
        "cited ok": f"{pct(x['cited_numbers_in_evidence'], 2)}\\%",
        "unpredicted": f"{pct(x['explanations_naming_unpredicted_emotion'], 2)}\\%",
    }
    er = x["eraser"]
    for metric in ("comprehensiveness", "sufficiency"):
        cells[f"{metric} stated"] = f"{er['stated_modality'][metric]:.3f}"
        cells[f"{metric} majority random"] = f"{er['always_majority_modality'][metric]:.3f} / {er['random_modality'][metric]:.3f}"
        cells[f"{metric} top"] = f"{er['measured_top_modality'][metric]:.3f}"
    rows = [line for line in TEX.split("\n") if line.count("&") == 2 and line.rstrip().endswith("\\\\")]
    table_rows = rows[rows.index(" & Eval & Test \\\\") + 1:]
    values = [row.split("&")[column].strip().removesuffix("\\\\").strip() for row in table_rows[:len(cells)]]
    for (label, want), got in zip(cells.items(), values):
        if want == got:
            oks += 1
        else:
            fails.append(f"explanation {split} {label}: table has {got!r}, source {want!r}")
    assert x["stated_modality_counts"]["person"] / x["n_usable"] > 0.99, split
    assert er["stated_modality"]["sufficiency"] >= er["always_majority_modality"]["sufficiency"], split
ev, et = explain["eval"], explain["test"]
expect("eval person share", f"names the person in {pct(ev['stated_modality_counts']['person'] / ev['n_usable'])}\\% of usable eval")
expect("eval measured person", f"for only {pct(ev['measured_top_modality_counts']['person'] / ev['n_usable'])}\\% of people")
expect("test person share", f"names the person in {pct(et['stated_modality_counts']['person'] / et['n_usable'])}\\%, so")
expect("abstract person share", f"in {pct((ev['stated_modality_counts']['person'] + et['stated_modality_counts']['person']) / (ev['n_usable'] + et['n_usable']))}\\% of cases")

eb = json.loads((ROOT / "logs" / "explanation_bootstrap.json").read_text(encoding="utf-8"))
assert eb["eval"]["majority_modality"] == "context" and eb["test"]["majority_modality"] == "person"
for split in ("eval", "test"):
    n = explain[split]["n_usable"]
    for key in eb[split]:
        if key != "majority_modality":
            assert eb[split][key]["n_people"] == n, (split, key)
    assert eb[split]["agreement_stated_vs_majority"]["ci95_upper"] > 0, split
    assert eb[split]["comprehensiveness_stated_vs_majority"]["ci95_upper"] > 0, split
    assert eb[split]["comprehensiveness_stated_vs_random"]["ci95_lower"] > 0, split


def ci_text(b, scale, d):
    return f"({signed(scale * b['ci95_lower'], d)} to {signed(scale * b['ci95_upper'], d)})"


a_ev, a_te = eb["eval"]["agreement_stated_vs_majority"], eb["test"]["agreement_stated_vs_majority"]
c_ev, c_te = eb["eval"]["comprehensiveness_stated_vs_majority"], eb["test"]["comprehensiveness_stated_vs_majority"]
r_ev, r_te = eb["eval"]["comprehensiveness_stated_vs_random"], eb["test"]["comprehensiveness_stated_vs_random"]
expect("agreement bootstrap", f"{signed(100 * a_ev['difference_a_minus_b'])} points on eval (95\\% CI "
       f"{signed(100 * a_ev['ci95_lower'])} to {signed(100 * a_ev['ci95_upper'])}) and "
       f"{signed(100 * a_te['difference_a_minus_b'], 2)} on test {ci_text(a_te, 100, 2)}")
expect("comprehensiveness bootstrap", f"{signed(c_ev['difference_a_minus_b'], 3)} on eval {ci_text(c_ev, 1, 3)} and "
       f"{abs(c_te['difference_a_minus_b']):.4f} on test {ci_text(c_te, 1, 4)}")
expect("random bootstrap", f"higher by {r_ev['difference_a_minus_b']:.3f} on eval ({r_ev['ci95_lower']:.3f} to {r_ev['ci95_upper']:.3f}) "
       f"and {r_te['difference_a_minus_b']:.3f} on test ({r_te['ci95_lower']:.3f} to {r_te['ci95_upper']:.3f})")
expect("test cited", f"({pct(et['cited_numbers_in_evidence'], 2)}\\% of {et['cited_numbers_total']:,})")
expect("test unpredicted", f"{pct(et['explanations_naming_unpredicted_emotion'], 2)}\\% of test explanations")

print(f"checked OK: {oks}")
if fails:
    print("MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all numbers in main.tex match the sources")
