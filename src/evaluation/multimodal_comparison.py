"""Prediction-only dev/eval comparison for Notebook 05; no test-data access."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import platform
from pathlib import Path

import numpy as np

from src.datasets.emotic_dataset import EMOTION_LABELS
from src.utils.run_records import sha256
from .emotion_metrics import (multilabel_metrics, select_per_class_thresholds,
    threshold_boundary_report, select_constant_top_k, constant_top_k_predictions,
    paired_image_bootstrap)
from .selection import mix_probabilities, select_fusion_weight, per_sample_top_k, select_sample_top_k

SOURCE_RUN = "outputs/multimodal_pipeline/20261110-074559"
RESNET_RUN = "outputs/multimodal_comparison/20261109-090105-378575"
RESNETS = ("person_only", "context_only", "person_context")
LIMITATIONS = [
    "ResNet chỉ có một seed.",
    "Checkpoint ResNet chọn trên toàn bộ PAMI val, bao gồm eval; model 04 chọn trên dev.",
    "Feature 04 được mã hóa lại; đây là run chính thức mới, không yêu cầu bằng H100 từng bit.",
    "Test của 04 đã được bật trước phân tích này; không dùng 05 để đổi model, seed hay ngưỡng 04.",
    "Ensemble, late fusion và top-k theo mẫu chỉ là phân tích eval; không được chấm test sau đó.",
    "Bootstrap đo bất định do lấy mẫu ảnh, không loại bỏ thiên lệch chọn checkpoint ResNet.",
]


def safe(value):
    """Convert NumPy values to strict JSON, using null for undefined AP."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def required_inputs(root, source_run=SOURCE_RUN, resnet_run=RESNET_RUN):
    """Return an explicit validation-only allowlist, derived from source config."""
    root = Path(root)
    names = [f"{source_run}/{p}" for p in (
        "config.json", "samples.json", "train/metrics.json", "environment.txt", "code_snapshot.json")]
    config_path = root / source_run / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        names += [f"{source_run}/train/predictions/{name}_seed{seed}.npz"
                  for name in config["variants"] for seed in config["seeds"]]
        names += [f"{source_run}/features/{config['image_backbone']}/val.npz"]
        names += [p.relative_to(root).as_posix() for p in sorted(
            (root / source_run).glob("code_snapshot_*.json"))]
    names += [f"{resnet_run}/{p}" for p in (
        "predictions_dev.npz", "predictions_eval.npz", "metrics.json", "config.json", "selected_thresholds.json")]
    assert all("test" not in name.lower() for name in names), "Không được đọc dữ liệu test."
    return list(dict.fromkeys(names))


class ComparisonStudy:
    """Run independently reviewable comparison stages from immutable predictions."""

    def __init__(self, root, commit, source_run=SOURCE_RUN, resnet_run=RESNET_RUN):
        self.root = Path(root)
        self.source_run, self.resnet_run = source_run, resnet_run
        self.artifacts = {}
        self.source_config = self.read_json(f"{source_run}/config.json")
        cfg = self.source_config
        assert cfg["category_policy"] == "union"
        self.config = {"source_run": source_run, "resnet_run": resnet_run,
            "primary_variant": cfg["primary_variant"], "variants": list(cfg["variants"]), "seeds": cfg["seeds"],
            "threshold_grid": cfg["threshold_grids"]["extended"],
            "threshold_grids": cfg["threshold_grids"], "bootstrap_resamples": 2000,
            "bootstrap_seed": 42, "bootstrap_metrics": ["map", "macro_f1"],
            "fusion_weight_grid": np.linspace(0, 1, 21).tolist(), "top_k_grid": list(range(1, 27)),
            "error_examples_per_direction": 10, "git_commit": commit}
        assert self.config["primary_variant"] == "full"
        required = required_inputs(self.root, source_run, resnet_run)
        missing = [name for name in required if not (self.root / name).is_file()]
        assert not missing, f"Thiếu artifact: {missing}"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self.out = self.root / "outputs/multimodal_comparison" / stamp
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "figures").mkdir()
        self.write("config.json", self.config)
        (self.out / "environment.txt").write_text(
            f"python={platform.python_version()}\nnumpy={np.__version__}\ngit_commit={commit}\ncpu_only=True\n",
            encoding="utf-8")
        self.probs, self.thresholds, self.results = {p: {} for p in ("dev", "eval")}, {}, {}
        self.rows, self.checks = [], []
        self.model_info = {}

    def path(self, name):
        """Resolve, audit and hash every source file before reading it."""
        path = (self.root / name).resolve()
        rel = path.relative_to(self.root.resolve()).as_posix()
        assert "test" not in rel.lower(), f"Cấm đọc test: {rel}"
        assert rel in required_inputs(self.root, self.source_run, self.resnet_run), rel
        if rel not in self.artifacts:
            self.artifacts[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    def read_json(self, name):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write(self, name, payload):
        (self.out / name).write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2,
                                              allow_nan=False), encoding="utf-8")

    def csv(self, name, rows):
        rows = safe(rows)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (self.out / name).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def load_sources(self):
        """Check labels, image partitions, class order and all prediction rows."""
        samples = self.read_json(f"{self.source_run}/samples.json")
        assert samples["validation_split_unit"] == "relative_image_path"
        self.splits = {part: samples["splits"][part] for part in ("dev", "eval")}
        self.targets, self.images = {}, {}
        for part in self.splits:
            with np.load(self.path(f"{self.resnet_run}/predictions_{part}.npz"), allow_pickle=False) as data:
                assert data["sample_ids"].tolist() == self.splits[part]
                assert data["class_names"].tolist() == list(EMOTION_LABELS)
                self.targets[part], self.images[part] = data["targets"].copy(), data["image_ids"].copy()
                for name in RESNETS:
                    self.probs[part][name] = data[name].copy()
                if part == "eval":
                    for name in RESNETS:
                        self.thresholds[name] = {"extended": data[f"{name}_thresholds"].copy(),
                                                "legacy": data[f"{name}_legacy_thresholds"].copy()}
            ids = self.splits[part]
            assert len(ids) == len(set(ids)) and all(sid.startswith("val:") for sid in ids)
            assert self.targets[part].shape == (len(ids), len(EMOTION_LABELS))
            assert np.isin(self.targets[part], [0, 1]).all()
        assert not set(self.images["dev"]) & set(self.images["eval"])
        assert not set(self.splits["dev"]) & set(self.splits["eval"])
        with np.load(self.path(f"{self.source_run}/features/{self.source_config['image_backbone']}/val.npz"),
                     allow_pickle=False) as data:
            ids = data["sample_ids"].tolist()
            assert len(ids) == len(set(ids))
            assert ids == [sid for sid in samples["sample_ids"] if sid.startswith("val:")]
            lookup = {sid: i for i, sid in enumerate(ids)}
            labels = data["labels"]  # Do not materialize image features.
            for part in self.splits:
                assert np.array_equal(labels[[lookup[sid] for sid in self.splits[part]]], self.targets[part])
        for name in self.config["variants"]:
            for seed in self.config["seeds"]:
                key = f"{name}_seed{seed}"
                with np.load(self.path(f"{self.source_run}/train/predictions/{key}.npz"), allow_pickle=False) as data:
                    for part in self.splits:
                        assert data[f"{part}_sample_ids"].tolist() == self.splits[part]
                        self.probs[part][key] = data[f"{part}_probabilities"].copy()
                    self.thresholds[key] = {grid: data[f"thresholds_{grid}"].copy()
                                            for grid in self.config["threshold_grids"]}
                self.model_info[key] = {"variant": name, "seed": seed, "origin": "04"}
        for name in RESNETS:
            self.model_info[name] = {"variant": name, "seed": None, "origin": "ResNet"}
        for values in self.probs.values():
            for name, array in values.items():
                assert array.shape == self.targets["dev" if values is self.probs["dev"] else "eval"].shape
                assert np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(), name
        # Read provenance contents as well as record byte hashes.
        for name in required_inputs(self.root, self.source_run, self.resnet_run):
            if "code_snapshot" in name:
                self.read_json(name)
            elif name.endswith("environment.txt"):
                self.path(name).read_text(encoding="utf-8")
        self.read_json(f"{self.resnet_run}/config.json")
        self.write("source_artifacts.json", {"files": self.artifacts, "source_run": self.source_run,
                                            "resnet_run": self.resnet_run})
        print(f"Đã kiểm tra nhãn, thứ tự mẫu, {len(self.model_info) - len(RESNETS)} prediction và chia theo ảnh.", flush=True)

    def reproduce(self):
        """Stop on any metric or threshold mismatch before doing new analysis."""
        source = self.read_json(f"{self.source_run}/train/metrics.json")
        resnet = self.read_json(f"{self.resnet_run}/metrics.json")
        selected = self.read_json(f"{self.resnet_run}/selected_thresholds.json")
        for grid, candidates in self.config["threshold_grids"].items():
            for key, info in self.model_info.items():
                chosen = select_per_class_thresholds(self.targets["dev"], self.probs["dev"][key], candidates)
                # Legacy ResNet serialized decimal-rounded grid points; 04 stores NumPy grid points.
                np.testing.assert_allclose(chosen, self.thresholds[key][grid], atol=1e-12, rtol=0,
                                           err_msg=f"{key} {grid}")
                if info["origin"] == "ResNet":
                    field = "thresholds" if grid == "extended" else "legacy_thresholds"
                    np.testing.assert_array_equal(self.thresholds[key][grid], selected[field][key])
                for part in self.splits:
                    result = self.evaluate(key, grid, part)
                    if info["origin"] == "04":
                        old = [r for r in source["runs"] if (r["variant"], r["seed"], r["grid"], r["split"]) ==
                               (info["variant"], info["seed"], grid, part)]
                        assert len(old) == 1
                        old = old[0]
                    elif part == "eval":
                        old = resnet["eval" if grid == "extended" else "eval_legacy_grid"][key]
                    else:
                        continue
                    for metric in ("map", "macro_f1", "micro_f1"):
                        assert np.isclose(result[metric], old[metric], atol=1e-9, rtol=0), (key, grid, part, metric)
                    self.checks.append({"model": key, "grid": grid, "part": part, "passed": True})
        self.write("reproduction_checks.json", {"passed": True, "checks": self.checks,
                                                "reference": "04 mới và ResNet đã lưu; không so H100"})
        self.write("source_artifacts.json", {"files": self.artifacts, "source_run": self.source_run,
                                            "resnet_run": self.resnet_run})
        print(f"Tái lập đạt: {len(self.checks)} bộ metric; tất cả ngưỡng chọn lại khớp.", flush=True)

    def evaluate(self, name, grid="extended", part="eval", mask=None, predictions=None):
        target, scores = self.targets[part], self.probs[part][name]
        if mask is not None:
            target, scores = target[mask], scores[mask]
        return multilabel_metrics(target, scores, thresholds=self.thresholds[name][grid],
                                  predictions=predictions, class_names=EMOTION_LABELS)

    def main_table(self):
        """Recompute all main metrics and summarize population SD across seeds."""
        for grid in self.config["threshold_grids"]:
            for key, info in self.model_info.items():
                result = self.evaluate(key, grid)
                self.results[key, grid] = result
                row = {"model": key, **info, "grid": grid, "summary": False,
                       **{k: v for k, v in result.items() if k != "per_class"},
                       "thresholds_at_grid_min": threshold_boundary_report(
                           self.targets["dev"], self.thresholds[key][grid], self.config["threshold_grids"][grid])["at_min"]}
                self.rows.append(row)
            for variant in self.config["variants"]:
                subset = [r for r in self.rows if r["variant"] == variant and r["grid"] == grid and not r["summary"]]
                keys = [k for k, v in subset[0].items() if isinstance(v, (int, float, np.number))
                        and k not in ("seed", "summary")]
                self.rows.append({"model": variant, "variant": variant, "seed": None,
                    "grid": grid, "summary": True, "origin": "04 mean ± SD",
                    **{k: float(np.mean([r[k] for r in subset])) for k in keys},
                    **{k + "_sd": float(np.std([r[k] for r in subset])) for k in keys}})
        self.constant = select_constant_top_k(self.targets["dev"])
        self.constant_binary = {p: constant_top_k_predictions(len(self.targets[p]), self.constant["label_order"],
            self.constant["k"], len(EMOTION_LABELS)) for p in self.splits}
        # Prevalence ranking is dev-only; constant-score AP is still well-defined on eval.
        prevalence = self.targets["dev"].mean(axis=0)
        for part in self.splits:
            self.probs[part]["constant_top_k"] = np.tile(prevalence, (len(self.targets[part]), 1))
        self.thresholds["constant_top_k"] = {grid: np.full(26, .5) for grid in self.config["threshold_grids"]}
        for grid in self.config["threshold_grids"]:
            result = self.evaluate("constant_top_k", grid, predictions=self.constant_binary["eval"])
            self.results["constant_top_k", grid] = result
            self.rows.append({"model": "constant_top_k", "variant": "constant_top_k", "seed": None,
                "origin": "dev prevalence", "grid": grid, "summary": False,
                **{k: v for k, v in result.items() if k != "per_class"}})
        self.save_metrics()

    def supplementary(self):
        """Freeze all additional choices on dev before computing their eval metrics."""
        full_keys = [f"full_seed{s}" for s in self.config["seeds"]]
        for part in self.splits:
            self.probs[part]["full_ensemble"] = np.mean([self.probs[part][key] for key in full_keys], axis=0)
        self.fusion = select_fusion_weight(self.targets["dev"], self.probs["dev"]["full_ensemble"],
                                           self.probs["dev"]["person_context"], self.config["fusion_weight_grid"])
        for part in self.splits:
            self.probs[part]["late_fusion"] = mix_probabilities(self.probs[part]["full_ensemble"],
                self.probs[part]["person_context"], self.fusion["weight"])
        self.top_k = {}
        for name in ("full_ensemble", "late_fusion"):
            self.thresholds[name] = {grid: select_per_class_thresholds(
                self.targets["dev"], self.probs["dev"][name], candidates)
                for grid, candidates in self.config["threshold_grids"].items()}
        for name in self.thresholds:
            if name != "constant_top_k":
                self.top_k[name] = select_sample_top_k(self.targets["dev"], self.probs["dev"][name], self.config["top_k_grid"])
        for name in ("full_ensemble", "late_fusion"):
            for grid in self.config["threshold_grids"]:
                result = self.evaluate(name, grid)
                self.results[name, grid] = result
                self.rows.append({"model": name, "origin": "eval only", "grid": grid, "summary": False,
                                  "thresholds_at_grid_min": threshold_boundary_report(
                                      self.targets["dev"], self.thresholds[name][grid], self.config["threshold_grids"][grid])["at_min"],
                                  **{k: v for k, v in result.items() if k != "per_class"}})
        for name, selection in self.top_k.items():
            result = self.evaluate(name, predictions=per_sample_top_k(self.probs["eval"][name], selection["k"]))
            self.results[name, "sample_top_k"] = result
            self.rows.append({"model": name, "origin": "eval only", "grid": "sample_top_k", "summary": False,
                              **{k: v for k, v in result.items() if k != "per_class"}})
        self.write("selected_parameters.json", {"selected_on": "dev", "thresholds": self.thresholds,
                   "constant_top_k": self.constant, "late_fusion": self.fusion, "sample_top_k": self.top_k,
                   "supplementary_scope": "eval only; không chấm test"})
        for part in self.splits:
            np.savez_compressed(self.out / f"predictions_{part}.npz", sample_ids=np.asarray(self.splits[part]),
                image_ids=self.images[part], targets=self.targets[part], class_names=np.asarray(EMOTION_LABELS),
                **self.probs[part], constant_predictions=self.constant_binary[part],
                **{f"{name}_thresholds_{grid}": value for name, grids in self.thresholds.items() for grid, value in grids.items()})
        self.save_metrics()
        print("Đã khóa w, k và ngưỡng trên dev; lưu prediction bổ sung chỉ để phân tích eval.", flush=True)

    def save_metrics(self):
        self.csv("metrics.csv", self.rows)
        self.csv("per_class_metrics.csv", [{"model": name, "grid": grid, **row}
            for (name, grid), result in self.results.items() for row in result["per_class"]])
        self.write("metrics.json", {"scope": "PAMI val dev/eval only; union; no test access",
            "limitations": LIMITATIONS, "rows": self.rows,
            "eval": {f"{name}/{grid}": result for (name, grid), result in self.results.items()}})

    def bootstrap(self):
        """Save every completed paired comparison and print progress on CPU."""
        jobs = []
        for seed in self.config["seeds"]:
            a = f"full_seed{seed}"
            jobs += [(a, f"{name}_seed{seed}", seed) for name in self.config["variants"] if name != "full"]
            jobs += [(a, name, seed) for name in RESNETS]
        jobs += [(name, "person_context", None) for name in ("full_ensemble", "late_fusion")]
        results = []
        for index, (a, b, seed) in enumerate(jobs, 1):
            for metric in self.config["bootstrap_metrics"]:
                pa, pb = self.probs["eval"][a], self.probs["eval"][b]
                if metric == "macro_f1":
                    pa, pb = pa >= self.thresholds[a]["extended"], pb >= self.thresholds[b]["extended"]
                print(f"Bootstrap {index}/{len(jobs)}: {a} − {b}, {metric}", flush=True)
                value = paired_image_bootstrap(self.targets["eval"], pa, pb, self.images["eval"],
                    n_resamples=self.config["bootstrap_resamples"], seed=self.config["bootstrap_seed"], metric=metric)
                excludes = value["ci95_lower"] > 0 or value["ci95_upper"] < 0
                results.append({"a": a, "b": b, "seed": seed, "metric": metric, **value,
                    "ci_excludes_zero": excludes,
                    "interpretation": "CI không chứa 0" if excludes else "không phát hiện đóng góp rõ"})
                self.write("bootstrap.json", {"complete": False, "comparisons": results})
        summaries = []
        for b in [v for v in self.config["variants"] if v != "full"] + list(RESNETS):
            for metric in self.config["bootstrap_metrics"]:
                rows = [r for r in results if r["seed"] is not None and r["metric"] == metric
                        and (r["b"] == b or r["b"] == f"{b}_seed{r['seed']}")]
                summaries.append({"comparison": f"full − {b}", "metric": metric, "seeds": rows,
                    "seeds_ci_excludes_zero": sum(r["ci_excludes_zero"] for r in rows),
                    "n_seeds": len(rows)})
        self.boot_results = results
        self.write("bootstrap.json", {"complete": True, "comparisons": results, "seed_summary": summaries,
            "note": "CI riêng từng seed; không gộp p-value. CI chứa 0: không phát hiện đóng góp rõ."})

    def diagnostics(self):
        """Report class AP deltas, image crowd sizes and deterministic error examples."""
        counts = Counter(np.concatenate(list(self.images.values())).tolist())
        people = np.asarray([counts[x] for x in self.images["eval"]])
        groups = {}
        for label, mask in {"1": people == 1, "2": people == 2, "3–4": (people >= 3) & (people <= 4), "≥5": people >= 5}.items():
            groups[label] = {"people": int(mask.sum()), "images": len(set(self.images["eval"][mask])), "metrics": {}}
            if mask.any():
                for name in self.thresholds:
                    prediction = self.constant_binary["eval"][mask] if name == "constant_top_k" else None
                    result = self.evaluate(name, mask=mask, predictions=prediction)
                    groups[label]["metrics"][name] = {k: result[k] for k in ("map", "macro_f1")}
                for variant in self.config["variants"]:
                    rows = [groups[label]["metrics"][f"{variant}_seed{s}"] for s in self.config["seeds"]]
                    groups[label]["metrics"][variant] = {k: float(np.mean([r[k] for r in rows])) for k in ("map", "macro_f1")}
        self.groups = groups
        self.write("people_group_metrics.json", groups)
        rows = []
        for i, label in enumerate(EMOTION_LABELS):
            ap = np.mean([self.results[f"full_seed{s}", "extended"]["per_class"][i]["ap"] for s in self.config["seeds"]])
            ref = self.results["person_context", "extended"]["per_class"][i]["ap"]
            rows.append({"class_name": label, "support_dev": int(self.targets["dev"][:, i].sum()),
                "support_eval": int(self.targets["eval"][:, i].sum()), "ap_full_mean": ap,
                "ap_resnet_c": ref, "delta_full_minus_resnet_c": ap - ref})
        self.ap_delta = rows
        self.csv("per_class_ap_delta.csv", rows)
        a = f"full_seed{self.config['seeds'][0]}"
        # Fixed seed is specified for error inspection; never select it on eval.
        pa = self.probs["eval"][a] >= self.thresholds[a]["extended"]
        pb = self.probs["eval"]["person_context"] >= self.thresholds["person_context"]["extended"]
        y = self.targets["eval"]
        def sample_f1(pred):
            den = y.sum(1) + pred.sum(1)
            return np.divide(2 * (y * pred).sum(1), den, out=np.zeros(len(y)), where=den > 0)
        delta = sample_f1(pa) - sample_f1(pb)
        examples = {"full_model": a, "selection_rule": "sample-F1 delta, then sample_id; strict direction"}
        for name, sign in (("full_better", 1), ("resnet_better", -1)):
            indices = sorted([i for i in range(len(y)) if sign * delta[i] > 0],
                             key=lambda i: (-sign * delta[i], self.splits["eval"][i]))[:self.config["error_examples_per_direction"]]
            examples[name] = [{"sample_id": self.splits["eval"][i], "image_id": self.images["eval"][i],
                "target": y[i], "full_prediction": pa[i], "resnet_prediction": pb[i]} for i in indices]
        self.write("error_examples.json", examples)

    def figures(self):
        """Export summary bars, bootstrap intervals, AP deltas and crowd-size plots."""
        import matplotlib.pyplot as plt
        main = [r for r in self.rows if r["grid"] == "extended" and
                (r.get("summary") or r["model"] in (*RESNETS, "constant_top_k", "full_ensemble", "late_fusion"))]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, metric in zip(axes, ("map", "macro_f1")):
            ax.barh([r["model"] for r in main], [r[metric] for r in main],
                    xerr=[r.get(metric + "_sd", 0) for r in main])
            ax.set_title(f"Eval — {metric} (SD qua seed)")
        fig.tight_layout(); fig.savefig(self.out / "figures/comparison.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(18, 13))
        for ax, metric in zip(axes, ("map", "macro_f1")):
            rows = [r for r in self.boot_results if r["metric"] == metric]
            for i, row in enumerate(rows):
                ax.plot([row["ci95_lower"], row["ci95_upper"]], [i, i], color="steelblue")
                ax.plot(row["difference_a_minus_b"], i, "o", color="navy")
            ax.set_yticks(range(len(rows)), [r["a"] + " − " + r["b"] for r in rows], fontsize=7)
            ax.axvline(0, color="grey", linestyle="--"); ax.set_title(metric)
        fig.tight_layout(); fig.savefig(self.out / "figures/bootstrap.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh([r["class_name"] for r in self.ap_delta], [r["delta_full_minus_resnet_c"] for r in self.ap_delta])
        ax.axvline(0, color="grey"); ax.set_title("AP full trung bình seed − ResNet C")
        fig.tight_layout(); fig.savefig(self.out / "figures/ap_delta.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric in zip(axes, ("map", "macro_f1")):
            for name in (*RESNETS, *self.config["variants"], "constant_top_k"):
                ax.plot(list(self.groups), [g["metrics"].get(name, {}).get(metric, np.nan) for g in self.groups.values()], label=name)
            ax.set_title(metric); ax.set_xlabel("Số người trong ảnh")
        axes[-1].legend(fontsize=7); fig.tight_layout()
        fig.savefig(self.out / "figures/people_groups.png", dpi=160); plt.close(fig)
        print("Hoàn tất Notebook 05:", self.out, flush=True)


__all__ = ["ComparisonStudy", "required_inputs", "SOURCE_RUN", "RESNET_RUN"]
