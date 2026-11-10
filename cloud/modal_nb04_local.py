"""Run an auditable local NB04 snapshot on the existing emotic volume."""
import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "04_VLM_Pilot.ipynb"
SOURCE_RUN = "outputs/qwen3_multimodal/20261104-071259-h100"
VOLUME = os.environ.get("NB04_VOLUME", "emotic")


def manifest():
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "notebooks" / NOTEBOOK, ROOT / "requirements.txt", Path(__file__).resolve()]
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"source_mode": "local_snapshot", "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


SNAPSHOT = manifest() if modal.is_local() else json.loads(os.environ["NB04_SNAPSHOT"])
app = modal.App("emotic-nb04")
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
small_image = modal.Image.debian_slim(python_version="3.12").env({"NB04_SNAPSHOT": json.dumps(SNAPSHOT)})
image = (
    small_image.apt_install("git")
    .pip_install("torch==2.11.0", "torchvision", "transformers==5.22.0", "accelerate",
                 "numpy>=1.24,<3", "scipy>=1.10,<2", "Pillow>=10,<13", "matplotlib>=3.7,<4",
                 "PyYAML>=6,<7", "scikit-learn>=1.3,<2", "tqdm>=4.66,<5",
                 "papermill", "ipykernel", "nbformat")
    .env({"HF_HOME": "/vol/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "TQDM_MININTERVAL": "60", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4"})
    .add_local_dir(str(ROOT / "src"), "/repo/src", ignore=["**/__pycache__/**"])
    .add_local_file(str(ROOT / "notebooks" / NOTEBOOK), "/repo/notebooks/" + NOTEBOOK)
) if modal.is_local() else small_image


@app.function(image=small_image, volumes={"/vol": volume}, timeout=120)
def preflight(run_id: str = "", run_test: bool = False):
    root = Path("/vol")
    required = [f"outputs/multimodal_pipeline/{run_id}/config.json"] if run_id else []
    required += ["data/raw/raw/emotic/PAMI/Annotations/Annotations/Annotations.mat",
                "data/raw/raw/emotic/PAMI/pami_emotic.zip", "data/raw/raw/emotic/CVPR/cvpr_emotic.zip"]
    required += [f"{SOURCE_RUN}/{name}" for name in (
        "config.json", "prompts.json", "samples.json", "qwen_cues.jsonl", "qwen_cues_test.jsonl",
        "test_results.json", "evidence_packages.json", "explanations.jsonl")]
    if run_test:
        if not run_id:
            raise ValueError("Test requires an existing finalized run_id.")
        config_path = root / f"outputs/multimodal_pipeline/{run_id}/config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            resnet = config["resnet_run"]
            required.append(f"{resnet}/metrics/thresholds.json")
            required += [f"{resnet}/predictions/{name}_test.npz" for name in
                         ("person_only", "context_only", "person_context")]
            required += [f"outputs/multimodal_pipeline/{run_id}/train/checkpoints/{name}_seed{seed}.pt"
                         for name in config["variants"] for seed in config["seeds"]]
    return {"volume": VOLUME, "missing": [p for p in required if not (root / p).is_file()]}


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=24 * 3600,
              volumes={"/vol": volume}, max_containers=1)
def run_notebook(run_id: str = "", run_test: bool = False):
    import logging
    import sys
    import threading
    import time
    from datetime import datetime, timezone

    import nbformat
    import papermill

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path("/vol/outputs/modal_executions") / (stamp + "-local")
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    nb = nbformat.read("/repo/notebooks/" + NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert "drive.mount" in code[0].source
    code[0].source = ("from pathlib import Path\nimport subprocess, sys\n"
                      "DRIVE_ROOT = Path('/vol')\nCODE_ROOT = Path('/repo')\n"
                      "sys.path.insert(0, str(CODE_ROOT))\n")
    original = ('GIT_COMMIT = subprocess.run(\n'
                '    ["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],\n'
                '    check=True, capture_output=True, text=True,\n).stdout.strip()')
    assert original in code[1].source
    code[1].source = code[1].source.replace(original, f"GIT_COMMIT = {'snapshot-sha256:' + SNAPSHOT['sha256']!r}")
    # This account has cues but no H100 feature caches. Re-encode rather than claim a cache import.
    old = '"feature_import_from": H100_RUN,'
    assert code[1].source.count(old) == 1
    code[1].source = code[1].source.replace(old, '"feature_import_from": None,')
    assert "RUN_ID = None" in code[1].source
    assert "RUN_TEST = False" in code[1].source
    # Resume an existing run (same PIPELINE_CONFIG) and optionally score the test set.
    code[1].source = code[1].source.replace("RUN_ID = None", f"RUN_ID = {run_id or None!r}", 1)
    code[1].source = code[1].source.replace("RUN_TEST = False", f"RUN_TEST = {run_test!r}", 1)
    # Keep the snapshot of every code version that touched the run.
    snapshot_name = f"code_snapshot_{SNAPSHOT['sha256'][:12]}.json" if run_id else "code_snapshot.json"
    code[1].source += (
        f"\nwrite_json(RUN_DIR / {snapshot_name!r}, {SNAPSHOT!r})\n"
        f"write_json(Path({str(out / 'pipeline_run.json')!r}), {{'run_id': RUN_ID}})\n"
    )
    for cell in code[1:]:
        cell.source = cell.source.replace('Path("/content/emotic")', 'Path("/tmp/emotic")')
        # Compare metrics even when the legacy caches are unavailable.
        cell.source = cell.source.replace(
            'if feature_source is None or not h100_results.is_file():', 'if not h100_results.is_file():')
    assert not any("google.colab" in c.source or "/content" in c.source for c in code)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, out / "input.ipynb")
    (out / "code_snapshot.json").write_text(json.dumps(SNAPSHOT, indent=2))
    status = {"state": "running", "run_id": run_id or None, "snapshot_sha256": SNAPSHOT["sha256"],
              "feature_import_from": None, "run_test": run_test}
    status_path = out / "status.json"
    status_path.write_text(json.dumps(status, indent=2))
    volume.commit()
    print(f"Execution directory: {out}", flush=True)

    def sync():
        while True:
            time.sleep(300)
            volume.commit()

    threading.Thread(target=sync, daemon=True).start()
    try:
        papermill.execute_notebook(str(out / "input.ipynb"), str(out / NOTEBOOK),
                                   kernel_name="python3", cwd="/repo", log_output=True,
                                   progress_bar=False, request_save_on_cell_execute=True)
        status["state"] = "completed"
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        link = out / "pipeline_run.json"
        if link.is_file():
            status.update(json.loads(link.read_text()))
        status_path.write_text(json.dumps(status, indent=2))
        volume.commit()
    return {"executed_notebook": str(out / NOTEBOOK), **status}


@app.local_entrypoint()
def main(check_only: bool = False, run_id: str = "", run_test: bool = False):
    report = preflight.remote(run_id, run_test)
    print(json.dumps(report, indent=2), flush=True)
    if report["missing"]:
        raise FileNotFoundError("Missing inputs; no GPU allocated.")
    if not check_only:
        print(run_notebook.remote(run_id, run_test))
