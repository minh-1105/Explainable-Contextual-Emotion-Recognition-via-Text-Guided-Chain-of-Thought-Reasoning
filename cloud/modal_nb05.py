"""Run prediction-only Notebook 05 on CPU with an audited local snapshot."""
import hashlib
import json
import os
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "05_Compare_Multimodal_Baselines.ipynb"
VOLUME_NAME = os.environ.get("NB05_VOLUME", "emotic")
SOURCE_RUN = "outputs/multimodal_pipeline/20261110-074559"
RESNET_RUN = "outputs/multimodal_comparison/20261109-090105-378575"
SOURCE_FILES = ("config.json", "samples.json", "train/metrics.json", "environment.txt", "code_snapshot.json")


def snapshot_manifest():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "notebooks" / NOTEBOOK,
        ROOT / "requirements.txt", Path(__file__).resolve()]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"files": hashes, "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "source_mode": "local_snapshot"}


SNAPSHOT = snapshot_manifest() if modal.is_local() else json.loads(os.environ["NB05_SNAPSHOT"])
app = modal.App("emotic-nb05")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
base = (modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy>=1.24,<3", "scipy>=1.10,<2", "scikit-learn==1.9.1", "matplotlib>=3.7,<4",
                     "torch==2.11.0", "Pillow>=10,<13", "papermill", "ipykernel", "nbformat")
        .env({"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "NB05_SNAPSHOT": json.dumps(SNAPSHOT)}))
image = (base.add_local_dir(str(ROOT / "src"), "/repo/src", ignore=["**/__pycache__/**"])
         .add_local_file(str(ROOT / "notebooks" / NOTEBOOK), "/repo/notebooks/" + NOTEBOOK)
         .add_local_file(str(ROOT / "requirements.txt"), "/repo/requirements.txt")) if modal.is_local() else base


def inspect_inputs(root):
    from src.evaluation.multimodal_comparison import required_inputs
    names = required_inputs(root, SOURCE_RUN, RESNET_RUN)
    assert all("test" not in p.lower() for p in names)
    missing = [p for p in names if not (root / p).is_file()]
    return {"ready": not missing, "missing": missing, "source_run": SOURCE_RUN,
            "resnet_run": RESNET_RUN, "volume": VOLUME_NAME, "cpu_only": True}


@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def preflight():
    import sys
    sys.path.insert(0, "/repo")
    return inspect_inputs(Path("/vol"))


@app.function(image=image, cpu=8, memory=16384, volumes={"/vol": volume},
              timeout=24 * 3600, max_containers=1)
def run_notebook():
    import logging
    import sys
    import threading
    import time
    from datetime import datetime, timezone
    import nbformat
    import papermill

    sys.path.insert(0, "/repo")
    report = inspect_inputs(Path("/vol"))
    if not report["ready"]:
        raise FileNotFoundError(json.dumps(report))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    executed = Path("/vol/outputs/executed_notebooks") / f"nb05-{stamp}"
    executed.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(executed / "run.log", encoding="utf-8")])
    nb = nbformat.read("/repo/notebooks/" + NOTEBOOK, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert "drive.mount" in code[0].source and "CODE_ROOT" in code[0].source
    code[0].source = ("from pathlib import Path\nimport subprocess, sys\n"
                      "DRIVE_ROOT = Path('/vol')\nCODE_ROOT = Path('/repo')\n"
                      "sys.path.insert(0, str(CODE_ROOT))\n")
    original = 'GIT_COMMIT = subprocess.check_output(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"], text=True).strip()'
    assert code[1].source.count(original) == 1
    code[1].source = code[1].source.replace(original, f"GIT_COMMIT = {'snapshot-sha256:' + SNAPSHOT['sha256']!r}")
    code[1].source += (f"\n(RUN_DIR / 'code_snapshot.json').write_text({json.dumps(SNAPSHOT, indent=2)!r}, encoding='utf-8')\n"
        f"Path({str(executed / 'pipeline_run.json')!r}).write_text(json.dumps({{'run_dir': str(RUN_DIR)}}))\n")
    nbformat.write(nb, executed / "input.ipynb")
    (executed / "code_snapshot.json").write_text(json.dumps(SNAPSHOT, indent=2))
    status = {"state": "running", "snapshot_sha256": SNAPSHOT["sha256"], "cpu_only": True}
    def write_status():
        (executed / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
        volume.commit()
    write_status()
    def sync():
        while True:
            time.sleep(300)
            volume.commit()
    threading.Thread(target=sync, daemon=True).start()
    print("Execution:", executed, flush=True)
    try:
        papermill.execute_notebook(str(executed / "input.ipynb"), str(executed / NOTEBOOK),
            cwd="/repo", kernel_name="python3", log_output=True, progress_bar=False)
        status["state"] = "completed"
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        link = executed / "pipeline_run.json"
        if link.is_file():
            status.update(json.loads(link.read_text()))
        write_status()
    return {**status, "execution_dir": str(executed)}


@app.local_entrypoint()
def main(check_only: bool = False):
    report = preflight.remote()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["ready"]:
        raise FileNotFoundError("Thiếu prediction nguồn; chưa chạy notebook.")
    if not check_only:
        print(run_notebook.remote())
