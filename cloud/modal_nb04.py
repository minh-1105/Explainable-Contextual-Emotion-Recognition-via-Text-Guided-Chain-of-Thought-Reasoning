import modal

REPO_URL = "https://github.com/minh-1105/Explainable-Contextual-Emotion-Recognition-via-Text-Guided-Chain-of-Thought-Reasoning.git"
NOTEBOOK = "notebooks/04_VLM_Pilot.ipynb"

volume = modal.Volume.from_name("ecer-nb04")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0", "torchvision", "transformers==5.22.0", "accelerate",
        "numpy>=1.24,<3", "scipy>=1.10,<2", "Pillow>=10,<13", "matplotlib>=3.7,<4",
        "PyYAML>=6,<7", "scikit-learn>=1.3,<2", "tqdm>=4.66,<5",
        "papermill", "ipykernel", "nbformat",
    )
    .env({"HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false", "TQDM_MININTERVAL": "60"})
)
app = modal.App("ecer-nb04", image=image)

PRELUDE = '''from pathlib import Path
import subprocess
import sys

DRIVE_ROOT = Path("/vol")
CODE_ROOT = Path("/repo")
sys.path.insert(0, str(CODE_ROOT))
print("Modal runtime: Drive root", DRIVE_ROOT, "| code root", CODE_ROOT)
'''


def modal_notebook(notebook):
    """Point a Colab notebook (04 by default) at the volume instead of Drive and /content."""
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "drive.mount" in code[0].source and "CODE_ROOT" in code[0].source
    code[0].source = PRELUDE
    for cell in code[1:]:
        if cell.source.strip().startswith("from google.colab import drive"):
            cell.source = ""
        cell.source = cell.source.replace('Path("/content/emotic")', 'Path("/tmp/emotic")')
    assert sum('Path("/tmp/emotic")' in cell.source for cell in code) == 1
    assert not any("/content" in cell.source or "google.colab" in cell.source for cell in code)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


@app.function(gpu="H100", cpu=8, memory=65536, timeout=24 * 3600, volumes={"/vol": volume},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run(branch: str = "main", notebook: str = NOTEBOOK):
    import logging
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    import nbformat
    import papermill

    started_at = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = Path("/vol/outputs/modal_executions") / started_at
    out_dir.mkdir(parents=True)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    subprocess.run(["git", "clone", "--branch", branch, REPO_URL, "/repo"], check=True)
    commit = subprocess.run(["git", "-C", "/repo", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    logging.info("branch %s commit %s notebook %s", branch, commit, notebook)
    source = out_dir / "input.ipynb"
    nbformat.write(modal_notebook(nbformat.read(f"/repo/{notebook}", as_version=4)), source)

    def commit_loop():
        while True:
            time.sleep(300)
            volume.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    executed = out_dir / Path(notebook).name
    try:
        papermill.execute_notebook(
            str(source), str(executed), kernel_name="python3", cwd="/repo",
            log_output=True, progress_bar=False, request_save_on_cell_execute=True,
        )
    finally:
        volume.commit()
    logging.info("executed notebook: %s", executed)
