import modal

REPO_URL = "https://github.com/minh-1105/Explainable-Contextual-Emotion-Recognition-via-Text-Guided-Chain-of-Thought-Reasoning.git"

volume = modal.Volume.from_name("ecer-nb04")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0", "torchvision", "transformers==5.22.0", "accelerate",
        "numpy>=1.24,<3", "scipy>=1.10,<2", "Pillow>=10,<13", "matplotlib>=3.7,<4",
        "PyYAML>=6,<7", "scikit-learn>=1.3,<2", "tqdm>=4.66,<5",
    )
    .env({"HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("ecer-nb04", image=image)

PRELUDE = '''
from pathlib import Path
import subprocess
import sys
DRIVE_ROOT = Path("/vol")
CODE_ROOT = Path("/repo")
sys.path.insert(0, str(CODE_ROOT))
'''


@app.function(gpu="H100", cpu=8, memory=65536, timeout=24 * 3600, volumes={"/vol": volume},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run(branch: str = "main"):
    import json
    import subprocess
    import threading
    import time

    subprocess.run(["git", "clone", "--branch", branch, REPO_URL, "/repo"], check=True)
    notebook = json.load(open("/repo/notebooks/04_VLM_Pilot.ipynb", encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "drive.mount" in cells[0] and "CODE_ROOT" in cells[0]
    assert cells[1].strip().startswith("from google.colab import drive")
    assert cells[2].count('Path("/content/emotic")') == 1
    cells = [PRELUDE, cells[2].replace('Path("/content/emotic")', 'Path("/tmp/emotic")')] + cells[3:]
    assert not any("/content" in cell or "google.colab" in cell for cell in cells)

    def commit_loop():
        while True:
            time.sleep(300)
            volume.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    namespace = {"__name__": "__main__"}
    for number, source in enumerate(cells):
        started = time.time()
        print(f"\n===== code cell {number + 1}/{len(cells)} =====\n{source.splitlines()[0][:120]}", flush=True)
        exec(compile(source, f"<cell {number + 1}>", "exec"), namespace)
        volume.commit()
        print(f"===== cell {number + 1} xong sau {time.time() - started:.0f}s =====", flush=True)
