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
    .env({"HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false", "TQDM_MININTERVAL": "60"})
)


class TimestampedTee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.line_start = True

    def write(self, text):
        import datetime

        self.stream.write(text)
        for piece in text.replace("\r", "\n").splitlines(keepends=True):
            content = piece.rstrip("\n")
            if content.strip() or content and not self.line_start:
                if self.line_start:
                    self.log_file.write(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ "))
                self.log_file.write(content)
                self.line_start = False
            if piece.endswith("\n") and not self.line_start:
                self.log_file.write("\n")
                self.line_start = True
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return False
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
    import shutil
    import subprocess
    import sys
    import threading
    import time

    started_at = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log_path = f"/tmp/modal_{started_at}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = TimestampedTee(sys.__stdout__, log_file)
    sys.stderr = TimestampedTee(sys.__stderr__, log_file)
    subprocess.run(["git", "clone", "--branch", branch, REPO_URL, "/repo"], check=True)
    notebook = json.load(open("/repo/notebooks/04_VLM_Pilot.ipynb", encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "drive.mount" in cells[0] and "CODE_ROOT" in cells[0]
    assert cells[1].strip().startswith("from google.colab import drive")
    assert cells[2].count('Path("/content/emotic")') == 1
    cells = [PRELUDE, cells[2].replace('Path("/content/emotic")', 'Path("/tmp/emotic")')] + cells[3:]
    assert not any("/content" in cell or "google.colab" in cell for cell in cells)

    namespace = {"__name__": "__main__"}

    def save_log():
        if "RUN_DIR" in namespace:
            log_file.flush()
            target = namespace["RUN_DIR"] / "logs"
            target.mkdir(exist_ok=True)
            shutil.copyfile(log_path, target / f"modal_{started_at}.log")
        volume.commit()

    def commit_loop():
        while True:
            time.sleep(300)
            save_log()

    threading.Thread(target=commit_loop, daemon=True).start()
    try:
        for number, source in enumerate(cells):
            started = time.time()
            print(f"\n===== code cell {number + 1}/{len(cells)} =====\n{source.splitlines()[0][:120]}", flush=True)
            exec(compile(source, f"<cell {number + 1}>", "exec"), namespace)
            print(f"===== cell {number + 1} xong sau {time.time() - started:.0f}s =====", flush=True)
            save_log()
    except BaseException:
        import traceback

        traceback.print_exc()
        save_log()
        raise
