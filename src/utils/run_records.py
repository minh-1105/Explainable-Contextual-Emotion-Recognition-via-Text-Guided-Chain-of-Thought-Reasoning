"""Run directories, config checks, environment records and hashes for notebook 04."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def open_run(output_root: Path, run_id: str | None, config: dict) -> tuple[str, Path]:
    """Create a new run (``run_id=None``) or reopen one whose config matches.

    ``git_commit`` is the only key allowed to differ on resume.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_root) / run_id
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config)
        return run_id, run_dir
    run_dir = Path(output_root) / run_id
    saved = read_json(run_dir / "config.json")
    strip = lambda payload: {key: value for key, value in payload.items() if key != "git_commit"}
    assert strip(saved) == strip(config), "Config khác run cũ; hãy tạo run mới (RUN_ID = None)."
    return run_id, run_dir


def record_environment(run_dir: Path, git_commit: str, extra: dict[str, str]) -> None:
    """Append the environment to ``environment.txt`` the first time a commit runs."""
    path = Path(run_dir) / "environment.txt"
    text = "".join(f"{key}={value}\n" for key, value in extra.items()) + f"git_commit={git_commit}\n"
    if not path.is_file():
        path.write_text(text, encoding="utf-8")
    elif f"git_commit={git_commit}\n" not in path.read_text(encoding="utf-8"):
        with path.open("a", encoding="utf-8") as file:
            file.write(text)


def check_or_write_json(path: Path, payload) -> None:
    """Write ``payload`` once; later runs must produce the same content."""
    path = Path(path)
    if path.is_file():
        assert read_json(path) == payload, f"{path.name} khác lần chạy trước."
    else:
        write_json(path, payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["check_or_write_json", "open_run", "read_json", "record_environment", "sha256", "write_json"]
