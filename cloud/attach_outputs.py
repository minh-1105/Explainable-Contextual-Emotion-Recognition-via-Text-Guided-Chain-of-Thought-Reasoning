import copy
import json
import sys
from pathlib import Path

import nbformat

from modal_nb04 import NOTEBOOK, modal_notebook

executed_path = Path(sys.argv[1])
repo_path = Path(__file__).resolve().parent.parent / NOTEBOOK
repo = json.loads(repo_path.read_text(encoding="utf-8"))
executed = json.loads(executed_path.read_text(encoding="utf-8"))
expected = modal_notebook(nbformat.from_dict(copy.deepcopy(repo)))

repo_code = [cell for cell in repo["cells"] if cell["cell_type"] == "code"]
executed_code = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
expected_code = [cell for cell in expected.cells if cell.cell_type == "code"]
assert len(repo_code) == len(executed_code) == len(expected_code)
for number, (target, run, reference) in enumerate(zip(repo_code, executed_code, expected_code)):
    assert "".join(run["source"]) == reference.source, f"code cell {number + 1} differs from the executed notebook"
    if number < 2:
        target["outputs"], target["execution_count"] = [], None
        continue
    assert not any(output["output_type"] == "error" for output in run["outputs"]), f"code cell {number + 1} failed"
    target["outputs"] = run["outputs"]
    target["execution_count"] = run["execution_count"]
    target.setdefault("metadata", {})["papermill"] = run["metadata"]["papermill"]
repo["metadata"]["papermill"] = executed["metadata"]["papermill"]
repo_path.write_text(json.dumps(repo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("attached outputs of", len(repo_code) - 2, "code cells to", repo_path)
