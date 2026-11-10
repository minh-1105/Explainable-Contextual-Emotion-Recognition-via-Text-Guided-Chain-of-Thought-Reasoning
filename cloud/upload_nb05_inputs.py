"""Upload only the existing artifacts needed by Notebook 05, without overwriting."""
import argparse
from pathlib import Path

import modal

SOURCE_FILES = ("source_artifacts.json", "config.json", "environment.txt",
                "samples.json", "metrics.json", "predictions_eval.npz", "comparison.json")
MODES = ("person_only", "context_only", "person_context")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--volume", default="emotic")
    args = parser.parse_args()
    pairs = []
    run = Path("outputs/qwen3_multimodal/20261104-071259-h100")
    pairs += [(args.source / run / name, "/" + (run / name).as_posix()) for name in SOURCE_FILES]
    for mode in MODES:
        candidates = [p for base in ("models", "checkpoints")
                      for p in (args.source / base / f"emotic_{mode}_resnet50").glob("*")
                      if (p / "model.pt").is_file()]
        if not candidates:
            raise FileNotFoundError(f"Missing export: {mode}")
        export = sorted(candidates, key=lambda p: (p.name, str(p)))[-1]
        pairs += [(export / name, "/" + (export / name).relative_to(args.source).as_posix())
                  for name in ("model.pt", "config.json", "label_map.json")]
    for local, _ in pairs:
        if not local.is_file():
            raise FileNotFoundError(local)
    print(f"Uploading {len(pairs)} files ({sum(p.stat().st_size for p, _ in pairs):,} bytes)", flush=True)
    volume = modal.Volume.from_name(args.volume, create_if_missing=False)
    with volume.batch_upload(force=False) as batch:
        for local, remote in pairs:
            batch.put_file(str(local), remote)
    print("Uploaded Notebook 05 inputs.", flush=True)


if __name__ == "__main__":
    main()
