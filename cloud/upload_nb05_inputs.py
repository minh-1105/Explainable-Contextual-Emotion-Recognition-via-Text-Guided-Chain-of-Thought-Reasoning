"""Upload only the existing artifacts needed by Notebook 05, without overwriting."""
import argparse
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.multimodal_comparison import required_inputs, SOURCE_RUN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--volume", default="emotic")
    args = parser.parse_args()
    if not (args.source / SOURCE_RUN / "config.json").is_file():
        raise FileNotFoundError("Missing source pipeline config; export the new 04 artifacts first.")
    pairs = [(args.source / name, "/" + name) for name in required_inputs(args.source)]
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
