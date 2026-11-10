"""Split PAMI val into dev/eval by image, as Notebook 04 does."""

from __future__ import annotations

import random
from collections import defaultdict


def split_by_image(train_dataset, val_dataset, seed: int, dev_fraction: float) -> dict:
    """Return the ``samples.json`` payload of Notebook 04 for full train and val.

    Image paths are sorted, shuffled with ``seed`` and the first ``dev_fraction``
    go to dev, so every person of one image lands in the same partition.
    """
    indices_by_image = defaultdict(list)
    for index, sample in enumerate(val_dataset.samples):
        indices_by_image[sample.relative_image_path].append(index)
    image_ids = sorted(indices_by_image)
    random.Random(seed).shuffle(image_ids)
    dev_end = round(len(image_ids) * dev_fraction)
    parts = {"dev": image_ids[:dev_end], "eval": image_ids[dev_end:]}
    splits = {"train": [sample.sample_id for sample in train_dataset.samples]}
    for name, images in parts.items():
        splits[name] = [val_dataset.samples[index].sample_id for image in images for index in indices_by_image[image]]
    sample_ids = splits["train"] + [sample.sample_id for sample in val_dataset.samples]
    assert len(sample_ids) == len(set(sample_ids))
    assert set(splits["dev"]) | set(splits["eval"]) == {sample.sample_id for sample in val_dataset.samples}
    return {
        "sample_ids": sample_ids,
        "splits": splits,
        "counts": {
            "train_people": len(splits["train"]),
            "dev_people": len(splits["dev"]),
            "eval_people": len(splits["eval"]),
            "dev_images": len(parts["dev"]),
            "eval_images": len(parts["eval"]),
        },
        "validation_split_unit": "relative_image_path",
    }


__all__ = ["split_by_image"]
