"""Frozen image backbones and the frozen BERT node encoder used by Notebook 04.

``vit_b_16_imagenet1k_v1`` reproduces the ViT features of Notebook 04. The CLIP
and SigLIP backbones return the pooled output of the vision tower.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from src.reasoning.qwen_cues import NODE_TYPE

IMAGE_BACKBONES = {
    "vit_b_16_imagenet1k_v1": {"library": "torchvision"},
    "clip_vit_b_16": {"library": "transformers", "model_id": "openai/clip-vit-base-patch16"},
    "siglip_base_patch16_224": {"library": "transformers", "model_id": "google/siglip-base-patch16-224"},
}


class FrozenImageEncoder:
    """Map a list of PIL images to a float32 array of pooled features."""

    def __init__(self, name: str, device: str = "cuda"):
        if name not in IMAGE_BACKBONES:
            raise ValueError(f"Unknown backbone {name}; choose one of {sorted(IMAGE_BACKBONES)}")
        self.name, self.device = name, device
        if name == "vit_b_16_imagenet1k_v1":
            from torchvision.models import ViT_B_16_Weights, vit_b_16

            weights = ViT_B_16_Weights.IMAGENET1K_V1
            transform = weights.transforms()
            model = vit_b_16(weights=weights)
            model.heads = nn.Identity()
            self._preprocess = lambda images: torch.stack([transform(image) for image in images])
            self._forward = model
        else:
            model_id = IMAGE_BACKBONES[name]["model_id"]
            if name.startswith("clip"):
                from transformers import CLIPImageProcessor as Processor, CLIPVisionModel as Model
            else:
                from transformers import SiglipImageProcessor as Processor, SiglipVisionModel as Model
            processor = Processor.from_pretrained(model_id)
            model = Model.from_pretrained(model_id)
            self._preprocess = lambda images: processor(images=list(images), return_tensors="pt")["pixel_values"]
            self._forward = lambda pixels: model(pixel_values=pixels).pooler_output
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, images: Sequence) -> np.ndarray:
        return self._forward(self._preprocess(images).to(self.device)).float().cpu().numpy()

    def close(self) -> None:
        del self.model, self._forward
        torch.cuda.empty_cache()


def encode_person_context(
    load_item: Callable[[int], dict],
    indices: Sequence[int],
    encoder: FrozenImageEncoder,
    batch_size: int = 64,
    workers: int = 8,
    progress: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Encode the person crop and the full image of each dataset index."""
    person, context, labels = [], [], []
    starts = range(0, len(indices), batch_size)
    with ThreadPoolExecutor(workers) as pool:
        for start in (progress(starts) if progress else starts):
            items = list(pool.map(load_item, indices[start:start + batch_size]))
            person.append(encoder([item["person_image"] for item in items]))
            context.append(encoder([item["context_image"] for item in items]))
            labels.extend(item["labels"].numpy() for item in items)
    return {
        "person_features": np.concatenate(person),
        "context_features": np.concatenate(context),
        "labels": np.asarray(labels, dtype=np.float32),
    }


@torch.inference_mode()
def encode_node_lists(
    node_lists: Sequence[Sequence[tuple[str, str]]],
    model_id: str,
    max_nodes: int,
    max_length: int,
    batch_size: int = 256,
    device: str = "cuda",
    progress: Callable | None = None,
) -> dict[str, np.ndarray]:
    """BERT [CLS] embedding of every node; padded to ``max_nodes`` with a mask."""
    from transformers import BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    model = BertModel.from_pretrained(model_id).to(device).eval()
    flat = [(row, column, kind, text) for row, nodes in enumerate(node_lists) for column, (kind, text) in enumerate(nodes)]
    features = np.zeros((len(node_lists), max_nodes, model.config.hidden_size), dtype=np.float32)
    mask = np.zeros((len(node_lists), max_nodes), dtype=np.bool_)
    types = np.zeros_like(mask, dtype=np.int64)
    starts = range(0, len(flat), batch_size)
    for start in (progress(starts) if progress else starts):
        batch = flat[start:start + batch_size]
        tokens = tokenizer([entry[3] for entry in batch], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
        embedding = model(**tokens).last_hidden_state[:, 0].float().cpu().numpy()
        for (row, column, kind, _), vector in zip(batch, embedding):
            features[row, column] = vector
            mask[row, column] = True
            types[row, column] = NODE_TYPE[kind]
    del model
    torch.cuda.empty_cache()
    return {"node_features": features, "node_mask": mask, "node_types": types}


__all__ = ["FrozenImageEncoder", "IMAGE_BACKBONES", "encode_node_lists", "encode_person_context"]
