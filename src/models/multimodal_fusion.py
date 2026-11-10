"""Graph Transformer + Multimodal Transformer over frozen features (Notebook 04).

Parameter names match ``model.pt`` of run ``20261104-071259-h100``, so its
state dict loads with ``strict=True``.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.emotion_metrics import multilabel_metrics
from src.reasoning.qwen_cues import NODE_TYPE

TOKEN_INDEX = {"person": 1, "context": 2, "graph": 3}


class SemanticGraphTransformer(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_layers: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, model_dim)
        self.type_embedding = nn.Embedding(len(NODE_TYPE), model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=8, dim_feedforward=model_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features, mask, node_types):
        tokens = self.projection(features) + self.type_embedding(node_types)
        tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        return self.norm(pooled)


class EmotionMultimodalTransformer(nn.Module):
    def __init__(self, visual_dim, text_dim, model_dim, graph_layers, fusion_layers, n_classes):
        super().__init__()
        self.person_projection = nn.Linear(visual_dim, model_dim)
        self.context_projection = nn.Linear(visual_dim, model_dim)
        self.graph = SemanticGraphTransformer(text_dim, model_dim, graph_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.modality_embedding = nn.Parameter(torch.randn(1, 4, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=8, dim_feedforward=model_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(0.2), nn.Linear(model_dim, n_classes))

    def forward(self, person, context, nodes, node_mask, node_types, drop_modality=None, keep_only=None):
        """Zero the token of ``drop_modality``, or every modality token except ``keep_only``."""
        graph = self.graph(nodes, node_mask, node_types)
        batch_size = person.shape[0]
        tokens = torch.stack(
            [
                self.cls.expand(batch_size, -1, -1)[:, 0],
                self.person_projection(person),
                self.context_projection(context),
                graph,
            ],
            dim=1,
        ) + self.modality_embedding
        if drop_modality is not None:
            tokens[:, TOKEN_INDEX[drop_modality]] = 0
        if keep_only is not None:
            for name, index in TOKEN_INDEX.items():
                if name != keep_only:
                    tokens[:, index] = 0
        return self.head(self.fusion(tokens)[:, 0])


def build_model(config: dict, visual_dim: int, text_dim: int, n_classes: int) -> EmotionMultimodalTransformer:
    return EmotionMultimodalTransformer(
        visual_dim, text_dim, config["fusion_dim"], config["graph_layers"], config["fusion_layers"], n_classes,
    )


def drop_node_kinds(node_mask: torch.Tensor, node_types: torch.Tensor, kinds: Sequence[str]) -> torch.Tensor:
    """Mask out graph nodes of the given kinds; the situation node always remains."""
    assert "situation" not in kinds
    if not kinds:
        return node_mask
    dropped = torch.isin(node_types, torch.tensor([NODE_TYPE[kind] for kind in kinds]))
    return node_mask & ~dropped


def make_loader(tensors: Sequence[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    """Batches of (person, context, nodes, node_mask, node_types, labels)."""
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def positive_weight(train_targets: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    """negatives / positives per label, clipped to [1, max_weight]."""
    positives = train_targets.sum(0)
    negatives = len(train_targets) - positives
    weight = np.divide(negatives, positives, out=np.ones_like(negatives), where=positives > 0)
    return torch.from_numpy(np.clip(weight, 1.0, max_weight)).float()


@torch.inference_mode()
def predict_probabilities(model, loader, device="cuda", drop_modality=None, keep_only=None) -> np.ndarray:
    model.eval()
    return np.concatenate([
        model(*[value.to(device) for value in batch[:5]], drop_modality=drop_modality, keep_only=keep_only)
        .sigmoid().cpu().numpy()
        for batch in loader
    ])


def train_fusion(
    model,
    loaders: dict[str, DataLoader],
    dev_targets: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: torch.Tensor,
    drop_modality: str | None = None,
    class_names: Sequence[str] | None = None,
    device: str = "cuda",
    log: Callable[[dict], None] = print,
) -> tuple[dict, list[dict], float]:
    """Train with AdamW and weighted BCE; keep the epoch with the highest dev mAP.

    The model is left holding the best state. Returns that state, the per-epoch
    history and the best dev mAP. The caller seeds torch before building the
    model, as Notebook 04 did, so initialisation and shuffling are reproducible.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_map, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*[value.to(device) for value in batch[:5]], drop_modality=drop_modality)
            loss = criterion(logits, batch[5].to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        dev_scores = predict_probabilities(model, loaders["dev"], device, drop_modality)
        dev_map = multilabel_metrics(dev_targets, dev_scores, class_names=class_names)["map"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_map": dev_map,
               "seconds": round(time.time() - started, 1)}
        history.append(row)
        log(row)
        if dev_map > best_map:
            best_map = dev_map
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_state, history, best_map


__all__ = [
    "EmotionMultimodalTransformer", "SemanticGraphTransformer", "TOKEN_INDEX", "build_model", "drop_node_kinds",
    "make_loader", "positive_weight", "predict_probabilities", "train_fusion",
]
