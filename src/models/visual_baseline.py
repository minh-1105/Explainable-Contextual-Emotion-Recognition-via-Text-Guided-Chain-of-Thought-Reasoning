"""Checkpoint-compatible ResNet baselines used in Notebook 03."""
import torch
from torch import nn
from torchvision.models import resnet50

EXPERIMENTS = ("person_only", "context_only", "person_context")

def make_encoder(freeze=False):
    backbone = resnet50(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    if freeze:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    return backbone, feature_dim


class VisualBaseline(nn.Module):
    def __init__(self, mode, num_classes=26, freeze_backbone=False, dropout=0.3):
        super().__init__()
        if mode not in EXPERIMENTS:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        if mode in {"person_only", "person_context"}:
            self.person_encoder, person_dim = make_encoder(freeze_backbone)
        if mode in {"context_only", "person_context"}:
            self.context_encoder, context_dim = make_encoder(freeze_backbone)
        input_dim = (
            person_dim if mode == "person_only" else
            context_dim if mode == "context_only" else
            person_dim + context_dim
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, person_image, context_image):
        if self.mode == "person_only":
            features = self.person_encoder(person_image)
        elif self.mode == "context_only":
            features = self.context_encoder(context_image)
        else:
            features = torch.cat(
                (self.person_encoder(person_image), self.context_encoder(context_image)),
                dim=1,
            )
        return self.classifier(features)
