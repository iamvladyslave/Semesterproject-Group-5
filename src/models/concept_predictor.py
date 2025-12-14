from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ConceptBackboneConfig:
    """Configuration for building the EfficientNetV2 backbone."""

    name: str = "efficientnet_v2_s"
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False


class ConceptPredictor(nn.Module):
    """
    Concept prediction head on top of a pretrained EfficientNetV2 backbone.

    The backbone's classifier is replaced with a linear layer producing
    `num_concepts` logits. Sigmoid activation is applied in loss/metric code.
    """

    def __init__(
        self,
        num_concepts: int,
        backbone_cfg: Optional[ConceptBackboneConfig] = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        backbone_cfg = backbone_cfg or ConceptBackboneConfig()

        self.backbone = self._build_backbone(backbone_cfg)
        if backbone_cfg.freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def _build_backbone(self, cfg: ConceptBackboneConfig) -> nn.Module:
        weights = None
        if cfg.pretrained:
            # TorchVision API maps weights to enum types; fall back gracefully if unavailable.
            weight_attr = f"{cfg.name.upper()}_Weights"
            if hasattr(models, weight_attr):
                weights_enum = getattr(models, weight_attr)
                weights = getattr(weights_enum, "IMAGENET1K_V1", None) or getattr(
                    weights_enum, "DEFAULT", None
                )
        constructor = getattr(models, cfg.name)
        backbone = constructor(weights=weights)

        # EfficientNetV2 classifier is a Sequential with Dropout + Linear.
        if not hasattr(backbone, "classifier"):
            raise AttributeError(f"Backbone {cfg.name} missing classifier attribute")
        classifier = backbone.classifier
        if not isinstance(classifier, nn.Sequential) or not classifier:
            raise ValueError(f"Unexpected classifier format for {cfg.name}: {classifier}")

        in_features = classifier[-1].in_features  # type: ignore[arg-type]
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features, self.num_concepts),
        )
        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
