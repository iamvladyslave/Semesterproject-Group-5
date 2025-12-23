import torch
import torch.nn as nn


class LabelPredictor(nn.Module):

    def __init__(
        self,
        num_concepts: int,
        num_classes: int = 43,
        hidden_dims=(256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = num_concepts
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.net(concepts)