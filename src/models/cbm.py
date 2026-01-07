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


class cbm_model(nn.Module):
    # CBM Model bestehend aus Concept Predictor und Label Predictor
    #binary_concepts: ob die Konzepte binär sind, binary_treshold: Schwellenwert für die Binarisierung der Konzepte
    def __init__(self, concept_predictor: nn.Module, label_predictor: nn.Module, binarization_threshold: float = 0.5, binary_concepts: bool = False):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor
        self.binarization_threshold = binarization_threshold
        self.binary_concepts = binary_concepts
    
    def forward(self, images):
        #concept prediction
        concepts = self.concept_predictor(images)
        #concept activation
        concept_probs = torch.sigmoid(concepts)
        #binarization der Konzepte wenn binary_concepts True ist
        if self.binary_concepts:
            concept_probs = (concept_probs >= self.binarization_threshold).float()
            concepts = concept_probs
        else:
            concepts = concept_probs
        #label prediction
        label_logits = self.label_predictor(concepts)

        return label_logits, concept_probs, concepts


ConceptBottleneckModel = cbm_model
