import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self):
        #initialize BCEWithLogitsLoss
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    #forward method
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)