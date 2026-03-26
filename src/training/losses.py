import torch
import torch.nn as nn
#verwenden wir das überhaupt???

class Loss(nn.Module):
    def __init__(self):
        #initialize BCEWithLogitsLoss
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    #forward method
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)