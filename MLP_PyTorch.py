import torch
from torch import nn

class PyTorchMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers = nn.Sequential(
            nn.Linear(num_features, 50),
            nn.ReLU(),

            nn.Linear(50, 25),
            nn.ReLU(),

            nn.Linear(25, num_classes)
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits