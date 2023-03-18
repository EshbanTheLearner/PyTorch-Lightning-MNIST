import torch
import torch.nn.functional as F
from MLP_PyTorch import PyTorchMLP
from utils import get_data, compute_accuracy
from watermark import watermark


def compute_total_loss(model, dataloader, device=None):
    if device is None:
        device = torch.device(device)
    
    model = model.eval()
    loss = 0.0
    examples = 0.0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
            batch_loss = F.cross_entropy(logits, labels, reduction="sum")
        loss += batch_loss.item()
        examples += logits.shape[0]
    
    return loss/examples

def train(model, optimizer, train_loader, val_loader, num_epochs=10, seed=42, device=None):
    if device is None:
        device = torch.device(device)
    
    torch.manual_seed(seed)

    for epoch in range(num_epochs):
        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not batch_idx % 250:
                val_loss = compute_total_loss(model, val_loader, device=device)
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:.03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train Batch Loss: {loss:.4f}"
                    f" | Val Total Loss: {val_loss:.4f}"
                )

