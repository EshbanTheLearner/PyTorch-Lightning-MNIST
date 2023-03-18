import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

def get_data():
    train_data = datasets.MNIST(
        root="./mnist",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root="./mnist",
        train=False,
        transform=transforms.ToTensor()
    )

    train_data, val_data = random_split(train_data, lengths=[55000, 5000])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def compute_accuracy(model, dataloader, device=None):
    if device is None:
        device = torch.device("cpu")
    
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (fetures, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predicitions = torch.argmax(logits, dim=1)
        compare = labels == predicitions
        correct += torch.sum(compare)
        total_examples += len(compare)
    
    return correct / total_examples