from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

from collections import Counter

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

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

print("\nTraining Label Distribution:")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\Validation Label Distribution:")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\Test Label Distribution:")
print(sorted(test_counter.items()))