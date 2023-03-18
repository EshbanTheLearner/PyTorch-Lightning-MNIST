from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

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