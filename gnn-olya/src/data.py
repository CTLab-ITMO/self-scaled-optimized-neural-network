"""
Единый модуль загрузки и подготовки MNIST:
- Фиксированный seed
- Разделение: 50 000 train / 10 000 val / 10 000 test
- Одинаковый DataLoader и нормализация для всех экспериментов
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_size: int = 10000,
    seed: int = 42,
    num_workers: int = 2
):
    # Стандартная нормализация для MNIST (mean=0.1307, std=0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader