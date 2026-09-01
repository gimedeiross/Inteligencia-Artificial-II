from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from config import (
    DATASET_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    SEED,
)


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


def transform_train(example):

    example["pixel_values"] = [
        train_transform(
            image.convert("RGB")
        )
        for image in example["image"]
    ]

    return example


def transform_eval(example):

    example["pixel_values"] = [
        eval_transform(
            image.convert("RGB")
        )
        for image in example["image"]
    ]

    return example


def collate_fn(batch):

    images = torch.stack([
        item["pixel_values"]
        for item in batch
    ])

    labels = torch.tensor([
        item["label"]
        for item in batch
    ])

    return images, labels


def calculate_class_weights(dataset):

    labels = dataset["train"]["label"]

    counts = torch.bincount(
        torch.tensor(labels),
        minlength=NUM_CLASSES
    ).float()

    weights = len(labels) / (
        NUM_CLASSES * counts
    )

    return weights


def load_galaxy_dataset():

    print(
        f"Carregando dataset: {DATASET_NAME}"
    )

    dataset = load_dataset(
        DATASET_NAME
    )

    if "validation" not in dataset:

        split = dataset["train"].train_test_split(
            test_size=0.15,
            seed=SEED
        )

        train_validation = split["train"]

        test_dataset = split["test"]

        validation_split = (
            train_validation.train_test_split(
                test_size=0.1765,
                seed=SEED
            )
        )

        dataset = {
            "train": validation_split["train"],
            "validation": validation_split["test"],
            "test": test_dataset,
        }

    class_weights = calculate_class_weights(
        dataset
    )

    print("\nPesos das classes:")

    for index, weight in enumerate(
        class_weights
    ):

        print(
            f"Classe {index}: "
            f"{weight:.4f}"
        )

    dataset["train"].set_transform(
        transform_train
    )

    dataset["validation"].set_transform(
        transform_eval
    )

    dataset["test"].set_transform(
        transform_eval
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
        "class_weights": class_weights,
    }