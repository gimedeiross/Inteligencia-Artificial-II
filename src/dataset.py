from datasets import load_dataset
from torchvision import transforms

from .config import DATASET_NAME, IMAGE_SIZE


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform, test_transform


def load_galaxy_zoo():

    dataset = load_dataset(DATASET_NAME)

    print(dataset)

    return dataset


def prepare_dataset(dataset):

    train_transform, test_transform = get_transforms()

    def train_transforms(examples):

        examples["pixel_values"] = [
            train_transform(image.convert("RGB"))
            for image in examples["image"]
        ]

        return examples

    def test_transforms(examples):

        examples["pixel_values"] = [
            test_transform(image.convert("RGB"))
            for image in examples["image"]
        ]

        return examples

    dataset["train"].set_transform(train_transforms)
    dataset["test"].set_transform(test_transforms)

    return dataset


def get_dataset():

    dataset = load_galaxy_zoo()

    dataset = prepare_dataset(dataset)

    return dataset