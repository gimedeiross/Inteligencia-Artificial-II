import os
import math

import matplotlib.pyplot as plt

from datasets import load_dataset

from config import (
    DATASET_NAME,
    CLASS_NAMES,
    RESULTS_DIR,
)


def load_data():
    print(
        f"Carregando dataset: {DATASET_NAME}"
    )

    return load_dataset(
        DATASET_NAME
    )


def show_samples(
    dataset,
    samples_per_class=4
):

    images = []
    labels = []

    for class_id in range(
        len(CLASS_NAMES)
    ):

        class_samples = []

        for example in dataset:

            if example["label"] == class_id:

                class_samples.append(
                    example
                )

                if len(class_samples) >= samples_per_class:
                    break

        for example in class_samples:
            images.append(
                example["image"]
            )
            labels.append(
                class_id
            )

    columns = samples_per_class
    rows = len(CLASS_NAMES)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 24)
    )

    for index, (image, label) in enumerate(
        zip(images, labels)
    ):

        row = index // columns
        column = index % columns

        axis = axes[row][column]

        axis.imshow(image)
        axis.set_title(
            CLASS_NAMES[label]
        )
        axis.axis("off")

    figure.suptitle(
        "Amostras do Galaxy Zoo Dataset",
        fontsize=16
    )

    figure.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "dataset_samples.png"
    )

    figure.savefig(
        output_path,
        dpi=300
    )

    plt.close(figure)

    print(
        f"Amostras salvas em: {output_path}"
    )


def plot_class_distribution(dataset):

    counts = [
        0
        for _ in CLASS_NAMES
    ]

    for label in dataset["label"]:
        counts[label] += 1

    figure = plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        CLASS_NAMES,
        counts
    )

    plt.xlabel("Classe")
    plt.ylabel("Quantidade de imagens")
    plt.title(
        "Distribuição das classes - Galaxy Zoo"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    figure.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "class_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=300
    )

    plt.close(figure)

    print(
        f"Distribuição salva em: {output_path}"
    )


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    dataset = load_data()

    print("\nGerando amostras...")
    show_samples(
        dataset["train"]
    )

    print("\nGerando distribuição...")
    plot_class_distribution(
        dataset["train"]
    )


if __name__ == "__main__":
    main()