import os

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
    class_samples = {
        class_id: []
        for class_id in range(len(CLASS_NAMES))
    }

    for example in dataset:
        label = example["label"]

        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(example)

        if all(
            len(samples) >= samples_per_class
            for samples in class_samples.values()
        ):
            break

    columns = samples_per_class
    rows = len(CLASS_NAMES)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 24)
    )

    for row, class_id in enumerate(CLASS_NAMES):
        samples = class_samples[row]

        for column, example in enumerate(samples):
            axis = axes[row][column]

            axis.imshow(example["image"])
            axis.set_title(
                CLASS_NAMES[row]
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
        dpi=300,
        bbox_inches="tight"
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