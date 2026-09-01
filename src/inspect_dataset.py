from datasets import load_dataset
from collections import Counter

from config import DATASET_NAME, CLASS_NAMES


def main():
    print(f"Carregando dataset: {DATASET_NAME}\n")

    dataset = load_dataset(DATASET_NAME)

    print("=== SPLITS ===")
    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} exemplos")

    print("\n=== ESTRUTURA ===")
    print(dataset)

    # Usa o primeiro split disponível para inspeção
    split_name = list(dataset.keys())[0]
    split = dataset[split_name]

    print(f"\n=== FEATURES ({split_name}) ===")
    print(split.features)

    print("\n=== PRIMEIRO EXEMPLO ===")
    example = split[0]

    for key, value in example.items():
        if key == "image":
            print(f"{key}: {value}")
            print(f"  Tipo: {type(value)}")
            print(f"  Tamanho: {value.size}")
            print(f"  Modo: {value.mode}")
        else:
            print(f"{key}: {value}")

    # Procura coluna de label
    label_column = None

    for column in split.column_names:
        if column in ["label", "labels", "class"]:
            label_column = column
            break

    if label_column is None:
        print("\nNenhuma coluna de label encontrada automaticamente.")
        return

    print(f"\n=== DISTRIBUIÇÃO DAS CLASSES ===")

    labels = split[label_column]
    counter = Counter(labels)

    for label, count in sorted(counter.items()):
        if isinstance(label, int) and label < len(CLASS_NAMES):
            name = CLASS_NAMES[label]
        else:
            name = str(label)

        percentage = count / len(labels) * 100

        print(
            f"{label}: {name:<25} "
            f"{count:>6} imagens "
            f"({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()