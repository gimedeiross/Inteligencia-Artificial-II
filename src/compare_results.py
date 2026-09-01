import json
import os

import matplotlib.pyplot as plt

from config import RESULTS_DIR, CLASS_NAMES


def load_results():
    results = {}

    for model_name in ["resnet", "googlenet", "mobilenet"]:
        path = os.path.join(
            RESULTS_DIR,
            model_name,
            "metrics.json"
        )

        if not os.path.exists(path):
            print(
                f"Aviso: resultado de {model_name} "
                f"não encontrado."
            )
            continue

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            results[model_name] = json.load(file)

    return results


def create_comparison_table(results):
    table = []

    for model_name, result in results.items():

        metrics = result["metrics"]
        parameters = result["parameters"]
        training = result["training"]

        table.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "parameters": parameters["total"],
            "training_time_seconds": training[
                "training_time_seconds"
            ],
            "evaluation_time_seconds": training[
                "evaluation_time_seconds"
            ],
        })

    return table


def save_comparison(results):
    comparison = create_comparison_table(results)

    output_path = os.path.join(
        RESULTS_DIR,
        "comparison.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Comparação salva em: {output_path}"
    )

    return comparison


def print_comparison(comparison):

    print("\n" + "=" * 90)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 90)

    print(
        f"{'Modelo':<12}"
        f"{'Accuracy':>12}"
        f"{'F1 Macro':>12}"
        f"{'F1 Weighted':>14}"
        f"{'Parâmetros':>16}"
        f"{'Tempo (s)':>14}"
    )

    print("-" * 90)

    for result in comparison:

        print(
            f"{result['model']:<12}"
            f"{result['accuracy']:>12.4f}"
            f"{result['f1_macro']:>12.4f}"
            f"{result['f1_weighted']:>14.4f}"
            f"{result['parameters']:>16,}"
            f"{result['training_time_seconds']:>14.2f}"
        )


def create_comparison_chart(comparison):

    models = [
        result["model"].capitalize()
        for result in comparison
    ]

    accuracy = [
        result["accuracy"]
        for result in comparison
    ]

    f1_macro = [
        result["f1_macro"]
        for result in comparison
    ]

    x = range(len(models))
    width = 0.35

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.bar(
        [value - width / 2 for value in x],
        accuracy,
        width,
        label="Accuracy"
    )

    axis.bar(
        [value + width / 2 for value in x],
        f1_macro,
        width,
        label="F1 Macro"
    )

    axis.set_xticks(x)
    axis.set_xticklabels(models)

    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_title(
        "Comparação de desempenho dos modelos"
    )

    axis.legend()

    figure.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300
    )

    plt.close(figure)

    print(
        f"Gráfico salvo em: {output_path}"
    )


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = load_results()

    if not results:
        print(
            "Nenhum resultado encontrado."
        )
        print(
            "Execute primeiro o treinamento."
        )
        return

    comparison = save_comparison(
        results
    )

    print_comparison(
        comparison
    )

    create_comparison_chart(
        comparison
    )


if __name__ == "__main__":
    main()