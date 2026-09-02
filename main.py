import json
import os
import time

import torch
import torch.nn as nn

from config import (
    NUM_CLASSES,
    CLASS_NAMES,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PRETRAINED,
    EARLY_STOPPING_PATIENCE,
    RESULTS_DIR,
    MODELS,
    USE_CLASS_WEIGHTS,
    SEED
)

from dataset import load_galaxy_dataset
from evaluate import (
    evaluate_model,
    save_training_curves,
)

from train import train_model

from utils import (
    set_seed,
    get_device,
    count_parameters,
    save_json,
)


from models.resnet import (
    create_model as create_resnet
)

from models.googlenet import (
    create_model as create_googlenet
)

from models.mobilenet import (
    create_model as create_mobilenet
)


MODEL_CREATORS = {
    "resnet": create_resnet,
    "googlenet": create_googlenet,
    "mobilenet": create_mobilenet,
}


def create_results_directory(
    model_name
):

    directory = os.path.join(
        RESULTS_DIR,
        model_name
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    return directory


def create_criterion(
    class_weights,
    device
):

    if USE_CLASS_WEIGHTS:

        class_weights = (
            class_weights.to(device)
        )

        return nn.CrossEntropyLoss(
            weight=class_weights
        )

    return nn.CrossEntropyLoss()


def train_single_model(
    model_name,
    loaders,
    device
):

    print("\n")
    print("=" * 60)
    print(
        f"TREINANDO: {model_name.upper()}"
    )
    print("=" * 60)

    results_dir = create_results_directory(
        model_name
    )

    model = MODEL_CREATORS[
        model_name
    ](
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED
    )

    model = model.to(
        device
    )

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    print(
        f"Parâmetros totais: "
        f"{total_parameters:,}"
    )

    print(
        f"Parâmetros treináveis: "
        f"{trainable_parameters:,}"
    )

    criterion = create_criterion(
        loaders["class_weights"],
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    checkpoint_path = os.path.join(
        results_dir,
        "best_model.pth"
    )

    start_time = time.perf_counter()

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    save_training_curves(
        history,
        results_dir
    )

    print("\nAvaliando no conjunto de teste...")

    test_start = time.perf_counter()

    raw_metrics = evaluate_model(
        model=model,
        test_loader=loaders["test"],
        device=device,
        class_names=CLASS_NAMES,
        results_dir=results_dir
    )

    test_time = (
        time.perf_counter()
        - test_start
    )

    result = {
        "model": model_name,

        "metrics": raw_metrics,

        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },

        "training": {
            "training_time_seconds": training_time,
            "evaluation_time_seconds": test_time,
            "epochs_completed": history["epochs_completed"],
            "best_validation_accuracy": history["best_validation_accuracy"],
            "best_validation_f1_macro": history["best_validation_f1_macro"],
            "max_gpu_memory_gb": history["max_gpu_memory_gb"],
            "pretrained": PRETRAINED,
            "class_weights": USE_CLASS_WEIGHTS,
        },
    }

    save_json(
        history,
        os.path.join(
            results_dir,
            "history.json"
        )
    )

    save_json(
        result,
        os.path.join(
            results_dir,
            "metrics.json"
        )
    )

    print("\nResultados:")
    print(
        f"Accuracy: "
        f"{raw_metrics['accuracy']:.4f}"
    )

    print(
        f"F1 Macro: "
        f"{raw_metrics['f1_macro']:.4f}"
    )

    print(
        f"F1 Weighted: "
        f"{raw_metrics['f1_weighted']:.4f}"
    )

    return result


def save_comparison(
    results
):
    # Nome diferente de "comparison.json" para não colidir com
    # o arquivo gerado por compare_results.py, que usa um
    # formato de tabela achatado a partir destes mesmos dados.
    path = os.path.join(
        RESULTS_DIR,
        "training_summary.json"
    )

    save_json(
        results,
        path
    )

    print(
        f"\nResumo do treinamento salvo em: {path}"
    )


def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 60)
    print("PROJETO IA 2 - GALAXY ZOO")
    print("=" * 60)

    print(
        f"Dispositivo: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.cuda.reset_peak_memory_stats()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("\nCarregando dados...")

    loaders = load_galaxy_dataset()

    results = []

    for model_name in MODELS:

        metrics = train_single_model(
            model_name,
            loaders,
            device
        )

        results.append(
            metrics
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    save_comparison(
        results
    )

    print("\n")
    print("=" * 60)
    print("COMPARAÇÃO FINAL")
    print("=" * 60)

    for result in results:

        metrics = result["metrics"]
        training = result["training"]

        print(
            f"\n{result['model'].upper()}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"F1 Macro: "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"F1 Weighted: "
            f"{metrics['f1_weighted']:.4f}"
        )

        print(
            f"Tempo: "
            f"{training['training_time_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()