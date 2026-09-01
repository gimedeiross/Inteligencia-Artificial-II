import argparse

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    CLASS_NAMES,
    RESULTS_DIR,
)

from .dataset import get_dataset
from .models import create_model
from .utils import get_device


@torch.no_grad()
def predict(
    model,
    dataloader,
    device
):

    model.eval()

    predictions = []
    labels = []

    for batch in dataloader:

        images = batch["pixel_values"].to(device)

        batch_labels = batch["label"]

        outputs = model(images)

        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        batch_predictions = (
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
        )

        predictions.extend(
            batch_predictions
        )

        labels.extend(
            batch_labels.numpy()
        )

    return (
        np.array(labels),
        np.array(predictions)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet",
            "googlenet"
        ]
    )

    args = parser.parse_args()

    device = get_device()

    dataset = get_dataset()

    test_loader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = create_model(
        args.model,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            f"{RESULTS_DIR}/{args.model}.pth",
            map_location=device
        )
    )

    model.to(device)

    y_true, y_pred = predict(
        model,
        test_loader,
        device
    )

    # -------------------------
    # Métricas
    # -------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nRESULTADOS")
    print("=" * 40)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    # -------------------------
    # Relatório
    # -------------------------

    print("\nRELATÓRIO POR CLASSE")
    print("=" * 40)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # -------------------------
    # Matriz de confusão
    # -------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nMATRIZ DE CONFUSÃO")
    print(matrix)


if __name__ == "__main__":
    main()