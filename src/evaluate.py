import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Métricas de avaliação vêm do scikit-learn em vez de calculadas
# manualmente: são implementações padrão da literatura, já tratam
# casos-limite (ex.: `zero_division=0` para classes sem nenhuma
# previsão, relevante aqui pelo desbalanceamento do Galaxy Zoo) e
# evitam reimplementar precision/recall/F1 e classification_report.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    test_loader,
    device,
    class_names,
    results_dir
):

    model.eval()

    predictions = []

    labels = []

    with torch.no_grad():

        for images, batch_labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            batch_predictions = (
                outputs.argmax(dim=1)
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision_weighted = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    save_confusion_matrix(
        matrix,
        class_names,
        results_dir
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "precision_weighted": float(
            precision_weighted
        ),

        "recall_weighted": float(
            recall_weighted
        ),

        "f1_weighted": float(
            f1_weighted
        ),

        "precision_macro": float(
            precision_macro
        ),

        "recall_macro": float(
            recall_macro
        ),

        "f1_macro": float(
            f1_macro
        ),

        "classification_report": report,
    }


def save_confusion_matrix(
    matrix,
    class_names,
    results_dir
):

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    image = axis.imshow(
        matrix
    )

    figure.colorbar(
        image
    )

    axis.set(
        xticks=np.arange(
            len(class_names)
        ),
        yticks=np.arange(
            len(class_names)
        ),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix"
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            axis.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            results_dir,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close(
        figure
    )


def save_training_curves(
    history,
    results_dir
):

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # Loss

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training and Validation Loss"
    )
    plt.legend()

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            results_dir,
            "loss_curve.png"
        ),
        dpi=300
    )

    plt.close(
        figure
    )

    # Accuracy

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            results_dir,
            "accuracy_curve.png"
        ),
        dpi=300
    )

    plt.close(
        figure
    )

    # F1 Macro

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_f1_macro"],
        label="Train F1 Macro"
    )

    plt.plot(
        epochs,
        history["val_f1_macro"],
        label="Validation F1 Macro"
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title(
        "Training and Validation F1 Macro"
    )

    plt.legend()

    figure.tight_layout()

    figure.savefig(
        os.path.join(
            results_dir,
            "f1_macro_curve.png"
        ),
        dpi=300
    )

    plt.close(
        figure
    )