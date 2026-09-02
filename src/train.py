import time

import torch
from sklearn.metrics import f1_score

from utils import get_gpu_memory


def get_logits(outputs):
    """
    Extrai os logits principais das diferentes arquiteturas.

    Para o GoogLeNet, que possui saídas auxiliares durante
    o treinamento, utiliza apenas a saída principal para
    validação e cálculo das métricas.
    """

    if hasattr(outputs, "logits"):
        return outputs.logits

    return outputs


def compute_training_loss(outputs, labels, criterion):
    """
    Calcula a loss durante o treinamento.

    Para o GoogLeNet, utiliza a saída principal e as duas
    saídas auxiliares.

    Para ResNet e MobileNet, utiliza apenas a saída principal.
    """

    if hasattr(outputs, "logits"):

        main_loss = criterion(
            outputs.logits,
            labels
        )

        loss = main_loss

        if (
            hasattr(outputs, "aux_logits1")
            and outputs.aux_logits1 is not None
        ):
            aux1_loss = criterion(
                outputs.aux_logits1,
                labels
            )

            loss += 0.3 * aux1_loss

        if (
            hasattr(outputs, "aux_logits2")
            and outputs.aux_logits2 is not None
        ):
            aux2_loss = criterion(
                outputs.aux_logits2,
                labels
            )

            loss += 0.3 * aux2_loss

        return loss, outputs.logits

    loss = criterion(
        outputs,
        labels
    )

    return loss, outputs


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    """
    Treina o modelo durante uma época.
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss, logits = compute_training_loss(
            outputs,
            labels,
            criterion
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        total_loss / total,
        correct / total,
        f1_macro
    )


def validate(
    model,
    loader,
    criterion,
    device
):
    """
    Avalia o modelo no conjunto de validação.

    Durante a validação são utilizados apenas os logits
    principais, inclusive no GoogLeNet.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            logits = get_logits(outputs)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        total_loss / total,
        correct / total,
        f1_macro
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    checkpoint_path
):
    """
    Executa o treinamento completo com early stopping.

    O melhor modelo é definido pela maior acurácia
    no conjunto de validação.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_macro": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "epoch_time_seconds": [],
    }

    best_val_f1_macro = 0.0
    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    total_start = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):

        epoch_start = time.perf_counter()

        train_loss, train_accuracy, train_f1_macro = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy, val_f1_macro = validate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_f1_macro"].append(
            train_f1_macro
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_f1_macro"].append(
            val_f1_macro
        )

        history["epoch_time_seconds"].append(
            epoch_time
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train F1: {train_f1_macro:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1_macro:.4f}"
        )

        if val_f1_macro > best_val_f1_macro:

            best_val_f1_macro = val_f1_macro
            best_val_accuracy = val_accuracy

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "  → Melhor modelo salvo."
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                "  → Early stopping."
            )

            break

    history[
        "total_training_time_seconds"
    ] = (
        time.perf_counter()
        - total_start
    )

    history[
        "best_validation_accuracy"
    ] = best_val_accuracy

    history[
        "best_validation_f1_macro"
    ] = best_val_f1_macro

    history[
        "epochs_completed"
    ] = len(
        history["train_loss"]
    )

    history[
        "max_gpu_memory_gb"
    ] = get_gpu_memory()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    return history