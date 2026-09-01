import time

import torch


def get_logits(outputs):
    """
    Extrai os logits principais de diferentes arquiteturas.

    Para modelos como GoogLeNet, que possuem saídas auxiliares,
    utiliza apenas a saída principal para cálculo das métricas.
    """

    if hasattr(outputs, "logits"):
        return outputs.logits

    return outputs


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

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

        # GoogLeNet retorna:
        # GoogLeNetOutputs(
        #     logits,
        #     aux_logits2,
        #     aux_logits1
        # )
        #
        # Durante o treinamento, utilizamos
        # também as saídas auxiliares.

        if hasattr(outputs, "logits"):

            logits = outputs.logits

            loss = criterion(
                logits,
                labels
            )

            if (
                hasattr(outputs, "aux_logits1")
                and outputs.aux_logits1 is not None
            ):
                loss += 0.3 * criterion(
                    outputs.aux_logits1,
                    labels
                )

            if (
                hasattr(outputs, "aux_logits2")
                and outputs.aux_logits2 is not None
            ):
                loss += 0.3 * criterion(
                    outputs.aux_logits2,
                    labels
                )

        else:

            logits = outputs

            loss = criterion(
                logits,
                labels
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

    return (
        total_loss / total,
        correct / total
    )


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

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

    return (
        total_loss / total,
        correct / total
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

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
    }

    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    total_start = time.perf_counter()

    for epoch in range(epochs):

        epoch_start = time.perf_counter()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_accuracy = validate(
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

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["epoch_time_seconds"].append(
            epoch_time
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

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

        if (
            epochs_without_improvement
            >= patience
        ):

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
        "epochs_completed"
    ] = len(
        history["train_loss"]
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    return history