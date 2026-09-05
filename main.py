import argparse
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
    FREEZE_BACKBONE,
    EARLY_STOPPING_PATIENCE,
    RESULTS_DIR,
    MODELS,
    USE_CLASS_WEIGHTS,
    SEED
)

from src.dataset import load_galaxy_dataset
from src.evaluate import (
    evaluate_model,
    save_training_curves,
)

from src.train import train_model

from src.utils import (
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


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Treina e avalia arquiteturas CNN "
            "(ResNet18, GoogLeNet, MobileNetV3-Small) "
            "no Galaxy Zoo Dataset."
        )
    )

    parser.add_argument(
        "--model",
        choices=["resnet", "googlenet", "mobilenet", "all"],
        default="all",
        help=(
            "Qual modelo treinar. Use 'resnet', 'googlenet' ou "
            "'mobilenet' para testar rapidamente se o pipeline está "
            "funcionando com um único modelo. Use 'all' (padrão) para "
            "treinar os três sequencialmente e gerar a comparação final."
        ),
    )

    return parser.parse_args()


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
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE
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
        f"{trainable_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.1f}%)"
    )

    criterion = create_criterion(
        loaders["class_weights"],
        device
    )

    # Com o backbone congelado, model.parameters() ainda inclui
    # os tensores com requires_grad=False; passá-los ao otimizador
    # não quebra nada, mas é desnecessário. Filtramos para treinar
    # apenas o que de fato tem gradiente (a cabeça de classificação).
    trainable_params = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
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
            "freeze_backbone": FREEZE_BACKBONE,
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

    # `results` contém só os modelos rodados NESTA execução
    # (ex.: um único modelo, se `--model resnet/googlenet/
    # mobilenet` foi usado). Sem o merge abaixo, cada execução
    # sobrescreveria o arquivo inteiro com só esse(s) modelo(s),
    # apagando os resultados de execuções anteriores com outros
    # modelos — problema real quando main.py é rodado uma vez
    # por modelo em vez de com `--model all`.
    #
    # Carrega o que já existe no arquivo (se existir) e indexa
    # por nome do modelo, para poder mesclar com o resultado
    # novo sem perder os outros.
    existing_results = []

    if os.path.exists(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            existing_results = json.load(file)

    combined_by_model = {
        result["model"]: result
        for result in existing_results
    }

    # Resultados desta execução sobrescrevem, por nome de modelo,
    # qualquer resultado anterior do MESMO modelo (ex.: rodar
    # `--model resnet` de novo atualiza só a entrada do resnet),
    # mas preservam os resultados de modelos diferentes que já
    # estavam salvos.
    for result in results:

        combined_by_model[result["model"]] = result

    merged_results = list(
        combined_by_model.values()
    )

    save_json(
        merged_results,
        path
    )

    print(
        f"\nResumo do treinamento salvo em: {path}"
    )


def main():

    args = parse_args()

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

    # --model all (padrão) treina os três sequencialmente, na ordem
    # definida em config.MODELS. --model resnet/googlenet/mobilenet
    # treina só o modelo escolhido, útil para validar rapidamente
    # se o pipeline está funcionando antes de rodar o experimento
    # completo.
    models_to_run = (
        MODELS
        if args.model == "all"
        else [args.model]
    )

    print(
        f"\nModelo(s) selecionado(s): {', '.join(models_to_run)}"
    )

    print("\nCarregando dados...")

    loaders = load_galaxy_dataset()

    results = []

    for model_name in models_to_run:

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