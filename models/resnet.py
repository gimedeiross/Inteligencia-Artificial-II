from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

import torch.nn as nn


def create_model(
    num_classes,
    pretrained=True,
    freeze_backbone=False
):
    """
    Cria o ResNet18 para transfer learning.

    Se `freeze_backbone=True`, todos os parâmetros pré-treinados
    são congelados e apenas a nova camada `fc` (substituída para
    `num_classes`) é treinada. Se `freeze_backbone=False`, a rede
    inteira é treinada (fine-tuning) a partir dos pesos
    pré-treinados.
    """

    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

    # A nova camada é criada depois do freeze, então seus
    # parâmetros nascem com requires_grad=True e são sempre
    # treináveis, mesmo com o backbone congelado.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model