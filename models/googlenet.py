from torchvision.models import (
    googlenet,
    GoogLeNet_Weights,
)

import torch.nn as nn


def create_model(
    num_classes,
    pretrained=True,
    freeze_backbone=False
):
    """
    Cria o GoogLeNet para transfer learning.

    Se `freeze_backbone=True`, todos os parâmetros pré-treinados
    são congelados. Em seguida, `fc`, `aux1.fc2` e `aux2.fc2` são
    substituídos para `num_classes` e, por serem camadas novas,
    nascem treináveis independente do freeze — são a "cabeça" de
    classificação principal e auxiliar da rede.
    """

    weights = (
        GoogLeNet_Weights.DEFAULT
        if pretrained
        else None
    )

    model = googlenet(
        weights=weights,
        aux_logits=True
    )

    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.aux1.fc2 = nn.Linear(
        model.aux1.fc2.in_features,
        num_classes
    )

    model.aux2.fc2 = nn.Linear(
        model.aux2.fc2.in_features,
        num_classes
    )

    return model