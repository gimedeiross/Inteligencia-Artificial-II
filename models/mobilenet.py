from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

import torch.nn as nn


def create_model(
    num_classes,
    pretrained=True,
    freeze_backbone=False
):
    """
    Cria o MobileNetV3-Small para transfer learning.

    Se `freeze_backbone=True`, congela apenas `model.features`
    (o extrator de características convolucional) e mantém todo
    o `model.classifier` treinável — não só a última camada — já
    que a cabeça do MobileNetV3 é composta por múltiplas camadas
    (Linear -> Hardswish -> Dropout -> Linear) que costumam se
    beneficiar de serem treinadas juntas.
    """

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    if freeze_backbone:

        for parameter in model.features.parameters():
            parameter.requires_grad = False

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    return model