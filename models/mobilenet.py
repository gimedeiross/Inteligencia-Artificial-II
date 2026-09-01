from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

import torch.nn as nn


def create_model(num_classes, pretrained=True):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    return model