from torchvision.models import (
    googlenet,
    GoogLeNet_Weights,
)

import torch.nn as nn


def create_model(num_classes, pretrained=True):

    weights = (
        GoogLeNet_Weights.DEFAULT
        if pretrained
        else None
    )

    model = googlenet(
        weights=weights,
        aux_logits=True
    )

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