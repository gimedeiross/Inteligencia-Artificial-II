import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    googlenet,
    GoogLeNet_Weights,
)


def create_resnet(num_classes):

    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def create_googlenet(num_classes):

    model = googlenet(
        weights=GoogLeNet_Weights.DEFAULT,
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


def create_model(name, num_classes):

    if name == "resnet":
        return create_resnet(num_classes)

    if name == "googlenet":
        return create_googlenet(num_classes)

    raise ValueError(
        "Modelo inválido. Use 'resnet' ou 'googlenet'."
    )