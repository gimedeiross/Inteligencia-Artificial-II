# create_model() é a ÚNICA função deste arquivo. É chamada por
# main.py (via MODEL_CREATORS["resnet"]) uma vez por execução,
# para instanciar o ResNet18 já adaptado ao Galaxy Zoo.
#
# Por que ResNet18 no experimento:
# é a arquitetura mais "clássica" das três (conexões residuais,
# CNN pura, sem branches extras), serve como baseline de
# comparação para GoogLeNet (Inception + aux losses) e
# MobileNetV3 (arquitetura voltada a mobile/eficiência).

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

    Ativação: o backbone usa ReLU (inplace) após cada camada de
    BatchNorm dentro dos blocos residuais — arquitetura padrão do
    ResNet, herdada do torchvision e não modificada aqui. A cabeça
    nova (`model.fc`) é uma única Linear SEM nenhuma ativação:
    o vetor de features (pós average pooling) vai direto para a
    saída, um classificador linear puro.
    """

    # Pesos ImageNet (torchvision). Usados para os três modelos
    # do projeto (mesmo racional: reaproveitar filtros de baixo
    # nível já aprendidos, já que o Galaxy Zoo tem ~156 mil
    # imagens — pouco para treinar uma CNN do zero).
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    # Congela TODO o backbone (todas as camadas convolucionais e
    # de batchnorm pré-treinadas). No ResNet18 isso é simples
    # porque a rede é sequencial/residual, sem branches — ao
    # contrário do GoogLeNet (que tem branches auxiliares) e do
    # MobileNet (que congela só `features`, não o classifier
    # inteiro). Essa diferença de granularidade do freeze entre
    # os três modelos está documentada no README.
    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

    # Substitui a última camada (`fc`), que na ImageNet tem 1000
    # saídas, por uma nova Linear com `num_classes` (8, as classes
    # do Galaxy Zoo). É um classificador LINEAR puro — sem
    # ativação extra entre o pooling e a saída — mantendo a
    # comparação com GoogLeNet igual nesse aspecto (só o MobileNet
    # herda uma cabeça não-linear, ver mobilenet.py).
    #
    # A nova camada é criada depois do freeze, então seus
    # parâmetros nascem com requires_grad=True e são sempre
    # treináveis, mesmo com o backbone congelado.
    #
    # Sem ativação aqui: nn.Linear direto, sem ReLU/Hardswish
    # entre as features e a saída (diferente do MobileNetV3, cuja
    # cabeça original já tem Hardswish — ver mobilenet.py).
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model