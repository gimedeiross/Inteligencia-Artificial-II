# create_model() é a ÚNICA função deste arquivo. É chamada por
# main.py (via MODEL_CREATORS["mobilenet"]).
#
# Por que MobileNetV3-Small no experimento:
# é a arquitetura voltada a eficiência (poucos parâmetros, pensada
# para mobile/edge), serve para comparar custo computacional e
# tempo de treino/inferência contra ResNet18 e GoogLeNet, que são
# mais pesados. Também é a única das três com ativação Hardswish
# (além de ReLU) e cujo classificador original já tem mais de uma
# camada — isso muda como o freeze e a cabeça nova são tratados
# abaixo, diferente do ResNet/GoogLeNet.

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

    Ativação: é a única das três arquiteturas do projeto que NÃO
    usa só ReLU. O backbone (`model.features`) mistura ReLU (nas
    camadas iniciais, mais barata computacionalmente) e Hardswish
    (nas camadas finais, mais expressiva) — decisão de design do
    paper original do MobileNetV3, herdada do torchvision. A
    cabeça (`model.classifier`) também herda essa mistura: sua
    estrutura original é `Linear -> Hardswish -> Dropout ->
    Linear`, e como só a ÚLTIMA Linear é substituída abaixo, a
    Hardswish intermediária continua ativa. É a única das três
    cabeças do projeto com uma não-linearidade entre as features
    e a saída (ResNet e GoogLeNet ficam com cabeça linear pura,
    ver resnet.py e googlenet.py).
    """

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    # Diferença chave em relação a resnet.py/googlenet.py: aqui
    # só `model.features` é congelado (o extrator convolucional),
    # não `model.parameters()` inteiro. `model.classifier` fica de
    # fora do freeze de propósito, porque a cabeça original do
    # MobileNetV3 já é uma mini-rede (Linear -> Hardswish ->
    # Dropout -> Linear), não uma única camada — então "congelar
    # tudo e treinar só a última Linear" deixaria a Linear
    # intermediária e o Hardswish presos com pesos da ImageNet.
    if freeze_backbone:

        for parameter in model.features.parameters():
            parameter.requires_grad = False

    # Substitui só a ÚLTIMA camada do classifier (1000 -> 8
    # classes). As camadas anteriores do classifier (a primeira
    # Linear e o Hardswish) são preservadas e continuam
    # treináveis — por isso a cabeça do MobileNet, ao contrário
    # da do ResNet/GoogLeNet, já tem uma não-linearidade (Hardswish)
    # entre as features e a saída final.
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )

    return model