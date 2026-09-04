# create_model() é a ÚNICA função deste arquivo. É chamada por
# main.py (via MODEL_CREATORS["googlenet"]).
#
# Por que GoogLeNet no experimento:
# é a única das três arquiteturas com módulos Inception (múltiplos
# tamanhos de filtro em paralelo) e saídas auxiliares — serve para
# comparar contra o ResNet18 (CNN sequencial simples) e o
# MobileNetV3 (voltado a eficiência/mobile). O uso das saídas
# auxiliares durante o treino é tratado em conjunto com train.py
# (funções get_logits e compute_training_loss), não aqui.

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

    Ativação: o backbone usa ReLU dentro de cada módulo Inception,
    incluindo os branches auxiliares (aux1/aux2) — arquitetura
    padrão do GoogLeNet, herdada do torchvision e não modificada
    aqui. As três cabeças novas (`fc`, `aux1.fc2`, `aux2.fc2`) são
    Linear SEM ativação extra: substituem só a última camada de
    cada branch original, então a ReLU que já existia ANTES dessas
    camadas (dentro do próprio branch auxiliar) continua ativa,
    mas não é adicionada nenhuma ativação nova por nós.
    """

    weights = (
        GoogLeNet_Weights.DEFAULT
        if pretrained
        else None
    )

    # aux_logits=True é obrigatório aqui: sem isso o torchvision
    # não cria os módulos aux1/aux2, e as duas linhas abaixo que
    # substituem aux1.fc2/aux2.fc2 quebrariam. É essa mesma flag
    # que faz o forward() do modelo, durante o treino, devolver um
    # objeto com .logits, .aux_logits1 e .aux_logits2 — usado em
    # train.py (compute_training_loss) para montar a loss composta
    # do GoogLeNet.
    model = googlenet(
        weights=weights,
        aux_logits=True
    )

    # Congela TODO o backbone pré-treinado (blocos Inception e as
    # duas cabeças auxiliares originais da ImageNet, antes de
    # serem substituídas abaixo). Mesma lógica do ResNet18: com
    # freeze_backbone=True, só a(s) cabeça(s) nova(s) treinam.
    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

    # Cabeça principal: 1000 classes ImageNet -> num_classes (8).
    # Classificador linear puro, igual ao ResNet18.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # Cabeças auxiliares (aux1 e aux2): existem em pontos
    # intermediários da rede e, no treino original do GoogLeNet,
    # ajudam o gradiente a se propagar em uma rede profunda
    # (mitigam vanishing gradient). Também precisam ser adaptadas
    # para num_classes, senão suas losses (calculadas em
    # train.py) ficariam incompatíveis com os rótulos do Galaxy
    # Zoo. Só são usadas durante o TREINO — na validação/teste o
    # modelo usa apenas a saída principal (ver get_logits em
    # train.py), então não afetam a avaliação final.
    model.aux1.fc2 = nn.Linear(
        model.aux1.fc2.in_features,
        num_classes
    )

    model.aux2.fc2 = nn.Linear(
        model.aux2.fc2.in_features,
        num_classes
    )

    return model