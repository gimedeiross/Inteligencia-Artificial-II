# DATASET
DATASET_NAME = "mrJordi0/galaxy-zoo-dataset"

NUM_CLASSES = 8

CLASS_NAMES = [
    "Round Elliptical",
    "In-between Elliptical",
    "Cigar-shaped Elliptical",
    "Edge-on Spiral",
    "Barred Spiral",
    "Unbarred Spiral",
    "Irregular",
    "Merger",
]

# IMAGENS
IMAGE_SIZE = 224

# TREINAMENTO
BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

# TRANSFER LEARNING
# PRETRAINED=True carrega pesos pré-treinados na ImageNet.
#
# FREEZE_BACKBONE controla se o backbone (features convolucionais
# pré-treinadas) fica congelado ou não:
# - True  -> feature extraction / "linear probing": só a cabeça
#            de classificação nova é treinada. Rápido e barato em
#            memória GPU, mas limita a acurácia, porque as
#            features continuam genéricas da ImageNet (objetos do
#            dia a dia) e não se especializam em morfologia de
#            galáxia.
# - False -> fine-tuning completo: parte de pesos pré-treinados
#            mas ajusta a rede inteira, permitindo que as próprias
#            features convolucionais se adaptem às texturas e
#            formas específicas do Galaxy Zoo. Mais lento e usa
#            mais memória GPU (muito mais parâmetros treináveis),
#            mas tende a melhorar a acurácia justamente nas
#            classes mais ambíguas (ex.: subtipos de elípticas e
#            presença/ausência de barra nas espirais), que exigem
#            características mais finas do que a ImageNet ensina.
#
# Mudado de True para False em [ver seção "Alterações" no
# README]: com o backbone congelado, o ResNet18 ficou preso a
# ~41% de acurácia (vs. 25% do baseline "sempre prever a classe
# majoritária") e as curvas de treino mostravam a validação
# achatando cedo — sinal de teto de capacidade, não de
# overfitting. Fine-tuning completo é o passo mais direto para
# tentar melhorar isso antes de comparar contra GoogLeNet e
# MobileNet.
PRETRAINED = True

FREEZE_BACKBONE = False

# BALANCEAMENTO
USE_CLASS_WEIGHTS = True

# REGULARIZAÇÃO
EARLY_STOPPING_PATIENCE = 3

# REPRODUTIBILIDADE
SEED = 42

# RESULTADOS
RESULTS_DIR = "results"

# MODELOS
MODELS = [
    "resnet",
    "googlenet",
    "mobilenet",
]