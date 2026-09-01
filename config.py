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
PRETRAINED = False

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