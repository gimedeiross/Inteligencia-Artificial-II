DATASET_NAME = "mrJordi0/galaxy-zoo-dataset"

NUM_CLASSES = 8
IMAGE_SIZE = 224

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

NUM_WORKERS = 4
SEED = 42

RESULTS_DIR = "results"

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