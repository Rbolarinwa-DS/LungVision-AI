from pathlib import Path

import random
import numpy as np
import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "chestxray14"

IMAGE_DIR = RAW_DATA_DIR / "dif_nih14"

PROCESSED_DIR = DATA_DIR / "processed"

SPLIT_DIR = DATA_DIR / "splits"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ==========================================================
# Dataset Configuration
# ==========================================================

TARGET_DISEASES = [
    "Atelectasis",
    "Consolidation",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax"
]

NUM_CLASSES = len(TARGET_DISEASES)


# ==========================================================
# Image Configuration
# ==========================================================

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)


# ==========================================================
# Training Configuration
# ==========================================================

BATCH_SIZE = 32

NUM_WORKERS = 4

PIN_MEMORY = torch.cuda.is_available()

RANDOM_STATE = 42

# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=RANDOM_STATE):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    
