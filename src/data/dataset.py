from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import (
    SPLIT_DIR,
    TARGET_DISEASES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


# ==========================================================
# Dataset
# ==========================================================

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transforms=None):

        self.dataframe = dataframe.reset_index(drop=True)

        self.transforms = transforms

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["Image Path"]

        image = cv2.imread(image_path)

        if image is None:

            raise FileNotFoundError(

                f"Unable to load image:\n{image_path}"

            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = row[TARGET_DISEASES].values.astype(np.float32)

        if self.transforms:

            image = self.transforms(image=image)["image"]

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


# ==========================================================
# Training Transform
# ==========================================================

train_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.10,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.4
    ),

    A.GaussianBlur(
        blur_limit=(3,5),
        p=0.2
    ),

    A.GaussNoise(
        std_range=(0.01, 0.05),
        p=0.2
    ),

    A.CLAHE(
        clip_limit=2.0,
        tile_grid_size=(8,8),
        p=0.3
    ),

    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    ToTensorV2()

])


# ==========================================================
# Validation/Test Transform
# ==========================================================

valid_transform = A.Compose([

    A.Resize(

        IMAGE_SIZE[0],
        IMAGE_SIZE[1]

    ),

    A.Normalize(

        mean=IMAGENET_MEAN,
        std=IMAGENET_STD

    ),

    ToTensorV2()

])


# ==========================================================
# DataLoader Factory
# ==========================================================

def get_dataloaders():

    train_df = pd.read_csv(

        SPLIT_DIR / "train.csv"

    )

    val_df = pd.read_csv(

        SPLIT_DIR / "val.csv"

    )

    test_df = pd.read_csv(

        SPLIT_DIR / "test.csv"

    )

    train_dataset = ChestXrayDataset(

        dataframe=train_df,

        transforms=train_transform

    )

    val_dataset = ChestXrayDataset(

        dataframe=val_df,

        transforms=valid_transform

    )

    test_dataset = ChestXrayDataset(

        dataframe=test_df,

        transforms=valid_transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY

    )

    return train_loader, val_loader, test_loader


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print("=" * 50)

    print("Dataset Summary")

    print("=" * 50)

    print(f"Training batches   : {len(train_loader)}")

    print(f"Validation batches : {len(val_loader)}")

    print(f"Testing batches    : {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print()

    print(f"Images Shape : {images.shape}")

    print(f"Labels Shape : {labels.shape}")

    print()

    print("Dataset loaded successfully.")