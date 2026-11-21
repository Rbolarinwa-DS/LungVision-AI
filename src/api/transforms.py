import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

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