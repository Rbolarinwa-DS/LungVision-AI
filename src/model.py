import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from src.utils import NUM_CLASSES


# ==========================================================
# LungVision Model
# ==========================================================

class LungVisionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = efficientnet_b0(weights=None)
        
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(

    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),

    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),

    nn.Linear(256, NUM_CLASSES)

)

    # ------------------------------------------------------

    def freeze_backbone(self):
        """
        Freeze EfficientNet feature extractor.
        Only the classifier will train.
        """

        for param in self.model.features.parameters():
            param.requires_grad = False

    # ------------------------------------------------------

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone for fine tuning.
        """

        for param in self.model.features.parameters():
            param.requires_grad = True

    # ------------------------------------------------------

    def forward(self, x):

        return self.model(x)


# ==========================================================
# Factory
# ==========================================================

def create_model():

    return LungVisionModel()


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    model = create_model()

    dummy = torch.randn(1, 3, 224, 224)

    output = model(dummy)

    print("=" * 50)
    print("Model Loaded")
    print("=" * 50)

    print(output.shape)