from pathlib import Path
import uuid
import numpy as np
import torch
from PIL import Image

from src.model import create_model
from src.data.dataset import valid_transform
from src.utils import TARGET_DISEASES

from src.api.gradcam import GradCAMGenerator

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("artifacts/best_model.pth")

THRESHOLD = 0.45


class Predictor:

    def __init__(self):

        self.model = create_model()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transforms = valid_transform

        self.gradcam = GradCAMGenerator(
            self.model
        )

    def predict(self, image):

        original_image = image.convert("RGB")

        transformed = self.transforms(
            image=np.array(original_image)
        )["image"]

        image_tensor = (
            transformed
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            logits = self.model(image_tensor)

            probabilities = torch.sigmoid(
                logits
            )[0].cpu().numpy()

        predictions = {}

        highest_class = int(
            np.argmax(probabilities)
        )

        for disease, probability in zip(
            TARGET_DISEASES,
            probabilities
        ):

            predictions[disease] = {
                "probability": round(
                    float(probability),
                    4
                ),
                "detected": bool(
                    probability >= THRESHOLD
                )
            }

        gradcam_image = self.gradcam.generate(
            image_tensor=image_tensor,
            original_image=original_image,
            class_index=highest_class
        )

        output_dir = Path(
            "artifacts/gradcam"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            uuid.uuid4().hex + ".png"
        )

        save_path = output_dir / filename

        gradcam_image.save(save_path)

        return {
            "predictions": predictions,
            "gradcam": f"/gradcam/{filename}"
        }


predictor = Predictor()