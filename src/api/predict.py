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

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = create_model()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
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

    def predict(self, image: Image.Image):

        original_image = image.convert("RGB")

        transformed = self.transforms(
            image=np.array(original_image)
        )["image"]

        image_tensor = (
            transformed
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.inference_mode():

            logits = self.model(image_tensor)

            probabilities = torch.sigmoid(
                logits
            )[0].cpu().numpy()

        highest_class = int(
            np.argmax(probabilities)
        )

        predictions = {}

        ranked = sorted(
            zip(TARGET_DISEASES, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        for disease, probability in ranked:

            predictions[disease] = {
                "probability": round(
                    float(probability),
                    4
                ),
                "detected": probability >= THRESHOLD
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

        filename = f"{uuid.uuid4().hex}.png"

        save_path = output_dir / filename

        gradcam_image.save(save_path)

        return {
            "predictions": predictions,
            "gradcam": f"/gradcam/{filename}"
        }


predictor = Predictor()
