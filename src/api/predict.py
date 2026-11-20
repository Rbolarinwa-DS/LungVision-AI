
from pathlib import Path
import base64
import io
import time

import numpy as np
import torch
from PIL import Image

from src.model import create_model
from src.api.transforms import valid_transform
from src.utils import TARGET_DISEASES
from src.api.gradcam import GradCAMGenerator


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.pth"

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

        total_start = time.perf_counter()

        # Preprocessing
        t = time.perf_counter()

        original_image = image.convert("RGB")

        transformed = self.transforms(
            image=np.array(original_image)
        )["image"]

        image_tensor = (
            transformed
            .unsqueeze(0)
            .to(DEVICE)
        )

        print(
            f"[TIMING] Preprocess: {time.perf_counter() - t:.3f}s"
        )

        # Inference
        t = time.perf_counter()

        with torch.inference_mode():

            logits = self.model(image_tensor)

            probabilities = torch.sigmoid(
                logits
            )[0].cpu().numpy()

        print(
            f"[TIMING] Inference: {time.perf_counter() - t:.3f}s"
        )

        # Prediction formatting
        t = time.perf_counter()

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

        print(
            f"[TIMING] Formatting: {time.perf_counter() - t:.3f}s"
        )

        # Grad-CAM
        t = time.perf_counter()

        gradcam_image = self.gradcam.generate(
            image_tensor=image_tensor,
            original_image=original_image,
            class_index=highest_class
        )

        print(
            f"[TIMING] GradCAM: {time.perf_counter() - t:.3f}s"
        )

        # Encode Grad-CAM image in memory
        t = time.perf_counter()

        image_buffer = io.BytesIO()

        gradcam_image.save(
            image_buffer,
            format="PNG"
        )

        image_base64 = base64.b64encode(
            image_buffer.getvalue()
        ).decode("ascii")

        gradcam_data_uri = (
            f"data:image/png;base64,{image_base64}"
        )

        print(
            f"[TIMING] Image encoding: {time.perf_counter() - t:.3f}s"
        )

        print(
            f"[TIMING] TOTAL: {time.perf_counter() - total_start:.3f}s"
        )

        return {
            "predictions": predictions,
            "gradcam": gradcam_data_uri
        }


predictor = Predictor()
