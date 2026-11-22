from pathlib import Path
import uuid
import time

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

        total_start = time.perf_counter()

        # ---------------------------------
        # Preprocessing
        # ---------------------------------
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

        # ---------------------------------
        # Inference
        # ---------------------------------
        t = time.perf_counter()

        with torch.inference_mode():

            logits = self.model(image_tensor)

            probabilities = torch.sigmoid(
                logits
            )[0].cpu().numpy()

        print(
            f"[TIMING] Inference: {time.perf_counter() - t:.3f}s"
        )

        # ---------------------------------
        # Prediction formatting
        # ---------------------------------
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

        # ---------------------------------
        # GradCAM
        # ---------------------------------
        t = time.perf_counter()

        gradcam_image = self.gradcam.generate(
            image_tensor=image_tensor,
            original_image=original_image,
            class_index=highest_class
        )

        print(
            f"[TIMING] GradCAM: {time.perf_counter() - t:.3f}s"
        )

        # ---------------------------------
        # Save image
        # ---------------------------------
        t = time.perf_counter()

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

        print(
            f"[TIMING] Save Image: {time.perf_counter() - t:.3f}s"
        )

        # ---------------------------------
        # Total
        # ---------------------------------
        print(
            f"[TIMING] TOTAL: {time.perf_counter() - total_start:.3f}s"
        )

        return {
            "predictions": predictions,
            "gradcam": f"/gradcam/{filename}"
        }

predictor = Predictor()