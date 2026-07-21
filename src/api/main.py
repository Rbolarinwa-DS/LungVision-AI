from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image

from src.api.predict import predictor
from src.api.schemas import PredictionResponse


app = FastAPI(
    title="LungVision AI API",
    description="Explainable Multi-Label Chest X-ray Disease Classification",
    version="1.0.0"
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Grad-CAM Directory
# ---------------------------------------------------------

Path("artifacts/gradcam").mkdir(
    parents=True,
    exist_ok=True
)

app.mount(
    "/gradcam",
    StaticFiles(directory="artifacts/gradcam"),
    name="gradcam"
)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "LungVision AI Backend Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        result = predictor.predict(image)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
