from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image

from src.api.predict import predictor
from src.api.schemas import PredictionResponse


app = FastAPI(
    title="LungVision AI API",
    description="Explainable Multi-Class Chest X-ray Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("artifacts/gradcam").mkdir(
    parents=True,
    exist_ok=True
)

app.mount(
    "/gradcam",
    StaticFiles(directory="artifacts/gradcam"),
    name="gradcam"
)


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

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predictor.predict(image)

    return result