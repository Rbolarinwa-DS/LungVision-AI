# 🫁 LungVision AI

> **Explainable Multi-Label Chest X-ray Disease Classification using Deep Learning**

LungVision AI is an end-to-end medical AI application designed to detect multiple thoracic diseases from chest X-ray images using transfer learning and explainable artificial intelligence (XAI). The system leverages EfficientNet-B0 for feature extraction, FastAPI for deployment, and Grad-CAM to visualize the image regions influencing each prediction.

---

# Project Overview

Early diagnosis of thoracic diseases through chest radiography remains one of the most important tasks in modern healthcare. Interpreting chest X-rays requires significant clinical expertise and can be time-consuming.

LungVision AI assists this process by automatically identifying multiple chest abnormalities from a single X-ray image while providing visual explanations to improve transparency and trust in model predictions.

The application predicts the presence of the following diseases:

- Atelectasis
- Consolidation
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumothorax

Unlike traditional single-label classifiers, LungVision AI performs **multi-label classification**, allowing multiple diseases to be detected simultaneously from a single radiograph.

---

# Features

- Multi-label chest X-ray disease classification
- EfficientNet-B0 Transfer Learning
- Grad-CAM Explainability
- FastAPI REST API
- Production-ready inference pipeline
- Confidence scores for every disease
- Interactive API documentation (Swagger)
- Optimized inference using PyTorch

---

# Project Architecture

```
                Chest X-ray
                     │
                     ▼
          Image Preprocessing
                     │
                     ▼
          EfficientNet-B0 Backbone
                     │
                     ▼
          Multi-label Classifier Head
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
 Disease Probabilities         Grad-CAM Heatmap
      ▼                             ▼
           FastAPI REST API
                     │
                     ▼
               Frontend Client
```

---

# Dataset

**Dataset**

NIH ChestX-ray14

Original Dataset:

- 112,120 Chest X-ray Images
- 30,000+ Patients

For this project, the dataset was filtered to include only seven clinically relevant thoracic diseases.

Selected Diseases:

- Atelectasis
- Consolidation
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumothorax

---

# Model Architecture

Backbone:

- EfficientNet-B0
- ImageNet Pretrained Weights

Classifier Head:

```
EfficientNet Features
        │
Linear (1280 → 512)
BatchNorm
ReLU
Dropout(0.4)

        │
Linear (512 → 256)
BatchNorm
ReLU
Dropout(0.3)

        │
Linear (256 → 7)
Sigmoid
```

Loss Function:

- BCEWithLogitsLoss

Optimizer:

- AdamW

Regularization:

- Weight Decay
- Dropout
- Batch Normalization

---

# Data Augmentation

Training images undergo several augmentation techniques:

- Horizontal Flip
- Random Brightness & Contrast
- Shift
- Scale
- Rotation
- CLAHE
- Gaussian Blur
- Gaussian Noise
- ImageNet Normalization

---

# Evaluation

Evaluation Metric:

Primary Metric:

- Macro F1 Score

Additional Metrics:

- Precision
- Recall
- Classification Report
- Multi-label Confusion Matrix

## Final Test Results

| Metric | Score |
|---------|--------|
| Precision | **0.3352** |
| Recall | **0.7149** |
| Macro F1 | **0.4425** |

---

# Per-Class Performance

| Disease | Precision | Recall | F1 |
|----------|----------:|-------:|------:|
| Atelectasis | 0.36 | 0.74 | 0.48 |
| Consolidation | 0.15 | 0.80 | 0.25 |
| Effusion | 0.52 | 0.81 | 0.63 |
| Infiltration | 0.51 | 0.75 | 0.60 |
| Mass | 0.21 | 0.56 | 0.31 |
| Nodule | 0.19 | 0.60 | 0.29 |
| Pneumothorax | 0.42 | 0.75 | 0.54 |

---

# Explainability

Medical AI should not behave like a black box.

LungVision AI integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight the regions of the chest X-ray that contribute most to each disease prediction.

Benefits include:

- Improved interpretability
- Increased clinician trust
- Visual validation of model attention
- Explainable decision support

---

# Backend

Framework:

- FastAPI

Endpoints:

```
GET /
```

Health endpoint.

```
POST /predict
```

Upload a chest X-ray image and receive:

- Disease probabilities
- Disease predictions
- Grad-CAM visualization

---

# Tech Stack

### Machine Learning

- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-Learn

### Computer Vision

- OpenCV
- Albumentations
- Pillow

### Explainability

- Grad-CAM

### Backend

- FastAPI
- Uvicorn

### Deployment

- Render

---

# Project Structure

```
LungVision-AI
│
├── artifacts/
│   ├── best_model.pth
│   └── training_curves.png
│
├── src/
│   ├── api/
│   ├── data/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── runtime.txt
├── render.yaml
└── README.md
```

---

# Running Locally

Clone the repository

```bash
git clone https://github.com/yourusername/LungVision-AI.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
python -m uvicorn src.api.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

# Future Improvements

- Support additional thoracic diseases
- Threshold optimization per disease
- Model ensembling
- DICOM image support
- Docker containerization
- Authentication
- Cloud deployment optimization
- Model monitoring
- Clinical validation

---

# Limitations

- Trained on the NIH ChestX-ray14 dataset.
- Performance varies across disease classes due to dataset imbalance.
- Some thoracic diseases exhibit overlapping radiographic characteristics, making differentiation challenging.
- Intended as a decision-support tool and **not** a replacement for professional medical diagnosis.

---

# License

This project is intended for educational and research purposes.

---

# Author

**Rahman Bolarinwa**

Machine Learning Engineer

- GitHub: https://github.com/Rbolarinwa-DS
- x handle :https//x.com/R_mzy001
- 
