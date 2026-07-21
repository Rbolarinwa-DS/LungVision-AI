from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix
)

from src.model import create_model
from src.data.dataset import get_dataloaders
from src.utils import (
    ARTIFACTS_DIR,
    TARGET_DISEASES
)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device : {DEVICE}")

# ==========================================================
# Best Model
# ==========================================================

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"

# Default threshold (will be replaced by the best one found)
THRESHOLD = 0.50

# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    if not BEST_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{BEST_MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print("\n✓ Best model loaded successfully.")
    print(f"Checkpoint Epoch : {checkpoint['epoch']}")

    return model

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model):

    _, _, test_loader = get_dataloaders()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.vstack(all_probabilities)

    labels = np.vstack(all_labels)

    return probabilities, labels


# ==========================================================
# Find Best Threshold
# ==========================================================

def find_best_threshold(probabilities, labels):

    print("\n" + "=" * 60)
    print("SEARCHING FOR BEST THRESHOLD")
    print("=" * 60)

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in np.arange(0.20, 0.81, 0.05):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )

        print(
            f"Threshold = {threshold:.2f} | "
            f"Macro F1 = {macro_f1:.4f}"
        )

        if macro_f1 > best_f1:

            best_f1 = macro_f1
            best_threshold = threshold

    print("\n" + "-" * 60)
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Macro F1  : {best_f1:.4f}")
    print("-" * 60)

    return best_threshold

# ==========================================================
# Print Evaluation Metrics
# ==========================================================

def print_metrics(predictions, labels):

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Macro F1  : {macro_f1:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            labels,
            predictions,
            target_names=TARGET_DISEASES,
            zero_division=0
        )
    )


# ==========================================================
# Confusion Matrix (Per Disease)
# ==========================================================

def print_confusion_matrices(predictions, labels):

    matrices = multilabel_confusion_matrix(
        labels,
        predictions
    )

    print("\n" + "=" * 60)
    print("PER-CLASS CONFUSION MATRICES")
    print("=" * 60)

    for disease, matrix in zip(TARGET_DISEASES, matrices):

        tn, fp, fn, tp = matrix.ravel()

        print(f"\n{disease}")
        print("-" * 40)
        print(f"True Positive : {tp}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")
        print(f"True Negative : {tn}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("LungVision AI Evaluation")
    print("=" * 60)

    model = load_model()

    probabilities, labels = evaluate_model(model)

    best_threshold = find_best_threshold(
        probabilities,
        labels
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print_metrics(
        predictions,
        labels
    )

    print_confusion_matrices(
        predictions,
        labels
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":

    main()