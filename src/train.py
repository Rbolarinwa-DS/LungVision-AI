from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from src.model import create_model
from src.data.dataset import get_dataloaders

from src.utils import (
    ARTIFACTS_DIR,
    RANDOM_STATE,
    set_seed
)

# ==========================================================
# Reproducibility
# ==========================================================

set_seed(RANDOM_STATE)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print(f"\nUsing device : {DEVICE}")

# ==========================================================
# Training Configuration
# ==========================================================

STAGE1_EPOCHS = 10

STAGE2_EPOCHS = 20

LEARNING_RATE_STAGE1 = 1e-3

LEARNING_RATE_STAGE2 = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"

ARTIFACTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)

# ==========================================================
# Loss Function
# ==========================================================

POS_WEIGHT = torch.tensor(
    [
        2.7177,
        7.0730,
        1.4856,
        1.5374,
        7.1561,
        6.7951,
        5.7929
    ],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=POS_WEIGHT
)

# ==========================================================
# Optimizer Builder
# ==========================================================

def build_optimizer(

    model,

    learning_rate

):

    optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=learning_rate,
    weight_decay=5e-5
)

    scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.2,
    patience=3
)

    return optimizer, scheduler


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(

    outputs,

    labels,

    threshold=0.4

):

    probabilities = torch.sigmoid(outputs)

    predictions = (

        probabilities >= threshold

    ).float()

    predictions = predictions.cpu().numpy()

    labels = labels.cpu().numpy()

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

    return (

        precision,

        recall,

        macro_f1

    )
    
# ==========================================================
# Training Loop
# ==========================================================

def train_one_epoch(

    model,

    dataloader,

    optimizer

):

    model.train()

    running_loss = 0.0

    all_outputs = []

    all_labels = []

    progress_bar = tqdm(

        dataloader,

        desc="Training",

        leave=False

    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(

            outputs,

            labels

        )

        loss.backward()

        optimizer.step()

        running_loss += (

            loss.item()

            * images.size(0)

        )

        all_outputs.append(

            outputs.detach()

        )

        all_labels.append(

            labels.detach()

        )

        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    epoch_loss = (

        running_loss

        / len(dataloader.dataset)

    )

    outputs = torch.cat(

        all_outputs

    )

    labels = torch.cat(

        all_labels

    )

    precision, recall, macro_f1 = calculate_metrics(

        outputs,

        labels

    )

    return (

        epoch_loss,

        precision,

        recall,

        macro_f1

    )


# ==========================================================
# Validation Loop
# ==========================================================

def validate_one_epoch(

    model,

    dataloader

):

    model.eval()

    running_loss = 0.0

    all_outputs = []

    all_labels = []

    with torch.no_grad():

        progress_bar = tqdm(

            dataloader,

            desc="Validation",

            leave=False

        )

        for images, labels in progress_bar:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(

                outputs,

                labels

            )

            running_loss += (

                loss.item()

                * images.size(0)

            )

            all_outputs.append(

                outputs

            )

            all_labels.append(

                labels

            )

    epoch_loss = (

        running_loss

        / len(dataloader.dataset)

    )

    outputs = torch.cat(

        all_outputs

    )

    labels = torch.cat(

        all_labels

    )

    precision, recall, macro_f1 = calculate_metrics(

        outputs,

        labels

    )

    return (

        epoch_loss,

        precision,

        recall,

        macro_f1

    )


# ==========================================================
# Early Stopping
# ==========================================================

class EarlyStopping:

    def __init__(

        self,

        patience=EARLY_STOPPING_PATIENCE

    ):

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

        self.stop = False

    def __call__(

        self,

        validation_loss

    ):

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss

            self.counter = 0

        else:

            self.counter += 1

            print(

                f"EarlyStopping "

                f"{self.counter}/{self.patience}"

            )

            if self.counter >= self.patience:

                self.stop = True


# ==========================================================
# Save Best Model
# ==========================================================

def save_checkpoint(

    model,

    epoch,

    validation_loss

):

    checkpoint = {

        "epoch": epoch,

        "validation_loss": validation_loss,

        "model_state_dict": model.state_dict()

    }

    torch.save(

        checkpoint,

        BEST_MODEL_PATH

    )

    print(

        "\n✓ Best model saved."

    )


# ==========================================================
# Load Best Model
# ==========================================================

def load_checkpoint(

    model

):

    if not BEST_MODEL_PATH.exists():

        print(

            "\nNo previous checkpoint found."

        )

        return model

    checkpoint = torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    print(

        f"\nLoaded best model "

        f"(Epoch {checkpoint['epoch']})"

    )

    return model
# ==========================================================
# Plot Training History
# ==========================================================

def plot_history(history):

    epochs = range(

        1,

        len(history["train_loss"]) + 1

    )

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    plt.plot(

        epochs,

        history["train_loss"],

        label="Train"

    )

    plt.plot(

        epochs,

        history["val_loss"],

        label="Validation"

    )

    plt.title("Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(

        epochs,

        history["macro_f1"],

        label="Macro F1"

    )

    plt.title("Validation Macro F1")

    plt.xlabel("Epoch")

    plt.ylabel("Score")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        ARTIFACTS_DIR /

        "training_curves.png"

    )

    plt.close()


# ==========================================================
# Training Pipeline
# ==========================================================

def train_model(

    model,

    stage

):

    train_loader, val_loader, _ = get_dataloaders()
    
    if stage == 1:

        print("\n" + "="*60)

        print("Stage 1")

        print("Training classifier...")

        print("="*60)

        model.freeze_backbone()

        learning_rate = LEARNING_RATE_STAGE1

        epochs = STAGE1_EPOCHS

    else:

        print("\n" + "="*60)

        print("Stage 2")

        print("Fine-tuning backbone...")

        print("="*60)

        model.unfreeze_backbone()

        model = load_checkpoint(model)

        learning_rate = LEARNING_RATE_STAGE2

        epochs = STAGE2_EPOCHS

    optimizer, scheduler = build_optimizer(

        model,

        learning_rate

    )

    early_stopping = EarlyStopping()

    history = {

        "train_loss": [],

        "val_loss": [],

        "macro_f1": []

    }

    best_loss = float("inf")

    print(f"\nDevice : {DEVICE}")

    print(f"Epochs : {epochs}")

    start_time = time.time()

    for epoch in range(epochs):

        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_loss, _, _, _ = train_one_epoch(

            model,

            train_loader,

            optimizer

        )

        val_loss, precision, recall, macro_f1 = validate_one_epoch(

            model,

            val_loader

        )

        scheduler.step(val_loss)

        history["train_loss"].append(

            train_loss

        )

        history["val_loss"].append(

            val_loss

        )

        history["macro_f1"].append(

            macro_f1

        )

        print()

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Precision  : {precision:.4f}")

        print(f"Recall     : {recall:.4f}")

        print(f"Macro F1   : {macro_f1:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning Rate : {current_lr:.6f}")

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoint(

                model,

                epoch + 1,

                val_loss

            )

        early_stopping(val_loss)

        if early_stopping.stop:

            print("\nEarly stopping triggered.")

            break

    elapsed = (

        time.time()

        - start_time

    ) / 60

    print(

        f"\nTraining completed in "

        f"{elapsed:.2f} minutes."

    )

    plot_history(history)

    return model
# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("            LungVision AI")
    print("=" * 60)

    print("\nCreating model...")

    model = create_model().to(DEVICE)

    print("✓ Model created successfully.")

    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------

    model = train_model(

        model,

        stage=1

    )

    # ------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------

    model = train_model(

        model,

        stage=2

    )

    print("\n" + "=" * 60)
    print("Training Finished Successfully")
    print("=" * 60)

    print("\nSaved Files:")

    print(f"✓ Best Model       : {BEST_MODEL_PATH}")

    print(

        f"✓ Training Curves  : "

        f"{ARTIFACTS_DIR / 'training_curves.png'}"

    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()