from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ==========================================================
# Load Metadata
# ==========================================================

def load_metadata():

    metadata_path = Path("data/processed/processed_metadata.csv")

    return pd.read_csv(metadata_path)


# ==========================================================
# Patient-wise Split
# ==========================================================

def patient_split(df):

    patients = df["Patient ID"].unique()

    train_patients, temp_patients = train_test_split(
        patients,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_df = df[df["Patient ID"].isin(train_patients)].copy()

    val_df = df[df["Patient ID"].isin(val_patients)].copy()

    test_df = df[df["Patient ID"].isin(test_patients)].copy()

    return train_df, val_df, test_df


# ==========================================================
# Save Splits
# ==========================================================

def save_splits(train_df, val_df, test_df):

    output_dir = Path("data/splits")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)

    val_df.to_csv(output_dir / "val.csv", index=False)

    test_df.to_csv(output_dir / "test.csv", index=False)


# ==========================================================
# Summary
# ==========================================================

def print_summary(train_df, val_df, test_df):

    print("\nDataset Split Summary")
    print("-" * 40)

    print(f"Train Images : {len(train_df):,}")
    print(f"Validation   : {len(val_df):,}")
    print(f"Test         : {len(test_df):,}")

    print("-" * 40)

    print(f"Train Patients : {train_df['Patient ID'].nunique():,}")
    print(f"Val Patients   : {val_df['Patient ID'].nunique():,}")
    print(f"Test Patients  : {test_df['Patient ID'].nunique():,}")


# ==========================================================
# Main
# ==========================================================

def main():

    df = load_metadata()

    train_df, val_df, test_df = patient_split(df)

    save_splits(train_df, val_df, test_df)

    print_summary(train_df, val_df, test_df)


if __name__ == "__main__":
    main()