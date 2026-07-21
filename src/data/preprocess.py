from pathlib import Path
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

TARGET_DISEASES = [
    "Atelectasis",
    "Consolidation",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax",
]

IMAGE_DIR = Path("data/raw/chestxray14/dif_nih14")

# ==========================================================
# Data Loading
# ==========================================================

def load_metadata():
    """
    Load audited metadata.
    """

    metadata_path = Path("data/processed/metadata.csv")

    return pd.read_csv(metadata_path)


# ==========================================================
# Data Filtering
# ==========================================================

def contains_target_disease(label_string):
    """
    Return True if an image contains at least one
    target disease.
    """

    labels = label_string.split("|")

    return any(disease in labels for disease in TARGET_DISEASES)


def filter_target_diseases(df):
    """
    Keep only images containing one or more
    target diseases.
    """

    return df[
        df["Finding Labels"].apply(contains_target_disease)
    ].copy()


# ==========================================================
# Multi-label Encoding
# ==========================================================

def encode_labels(df):
    """
    Create one binary column per disease.
    """

    df = df.copy()

    for disease in TARGET_DISEASES:

        df[disease] = df["Finding Labels"].apply(
            lambda x: int(disease in x.split("|"))
        )

    return df


# ==========================================================
# Image Paths
# ==========================================================

def add_image_paths(df):
    """
    Add full image path for every image.
    """

    df = df.copy()

    df["Image Path"] = df["Image Index"].apply(
        lambda x: str(IMAGE_DIR / x)
    )

    return df


# ==========================================================
# Column Selection
# ==========================================================

def keep_required_columns(df):
    """
    Keep only columns needed throughout the project.
    """

    columns = [
        "Image Index",
        "Image Path",
        "Patient ID",
        "Patient Age",
        "Patient Sex",
        "View Position",
    ] + TARGET_DISEASES

    return df[columns]


# ==========================================================
# Save
# ==========================================================

def save_processed_metadata(df):

    output_path = Path("data/processed/processed_metadata.csv")

    df.to_csv(output_path, index=False)

    print(f"\nProcessed metadata saved to:")
    print(output_path)


# ==========================================================
# Main
# ==========================================================

def main():

    df = load_metadata()

    print("=" * 50)
    print("Original Dataset")
    print("=" * 50)
    print(f"Images : {len(df):,}")

    df = filter_target_diseases(df)

    print("\n" + "=" * 50)
    print("After Filtering")
    print("=" * 50)
    print(f"Images : {len(df):,}")

    df = encode_labels(df)

    df = add_image_paths(df)

    df = keep_required_columns(df)

    save_processed_metadata(df)

    print("\nProcessed Metadata Preview\n")

    print(df.head())


if __name__ == "__main__":
    main()