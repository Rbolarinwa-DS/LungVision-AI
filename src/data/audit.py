import os
import pandas as pd

# Load metadata
df = pd.read_csv(
    "data/raw/chestxray14/Data_Entry_2017_v2020.csv"
)

# Image folder
image_dir = "data/raw/chestxray14/dif_nih14"

# Get all downloaded images
image_files = set(os.listdir(image_dir))

print("Downloaded Images:", len(image_files))

# Match metadata to downloaded images
matched_df = df[df["Image Index"].isin(image_files)]

print("Matched Images:", len(matched_df))

print("Missing Images:", len(image_files) - len(matched_df))

import os
import pandas as pd

# ----------------------------
# Load metadata
# ----------------------------
df = pd.read_csv("data/raw/chestxray14/Data_Entry_2017_v2020.csv")

# ----------------------------
# Downloaded images
# ----------------------------
image_dir = "data/raw/chestxray14/dif_nih14"
image_files = set(os.listdir(image_dir))

# Keep only downloaded images
df = df[df["Image Index"].isin(image_files)]

# ----------------------------
# Target diseases
# ----------------------------
TARGET_DISEASES = [
    "Atelectasis",
    "Consolidation",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax"
]

# ----------------------------
# Count diseases
# ----------------------------
print("Disease Distribution")
print("-" * 40)

for disease in TARGET_DISEASES:
    count = df["Finding Labels"].str.contains(disease).sum()
    print(f"{disease:<18} {count:,}")

# ----------------------------
# No Finding
# ----------------------------
no_finding = (df["Finding Labels"] == "No Finding").sum()

print("-" * 40)
print(f"{'No Finding':<18} {no_finding:,}")

# ----------------------------
# Patient Statistics
# ----------------------------
unique_patients = df["Patient ID"].nunique()

print("\nPatient Statistics")
print("-" * 40)
print(f"Total Images      : {len(df):,}")
print(f"Unique Patients   : {unique_patients:,}")

avg_images = len(df) / unique_patients
print(f"Avg Images/Patient: {avg_images:.2f}")

# ----------------------------
# Save processed metadata
# ----------------------------
output_path = "data/processed/metadata.csv"

df.to_csv(output_path, index=False)

print(f"\nProcessed metadata saved to: {output_path}")
