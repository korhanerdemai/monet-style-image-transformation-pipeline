"""
src/data/make_dataset_splits.py
===============================
Script to deterministically split the raw dataset into static, reproducible
splits (Train, Validation, Test) and export them as manifest CSV files.

This solves the issue of dynamic runtime seeds failing to guarantee
reproducibility across different CPU architectures.
"""

import os
import glob
import pandas as pd

def main() -> None:
    raw_dir = "data/raw/monet_dataset"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Locate photo and monet directories
    photo_dir = os.path.join(raw_dir, "photo_jpg")
    if not os.path.exists(photo_dir):
        photo_dir = os.path.join(raw_dir, "photo")

    monet_dir = os.path.join(raw_dir, "monet_jpg")
    if not os.path.exists(monet_dir):
        monet_dir = os.path.join(raw_dir, "monet")

    if not os.path.exists(photo_dir):
        raise FileNotFoundError(f"Could not find photo directory at {photo_dir}")
    if not os.path.exists(monet_dir):
        raise FileNotFoundError(f"Could not find monet directory at {monet_dir}")

    # 2. Glob files and sort lexicographically for exact reproducibility
    photo_images = sorted(glob.glob(os.path.join(photo_dir, "*.jpg")))
    monet_images = sorted(glob.glob(os.path.join(monet_dir, "*.jpg")))

    print(f"Found {len(photo_images)} photos and {len(monet_images)} Monet paintings.")

    # 3. Standardize paths to use portable forward slashes
    photo_images = [p.replace("\\", "/") for p in photo_images]
    monet_images = [p.replace("\\", "/") for p in monet_images]

    # 4. Perform the exact split distributions:
    # Train: 5,000 photos, 250 Monet paintings
    # Val:   1,028 photos,  50 Monet paintings
    # Test:  1,000 photos,   0 Monet paintings
    if len(photo_images) < 7028:
        raise ValueError(f"Insufficient photo images: found {len(photo_images)}, need at least 7028.")
    if len(monet_images) < 300:
        raise ValueError(f"Insufficient Monet images: found {len(monet_images)}, need at least 300.")

    # Slice photo files
    train_photos = photo_images[0:5000]
    val_photos = photo_images[5000:6028]
    test_photos = photo_images[6028:7028]

    # Slice Monet files
    train_monet = monet_images[0:250]
    val_monet = monet_images[250:300]
    test_monet = monet_images[300:300]  # empty list (0 paintings)

    # 5. Construct dataset records
    train_records = (
        [{"image_path": p, "domain": "photo"} for p in train_photos] +
        [{"image_path": p, "domain": "monet"} for p in train_monet]
    )
    val_records = (
        [{"image_path": p, "domain": "photo"} for p in val_photos] +
        [{"image_path": p, "domain": "monet"} for p in val_monet]
    )
    test_records = (
        [{"image_path": p, "domain": "photo"} for p in test_photos] +
        [{"image_path": p, "domain": "monet"} for p in test_monet]
    )

    # 6. Save as CSVs using pandas
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)

    train_df.to_csv(os.path.join(processed_dir, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_manifest.csv"), index=False)

    print("\nDataset splits generated successfully:")
    print(f"  Train Set: {len(train_df)} total records ({len(train_photos)} photos, {len(train_monet)} monet)")
    print(f"  Val Set:   {len(val_df)} total records ({len(val_photos)} photos, {len(val_monet)} monet)")
    print(f"  Test Set:  {len(test_df)} total records ({len(test_photos)} photos, {len(test_monet)} monet)")

if __name__ == "__main__":
    main()
