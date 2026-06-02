"""
scripts/make_dataset_splits.py
==============================
Script to deterministically split the raw dataset into static, reproducible
splits (Train, Validation, Test) and export them as manifest CSV files.

This solves the issue of dynamic runtime seeds failing to guarantee
reproducibility across different CPU architectures.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    raw_path = Path("data/raw/monet_dataset")
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. Locate photo and monet directories
    photo_dir = raw_path / "photo_jpg"
    if not photo_dir.exists():
        photo_dir = raw_path / "photo"

    monet_dir = raw_path / "monet_jpg"
    if not monet_dir.exists():
        monet_dir = raw_path / "monet"

    if not photo_dir.exists():
        raise FileNotFoundError(f"Could not find photo directory at {photo_dir}")
    if not monet_dir.exists():
        raise FileNotFoundError(f"Could not find monet directory at {monet_dir}")

    # 2. Glob files and sort lexicographically for exact reproducibility
    photo_images = sorted(photo_dir.glob("*.jpg"))
    monet_images = sorted(monet_dir.glob("*.jpg"))

    print(f"Found {len(photo_images)} photos and {len(monet_images)} Monet paintings.")

    # 3. Standardize paths to use portable forward slashes
    photo_images_str = [path_val.as_posix() for path_val in photo_images]
    monet_images_str = [path_val.as_posix() for path_val in monet_images]

    # 4. Perform the exact split distributions:
    # Train: 5,000 photos, 250 Monet paintings
    # Val:   1,028 photos,  50 Monet paintings
    # Test:  1,000 photos,   0 Monet paintings
    if len(photo_images_str) < 7028:
        raise ValueError(
            f"Insufficient photo images: found {len(photo_images_str)}, need at least 7028."
        )
    if len(monet_images_str) < 300:
        raise ValueError(
            f"Insufficient Monet images: found {len(monet_images_str)}, need at least 300."
        )

    # Slice photo files
    train_photos = photo_images_str[0:5000]
    val_photos = photo_images_str[5000:6028]
    test_photos = photo_images_str[6028:7028]

    # Slice Monet files
    train_monet = monet_images_str[0:250]
    val_monet = monet_images_str[250:300]
    test_monet = monet_images_str[300:300]  # empty list (0 paintings)

    # 5. Construct dataset records
    train_records = [{"image_path": path_val, "domain": "photo"} for path_val in train_photos] + [
        {"image_path": path_val, "domain": "monet"} for path_val in train_monet
    ]
    val_records = [{"image_path": path_val, "domain": "photo"} for path_val in val_photos] + [
        {"image_path": path_val, "domain": "monet"} for path_val in val_monet
    ]
    test_records = [{"image_path": path_val, "domain": "photo"} for path_val in test_photos] + [
        {"image_path": path_val, "domain": "monet"} for path_val in test_monet
    ]

    # 6. Save as CSVs using pandas
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)

    train_df.to_csv(processed_path / "train_manifest.csv", index=False)
    val_df.to_csv(processed_path / "val_manifest.csv", index=False)
    test_df.to_csv(processed_path / "test_manifest.csv", index=False)

    print("\nDataset splits generated successfully:")
    print(
        f"  Train Set: {len(train_df)} total records "
        f"({len(train_photos)} photos, {len(train_monet)} monet)"
    )
    print(
        f"  Val Set:   {len(val_df)} total records "
        f"({len(val_photos)} photos, {len(val_monet)} monet)"
    )
    print(
        f"  Test Set:  {len(test_df)} total records "
        f"({len(test_photos)} photos, {len(test_monet)} monet)"
    )


if __name__ == "__main__":
    main()
