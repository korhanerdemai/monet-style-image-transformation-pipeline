"""
monet_pipeline/evaluation/evaluate_baseline.py
==============================================
Baseline evaluation pipeline for the Monet style transformation project.
Loads the test set manifest, runs each photo through the trained AdaIN baseline,
measures inference latency in milliseconds, calculates FID and MiFID quality scores,
and exports all results to metrics/baseline_metrics.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from monet_pipeline.evaluation.metrics import calculate_mifid, measure_latency
from monet_pipeline.models.baseline_adain import AdaINStyleTransfer


def _load_image_as_numpy(path: str | Path, size: int = 256) -> NDArray[np.uint8]:
    """Load a JPEG → resize → return uint8 HWC numpy array."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _tensor_to_numpy_uint8(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """Convert (1, 3, H, W) [-1, 1] tensor → uint8 HWC numpy array [0, 255]."""
    # Scale from [-1, 1] -> [0, 1]
    tensor = (tensor + 1.0) / 2.0
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cast(NDArray[np.uint8], (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def evaluate_baseline(
    test_manifest_path: str | Path = "data/processed/test_manifest.csv",
    val_manifest_path: str | Path = "data/processed/val_manifest.csv",
    weights_path: str | Path = "models/baseline_decoder.pth",
    n_photos: Optional[int] = None,
    batch_size: int = 16,
    image_size: int = 256,
    device: Optional[str] = None,
    seed: int = 42,
    output_path: str | Path = "metrics/baseline_metrics.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full AdaIN baseline evaluation and return the metrics dict.

    Parameters
    ----------
    test_manifest_path : str or Path
        Path to the test set CSV manifest containing photos.
    val_manifest_path : str or Path
        Path to the validation set CSV manifest containing reference Monet paintings.
    weights_path : str or Path
        Path to the saved baseline decoder weights (.pth).
    n_photos : int or None
        Number of landscape photos to stylize. If None, evaluates all photos in test manifest.
    batch_size : int
        InceptionV3 forward-pass batch size for FID/MiFID calculation.
    image_size : int
        Spatial resolution for all images. Default: 256.
    device : str or None
        ``"cuda"`` / ``"cpu"``. Auto-detected if None.
    seed : int
        Random seed for reproducible photo sampling.
    output_path : str or Path
        Path to write the JSON results file.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Full metrics dictionary including FID, MiFID, and Inference Latency.
    """
    random.seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    print(f"\n{'='*60}")
    print("  Monet CycleGAN — AdaIN Baseline Evaluation")
    print(f"{'='*60}")
    print(f"  Test Manifest : {test_manifest_path}")
    print(f"  Val Manifest  : {val_manifest_path}")
    print(f"  Weights Path  : {weights_path}")
    print(f"  Device        : {device}")
    print(f"  Image size    : {image_size}x{image_size}")
    print(f"{'='*60}\n")

    # 1. Collect file paths from the static manifests
    test_manifest_path = Path(test_manifest_path)
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test set manifest CSV not found at {test_manifest_path}")

    test_df = pd.read_csv(test_manifest_path)
    photo_files = [
        Path(photo_path)
        for photo_path in test_df[test_df["domain"] == "photo"]["image_path"].tolist()
    ]

    val_manifest_path = Path(val_manifest_path)
    if not val_manifest_path.exists():
        raise FileNotFoundError(f"Validation set manifest CSV not found at {val_manifest_path}")

    val_df = pd.read_csv(val_manifest_path)
    monet_files = [
        Path(monet_path)
        for monet_path in val_df[val_df["domain"] == "monet"]["image_path"].tolist()
    ]

    print(
        f"[Dataset] Found {len(monet_files)} Monet paintings in validation, "
        f"{len(photo_files)} photos in test."
    )

    # Sample subset of photos if n_photos is specified
    if n_photos is not None:
        sampled_photos = random.sample(photo_files, min(n_photos, len(photo_files)))
    else:
        sampled_photos = photo_files

    # Choose a deterministic style reference painting from the Monet files
    style_path = monet_files[0]
    print(f"[Dataset] Target photos: {len(sampled_photos)}; Style reference: {style_path.name}\n")

    # 2. Initialize AdaIN Style Transfer Model
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained decoder weights file not found at {weights_path}")

    print("[AdaIN] Initializing AdaIN model and loading decoder weights…")
    model = AdaINStyleTransfer().to(device_obj)
    model.eval()
    model.decoder.load_state_dict(torch.load(weights_path, map_location=device_obj))

    # Preprocess style image
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )

    style_img = Image.open(style_path).convert("RGB")
    style_tensor = transform(style_img).unsqueeze(0).to(device_obj) * 2.0 - 1.0

    # 3. Run feed-forward pass on each photo and measure latency
    stylized_images: List[NDArray[np.uint8]] = []
    latencies_ms: List[float] = []

    print("[Inference] Running feed-forward passes…")
    for photo_path in tqdm(sampled_photos, desc="AdaIN stylization", unit="img"):
        content_img = Image.open(photo_path).convert("RGB")
        content_tensor = transform(content_img).unsqueeze(0).to(device_obj) * 2.0 - 1.0

        # Run forward pass wrapping only the model execution in measure_latency
        with measure_latency() as tracker:
            with torch.no_grad():
                result_tensor = model(content_tensor, style_tensor)
                # Synchronize CUDA to measure actual device latency if GPU is used
                if device_obj.type == "cuda":
                    torch.cuda.synchronize()

        latencies_ms.append(tracker.latency_ms)
        stylized_images.append(_tensor_to_numpy_uint8(result_tensor))

        # Explicit cleanup
        del content_tensor, result_tensor
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    avg_latency = float(np.mean(latencies_ms))
    print(
        f"\n[Inference] Processed {len(stylized_images)} images. "
        f"Average Latency: {avg_latency:.2f} ms/image"
    )

    # Stack generated images HWC to (N, H, W, 3) uint8 numpy array
    generated_arr = np.stack(stylized_images, axis=0)

    # 4. Load reference Monet paintings as numpy arrays
    print("[Reference] Loading real Monet paintings for evaluation…")
    reference_images: List[NDArray[np.uint8]] = []
    for mp in tqdm(monet_files, desc="Loading reference paintings", unit="img"):
        reference_images.append(_load_image_as_numpy(mp, size=image_size))
    reference_arr = np.stack(reference_images, axis=0)
    print(f"[Reference] Loaded {len(reference_images)} reference paintings {reference_arr.shape}\n")

    # 5. Compute MiFID and FID
    print("[Metrics] Computing FID and MiFID quality scores…")
    mifid_results = calculate_mifid(
        generated_images=generated_arr,
        reference_images=reference_arr,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    fid_val = mifid_results["fid"]
    mifid_val = mifid_results["mifid"]

    # 6. Assemble and persist results in metrics/baseline_metrics.json
    full_results = {
        "phase": "baseline",
        "model": "AdaIN Style Transfer",
        "config": {
            "n_photos": len(sampled_photos),
            "image_size": image_size,
            "device": str(device),
            "seed": seed,
            "style_reference": style_path.name,
            "weights_path": str(weights_path),
        },
        "metrics": {
            "mifid": float(mifid_val),
            "fid": float(fid_val),
            "latency_ms": float(avg_latency),
            "cycle_consistency_loss": None,  # N/A for feed-forward AdaIN
            "identity_loss": None,  # N/A for feed-forward AdaIN
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as metrics_file:
        json.dump(full_results, metrics_file, indent=2)

    # 7. Print Summary
    print(f"\n{'='*60}")
    print("  BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    print("  Model             : AdaIN Style Transfer")
    print(f"  Photos stylized   : {len(sampled_photos)}")
    print(f"  Monet references  : {len(reference_images)}")
    print("  -- Metrics ----------------------------------")
    print(f"  FID               : {fid_val:.4f}")
    print(f"  MiFID             : {mifid_val:.4f}")
    print(f"  Inference Latency : {avg_latency:.2f} ms/image")
    print(f"{'='*60}")
    print(f"  Results saved to  : {output_path}")
    print(f"{'='*60}\n")

    return full_results


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate_baseline)
