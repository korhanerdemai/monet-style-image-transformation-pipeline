"""
src/evaluation/evaluate_baseline.py
====================================
Baseline evaluation pipeline for the Monet CycleGAN MLOps project.

Pipeline
--------
1. Load a fixed random subset of landscape photos and all Monet paintings
   from ``data/raw/monet_dataset/`` using the Phase 1 data loader.
2. Run each photo through the NST baseline (src/models/baseline_nst.py).
3. Compute the MiFID score comparing NST outputs vs. real Monet paintings
   (src/evaluation/metrics.py).
4. Print the MiFID score to the console and save full results to
   ``metrics/baseline_metrics.json``.

Usage
-----
Run from the repository root:

    python -m src.evaluation.evaluate_baseline

Or with CLI flags:

    python -m src.evaluation.evaluate_baseline \\
        --data_root data/raw/monet_dataset \\
        --n_photos 50 \\
        --nst_steps 150 \\
        --batch_size 16 \\
        --seed 42

Outputs
-------
metrics/baseline_metrics.json   — persistent record of scores
Console                         — live progress + final summary table
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.models.baseline_nst import NeuralStyleTransfer
from src.evaluation.metrics import calculate_mifid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_files(directory: str, ext: str = "*.jpg") -> List[str]:
    """Glob for image files under ``directory``."""
    files = sorted(glob.glob(os.path.join(directory, ext)))
    if not files:
        raise FileNotFoundError(f"No {ext!r} files found in: {directory!r}")
    return files


def _load_image_as_numpy(path: str, size: int = 256) -> np.ndarray:
    """Load a JPEG → resize → return uint8 HWC numpy array."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _tensor_to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) [0,1] tensor → uint8 HWC numpy array."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_baseline(
    data_root: str = "data/raw/monet_dataset",
    n_photos: int = 50,
    nst_steps: int = 150,
    nst_style_weight: float = 1e6,
    batch_size: int = 16,
    image_size: int = 256,
    device: Optional[str] = None,
    seed: int = 42,
    output_path: str = "metrics/baseline_metrics.json",
    verbose: bool = True,
) -> dict:
    """Run the full baseline evaluation and return the metrics dict.

    Parameters
    ----------
    data_root : str
        Root of the Monet dataset (contains ``monet_jpg/`` and ``photo_jpg/``).
    n_photos : int
        Number of landscape photos to stylize (sampled deterministically).
    nst_steps : int
        L-BFGS iterations per image in the NST baseline.
    nst_style_weight : float
        Style loss weight for NST.
    batch_size : int
        InceptionV3 forward-pass batch size for MiFID calculation.
    image_size : int
        Spatial resolution for all images. Default: 256.
    device : str or None
        ``"cuda"`` / ``"cpu"``. Auto-detected if None.
    seed : int
        Random seed for reproducible photo sampling.
    output_path : str
        Path to write the JSON results file.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Full metrics dictionary including FID, cosine distance, and MiFID.
    """
    random.seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    print(f"\n{'='*60}")
    print("  Monet CycleGAN — NST Baseline Evaluation")
    print(f"{'='*60}")
    print(f"  Data root  : {data_root}")
    print(f"  Device     : {device}")
    print(f"  Photos     : {n_photos}")
    print(f"  NST steps  : {nst_steps}")
    print(f"  Image size : {image_size}x{image_size}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Step 1 — Collect file paths
    # ------------------------------------------------------------------ #
    monet_dir = os.path.join(data_root, "monet_jpg")
    photo_dir = os.path.join(data_root, "photo_jpg")

    monet_files = _collect_files(monet_dir, "*.jpg")
    photo_files = _collect_files(photo_dir, "*.jpg")

    print(f"[Dataset] Found {len(monet_files)} Monet paintings, {len(photo_files)} photos.")

    # Sample a reproducible fixed subset of photos
    sampled_photos = random.sample(photo_files, min(n_photos, len(photo_files)))
    # Use one fixed random Monet painting as the style reference for each photo
    style_path = random.choice(monet_files)
    print(f"[Dataset] Sampled {len(sampled_photos)} photos; style reference: {Path(style_path).name}\n")

    # ------------------------------------------------------------------ #
    # Step 2 — Run NST on each photo
    # ------------------------------------------------------------------ #
    print("[NST] Initialising VGG-19 style transfer model …")
    nst_model = NeuralStyleTransfer(
        image_size=image_size,
        device=device,
        style_weight=nst_style_weight,
    )
    style_tensor = nst_model.load_image(style_path)

    stylized_images: List[np.ndarray] = []
    nst_start = time.time()

    for idx, photo_path in enumerate(tqdm(sampled_photos, desc="NST stylization", unit="img")):
        content_tensor = nst_model.load_image(photo_path)
        result_tensor = nst_model.transfer(
            content_tensor,
            style_tensor,
            steps=nst_steps,
            print_every=0,   # suppress per-step logs for batch runs
        )
        stylized_images.append(_tensor_to_numpy_uint8(result_tensor))

        # Free intermediate tensors explicitly
        del content_tensor, result_tensor
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    nst_elapsed = time.time() - nst_start
    print(f"\n[NST] Stylized {len(stylized_images)} images in {nst_elapsed:.1f}s "
          f"({nst_elapsed/len(stylized_images):.1f}s/img)\n")

    # Stack generated images: (N, H, W, 3) uint8
    generated_arr = np.stack(stylized_images, axis=0)

    # ------------------------------------------------------------------ #
    # Step 3 — Load reference Monet paintings
    # ------------------------------------------------------------------ #
    print("[Reference] Loading Monet paintings for MiFID …")
    reference_images: List[np.ndarray] = []
    for mp in tqdm(monet_files, desc="Loading Monet paintings", unit="img"):
        reference_images.append(_load_image_as_numpy(mp, size=image_size))
    reference_arr = np.stack(reference_images, axis=0)
    print(f"[Reference] Loaded {len(reference_images)} paintings  {reference_arr.shape}\n")

    # ------------------------------------------------------------------ #
    # Step 4 — Compute MiFID
    # ------------------------------------------------------------------ #
    mifid_results = calculate_mifid(
        generated_images=generated_arr,
        reference_images=reference_arr,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Assemble and persist results
    # ------------------------------------------------------------------ #
    full_results = {
        "phase": "baseline",
        "model": "NST (VGG-19)",
        "config": {
            "n_photos": len(sampled_photos),
            "nst_steps": nst_steps,
            "nst_style_weight": nst_style_weight,
            "image_size": image_size,
            "device": str(device),
            "seed": seed,
            "style_reference": Path(style_path).name,
            "data_root": data_root,
        },
        "timing": {
            "nst_total_seconds": round(nst_elapsed, 2),
            "nst_seconds_per_image": round(nst_elapsed / len(sampled_photos), 2),
        },
        "metrics": mifid_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(full_results, fp, indent=2)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("  BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Model             : NST (VGG-19, {nst_steps} steps)")
    print(f"  Photos stylized   : {len(sampled_photos)}")
    print(f"  Monet references  : {len(reference_images)}")
    print(f"  ── Metrics ──────────────────────────────────")
    print(f"  FID               : {mifid_results['fid']:.4f}")
    print(f"  Cosine distance   : {mifid_results['cosine_distance']:.6f}")
    print(f"  Cosine thresholded: {mifid_results['cosine_thresholded']:.6f}")
    print(f"  MiFID             : {mifid_results['mifid']:.4f}  ← primary metric")
    print(f"{'='*60}")
    print(f"  Results saved to  : {output_path}")
    print(f"{'='*60}\n")

    return full_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NST baseline and compute MiFID vs. Monet paintings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root", default="data/raw/monet_dataset",
        help="Dataset root containing monet_jpg/ and photo_jpg/."
    )
    parser.add_argument(
        "--n_photos", type=int, default=50,
        help="Number of landscape photos to stylize."
    )
    parser.add_argument(
        "--nst_steps", type=int, default=150,
        help="NST L-BFGS optimisation steps per image."
    )
    parser.add_argument(
        "--nst_style_weight", type=float, default=1e6,
        help="Style loss weight for NST."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="InceptionV3 batch size for MiFID."
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Spatial resolution for all images."
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device ('cuda' or 'cpu'). Auto-detected if not set."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible photo sampling."
    )
    parser.add_argument(
        "--output", default="metrics/baseline_metrics.json",
        help="Path to write the JSON results file."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose InceptionV3 progress output."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_baseline(
        data_root=args.data_root,
        n_photos=args.n_photos,
        nst_steps=args.nst_steps,
        nst_style_weight=args.nst_style_weight,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
        verbose=not args.quiet,
    )
