"""
monet_pipeline/evaluation/metrics.py
====================================
Memorization-informed Fréchet Inception Distance (MiFID) — PyTorch implementation.

MiFID Formula
-------------
    MiFID = FID / (cosine_thresholded + ε)

    where FID = ||μ₁-μ₂||² + Tr(Σ₁+Σ₂-2√(Σ₁Σ₂))
    and cosine_thresholded = cosine_distance  if < eps  else 1.0

Public API
----------
    InceptionV3FeatureExtractor  — nn.Module wrapping pretrained InceptionV3
    get_activations(...)          — batch-safe pool_3 feature extraction
    calculate_activation_statistics(...)
    calculate_frechet_distance(...)
    cosine_distance(...)
    distance_thresholding(...)
    calculate_mifid(...)          — primary entry point, returns dict
"""

from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy import linalg
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INCEPTION_OUTPUT_DIM: int = 2048
COSINE_DISTANCE_EPS: float = 0.1
FID_EPSILON: float = 1e-15


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------


class InceptionV3FeatureExtractor(nn.Module):
    """Returns pool_3 (2048-d) features from pretrained InceptionV3.

    The final FC layer is replaced with ``nn.Identity`` so the forward pass
    returns the flattened avgpool activations — equivalent to the
    ``pool_3:0`` tensor the reference notebook extracted from the frozen TF graph.

    Parameters
    ----------
    device : torch.device
        Device to load the model weights onto.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        # torchvision ≥0.13 raises ValueError if aux_logits=False is passed to
        # the constructor while using pretrained weights — it forces aux_logits=True
        # at construction time. Load with defaults, then disable the aux branch.
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Disable auxiliary classifier post-hoc (safe in eval mode)
        inception.aux_logits = False
        inception.AuxLogits = None
        # Replace the final classifier with Identity to expose pool_3 (2048-d) features
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        self.device = device
        # ImageNet normalisation expected by torchvision InceptionV3
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract pool_3 (2048-d) features.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Float32, shape ``(N, 3, H, W)``, values in ``[0, 1]``.
            H, W must be ≥ 75 (256 for Monet images).

        Returns
        -------
        torch.Tensor
            Shape ``(N, 2048)``.
        """
        input_tensor = self.normalize(input_tensor)
        out = self.model(input_tensor)
        # In training mode inception_v3 may return InceptionOutputs(logits, aux).
        # In eval mode (our case) it returns a plain tensor — guard anyway.
        if hasattr(out, "logits"):
            out = out.logits
        # Squeeze any residual spatial dims
        while out.ndim > 2:
            out = out.squeeze(-1)
        return cast(torch.Tensor, out)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _to_tensor(images: NDArray[Any], device: torch.device) -> torch.Tensor:
    """Convert numpy NHWC uint8/float image batch to NCHW float32 [0,1] tensor."""
    arr = images.astype(np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr /= 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def get_activations(
    images: NDArray[Any],
    extractor: InceptionV3FeatureExtractor,
    batch_size: int = 32,
    verbose: bool = False,
) -> NDArray[np.float32]:
    """Compute InceptionV3 pool_3 activations for an image array.

    Mirrors ``get_activations`` from the reference notebook but uses PyTorch.

    Parameters
    ----------
    images : np.ndarray
        Shape ``(N, H, W, 3)``. uint8 ``[0,255]`` or float32 ``[0,1]``.
    extractor : InceptionV3FeatureExtractor
        Pre-built extractor (avoids reloading weights per call).
    batch_size : int
        Images per forward pass. Reduce on low-VRAM GPUs.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2048)``, float32.
    """
    num_images = images.shape[0]
    batch_size = min(batch_size, num_images)
    n_batches = (num_images + batch_size - 1) // batch_size
    pred = np.empty((num_images, INCEPTION_OUTPUT_DIM), dtype=np.float32)

    itr = (
        tqdm(range(n_batches), desc="InceptionV3 features", unit="batch")
        if verbose
        else range(n_batches)
    )

    for i in itr:
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, num_images)
        batch_tensor = _to_tensor(images[start_idx:end_idx], extractor.device)
        with torch.no_grad():
            batch_features = extractor(batch_tensor)
        pred[start_idx:end_idx] = batch_features.cpu().numpy()
        del batch_tensor, batch_features
        if extractor.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return pred


def calculate_activation_statistics(
    images: NDArray[Any],
    extractor: InceptionV3FeatureExtractor,
    batch_size: int = 32,
    verbose: bool = False,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute mean, covariance, and raw feature matrix of InceptionV3 activations.

    Parameters
    ----------
    images : np.ndarray
        Shape ``(N, H, W, 3)``.
    extractor : InceptionV3FeatureExtractor
    batch_size : int
    verbose : bool

    Returns
    -------
    mu : np.ndarray, shape (2048,)
    sigma : np.ndarray, shape (2048, 2048)
    features : np.ndarray, shape (N, 2048)
        Raw feature vectors retained for memorization distance computation.
    """
    features = get_activations(images, extractor, batch_size=batch_size, verbose=verbose)
    mu = np.mean(features, axis=0).astype(np.float32)
    sigma = np.cov(features, rowvar=False).astype(np.float32)
    return mu, sigma, features


# ---------------------------------------------------------------------------
# Fréchet Distance
# ---------------------------------------------------------------------------


def calculate_frechet_distance(
    mu1: NDArray[Any],
    sigma1: NDArray[Any],
    mu2: NDArray[Any],
    sigma2: NDArray[Any],
    eps: float = 1e-6,
) -> float:
    """Fréchet Distance between two multivariate Gaussians (Sutherland stable version).

    FD = ||μ₁-μ₂||² + Tr(Σ₁ + Σ₂ - 2·√(Σ₁·Σ₂))

    Parameters
    ----------
    mu1, mu2 : np.ndarray, shape (2048,)
    sigma1, sigma2 : np.ndarray, shape (2048, 2048)
    eps : float
        Diagonal offset added when matrix product is near-singular.

    Returns
    -------
    float
        Fréchet distance (lower = better quality).
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        warnings.warn(f"Singular covariance product; adding eps={eps} to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Large imaginary component: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


# ---------------------------------------------------------------------------
# Memorization distance
# ---------------------------------------------------------------------------


def normalize_rows(input_matrix: NDArray[Any]) -> NDArray[np.float32]:
    """L2-normalise each row. Zero rows stay zero (no NaN produced)."""
    norm = np.linalg.norm(input_matrix, ord=2, axis=1, keepdims=True)
    return cast(NDArray[np.float32], np.nan_to_num(input_matrix / norm))


def cosine_distance(features1: NDArray[Any], features2: NDArray[Any]) -> float:
    """Mean minimum cosine distance — the memorization component of MiFID.

    For each generated image, finds its nearest reference image in cosine
    space. Low values (~0) indicate memorization; high values (~1) indicate
    diversity.

    Parameters
    ----------
    features1 : np.ndarray, shape (N, 2048)  — generated
    features2 : np.ndarray, shape (M, 2048)  — reference

    Returns
    -------
    float in [0, 1]
    """
    f1 = features1[np.sum(features1, axis=1) != 0]
    f2 = features2[np.sum(features2, axis=1) != 0]
    nf1 = normalize_rows(f1)
    nf2 = normalize_rows(f2)
    distances = 1.0 - np.abs(np.matmul(nf1, nf2.T))  # (N, M) cosine distance matrix
    return float(np.mean(np.min(distances, axis=1)))


def distance_thresholding(distance: float, eps: float = COSINE_DISTANCE_EPS) -> float:
    """Clamp memorization distance: values >= eps become 1.0 (no penalty).

    Parameters
    ----------
    distance : float  — raw cosine distance
    eps : float — threshold (default 0.1)

    Returns
    -------
    float
    """
    return distance if distance < eps else 1.0


# ---------------------------------------------------------------------------
# Top-level MiFID entry point
# ---------------------------------------------------------------------------


def calculate_mifid(
    generated_images: NDArray[Any],
    reference_images: NDArray[Any],
    batch_size: int = 32,
    device: Optional[str] = None,
    cosine_eps: float = COSINE_DISTANCE_EPS,
    fid_epsilon: float = FID_EPSILON,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute the full MiFID score.

    MiFID = FID / (distance_thresholded + fid_epsilon)

    Parameters
    ----------
    generated_images : np.ndarray
        Shape ``(N, H, W, 3)``. Stylized / generated images.
        Values: uint8 [0,255] or float32 [0,1]. H, W ≥ 75 (256 for Monet).
    reference_images : np.ndarray
        Shape ``(M, H, W, 3)``. Real reference images (Monet paintings).
    batch_size : int
        InceptionV3 forward-pass batch size.
    device : str or None
        ``"cuda"`` / ``"cpu"``. Auto-detected if None.
    cosine_eps : float
        Memorization threshold (default 0.1).
    fid_epsilon : float
        Denominator stabiliser (default 1e-15).
    verbose : bool
        Print progress and intermediate values.

    Returns
    -------
    dict
        Keys: ``fid``, ``cosine_distance``, ``cosine_thresholded``, ``mifid``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if verbose:
        print(
            f"[MiFID] device={dev}\n"
            f"  generated={generated_images.shape}\n"
            f"  reference={reference_images.shape}"
        )

    extractor = InceptionV3FeatureExtractor(dev)

    if verbose:
        print("[MiFID] Computing statistics for generated images …")
    mu1, sigma1, feat1 = calculate_activation_statistics(
        generated_images, extractor, batch_size=batch_size, verbose=verbose
    )

    if verbose:
        print("[MiFID] Computing statistics for reference images …")
    mu2, sigma2, feat2 = calculate_activation_statistics(
        reference_images, extractor, batch_size=batch_size, verbose=verbose
    )

    fid_val = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    dist_raw = cosine_distance(feat1, feat2)
    dist_thr = distance_thresholding(dist_raw, eps=cosine_eps)
    mifid = fid_val / (dist_thr + fid_epsilon)

    if verbose:
        print(
            f"[MiFID] FID={fid_val:.4f}  cos_raw={dist_raw:.6f}  "
            f"cos_thr={dist_thr:.6f}  MiFID={mifid:.4f}"
        )

    return {
        "fid": float(fid_val),
        "cosine_distance": float(dist_raw),
        "cosine_thresholded": float(dist_thr),
        "mifid": float(mifid),
    }


# ---------------------------------------------------------------------------
# Extended Metrics Suite (FID, L1, Latency)
# ---------------------------------------------------------------------------


def calculate_fid(
    generated_images: NDArray[Any],
    reference_images: NDArray[Any],
    batch_size: int = 32,
    device: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """Compute the standard Fréchet Inception Distance (FID).

    Parameters
    ----------
    generated_images : np.ndarray
        Shape ``(N, H, W, 3)``. Generated / stylized images.
    reference_images : np.ndarray
        Shape ``(M, H, W, 3)``. Real reference images.
    batch_size : int
        InceptionV3 forward-pass batch size.
    device : str or None
        ``"cuda"`` / ``"cpu"``. Auto-detected if None.
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Fréchet Inception Distance (lower = better quality).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    extractor = InceptionV3FeatureExtractor(dev)

    if verbose:
        print("[FID] Computing statistics for generated images …")
    mu1, sigma1, _ = calculate_activation_statistics(
        generated_images, extractor, batch_size=batch_size, verbose=verbose
    )

    if verbose:
        print("[FID] Computing statistics for reference images …")
    mu2, sigma2, _ = calculate_activation_statistics(
        reference_images, extractor, batch_size=batch_size, verbose=verbose
    )

    fid_val = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_val


def calculate_l1_loss(
    tensor_a: torch.Tensor | NDArray[Any], tensor_b: torch.Tensor | NDArray[Any]
) -> float:
    """Calculate the L1 Loss (Mean Absolute Error) between two tensors or arrays.

    Parameters
    ----------
    tensor_a : torch.Tensor or np.ndarray
        First tensor/array.
    tensor_b : torch.Tensor or np.ndarray
        Second tensor/array.

    Returns
    -------
    float
        L1 loss / mean absolute error.
    """
    if isinstance(tensor_a, torch.Tensor) and isinstance(tensor_b, torch.Tensor):
        return float(torch.mean(torch.abs(tensor_a - tensor_b)).item())

    x_np = np.asarray(tensor_a, dtype=np.float32)
    y_np = np.asarray(tensor_b, dtype=np.float32)
    return float(np.mean(np.abs(x_np - y_np)))


T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])


class measure_latency:
    """Context manager and decorator to track execution latency in milliseconds.

    Can be used as a context manager:
        with measure_latency() as tracker:
            model(input_tensor)
        print(tracker.latency_ms)

    Or as a decorator:
        @measure_latency()
        def run_inference():
            ...
    """

    def __init__(self, callback: Callable[[float], None] | None = None) -> None:
        """Initialize latency measurement wrapper.

        Parameters
        ----------
        callback : Callable[[float], None] or None
            Optional function called with the measured latency in milliseconds on completion.
        """
        self.callback = callback
        self.latency_ms = 0.0
        self.start = 0.0
        self.end = 0.0

    def __enter__(self) -> measure_latency:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end = time.perf_counter()
        self.latency_ms = (self.end - self.start) * 1000.0
        if self.callback is not None:
            self.callback(self.latency_ms)

    def __call__(self, func: T_Callable) -> T_Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return cast(T_Callable, wrapper)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def main(
    generated_dir: str,
    reference_dir: str,
    batch_size: int = 32,
    image_size: int = 256,
    device: Optional[str] = None,
) -> None:
    """Compute MiFID between two image directories."""
    from PIL import Image

    def _load(directory_path: str, size: int) -> NDArray[np.uint8]:
        path_obj = Path(directory_path)
        files = sorted(path_obj.glob("*.jpg")) + sorted(path_obj.glob("*.png"))
        loaded_images = []
        for file_path in files:
            img = Image.open(file_path).convert("RGB")
            resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
            loaded_images.append(np.array(resized_img))
        return np.stack(loaded_images)

    res = calculate_mifid(
        _load(generated_dir, image_size),
        _load(reference_dir, image_size),
        batch_size=batch_size,
        device=device,
    )
    print("\n=== MiFID Results ===")
    for key, val in res.items():
        print(f"  {key:25s}: {val:.6f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
