"""
src/data/data_loader.py
=======================
Modular PyTorch data loading pipeline for the CycleGAN Monet-style
image transformation project.

Refactored from notebooks/cyclegan-implementation.ipynb.

Classes
-------
CustomTransform
    Applies train-time augmentations (random crop, flip, jitter) or a simple
    resize for inference, then scales pixel values to [-1, 1].

CustomDataset
    A torch.utils.data.Dataset that reads JPEG images from a list of file paths
    and applies CustomTransform.

CycleGANDataModule
    A PyTorch Lightning LightningDataModule that wires together Monet and photo
    datasets for training, validation, testing, and prediction stages.

Usage
-----
    from src.data.data_loader import CycleGANDataModule

    dm = CycleGANDataModule(
        monet_dir="data/raw/monet_dataset/monet_jpg/*.jpg",
        photo_dir="data/raw/monet_dataset/photo_jpg/*.jpg",
        batch_size=1,
        sample_size=5,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms as T

try:
    import pytorch_lightning as L
    from pytorch_lightning.utilities import CombinedLoader
    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOAD_DIM: int = 286   # Resize target before random crop
DEFAULT_TARGET_DIM: int = 256  # Final spatial resolution
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_SAMPLE_SIZE: int = 5


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class CustomTransform:
    """Image transform pipeline for CycleGAN training and inference.

    During the ``"fit"`` stage applies:
        1. Resize to ``(load_dim, load_dim)``
        2. RandomCrop to ``(target_dim, target_dim)``
        3. RandomHorizontalFlip (p=0.5)
        4. ColorJitter (brightness, contrast, saturation, hue)

    During all other stages applies only a simple resize to ``(target_dim, target_dim)``.

    In both cases the output is scaled from ``[0, 1]`` to ``[-1, 1]``.

    Parameters
    ----------
    load_dim : int
        Spatial size to resize images before random cropping. Default: 286.
    target_dim : int
        Final spatial resolution of the output tensor. Default: 256.
    """

    def __init__(
        self,
        load_dim: int = DEFAULT_LOAD_DIM,
        target_dim: int = DEFAULT_TARGET_DIM,
    ) -> None:
        self.transform_train = T.Compose([
            T.Resize((load_dim, load_dim), antialias=True),
            T.RandomCrop((target_dim, target_dim)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
        ])
        self.transform_eval = T.Resize((target_dim, target_dim), antialias=True)

    def __call__(self, img: torch.Tensor, stage: Optional[str]) -> torch.Tensor:
        """Apply the transform.

        Parameters
        ----------
        img : torch.Tensor
            Float tensor with values in ``[0, 1]``, shape ``(C, H, W)``.
        stage : str or None
            Lightning stage string. ``"fit"`` triggers augmentation; all
            other values (``"test"``, ``"predict"``, ``None``) use eval transform.

        Returns
        -------
        torch.Tensor
            Float tensor with values in ``[-1, 1]``, shape ``(C, H, W)``.
        """
        if stage == "fit":
            img = self.transform_train(img)
        else:
            img = self.transform_eval(img)
        # Scale [0, 1] → [-1, 1]
        return img * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    """JPEG image dataset for CycleGAN.

    Reads images from ``filenames``, normalises them to ``[0, 1]``, and
    applies ``transform``.

    Parameters
    ----------
    filenames : list of str
        Absolute (or relative) paths to JPEG image files.
    transform : CustomTransform
        Transform to apply to each loaded image.
    stage : str or None
        Passed directly to ``transform.__call__`` to select aug vs. eval mode.
    """

    def __init__(
        self,
        filenames: List[str],
        transform: CustomTransform,
        stage: Optional[str],
    ) -> None:
        if not filenames:
            raise ValueError(
                "``filenames`` is empty — check your glob pattern and data paths."
            )
        self.filenames = filenames
        self.transform = transform
        self.stage = stage

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.filenames[idx]
        # read_image returns uint8 tensor (C, H, W) in [0, 255]
        img = read_image(img_path).float() / 255.0
        return self.transform(img, stage=self.stage)


# ---------------------------------------------------------------------------
# DataModule (requires pytorch-lightning)
# ---------------------------------------------------------------------------

def _require_lightning() -> None:
    if not _LIGHTNING_AVAILABLE:
        raise ImportError(
            "pytorch-lightning is required to use CycleGANDataModule. "
            "Install it with:  pip install pytorch-lightning"
        )


class CycleGANDataModule(L.LightningDataModule if _LIGHTNING_AVAILABLE else object):  # type: ignore[misc]
    """PyTorch Lightning DataModule for paired Monet ↔ photo datasets.

    Wraps ``CustomDataset`` for each domain and returns ``CombinedLoader``
    during training so that both domains are iterated in lockstep
    (``max_size_cycle`` mode).

    Parameters
    ----------
    monet_dir : str
        Glob pattern for Monet JPEG files,
        e.g. ``"data/raw/monet_dataset/monet_jpg/*.jpg"``.
    photo_dir : str
        Glob pattern for landscape photo JPEG files,
        e.g. ``"data/raw/monet_dataset/photo_jpg/*.jpg"``.
    batch_size : int
        Number of images per training batch. Default: 1.
    sample_size : int
        Number of images returned by ``val_dataloader`` / ``test_dataloader``.
        Default: 5.
    load_dim : int
        Resize dimension before random crop (train only). Default: 286.
    target_dim : int
        Final output resolution. Default: 256.
    num_workers : int or None
        DataLoader workers. Defaults to ``os.cpu_count()``.
    pin_memory : bool or None
        DataLoader pin_memory. Defaults to ``torch.cuda.is_available()``.
    """

    def __init__(
        self,
        monet_dir: str,
        photo_dir: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        load_dim: int = DEFAULT_LOAD_DIM,
        target_dim: int = DEFAULT_TARGET_DIM,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> None:
        _require_lightning()
        super().__init__()

        self.batch_size = batch_size
        self.sample_size = sample_size

        # Resolve file lists eagerly so we can fail fast on bad paths
        self.monet_filenames: List[str] = sorted(glob.glob(monet_dir))
        self.photo_filenames: List[str] = sorted(glob.glob(photo_dir))

        if not self.monet_filenames:
            raise FileNotFoundError(f"No Monet images found matching: {monet_dir!r}")
        if not self.photo_filenames:
            raise FileNotFoundError(f"No photo images found matching: {photo_dir!r}")

        self.transform = CustomTransform(load_dim=load_dim, target_dim=target_dim)

        self._loader_config: dict = {
            "num_workers": num_workers if num_workers is not None else (os.cpu_count() or 0),
            "pin_memory": pin_memory if pin_memory is not None else torch.cuda.is_available(),
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate dataset splits for the given ``stage``."""
        if stage == "fit":
            self.train_monet = CustomDataset(self.monet_filenames, self.transform, stage="fit")
            self.train_photo = CustomDataset(self.photo_filenames, self.transform, stage="fit")

        if stage in ("fit", "test", "predict", None):
            # Eval/validation dataset — no augmentation
            self.valid_photo = CustomDataset(self.photo_filenames, self.transform, stage=None)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> "CombinedLoader":
        """Return a CombinedLoader cycling over both Monet and photo batches."""
        loader_cfg = {
            "shuffle": True,
            "drop_last": True,
            "batch_size": self.batch_size,
            **self._loader_config,
        }
        loader_monet = DataLoader(self.train_monet, **loader_cfg)
        loader_photo = DataLoader(self.train_photo, **loader_cfg)
        return CombinedLoader({"monet": loader_monet, "photo": loader_photo}, mode="max_size_cycle")

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader (photo domain only, no augmentation)."""
        return DataLoader(
            self.valid_photo,
            batch_size=self.sample_size,
            **self._loader_config,
        )

    def test_dataloader(self) -> DataLoader:
        """Alias for ``val_dataloader``."""
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Return prediction DataLoader (single-image batches for inference)."""
        return DataLoader(
            self.valid_photo,
            batch_size=self.batch_size,
            **self._loader_config,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_data_root(
        cls,
        data_root: str = "data/raw/monet_dataset",
        **kwargs,
    ) -> "CycleGANDataModule":
        """Construct a ``CycleGANDataModule`` from a standard data root directory.

        Assumes the following layout under ``data_root``::

            monet_dataset/
            ├── monet_jpg/    ← Monet paintings (*.jpg)
            └── photo_jpg/    ← Landscape photos  (*.jpg)

        Parameters
        ----------
        data_root : str
            Path to the dataset root. Default: ``"data/raw/monet_dataset"``.
        **kwargs
            Forwarded to ``CycleGANDataModule.__init__``.
        """
        monet_dir = os.path.join(data_root, "monet_jpg", "*.jpg")
        photo_dir = os.path.join(data_root, "photo_jpg", "*.jpg")
        return cls(monet_dir=monet_dir, photo_dir=photo_dir, **kwargs)
