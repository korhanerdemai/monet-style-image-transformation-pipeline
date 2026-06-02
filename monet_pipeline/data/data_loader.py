"""
monet_pipeline/data/data_loader.py
==================================
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
    from monet_pipeline.data.data_loader import CycleGANDataModule

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

import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

try:
    import pytorch_lightning as L
    from pytorch_lightning.utilities import CombinedLoader

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTNING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOAD_DIM: int = 286  # Resize target before random crop
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
        self.transform_train = T.Compose(
            [
                T.Resize((load_dim, load_dim), antialias=True),
                T.RandomCrop((target_dim, target_dim)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
            ]
        )
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


class CustomDataset(Dataset[torch.Tensor]):
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
            raise ValueError("``filenames`` is empty — check your glob pattern and data paths.")
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
    train_manifest : str or Path
        Path to the training CSV manifest file.
        Default: ``"data/raw/monet_dataset/train_manifest.csv"``.
    val_manifest : str or Path
        Path to the validation CSV manifest file.
        Default: ``"data/raw/monet_dataset/val_manifest.csv"``.
    test_manifest : str or Path
        Path to the test CSV manifest file.
        Default: ``"data/raw/monet_dataset/test_manifest.csv"``.
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
        train_manifest: str | Path = "data/raw/monet_dataset/train_manifest.csv",
        val_manifest: str | Path = "data/raw/monet_dataset/val_manifest.csv",
        test_manifest: str | Path = "data/raw/monet_dataset/test_manifest.csv",
        batch_size: int = DEFAULT_BATCH_SIZE,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        load_dim: int = DEFAULT_LOAD_DIM,
        target_dim: int = DEFAULT_TARGET_DIM,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> None:
        _require_lightning()
        super().__init__()

        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.transform = CustomTransform(load_dim=load_dim, target_dim=target_dim)

        self._loader_config: dict[str, Any] = {
            "num_workers": num_workers if num_workers is not None else (os.cpu_count() or 0),
            "pin_memory": pin_memory if pin_memory is not None else torch.cuda.is_available(),
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate dataset splits for the given ``stage``."""
        if stage == "fit" or stage is None:
            # Eagerly load manifest files to fail-fast
            train_path = Path(self.train_manifest)
            val_path = Path(self.val_manifest)
            if not train_path.exists():
                raise FileNotFoundError(f"Train manifest not found: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"Val manifest not found: {val_path}")

            train_df = pd.read_csv(train_path)
            train_monet_paths = train_df[train_df["domain"] == "monet"]["image_path"].tolist()
            train_photo_paths = train_df[train_df["domain"] == "photo"]["image_path"].tolist()

            self.train_monet = CustomDataset(train_monet_paths, self.transform, stage="fit")
            self.train_photo = CustomDataset(train_photo_paths, self.transform, stage="fit")

            val_df = pd.read_csv(val_path)
            val_photo_paths = val_df[val_df["domain"] == "photo"]["image_path"].tolist()
            self.valid_photo = CustomDataset(val_photo_paths, self.transform, stage=None)

        if stage == "test":
            test_path = Path(self.test_manifest)
            if not test_path.exists():
                raise FileNotFoundError(f"Test manifest not found: {test_path}")
            test_df = pd.read_csv(test_path)
            test_photo_paths = test_df[test_df["domain"] == "photo"]["image_path"].tolist()
            self.test_photo = CustomDataset(test_photo_paths, self.transform, stage=None)

        if stage == "predict":
            predict_path = Path(self.test_manifest)
            if not predict_path.exists():
                raise FileNotFoundError(f"Test manifest not found: {predict_path}")
            test_df = pd.read_csv(predict_path)
            test_photo_paths = test_df[test_df["domain"] == "photo"]["image_path"].tolist()
            self.predict_photo = CustomDataset(test_photo_paths, self.transform, stage=None)

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

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return validation DataLoader (photo domain only, no augmentation)."""
        return DataLoader(
            self.valid_photo,
            batch_size=self.sample_size,
            **self._loader_config,
        )

    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return test DataLoader."""
        dataset = getattr(self, "test_photo", getattr(self, "valid_photo", None))
        if dataset is None:
            raise RuntimeError("Dataset for testing is not initialized. Run setup('test').")
        return DataLoader(
            dataset,
            batch_size=self.sample_size,
            **self._loader_config,
        )

    def predict_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return prediction DataLoader (single-image batches for inference)."""
        dataset = getattr(self, "predict_photo", getattr(self, "valid_photo", None))
        if dataset is None:
            raise RuntimeError("Dataset for prediction is not initialized. Run setup('predict').")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            **self._loader_config,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path = "data/processed",
        **kwargs: Any,
    ) -> "CycleGANDataModule":
        """Construct a ``CycleGANDataModule`` from a processed data directory containing manifests.

        Parameters
        ----------
        data_root : str or Path
            Path to the processed data directory. Default: ``"data/processed"``.
        **kwargs
            Forwarded to ``CycleGANDataModule.__init__``.
        """
        data_root_path = Path(data_root)
        train_manifest = data_root_path / "train_manifest.csv"
        val_manifest = data_root_path / "val_manifest.csv"
        test_manifest = data_root_path / "test_manifest.csv"
        return cls(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            **kwargs,
        )
