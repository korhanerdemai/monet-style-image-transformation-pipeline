"""
commands.py
===========
CLI commands for training and running the CycleGAN Monet style transformation pipeline.
Exposes entry points for training the baseline AdaIN decoder, CycleGAN, and running fast inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import fire
import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch
import torchvision.transforms as T
from hydra import compose, initialize
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger

from monet_pipeline.data.data_loader import CycleGANDataModule
from monet_pipeline.evaluation.evaluate_baseline import evaluate_baseline
from monet_pipeline.models.baseline_adain import AdaINStyleTransfer, adain
from monet_pipeline.models.cyclegan import CycleGAN
from monet_pipeline.models.losses import StyleTransferLoss


class MetricTracker(L.Callback):
    """Callback to collect logged metrics per epoch for robust in-memory plotting."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_metrics: List[Dict[str, float]] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics: Dict[str, float] = {}
        for k, v in trainer.logged_metrics.items():
            metrics[k] = float(v.item()) if hasattr(v, "item") else float(v)
        metrics["epoch"] = trainer.current_epoch + 1
        self.epoch_metrics.append(metrics)


def pull_data_dvc() -> None:
    """Pull the dataset using the DVC Python API (or CLI fallback)."""
    print("Checking dataset availability and initiating DVC pull...")
    try:
        from dvc.repo import Repo

        repo = Repo()
        print("Running dvc.repo.Repo().pull() targeting dataset...")
        repo.pull(targets=["data/raw/monet_dataset.dvc"])
        print("DVC Python API pull completed successfully.")
    except Exception as e:
        print(f"DVC Python API pull encountered an issue: {e}")
        print("Attempting DVC CLI fallback...")
        import subprocess

        try:
            subprocess.run(["dvc", "pull", "data/raw/monet_dataset.dvc"], check=True)
            print("DVC CLI fallback pull completed successfully.")
        except Exception as cli_err:
            print(f"DVC CLI fallback failed: {cli_err}")
            print(
                "Please ensure your DVC storage is configured and "
                "you have the required access permissions."
            )


def get_git_commit_id() -> str:
    """Retrieve the current Git commit hash, returning 'unknown' if not available."""
    import subprocess

    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except Exception:
        return "unknown"


def is_mlflow_server_running(tracking_uri: str) -> bool:
    """Check if the MLflow tracking server is reachable at host/port of tracking_uri."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(tracking_uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


def flatten_dict(
    target_dict: dict[str, Any], parent_key: str = "", sep: str = "/"
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary into a single-level dictionary."""
    items: list[tuple[str, Any]] = []
    for key, val in target_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, dict):
            items.extend(flatten_dict(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)


def train_baseline(
    epochs: int | None = None,
    batch_size: int | None = None,
    max_steps: int | None = None,
) -> None:
    """Train the AdaIN style transfer baseline decoder.

    Parameters
    ----------
    epochs : int, optional
        Number of epochs to train. Default: None.
    batch_size : int, optional
        Number of images per batch. Default: None.
    max_steps : int, optional
        Maximum number of steps per epoch (useful for smoke tests). Default: None.
    """
    pull_data_dvc()

    # Load configuration via Hydra Compose API
    try:
        initialize(config_path="configs", version_base=None)
    except ValueError:
        pass  # Hydra already initialized
    cfg = compose(config_name="config", overrides=["model=baseline"])

    # Fall back to Hydra values if CLI overrides are not provided
    epochs_val = epochs if epochs is not None else cfg.training.epochs
    epochs = epochs_val
    batch_size_val = batch_size if batch_size is not None else cfg.training.batch_size
    batch_size = batch_size_val
    lr_val = cfg.model.decoder_lr
    style_weight_val = cfg.model.style_weight

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize model and loss module
    model = AdaINStyleTransfer().to(device)
    # Freeze encoder and keep in eval mode; decoder is trainable and in train mode
    model.encoder.eval()
    model.decoder.train()

    criterion = StyleTransferLoss(style_weight=style_weight_val)

    # 2. Setup Adam optimizer targeting ONLY the decoder weights
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr_val)

    # 3. Setup data loader using static manifest CSV files
    dm = CycleGANDataModule(
        batch_size=batch_size_val,
        load_dim=cfg.preprocessing.load_dim,
        target_dim=cfg.preprocessing.target_dim,
        sample_size=cfg.preprocessing.sample_size,
        num_workers=0,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    weights_dir = Path("models")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "baseline_decoder.pth"

    print(f"Starting AdaIN Baseline Training on {device.upper()}:")
    print(f"  Epochs:      {epochs_val}")
    print(f"  Batch Size:  {batch_size_val}")
    print("  Optimizer:   Adam (Decoder only)")
    print(f"  Output path: {weights_path}\n")

    for epoch in range(epochs_val):
        running_loss = 0.0
        running_content = 0.0
        running_style = 0.0
        count = 0

        for batch_idx, batch_tuple in enumerate(train_loader):
            if max_steps is not None and batch_idx >= max_steps:
                print(f"Reached max_steps ({max_steps}) constraint. Stopping epoch early.")
                break

            # CombinedLoader returns (batch_dict, batch_idx, dataloader_idx) during direct iteration
            batch = batch_tuple[0]

            content_images = batch["photo"].to(device)
            style_images = batch["monet"].to(device)

            optimizer.zero_grad()

            # Normalize inputs
            norm_content = model.encoder.normalize(content_images)
            norm_style = model.encoder.normalize(style_images)

            # Forward pass
            content_feat = model.encoder(norm_content)
            style_feat = model.encoder(norm_style)

            # Bottleneck representation
            stylized_features = adain(content_feat, style_feat)

            # Reconstruct image
            stylized_images = model.decoder(stylized_features)

            # Extract features of generated image
            stylized_features_extracted = model.encoder(model.encoder.normalize(stylized_images))

            # Compute losses
            loss, content_loss, style_loss = criterion(
                stylized_features_extracted, stylized_features, style_feat
            )

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            batch_len = content_images.size(0)
            running_loss += loss.item() * batch_len
            running_content += content_loss.item() * batch_len
            running_style += style_loss.item() * batch_len
            count += batch_len

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} (Content: {content_loss.item():.4f}, "
                    f"Style: {style_loss.item():.4f})"
                )

        epoch_loss = running_loss / count
        epoch_content = running_content / count
        epoch_style = running_style / count
        print(
            f"\nEpoch [{epoch+1}/{epochs}] Finished! "
            f"Avg Loss: {epoch_loss:.4f} (Content: {epoch_content:.4f}, Style: {epoch_style:.4f})\n"
        )

    # Save weights
    torch.save(model.decoder.state_dict(), weights_path)
    print(f"Decoder weights saved successfully to {weights_path}")


def run_baseline(
    content_path: str,
    style_path: str,
    weights_path: str,
    output_path: str,
) -> None:
    """Run stylized inference on a single content and style image using the trained baseline.

    Parameters
    ----------
    content_path : str
        Path to the content image.
    style_path : str
        Path to the style image.
    weights_path : str
        Path to the saved decoder weight file (.pth).
    output_path : str
        Path where the stylized output image will be saved.
    """
    pull_data_dvc()

    # Load configuration via Hydra Compose API
    try:
        initialize(config_path="configs", version_base=None)
    except ValueError:
        pass  # Hydra already initialized
    cfg = compose(config_name="config", overrides=["model=baseline"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    content_path_obj = Path(content_path)
    style_path_obj = Path(style_path)
    weights_path_obj = Path(weights_path)
    output_path_obj = Path(output_path)

    if not content_path_obj.exists():
        raise FileNotFoundError(f"Content image not found: {content_path_obj}")
    if not style_path_obj.exists():
        raise FileNotFoundError(f"Style image not found: {style_path_obj}")
    if not weights_path_obj.exists():
        raise FileNotFoundError(f"Decoder weights not found: {weights_path_obj}")

    # 1. Initialize and load model
    model = AdaINStyleTransfer().to(device)
    model.eval()

    print(f"Loading decoder weights from {weights_path_obj}...")
    model.decoder.load_state_dict(torch.load(weights_path_obj, map_location=device))

    # 2. Image loading and transforms
    transform = T.Compose(
        [
            T.Resize((cfg.preprocessing.target_dim, cfg.preprocessing.target_dim)),
            T.ToTensor(),
        ]
    )

    content_img = Image.open(content_path_obj).convert("RGB")
    style_img = Image.open(style_path_obj).convert("RGB")

    # Rescale [0, 1] -> [-1, 1] to match the training data loading pipeline
    content_tensor = transform(content_img).unsqueeze(0).to(device) * 2.0 - 1.0
    style_tensor = transform(style_img).unsqueeze(0).to(device) * 2.0 - 1.0

    print("Running feed-forward style transfer pass...")
    with torch.no_grad():
        stylized_tensor = model(content_tensor, style_tensor)

        # Map output range [-1, 1] -> [0, 1] and clamp
        stylized_tensor = (stylized_tensor + 1.0) / 2.0
        stylized_tensor = torch.clamp(stylized_tensor, 0.0, 1.0).cpu().squeeze(0)

    # 3. Save the stylized image
    to_pil = T.ToPILImage()
    output_img = to_pil(stylized_tensor)

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_path_obj)

    print(f"Stylized output image saved to {output_path_obj}")


def train(fast_dev_run: bool | None = None) -> None:
    """Train the CycleGAN model using PyTorch Lightning, Hydra configuration, and MLflow.

    Parameters
    ----------
    fast_dev_run : bool, optional
        Override the fast_dev_run flag from Hydra configuration. Default: None.
    """
    pull_data_dvc()
    # 1. Load configuration via Hydra Compose API
    try:
        initialize(config_path="configs", version_base=None)
    except ValueError:
        pass  # Hydra already initialized
    cfg = compose(config_name="config", overrides=["model=cyclegan"])

    # Determine fast_dev_run value
    fdr_raw = cfg.training.fast_dev_run if fast_dev_run is None else fast_dev_run
    if isinstance(fdr_raw, str):
        if fdr_raw.lower() in ("true", "1"):
            fdr: bool | int = True
        elif fdr_raw.lower() in ("false", "0"):
            fdr = False
        else:
            try:
                fdr = int(fdr_raw)
            except ValueError:
                fdr = False
    else:
        fdr = fdr_raw

    print("Starting CycleGAN Training Phase:")
    print(f"  Generator Name:  {cfg.model.gen_name}")
    print(f"  ResBlocks:       {cfg.model.num_resblocks}")
    print(f"  Hidden Channels: {cfg.model.hid_channels}")
    print(f"  Batch Size:      {cfg.training.batch_size}")
    print(f"  Learning Rate:   {cfg.training.learning_rate}")
    print(f"  Max Epochs:      {cfg.training.epochs}")
    print(f"  Fast Dev Run:    {fdr}")

    # 2. Initialize DataModule using static configs
    datamodule = CycleGANDataModule(
        batch_size=cfg.training.batch_size,
        load_dim=cfg.preprocessing.load_dim,
        target_dim=cfg.preprocessing.target_dim,
        sample_size=cfg.preprocessing.sample_size,
        num_workers=0,  # Safest default for Windows and multiprocess synchronization
    )

    # 3. Initialize CycleGAN Lightning Module
    model = CycleGAN(
        gen_name=cfg.model.gen_name,
        num_resblocks=cfg.model.num_resblocks,
        hid_channels=cfg.model.hid_channels,
        lr=cfg.training.learning_rate,
        lambda_idt=cfg.model.lambda_idt,
        lambda_cycle=tuple(cfg.model.lambda_cycle),
        buffer_size=cfg.model.buffer_size,
        num_epochs=cfg.training.epochs,
        decay_epochs=cfg.training.decay_epochs,
    )

    # 4. Initialize MLFlow Logger and custom Metric Tracker
    tracking_uri = cfg.logging.tracking_uri
    if is_mlflow_server_running(tracking_uri):
        print(f"MLflow server detected at {tracking_uri}. Logging to tracking server.")
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.logging.experiment_name,
            save_dir=cfg.logging.save_dir,
            tracking_uri=tracking_uri,
        )
    else:
        print(
            f"MLflow server at {tracking_uri} is not reachable. "
            "Falling back to local file logging under ./mlruns"
        )
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.logging.experiment_name,
            save_dir=cfg.logging.save_dir,
        )

    # Log entire Hydra configuration and Git commit hash to MLflow
    from omegaconf import OmegaConf

    raw_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(raw_cfg_dict, dict):

        cfg_dict: dict[str, Any] = {str(k): v for k, v in raw_cfg_dict.items()}
        flat_hparams = flatten_dict(cfg_dict)
        flat_hparams["git_commit"] = get_git_commit_id()
        mlflow_logger.log_hyperparams(flat_hparams)

    metric_tracker = MetricTracker()

    # 5. Initialize Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        fast_dev_run=fdr,
        logger=mlflow_logger,
        callbacks=[metric_tracker],
        enable_checkpointing=not fdr,
    )

    # 6. Execute training
    print("Fitting model with PyTorch Lightning Trainer...")
    trainer.fit(model, datamodule=datamodule)

    # 7. Post-Training Hook: Retrieve metric history and save plots
    print("\n--- Executing Post-Training Hook: Generating Plots ---")
    plots_dir = Path(cfg.postprocessing.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epoch_metrics = metric_tracker.epoch_metrics
    if not epoch_metrics:
        print("Warning: No metrics recorded. Plots will not be generated.")
        return

    epochs = [metric_entry.get("epoch", idx + 1) for idx, metric_entry in enumerate(epoch_metrics)]

    # Plot 1: Training losses
    fig_sz = tuple(cfg.postprocessing.fig_size)
    plt.figure(figsize=fig_sz)
    gen_loss = [metric_entry.get("gen_loss", 0.0) for metric_entry in epoch_metrics]
    disc_m = [metric_entry.get("disc_loss_M", 0.0) for metric_entry in epoch_metrics]
    disc_p = [metric_entry.get("disc_loss_P", 0.0) for metric_entry in epoch_metrics]

    plt.plot(epochs, gen_loss, label="Generator Loss", color="royalblue", marker="o")
    plt.plot(epochs, disc_m, label="Discriminator M Loss", color="darkorange", marker="s")
    plt.plot(epochs, disc_p, label="Discriminator P Loss", color="forestgreen", marker="d")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CycleGAN Training Losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    loss_plot_path = plots_dir / "training_loss.png"
    plt.savefig(loss_plot_path, dpi=cfg.postprocessing.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss plot to: {loss_plot_path.absolute()}")

    # Plot 2: Validation metrics
    val_epochs = [
        metric_entry["epoch"]
        for metric_entry in epoch_metrics
        if "val_cycle_loss_P" in metric_entry
    ]
    val_cycle = [
        metric_entry["val_cycle_loss_P"]
        for metric_entry in epoch_metrics
        if "val_cycle_loss_P" in metric_entry
    ]
    val_gen = [
        metric_entry["val_gen_loss_P"]
        for metric_entry in epoch_metrics
        if "val_gen_loss_P" in metric_entry
    ]

    if val_cycle:
        plt.figure(figsize=fig_sz)
        plt.plot(val_epochs, val_cycle, label="Val Cycle Loss (Photo)", color="crimson", marker="x")
        plt.plot(val_epochs, val_gen, label="Val Gen Loss (Photo)", color="purple", marker="^")

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("CycleGAN Validation Metrics")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        val_plot_path = plots_dir / "validation_metrics.png"
        plt.savefig(val_plot_path, dpi=cfg.postprocessing.dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved validation metrics plot to: {val_plot_path.absolute()}")

    # Plot 3: Generator Loss Decomposition
    plt.figure(figsize=fig_sz)
    adv_loss = [metric_entry.get("total_adv_loss", 0.0) for metric_entry in epoch_metrics]
    idt_loss = [metric_entry.get("total_idt_loss", 0.0) for metric_entry in epoch_metrics]
    cycle_loss = [metric_entry.get("total_cycle_loss", 0.0) for metric_entry in epoch_metrics]

    plt.plot(epochs, adv_loss, label="Adversarial Loss", color="royalblue", marker="o")
    plt.plot(epochs, idt_loss, label="Identity Loss", color="crimson", marker="s")
    plt.plot(epochs, cycle_loss, label="Cycle consistency Loss", color="forestgreen", marker="d")

    plt.xlabel("Epoch")
    plt.ylabel("Loss Component Value")
    plt.title("CycleGAN Generator Loss Decomposition")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    decomp_plot_path = plots_dir / "generator_loss_decomposition.png"
    plt.savefig(decomp_plot_path, dpi=cfg.postprocessing.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved generator loss decomposition plot to: {decomp_plot_path.absolute()}")

    print("Post-training plotting hook completed successfully.")


if __name__ == "__main__":
    fire.Fire(
        {
            "train_baseline": train_baseline,
            "run_baseline": run_baseline,
            "evaluate_baseline": evaluate_baseline,
            "train": train,
        }
    )
