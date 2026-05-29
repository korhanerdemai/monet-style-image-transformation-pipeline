"""
commands.py
===========
CLI commands for training and running the CycleGAN Monet style transformation pipeline.
Exposes entry points for training the baseline AdaIN decoder and running fast inference.
"""

from __future__ import annotations

from pathlib import Path

import fire
import torch
import torchvision.transforms as T
from PIL import Image

from monet_pipeline.data.data_loader import CycleGANDataModule
from monet_pipeline.evaluation.evaluate_baseline import evaluate_baseline
from monet_pipeline.models.baseline_adain import AdaINStyleTransfer, adain
from monet_pipeline.models.losses import StyleTransferLoss


def train_baseline(epochs: int = 1, batch_size: int = 8, max_steps: int | None = None) -> None:
    """Train the AdaIN style transfer baseline decoder.

    Parameters
    ----------
    epochs : int
        Number of epochs to train. Default: 1.
    batch_size : int
        Number of images per batch. Default: 8.
    max_steps : int, optional
        Maximum number of steps per epoch (useful for smoke tests). Default: None.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize model and loss module
    model = AdaINStyleTransfer().to(device)
    # Freeze encoder and keep in eval mode; decoder is trainable and in train mode
    model.encoder.eval()
    model.decoder.train()

    criterion = StyleTransferLoss()

    # 2. Setup Adam optimizer targeting ONLY the decoder weights
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

    # 3. Setup data loader using static manifest CSV files
    dm = CycleGANDataModule(batch_size=batch_size, num_workers=0)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    weights_dir = Path("models")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "baseline_decoder.pth"

    print(f"Starting AdaIN Baseline Training on {device.upper()}:")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch Size:  {batch_size}")
    print("  Optimizer:   Adam (Decoder only)")
    print(f"  Output path: {weights_path}\n")

    for epoch in range(epochs):
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
            t = adain(content_feat, style_feat)

            # Reconstruct image
            g_t = model.decoder(t)

            # Extract features of generated image
            g_t_feat = model.encoder(model.encoder.normalize(g_t))

            # Compute losses
            loss, c_loss, s_loss = criterion(g_t_feat, t, style_feat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Metrics
            n = content_images.size(0)
            running_loss += loss.item() * n
            running_content += c_loss.item() * n
            running_style += s_loss.item() * n
            count += n

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} (Content: {c_loss.item():.4f}, "
                    f"Style: {s_loss.item():.4f})"
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
            T.Resize((256, 256)),
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


if __name__ == "__main__":
    fire.Fire(
        {
            "train_baseline": train_baseline,
            "run_baseline": run_baseline,
            "evaluate_baseline": evaluate_baseline,
        }
    )
