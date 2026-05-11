"""
src/models/baseline_nst.py
==========================
Neural Style Transfer (NST) baseline for the Monet CycleGAN MLOps pipeline.

Algorithm
---------
Gatys et al. (2015) — "A Neural Algorithm of Artistic Style"
https://arxiv.org/abs/1508.06576

Uses VGG-19 pretrained on ImageNet (matching the PyTorch framework from Phase 1).
Content and style features are extracted from specific VGG-19 convolutional
layers; gradient descent is run on the output image (initialised from the
content image) to minimise a weighted sum of content + style losses.

Optimisation is intentionally kept short (default: 150 iterations) so the
baseline evaluation over ~50-100 images completes in minutes on a GPU.

Usage
-----
Command-line (single image):
    python -m src.models.baseline_nst \
        --content path/to/photo.jpg \
        --style   path/to/monet.jpg \
        --output  path/to/result.jpg \
        --steps   150

From Python:
    from src.models.baseline_nst import NeuralStyleTransfer
    nst = NeuralStyleTransfer()
    result_tensor = nst.transfer(content_img_tensor, style_img_tensor, steps=150)
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------------------------
# VGG-19 layer names used for feature extraction
# ---------------------------------------------------------------------------

# Layer names in torchvision VGG-19 (features Sequential, 0-indexed)
# 'conv_N' → N-th convolution block encountered during forward pass.
CONTENT_LAYERS_DEFAULT: List[str] = ["conv_4"]
STYLE_LAYERS_DEFAULT: List[str] = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class ContentLoss(nn.Module):
    """Holds the target content feature map and computes MSE loss against it.

    The target is detached from the graph so it is treated as a constant.
    """

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target.detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x  # pass-through


class StyleLoss(nn.Module):
    """Holds the target style Gram matrix and computes MSE loss against it."""

    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()
        self.target = _gram_matrix(target_feature).detach()
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = _gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x  # pass-through


class Normalization(nn.Module):
    """ImageNet normalisation as an nn.Module (inserted at the start of the model)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------

def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a feature map.

    Parameters
    ----------
    x : torch.Tensor, shape (1, C, H, W)

    Returns
    -------
    torch.Tensor, shape (C, C) — normalised Gram matrix
    """
    b, c, h, w = x.size()
    f = x.view(b * c, h * w)
    G = torch.mm(f, f.t())
    return G.div(b * c * h * w)


# ---------------------------------------------------------------------------
# Build instrumented VGG-19
# ---------------------------------------------------------------------------

def _build_model(
    cnn: nn.Module,
    normalization: Normalization,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_layers: List[str],
    style_layers: List[str],
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss]]:
    """Insert ContentLoss and StyleLoss hooks into VGG-19 up to the last needed layer.

    Returns
    -------
    model : nn.Sequential
        Truncated VGG-19 with loss hooks inserted.
    content_losses : list of ContentLoss
    style_losses : list of StyleLoss
    """
    model = nn.Sequential(normalization)
    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []

    conv_idx = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_idx}"
            layer = nn.ReLU(inplace=False)   # inplace=True breaks gradient flow
        elif isinstance(layer, nn.MaxPool2d):
            # Replace MaxPool with AvgPool — produces smoother stylization
            name = f"pool_{conv_idx}"
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_idx}"
        else:
            name = f"unknown_{conv_idx}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_idx}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_idx}", sl)
            style_losses.append(sl)

        # Trim layers beyond the last needed one for efficiency
        all_target = content_layers + style_layers
        if name == max(all_target, key=lambda n: int(n.split("_")[1])):
            break

    return model, content_losses, style_losses


# ---------------------------------------------------------------------------
# Main NST class
# ---------------------------------------------------------------------------

class NeuralStyleTransfer:
    """VGG-19 based Neural Style Transfer baseline.

    Parameters
    ----------
    image_size : int
        Spatial resolution to resize inputs to. Default: 256 (matches Monet pipeline).
    device : str or None
        ``"cuda"`` / ``"cpu"``. Auto-detected if None.
    content_weight : float
        Weight for the content loss term (α). Default: 1.0.
    style_weight : float
        Weight for the style loss term (β). Default: 1e6.
    content_layers : list of str or None
        VGG-19 conv layer names for content representation.
    style_layers : list of str or None
        VGG-19 conv layer names for style representation.
    """

    def __init__(
        self,
        image_size: int = 256,
        device: Optional[str] = None,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        content_layers: Optional[List[str]] = None,
        style_layers: Optional[List[str]] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_size = image_size
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = content_layers or CONTENT_LAYERS_DEFAULT
        self.style_layers = style_layers or STYLE_LAYERS_DEFAULT

        # Load VGG-19 feature extractor (frozen)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self._cnn = vgg

        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    # ------------------------------------------------------------------
    # Image I/O helpers
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> torch.Tensor:
        """Load a JPEG/PNG from disk and return a (1,3,H,W) tensor in [0,1]."""
        img = Image.open(path).convert("RGB")
        return self._transform(img).unsqueeze(0).to(self.device)

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a (1,3,H,W) or (3,H,W) [0,1] tensor to a PIL Image."""
        t = tensor.squeeze(0).detach().cpu().clamp(0, 1)
        return transforms.ToPILImage()(t)

    # ------------------------------------------------------------------
    # Core transfer
    # ------------------------------------------------------------------

    def transfer(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        steps: int = 150,
        lr: float = 0.05,
        print_every: int = 50,
    ) -> torch.Tensor:
        """Run the NST optimisation loop.

        Parameters
        ----------
        content_img : torch.Tensor
            Shape ``(1, 3, H, W)``, values in ``[0, 1]``. Already on ``self.device``.
        style_img : torch.Tensor
            Shape ``(1, 3, H, W)``, values in ``[0, 1]``. Already on ``self.device``.
        steps : int
            Number of L-BFGS optimisation steps. Default: 150.
        lr : float
            Learning rate for the L-BFGS optimizer.
        print_every : int
            Log interval (steps). Set to 0 to suppress logging.

        Returns
        -------
        torch.Tensor
            Stylized image, shape ``(1, 3, H, W)``, clamped to ``[0, 1]``.
        """
        norm = Normalization(self.device)
        model, content_losses, style_losses = _build_model(
            copy.deepcopy(self._cnn),
            norm,
            content_img,
            style_img,
            self.content_layers,
            self.style_layers,
        )
        model.eval()

        # Optimise a copy of the content image directly (not model weights)
        output = content_img.clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS([output], lr=lr, max_iter=20)

        step = [0]

        def closure() -> float:
            with torch.no_grad():
                output.clamp_(0, 1)

            optimizer.zero_grad()
            model(output)   # forward populates .loss in each hook

            c_loss = self.content_weight * sum(cl.loss for cl in content_losses)
            s_loss = self.style_weight * sum(sl.loss for sl in style_losses)
            total = c_loss + s_loss
            total.backward()

            step[0] += 1
            if print_every and step[0] % print_every == 0:
                print(
                    f"  step {step[0]:4d}/{steps} | "
                    f"content={c_loss.item():.4f} | "
                    f"style={s_loss.item():.4f} | "
                    f"total={total.item():.4f}"
                )
            # Return a plain float — L-BFGS expects a scalar, not a tensor
            return total.item()

        # L-BFGS needs to re-evaluate the function multiple times per step
        while step[0] < steps:
            optimizer.step(closure)

        with torch.no_grad():
            output.clamp_(0, 1)

        return output.detach()


# ---------------------------------------------------------------------------
# Convenience function (single call, path-based)
# ---------------------------------------------------------------------------

def run_nst(
    content_path: str,
    style_path: str,
    output_path: Optional[str] = None,
    steps: int = 150,
    image_size: int = 256,
    device: Optional[str] = None,
    style_weight: float = 1e6,
    print_every: int = 50,
) -> torch.Tensor:
    """End-to-end NST: load images, transfer style, optionally save.

    Parameters
    ----------
    content_path : str
        Path to the landscape photo.
    style_path : str
        Path to the Monet painting.
    output_path : str or None
        If provided, saves the result as a JPEG.
    steps : int
        Optimisation iterations. Default: 150.
    image_size : int
        Resize resolution. Default: 256.
    device : str or None
    style_weight : float
    print_every : int
        Log interval. 0 = silent.

    Returns
    -------
    torch.Tensor, shape (1, 3, H, W), values in [0, 1]
    """
    nst = NeuralStyleTransfer(
        image_size=image_size, device=device, style_weight=style_weight
    )
    content = nst.load_image(content_path)
    style = nst.load_image(style_path)

    result = nst.transfer(content, style, steps=steps, print_every=print_every)

    if output_path:
        pil_img = nst.tensor_to_pil(result)
        pil_img.save(output_path, quality=95)
        print(f"Saved stylized image to {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Style Transfer baseline.")
    parser.add_argument("--content", required=True, help="Content image path.")
    parser.add_argument("--style", required=True, help="Style image path.")
    parser.add_argument("--output", default="stylized.jpg", help="Output path.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--style_weight", type=float, default=1e6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_nst(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        steps=args.steps,
        image_size=args.image_size,
        device=args.device,
        style_weight=args.style_weight,
    )
