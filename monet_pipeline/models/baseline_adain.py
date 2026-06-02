"""
monet_pipeline/models/baseline_adain.py
=======================================
Adaptive Instance Normalization (AdaIN) style transfer network.
Uses a pretrained ConvNeXt-Tiny encoder backbone and a symmetric convolutional
decoder to perform feed-forward, fast style transfer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.convnext import ConvNeXt_Tiny_Weights


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the channel-wise mean and standard deviation per instance.

    Parameters
    ----------
    feat : torch.Tensor
        Feature map tensor of shape ``(N, C, H, W)``.
    eps : float
        Small float to prevent division by zero. Default: 1e-5.

    Returns
    -------
    feat_mean : torch.Tensor
        Instance mean tensor of shape ``(N, C, 1, 1)``.
    feat_std : torch.Tensor
        Instance standard deviation tensor of shape ``(N, C, 1, 1)``.
    """
    size = feat.size()
    assert len(size) == 4, f"Expected 4D tensor, got shape {size}"
    batch_size, channels = size[:2]
    feat_var = feat.view(batch_size, channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    return feat_mean, feat_std


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Apply Adaptive Instance Normalization between content and style features.

    Calculates instance statistics and scales the content feature map to match
    the mean and variance of the style feature map.

    Parameters
    ----------
    content_feat : torch.Tensor
        Content image feature map of shape ``(N, C, H, W)``.
    style_feat : torch.Tensor
        Style image feature map of shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Stylized feature map of shape ``(N, C, H, W)``.
    """
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    normalized = (content_feat - content_mean) / content_std
    return style_std * normalized + style_mean


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Tiny encoder wrapper.

    Extracts intermediate features from Stage 2 of a pre-trained ConvNeXt-Tiny model.
    Weights are frozen during training.
    """

    def __init__(self) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)

        # Extract features up to Stage 2:
        # features[0]: Stem
        # features[1]: Stage 1
        # features[2]: Downsample
        # features[3]: Stage 2
        self.encoder = nn.Sequential(
            model.features[0],
            model.features[1],
            model.features[2],
            model.features[3],
        )

        # Freeze encoder completely
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from the input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Normalized image tensor of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map tensor of shape ``(N, 192, H/8, W/8)``.
        """
        return self.encoder(input_tensor)  # type: ignore[no-any-return]

    def normalize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Map images from range ``[-1, 1]`` to ``[0, 1]`` and apply ImageNet normalization.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image tensor of shape ``(N, 3, H, W)`` in range ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Normalized tensor in ImageNet space.
        """
        # Map [-1, 1] -> [0, 1]
        input_tensor_scaled = (input_tensor + 1.0) / 2.0

        # Normalise using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_tensor.device).view(1, 3, 1, 1)

        return (input_tensor_scaled - mean) / std


class AdaINDecoder(nn.Module):
    """Symmetric Convolutional Decoder.

    Upsamples features from Stage 2 bottleneck ``(192, 32, 32)``
    back to image space ``(3, 256, 256)`` using nearest-neighbor upsampling
    followed by convolutional layers with reflection padding.
    """

    def __init__(self) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            # Process bottleneck features at 32x32
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 192, kernel_size=3, padding=0),
            nn.ReLU(),
            # Upsample 32x32 -> 64x64
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 96, kernel_size=3, padding=0),
            nn.ReLU(),
            # Upsample 64x64 -> 128x128
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(96, 48, kernel_size=3, padding=0),
            nn.ReLU(),
            # Upsample 128x128 -> 256x256
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(48, 3, kernel_size=3, padding=0),
            nn.Tanh(),  # Maps output values to [-1, 1] matching dataset scale
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode stylized feature maps back to image space.

        Parameters
        ----------
        features : torch.Tensor
            Bottleneck feature map of shape ``(N, 192, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape ``(N, 3, 256, 256)`` in range ``[-1, 1]``.
        """
        return self.decoder(features)  # type: ignore[no-any-return]


class AdaINStyleTransfer(nn.Module):
    """End-to-end AdaIN Style Transfer Network.

    Wires together the pre-trained, frozen ConvNeXt-Tiny encoder,
    the Adaptive Instance Normalization layer, and the trainable convolutional decoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvNeXtEncoder()
        self.decoder = AdaINDecoder()

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Perform end-to-end feed-forward style transfer.

        Parameters
        ----------
        content : torch.Tensor
            Content image tensor of shape ``(N, 3, H, W)`` in range ``[-1, 1]``.
        style : torch.Tensor
            Style image tensor of shape ``(N, 3, H, W)`` in range ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Stylized image tensor of shape ``(N, 3, H, W)`` in range ``[-1, 1]``.
        """
        # Normalize inputs for ConvNeXt
        normalized_content = self.encoder.normalize(content)
        normalized_style = self.encoder.normalize(style)

        # Extract features
        content_feat = self.encoder(normalized_content)
        style_feat = self.encoder(normalized_style)

        # Perform AdaIN transformation
        stylized_features = adain(content_feat, style_feat)

        # Reconstruct image from stylized features
        stylized_images = self.decoder(stylized_features)
        return stylized_images  # type: ignore[no-any-return]
