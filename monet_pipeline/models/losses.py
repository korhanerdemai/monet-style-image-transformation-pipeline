"""
monet_pipeline/models/losses.py
==============================
Loss functions for style transfer.
Includes the StyleTransferLoss class that computes the Content Loss and Style Loss
between generated, target bottleneck, and original style feature maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monet_pipeline.models.baseline_adain import calc_mean_std


class StyleTransferLoss(nn.Module):
    """Loss module for Adaptive Instance Normalization Style Transfer.

    Calculates Content Loss (MSE between the generated image's encoder features and
    the stylized target bottleneck features) and Style Loss (MSE between the channel-wise
    mean and standard deviation of the generated and style features).
    """

    def __init__(self, content_weight: float = 1.0, style_weight: float = 10.0) -> None:
        """Initialize the loss module.

        Parameters
        ----------
        content_weight : float
            Loss coefficient for content reconstruction. Default: 1.0.
        style_weight : float
            Loss coefficient for style/texture matching. Default: 10.0.
        """
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        output_feat: torch.Tensor,
        target_feat: torch.Tensor,
        style_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined style transfer loss.

        Parameters
        ----------
        output_feat : torch.Tensor
            Encoder features of the generated image g(t) of shape ``(N, C, H, W)``.
        target_feat : torch.Tensor
            Stylized bottleneck target features t = AdaIN(content, style) of shape ``(N, C, H, W)``.
        style_feat : torch.Tensor
            Encoder features of the style image of shape ``(N, C, H, W)``.

        Returns
        -------
        total_loss : torch.Tensor
            Weighted total loss.
        content_loss : torch.Tensor
            Unweighted content loss.
        style_loss : torch.Tensor
            Unweighted style loss.
        """
        # 1. Content Loss: MSE between generated features and stylized target features
        content_loss = self.mse_loss(output_feat, target_feat)

        # 2. Style Loss: MSE of mean and std between generated features and style features
        output_mean, output_std = calc_mean_std(output_feat)
        style_mean, style_std = calc_mean_std(style_feat)

        mean_loss = self.mse_loss(output_mean, style_mean)
        std_loss = self.mse_loss(output_std, style_std)
        style_loss = mean_loss + std_loss

        # 3. Combined Loss
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss

        return total_loss, content_loss, style_loss
