"""
monet_pipeline/models/cyclegan.py
=================================
CycleGAN implementation in PyTorch Lightning.
Extracted and modularised from notebooks/cyclegan-implementation.ipynb.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsampling(nn.Module):
    """Downsampling convolutional block for CycleGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolving kernel. Default: 4.
    stride : int
        Stride of the convolution. Default: 2.
    padding : int
        Zero-padding added to both sides of the input. Default: 1.
    norm : bool
        If True, applies InstanceNorm2d. Default: True.
    lrelu : bool, optional
        If True, applies LeakyReLU(0.2). If False, applies ReLU.
        If None, no activation function is applied. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        lrelu: Optional[bool] = True,
    ) -> None:
        super().__init__()
        modules_list: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]
        if norm:
            modules_list.append(nn.InstanceNorm2d(out_channels, affine=True))
        if lrelu is not None:
            if lrelu:
                modules_list.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                modules_list.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class Upsampling(nn.Module):
    """Upsampling convolutional block for CycleGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolving kernel. Default: 4.
    stride : int
        Stride of the convolution. Default: 2.
    padding : int
        Zero-padding added to both sides of the input. Default: 1.
    output_padding : int
        Additional size added to one side of the output shape. Default: 0.
    dropout : bool
        If True, applies Dropout(0.5). Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        modules_list: List[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if dropout:
            modules_list.append(nn.Dropout(0.5))
        modules_list.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class ResBlock(nn.Module):
    """Residual block for Generator.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    kernel_size : int
        Size of the convolving kernel. Default: 3.
    padding : int
        Reflection padding added to both sides of the input. Default: 1.
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            Downsampling(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                lrelu=False,
            ),
            nn.ReflectionPad2d(padding),
            Downsampling(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                lrelu=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x + self.block(x))


class UNetGenerator(nn.Module):
    """U-Net Generator architecture for CycleGAN.

    Parameters
    ----------
    hid_channels : int
        Number of hidden channels.
    in_channels : int
        Number of input channels. Default: 3.
    out_channels : int
        Number of output channels. Default: 3.
    """

    def __init__(self, hid_channels: int, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.downsampling_path = nn.ModuleList(
            [
                Downsampling(in_channels, hid_channels, norm=False),  # 64
                Downsampling(hid_channels, hid_channels * 2),  # 128
                Downsampling(hid_channels * 2, hid_channels * 4),  # 256
                Downsampling(hid_channels * 4, hid_channels * 8),  # 512
                Downsampling(hid_channels * 8, hid_channels * 8),  # 512
                Downsampling(hid_channels * 8, hid_channels * 8),  # 512
                Downsampling(hid_channels * 8, hid_channels * 8),  # 512
                Downsampling(hid_channels * 8, hid_channels * 8, norm=False),  # 512
            ]
        )
        self.upsampling_path = nn.ModuleList(
            [
                Upsampling(hid_channels * 8, hid_channels * 8, dropout=True),
                Upsampling(hid_channels * 16, hid_channels * 8, dropout=True),
                Upsampling(hid_channels * 16, hid_channels * 8, dropout=True),
                Upsampling(hid_channels * 16, hid_channels * 8),
                Upsampling(hid_channels * 16, hid_channels * 4),
                Upsampling(hid_channels * 8, hid_channels * 2),
                Upsampling(hid_channels * 4, hid_channels),
            ]
        )
        self.feature_block = nn.Sequential(
            nn.ConvTranspose2d(hid_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)

        skips_rev = list(reversed(skips[:-1]))

        for up, skip in zip(self.upsampling_path, skips_rev):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return cast(torch.Tensor, self.feature_block(x))


class ResNetGenerator(nn.Module):
    """ResNet-based Generator architecture for CycleGAN.

    Parameters
    ----------
    hid_channels : int
        Number of hidden channels.
    in_channels : int
        Number of input channels. Default: 3.
    out_channels : int
        Number of output channels. Default: 3.
    num_resblocks : int
        Number of residual blocks. Default: 9.
    """

    def __init__(
        self,
        hid_channels: int,
        in_channels: int = 3,
        out_channels: int = 3,
        num_resblocks: int = 9,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            Downsampling(
                in_channels,
                hid_channels,
                kernel_size=7,
                stride=1,
                padding=0,
                lrelu=False,
            ),
            Downsampling(hid_channels, hid_channels * 2, kernel_size=3, lrelu=False),
            Downsampling(hid_channels * 2, hid_channels * 4, kernel_size=3, lrelu=False),
            *[ResBlock(hid_channels * 4) for _ in range(num_resblocks)],
            Upsampling(hid_channels * 4, hid_channels * 2, kernel_size=3, output_padding=1),
            Upsampling(hid_channels * 2, hid_channels, kernel_size=3, output_padding=1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(hid_channels, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


def get_gen(
    gen_name: str,
    hid_channels: int,
    num_resblocks: int,
    in_channels: int = 3,
    out_channels: int = 3,
) -> nn.Module:
    """Helper to return Generator module.

    Parameters
    ----------
    gen_name : str
        'unet' or 'resnet'.
    hid_channels : int
        Number of hidden channels.
    num_resblocks : int
        Number of residual blocks (for ResNet only).
    in_channels : int
        Input channels. Default: 3.
    out_channels : int
        Output channels. Default: 3.
    """
    if gen_name == "unet":
        return UNetGenerator(hid_channels, in_channels, out_channels)
    elif gen_name == "resnet":
        return ResNetGenerator(hid_channels, in_channels, out_channels, num_resblocks)
    else:
        raise NotImplementedError(f"Generator name '{gen_name}' not recognized.")


class Discriminator(nn.Module):
    """Discriminator network (PatchGAN) for CycleGAN.

    Parameters
    ----------
    hid_channels : int
        Number of hidden channels.
    in_channels : int
        Number of input channels. Default: 3.
    """

    def __init__(self, hid_channels: int, in_channels: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Downsampling(in_channels, hid_channels, norm=False),
            Downsampling(hid_channels, hid_channels * 2),
            Downsampling(hid_channels * 2, hid_channels * 4),
            Downsampling(hid_channels * 4, hid_channels * 8, stride=1),
            nn.Conv2d(hid_channels * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class ImageBuffer:
    """Image Buffer to store fake images for training Discriminators."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.curr_cap = 0
        self.buffer: List[torch.Tensor] = []

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.buffer_size == 0:
            return imgs

        return_imgs = []
        for img in imgs:
            img = img.unsqueeze(dim=0)
            if self.curr_cap < self.buffer_size:
                self.curr_cap += 1
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                p = float(np.random.uniform(low=0.0, high=1.0))
                if p > 0.5:
                    idx = int(np.random.randint(low=0, high=self.buffer_size))
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs, dim=0)


class CycleGAN(L.LightningModule):
    """CycleGAN model implemented as a PyTorch Lightning Module.

    Supports manual optimization to train Generator and Discriminator
    alternately.
    """

    def __init__(
        self,
        gen_name: str = "resnet",
        num_resblocks: int = 9,
        hid_channels: int = 64,
        lr: float = 0.0002,
        betas: Tuple[float, float] = (0.5, 0.999),
        lambda_idt: float = 0.5,
        lambda_cycle: Tuple[float, float] = (10.0, 10.0),
        buffer_size: int = 50,
        num_epochs: int = 50,
        decay_epochs: int = 25,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Define generators and discriminators
        self.gen_PM = get_gen(gen_name, hid_channels, num_resblocks)
        self.gen_MP = get_gen(gen_name, hid_channels, num_resblocks)
        self.disc_M = Discriminator(hid_channels)
        self.disc_P = Discriminator(hid_channels)

        # Initialize buffers to store fake images
        self.buffer_fake_M = ImageBuffer(buffer_size)
        self.buffer_fake_P = ImageBuffer(buffer_size)

        # Placeholders for intermediate calculations
        self.real_M: Optional[torch.Tensor] = None
        self.real_P: Optional[torch.Tensor] = None
        self.fake_M: Optional[torch.Tensor] = None
        self.fake_P: Optional[torch.Tensor] = None
        self.idt_M: Optional[torch.Tensor] = None
        self.idt_P: Optional[torch.Tensor] = None
        self.recon_M: Optional[torch.Tensor] = None
        self.recon_P: Optional[torch.Tensor] = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Translate photo to Monet-style image."""
        return cast(torch.Tensor, self.gen_PM(img))

    def init_weights(self) -> None:
        """Initialise weights with normal distribution."""

        def init_fn(m: nn.Module) -> None:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for net in [self.gen_PM, self.gen_MP, self.disc_M, self.disc_P]:
            net.apply(init_fn)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.init_weights()
            print("Model weights initialized.")

    def get_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine/Linear learning rate decay scheduler."""

        def lr_lambda(epoch: int) -> float:
            len_decay_phase = (
                float(self.hparams["num_epochs"]) - float(self.hparams["decay_epochs"]) + 1.0
            )
            curr_decay_step = max(0.0, float(epoch - float(self.hparams["decay_epochs"]) + 1.0))
            val = 1.0 - curr_decay_step / len_decay_phase
            return max(0.0, val)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        opt_config = {
            "lr": self.hparams["lr"],
            "betas": self.hparams["betas"],
        }
        opt_gen = torch.optim.Adam(
            list(self.gen_PM.parameters()) + list(self.gen_MP.parameters()),
            **opt_config,
        )
        opt_disc = torch.optim.Adam(
            list(self.disc_M.parameters()) + list(self.disc_P.parameters()),
            **opt_config,
        )
        optimizers: List[torch.optim.Optimizer] = [opt_gen, opt_disc]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def adv_criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)

    def recon_criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(y_hat, y)

    def get_adv_loss(self, fake: torch.Tensor, disc: nn.Module) -> torch.Tensor:
        fake_hat = disc(fake)
        real_labels = torch.ones_like(fake_hat)
        return self.adv_criterion(fake_hat, real_labels)

    def get_idt_loss(
        self, real: torch.Tensor, idt: torch.Tensor, lambda_cycle: float
    ) -> torch.Tensor:
        idt_loss = self.recon_criterion(idt, real)
        return float(self.hparams["lambda_idt"]) * lambda_cycle * idt_loss

    def get_cycle_loss(
        self, real: torch.Tensor, recon: torch.Tensor, lambda_cycle: float
    ) -> torch.Tensor:
        cycle_loss = self.recon_criterion(recon, real)
        return lambda_cycle * cycle_loss

    def get_gen_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate the total generator loss and dictionary of component losses."""
        assert self.fake_M is not None
        assert self.fake_P is not None
        assert self.real_M is not None
        assert self.real_P is not None
        assert self.idt_M is not None
        assert self.idt_P is not None
        assert self.recon_M is not None
        assert self.recon_P is not None

        # adversarial loss
        adv_loss_PM = self.get_adv_loss(self.fake_M, self.disc_M)
        adv_loss_MP = self.get_adv_loss(self.fake_P, self.disc_P)
        total_adv_loss = adv_loss_PM + adv_loss_MP

        # identity loss
        lambda_cycle = self.hparams["lambda_cycle"]
        idt_loss_MM = self.get_idt_loss(self.real_M, self.idt_M, lambda_cycle[0])
        idt_loss_PP = self.get_idt_loss(self.real_P, self.idt_P, lambda_cycle[1])
        total_idt_loss = idt_loss_MM + idt_loss_PP

        # cycle loss
        cycle_loss_MPM = self.get_cycle_loss(self.real_M, self.recon_M, lambda_cycle[0])
        cycle_loss_PMP = self.get_cycle_loss(self.real_P, self.recon_P, lambda_cycle[1])
        total_cycle_loss = cycle_loss_MPM + cycle_loss_PMP

        # combine losses
        gen_loss = total_adv_loss + total_idt_loss + total_cycle_loss

        loss_dict = {
            "gen_loss": gen_loss.item(),
            "adv_loss_PM": adv_loss_PM.item(),
            "adv_loss_MP": adv_loss_MP.item(),
            "total_adv_loss": total_adv_loss.item(),
            "idt_loss_MM": idt_loss_MM.item(),
            "idt_loss_PP": idt_loss_PP.item(),
            "total_idt_loss": total_idt_loss.item(),
            "cycle_loss_MPM": cycle_loss_MPM.item(),
            "cycle_loss_PMP": cycle_loss_PMP.item(),
            "total_cycle_loss": total_cycle_loss.item(),
        }
        return gen_loss, loss_dict

    def get_disc_loss(
        self, real: torch.Tensor, fake: torch.Tensor, disc: nn.Module
    ) -> torch.Tensor:
        # loss on real images
        real_hat = disc(real)
        real_labels = torch.ones_like(real_hat)
        real_loss = self.adv_criterion(real_hat, real_labels)

        # loss on fake images
        fake_hat = disc(fake.detach())
        fake_labels = torch.zeros_like(fake_hat)
        fake_loss = self.adv_criterion(fake_hat, fake_labels)

        # combine losses
        return (fake_loss + real_loss) * 0.5

    def get_disc_loss_M(self) -> torch.Tensor:
        assert self.fake_M is not None
        assert self.real_M is not None
        fake_M = self.buffer_fake_M(self.fake_M)
        return self.get_disc_loss(self.real_M, fake_M, self.disc_M)

    def get_disc_loss_P(self) -> torch.Tensor:
        assert self.fake_P is not None
        assert self.real_P is not None
        fake_P = self.buffer_fake_P(self.fake_P)
        return self.get_disc_loss(self.real_P, fake_P, self.disc_P)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)):
            batch_dict = batch[0]
        else:
            batch_dict = batch

        self.real_M = batch_dict["monet"]
        self.real_P = batch_dict["photo"]
        opt_gen, opt_disc = cast(Tuple[Any, Any], self.optimizers())

        # generate fake images
        self.fake_M = self.gen_PM(self.real_P)
        self.fake_P = self.gen_MP(self.real_M)

        # generate identity images
        self.idt_M = self.gen_PM(self.real_M)
        self.idt_P = self.gen_MP(self.real_P)

        # reconstruct images
        self.recon_M = self.gen_PM(self.fake_P)
        self.recon_P = self.gen_MP(self.fake_M)

        # train generators
        self.toggle_optimizer(opt_gen)
        gen_loss, gen_loss_dict = self.get_gen_loss()
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

        # train discriminators
        self.toggle_optimizer(opt_disc)
        disc_loss_M = self.get_disc_loss_M()
        disc_loss_P = self.get_disc_loss_P()
        opt_disc.zero_grad()
        self.manual_backward(disc_loss_M)
        self.manual_backward(disc_loss_P)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        # record training losses
        metrics = {
            "gen_loss": gen_loss,
            "disc_loss_M": disc_loss_M,
            "disc_loss_P": disc_loss_P,
            **{
                k: torch.tensor(v, device=self.device)
                for k, v in gen_loss_dict.items()
                if k != "gen_loss"
            },
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        real_P = batch
        fake_M = self.gen_PM(real_P)
        recon_P = self.gen_MP(fake_M)

        # Validation metrics
        val_cycle_loss_P = F.l1_loss(recon_P, real_P)

        fake_hat_M = self.disc_M(fake_M)
        val_gen_loss_P = F.mse_loss(fake_hat_M, torch.ones_like(fake_hat_M))

        metrics = {
            "val_cycle_loss_P": val_cycle_loss_P,
            "val_gen_loss_P": val_gen_loss_P,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        schedulers = cast(Any, self.lr_schedulers())
        curr_lr = schedulers[0].get_last_lr()[0]
        self.log("lr", curr_lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        schedulers = cast(Any, self.lr_schedulers())
        if isinstance(schedulers, list):
            for sch in schedulers:
                sch.step()
        elif schedulers is not None:
            schedulers.step()

        logged_values = self.trainer.progress_bar_metrics
        # Print progress in pure ASCII to avoid terminal encoding errors
        print(
            f"Epoch {self.current_epoch + 1} - "
            + " - ".join(f"{k}: {v:.5f}" for k, v in logged_values.items())
        )

    def on_train_end(self) -> None:
        print("Training ended.")
