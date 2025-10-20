from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1, padding: int | None = None, activation: nn.Module | None = None, norm: bool = True):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)]
        if norm:
            layers.append(nn.BatchNorm1d(out_ch))
        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)
        layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 4, stride: int = 2, padding: int = 1, activation: nn.Module | None = None, norm: bool = True):
        super().__init__()
        layers = [nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)]
        if norm:
            layers.append(nn.BatchNorm1d(out_ch))
        if activation is None:
            activation = nn.ReLU(inplace=True)
        layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator1D(nn.Module):
    def __init__(self, latent_dim: int, base_ch: int, out_length: int):
        super().__init__()
        # Map latent to a small spatial extent, then upsample via transposed convs
        self.init_len = max(8, 2 ** int(math.log2(out_length) - 4))  # heuristic start length
        self.project = nn.Sequential(
            nn.Linear(latent_dim, base_ch * self.init_len),
            nn.ReLU(inplace=True),
        )
        self.ups = nn.Sequential(
            DeconvBlock1D(base_ch, base_ch // 2),
            DeconvBlock1D(base_ch // 2, base_ch // 4),
            ConvBlock1D(base_ch // 4, base_ch // 4),
        )
        self.head = nn.Conv1d(base_ch // 4, 1, kernel_size=7, padding=3)
        self.out_length = out_length

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        x = self.project(z)  # (b, base_ch * init_len)
        x = x.view(b, -1, self.init_len)  # (b, C, L)
        x = self.ups(x)
        # Interpolate to exact out_length
        x = nn.functional.interpolate(x, size=self.out_length, mode="linear", align_corners=False)
        x = self.head(x)
        return x  # (b, 1, L)


class Discriminator1D(nn.Module):
    def __init__(self, base_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1D(1, base_ch, norm=False),
            ConvBlock1D(base_ch, base_ch, stride=2),
            ConvBlock1D(base_ch, base_ch * 2, stride=2),
            ConvBlock1D(base_ch * 2, base_ch * 4, stride=2),
            nn.Conv1d(base_ch * 4, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.view(x.size(0), 1)


if __name__ == "__main__":
    print("Testing Conv1D GAN models...")
    
    # Test parameters
    latent_dim = 128
    base_ch = 128
    out_length = 100
    batch_size = 4
    
    # Test Generator
    G = Generator1D(latent_dim=latent_dim, base_ch=base_ch, out_length=out_length)
    z = torch.randn(batch_size, latent_dim)
    fake = G(z)
    print(f"Generator: z.shape={z.shape} -> fake.shape={fake.shape}")
    
    # Test Discriminator
    D = Discriminator1D(base_ch=base_ch)
    real = torch.randn(batch_size, 1, out_length)
    pred_real = D(real)
    pred_fake = D(fake.detach())
    print(f"Discriminator: real.shape={real.shape} -> pred_real.shape={pred_real.shape}")
    print(f"Discriminator: fake.shape={fake.shape} -> pred_fake.shape={pred_fake.shape}")
    
    print("Model tests completed successfully!")


