import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import config
except ImportError:
    import config


class SinusoidalTimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ===== UNet Architecture =====


class ConvBlock(nn.Module):
    """Standard Conv1d block: Conv -> GroupNorm -> SiLU"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.silu1 = nn.SiLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.silu2 = nn.SiLU()

    def forward(self, x, t):
        h = self.silu1(self.norm1(self.conv1(x)))
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(-1)
        h = self.silu2(self.norm2(self.conv2(h)))
        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, t):
        h = self.conv(x, t)
        p = self.pool(h)
        return h, p


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            skip = torch.nn.functional.interpolate(
                skip, size=x.shape[-1], mode="linear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, t)
        return x


class UNet1D(nn.Module):
    """
    A simple MLP for modeling the vector field v(u, t) for breakpoints.
    Input:
        x (u_t): (Batch, 3) - breakpoints [depth1, tts1, tts_end]
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 3) - vector field for breakpoints
    """

    def __init__(self, dim=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Simple MLP for 3 breakpoint values
        self.mlp = nn.Sequential(
            nn.Linear(3 + time_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 3),
        )

    def forward(self, x, t):
        # x: (Batch, 3) or (Batch, 1, 3) - breakpoints
        # t: (Batch, 1)
        if x.dim() == 3:
            x = x.squeeze(1)  # (Batch, 3)

        t_emb = self.time_mlp(t.squeeze(-1))  # (Batch, time_emb_dim)

        # Concatenate x and time embedding
        x_with_time = torch.cat([x, t_emb], dim=-1)  # (Batch, 3 + time_emb_dim)

        # Predict vector field
        output = self.mlp(x_with_time)  # (Batch, 3)

        return output


# ===== FNO Architecture =====


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO"""

    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes1], self.weights1
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock(nn.Module):
    """FNO block with spectral convolution and time conditioning"""

    def __init__(self, modes, width, time_emb_dim):
        super().__init__()
        self.modes = modes
        self.width = width

        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, x, t_emb):
        x1 = self.conv(x)
        x2 = self.w(x)

        time_emb = self.time_mlp(t_emb).unsqueeze(-1)
        x = x1 + x2 + time_emb

        return F.gelu(x)


class FNO1D(nn.Module):
    """
    Simple MLP for modeling the vector field v(u, t) for breakpoints.
    (FNO not suitable for 3 values, using MLP instead)
    Input:
        x (u_t): (Batch, 3) - breakpoints [depth1, tts1, tts_end]
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 3) - vector field for breakpoints
    """

    def __init__(self, modes=16, width=64, time_emb_dim=128):
        super().__init__()
        self.width = width

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Simple MLP for 3 breakpoint values
        self.mlp = nn.Sequential(
            nn.Linear(3 + time_emb_dim, width),
            nn.SiLU(),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )

    def forward(self, x, t):
        # x: (Batch, 3) or (Batch, 1, 3) - breakpoints
        # t: (Batch, 1)
        if x.dim() == 3:
            x = x.squeeze(1)  # (Batch, 3)

        t_emb = self.time_mlp(t.squeeze(-1))  # (Batch, time_emb_dim)

        # Concatenate x and time embedding
        x_with_time = torch.cat([x, t_emb], dim=-1)  # (Batch, 3 + time_emb_dim)

        # Predict vector field
        output = self.mlp(x_with_time)  # (Batch, 3)

        return output


class MLP1D(nn.Module):
    """
    A dedicated MLP for modeling the vector field v(u, t) for breakpoints.
    Input:
        x (u_t): (Batch, 3) - breakpoints [depth1, tts1, tts_end]
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 3) - vector field for breakpoints
    """

    def __init__(self, dim=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Simple MLP for 3 breakpoint values
        self.mlp = nn.Sequential(
            nn.Linear(3 + time_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 3),
        )

    def forward(self, x, t):
        # x: (Batch, 3) or (Batch, 1, 3) - breakpoints
        # t: (Batch, 1)
        if x.dim() == 3:
            x = x.squeeze(1)  # (Batch, 3)

        t_emb = self.time_mlp(t.squeeze(-1))  # (Batch, time_emb_dim)

        # Concatenate x and time embedding
        x_with_time = torch.cat([x, t_emb], dim=-1)  # (Batch, 3 + time_emb_dim)

        # Predict vector field
        output = self.mlp(x_with_time)  # (Batch, 3)

        return output


# ===== Discriminator Architecture =====


class Discriminator1D(nn.Module):
    """
    Simple MLP Discriminator for GAN training on breakpoints.
    Input:
        x: (Batch, 3) - Breakpoint data [depth1, tts1, tts_end]
    Output:
        score: (Batch, 1) - Probability of being real (after sigmoid)
    """

    def __init__(self, dim=64):
        super().__init__()

        # Simple MLP for 3 breakpoint values
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        # x: (Batch, 3) or (Batch, 1, 3) - breakpoints
        if x.dim() == 3:
            x = x.squeeze(1)  # (Batch, 3)

        # Final classification
        score = self.mlp(x)  # (Batch, 1)

        return score


# ===== Model Factory =====


def create_model(model_type: str, config) -> nn.Module:
    """Factory function to create model based on config.

    Note: All models are actually MLPs since we work with 3 breakpoints.
    'unet' and 'fno' are kept for backward compatibility but use MLP architectures.
    """
    if model_type.lower() == "unet":
        return UNet1D(dim=config.unet_dim, time_emb_dim=config.time_emb_dim)
    elif model_type.lower() == "fno":
        return FNO1D(
            modes=config.fno_modes,
            width=config.fno_width,
            time_emb_dim=config.time_emb_dim,
        )
    elif model_type.lower() == "mlp":
        return MLP1D(dim=config.unet_dim, time_emb_dim=config.time_emb_dim)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'unet', 'fno', or 'mlp'."
        )


def create_discriminator(config) -> Discriminator1D:
    """Factory function to create discriminator based on config."""
    return Discriminator1D(dim=config.discriminator_dim)


if __name__ == "__main__":
    # Test models
    cfg = config.cfg

    print("Testing UNet1D (breakpoints)...")
    unet = UNet1D(dim=cfg.unet_dim, time_emb_dim=cfg.time_emb_dim)
    x = torch.randn(2, 3)  # Breakpoints: [depth1, tts1, tts_end]
    t = torch.randn(2, 1)
    out = unet(x, t)
    print(f"UNet input: {x.shape}, output: {out.shape}")
    print(f"UNet has {sum(p.numel() for p in unet.parameters()):,} parameters")

    print("\nTesting FNO1D (breakpoints)...")
    fno = FNO1D(modes=cfg.fno_modes, width=cfg.fno_width, time_emb_dim=cfg.time_emb_dim)
    out = fno(x, t)
    print(f"FNO input: {x.shape}, output: {out.shape}")
    print(f"FNO has {sum(p.numel() for p in fno.parameters()):,} parameters")

    print("\nTesting factory...")
    model = create_model("unet", cfg)
    print(f"Factory UNet output: {model(x, t).shape}")

    model = create_model("fno", cfg)
    print(f"Factory FNO output: {model(x, t).shape}")

    print("\nTesting MLP1D...")
    mlp = MLP1D(dim=cfg.unet_dim, time_emb_dim=cfg.time_emb_dim)
    mlp_out = mlp(x, t)
    print(f"MLP input: {x.shape}, output: {mlp_out.shape}")
    print(f"MLP has {sum(p.numel() for p in mlp.parameters()):,} parameters")

    model = create_model("mlp", cfg)
    print(f"Factory MLP output: {model(x, t).shape}")

    print("\nTesting Discriminator...")
    disc = create_discriminator(cfg)
    disc_out = disc(x)
    print(f"Discriminator input: {x.shape}, output: {disc_out.shape}")

    print("\nAll tests passed!")
