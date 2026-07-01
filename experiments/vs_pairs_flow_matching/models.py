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
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)  # Broadcast properly
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ===== MLP Architecture (best for 4D vectors) =====


class MLPFlowMatcher(nn.Module):
    """
    A simple MLP for modeling the vector field v(u, t) for 4D vs pairs.
    Input:
        x (u_t): (Batch, 1, 4)
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 1, 4)
    """

    def __init__(self, dim=128, layers=4, time_emb_dim=64):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Build MLP layers
        layers_list = []
        in_dim = 4 + time_emb_dim  # 4 for data, time_emb_dim for time

        for i in range(layers):
            out_dim = dim if i < layers - 1 else 4  # Output 4 dimensions
            layers_list.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = out_dim

        # Remove the last SiLU since we want raw output
        self.mlp = nn.Sequential(*layers_list[:-1])

    def forward(self, x, t):
        # x: (B, 1, 4)
        # t: (B, 1)

        t_emb = self.time_mlp(t.squeeze(-1))  # (B, time_emb_dim)

        # Flatten x for MLP
        x_flat = x.squeeze(1)  # (B, 4)

        # Concatenate with time embedding
        x_with_time = torch.cat([x_flat, t_emb], dim=1)  # (B, 4 + time_emb_dim)

        # Pass through MLP
        v_flat = self.mlp(x_with_time)  # (B, 4)

        # Reshape back to (B, 1, 4)
        v = v_flat.unsqueeze(1)

        return v


# ===== FNO Architecture for 4D Vectors =====


class SpectralConv4D(nn.Module):
    """Spectral Convolution layer for 4D vectors using FFT."""

    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes (max 3 for 4D due to Nyquist)

        self.scale = 1 / (in_channels * out_channels)
        # For 4D input, rfft outputs (1, 2, 2, 1) modes, so we have at most 3 modes
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, 4)
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, 3) for length 4

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOVectorBlock(nn.Module):
    """FNO block with spectral convolution and time conditioning for 4D vectors."""

    def __init__(self, modes, width, time_emb_dim):
        super().__init__()
        self.modes = modes
        self.width = width

        self.conv = SpectralConv4D(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, x, t_emb):
        # x: (B, width, 4)
        # Spectral convolution
        x1 = self.conv(x)
        x2 = self.w(x)

        # Add time conditioning
        time_emb = self.time_mlp(t_emb).unsqueeze(-1)  # (B, width, 1)
        x = x1 + x2 + time_emb

        return F.gelu(x)


class FNOVectorMatcher(nn.Module):
    """
    Fourier Neural Operator for modeling the vector field v(u, t) on 4D vs pairs.

    Input:
        x (u_t): (Batch, 1, 4)
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 1, 4)
    """

    def __init__(self, modes=3, width=64, time_emb_dim=64):
        super().__init__()
        self.modes = min(modes, 3)  # Limit to 3 for 4D vectors
        self.width = width

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Input projection
        self.fc0 = nn.Linear(1, self.width)

        # FNO layers (fewer layers for simplicity)
        self.fno_layers = nn.ModuleList(
            [FNOVectorBlock(self.modes, width, time_emb_dim) for _ in range(2)]
        )

        # Output projection
        self.fc1 = nn.Linear(width, width // 2)
        self.fc2 = nn.Linear(width // 2, 1)

    def forward(self, x, t):
        # x: (B, 1, 4)
        # t: (B, 1)

        t_emb = self.time_mlp(t.squeeze(-1))  # (B, time_emb_dim)

        # Transpose for FNO: (B, 4, 1) -> (B, 4, width)
        x = x.transpose(1, 2)  # (B, 4, 1)
        x = self.fc0(x)  # (B, 4, width)

        # Transpose back for spectral conv: (B, width, 4)
        x = x.transpose(1, 2)  # (B, width, 4)

        # Apply FNO layers
        for fno_layer in self.fno_layers:
            x = fno_layer(x, t_emb)

        # Transpose for final projection: (B, 4, width)
        x = x.transpose(1, 2)  # (B, 4, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (B, 4, 1)

        # Transpose back to output format: (B, 1, 4)
        x = x.transpose(1, 2)  # (B, 1, 4)

        return x


# ===== Model Factory =====


def create_model(model_type: str, config) -> nn.Module:
    """Factory function to create model based on config."""
    if model_type.lower() == "mlp":
        return MLPFlowMatcher(
            dim=config.mlp_dim,
            layers=config.mlp_layers,
            time_emb_dim=config.time_emb_dim,
        )
    elif model_type.lower() == "fno":
        return FNOVectorMatcher(
            modes=config.fno_modes,
            width=config.fno_width,
            time_emb_dim=config.time_emb_dim,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Use 'mlp' or 'fno' for vs pairs."
        )


if __name__ == "__main__":
    # Test model
    cfg = config.cfg

    print("Testing MLPFlowMatcher...")
    model_mlp = MLPFlowMatcher(
        dim=cfg.mlp_dim, layers=cfg.mlp_layers, time_emb_dim=cfg.time_emb_dim
    )
    x = torch.randn(2, 1, 4)
    t = torch.randn(2, 1)
    out = model_mlp(x, t)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"MLP has {sum(p.numel() for p in model_mlp.parameters()):,} parameters")

    print("\nTesting FNOVectorMatcher...")
    model_fno = FNOVectorMatcher(
        modes=cfg.fno_modes, width=cfg.fno_width, time_emb_dim=cfg.time_emb_dim
    )
    out = model_fno(x, t)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"FNO has {sum(p.numel() for p in model_fno.parameters()):,} parameters")

    print("\nTesting factory...")
    model = create_model("mlp", cfg)
    print(f"Factory MLP output: {model(x, t).shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    model = create_model("fno", cfg)
    print(f"Factory FNO output: {model(x, t).shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\nAll tests passed!")
