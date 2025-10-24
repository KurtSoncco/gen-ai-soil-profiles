import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        # First conv
        h = self.silu1(self.norm1(self.conv1(x)))
        
        # Add time embedding
        time_emb = self.time_mlp(t)  # (batch_size, out_channels)
        # Broadcast to match h shape: (batch_size, out_channels, length)
        h = h + time_emb.unsqueeze(-1) 
        
        # Second conv
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
        return h, p # Return skip connection


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, time_emb_dim) # For skip connection

    def forward(self, x, skip, t):
        x = self.up(x)
        # Ensure skip connection has the same length as upsampled x
        if x.shape[-1] != skip.shape[-1]:
            skip = torch.nn.functional.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1) # Concatenate skip connection
        x = self.conv(x, t)
        return x


class UNet1D(nn.Module):
    """
    A 1D UNet for modeling the vector field v(u, t).
    Input:
        x (u_t): (Batch, 1, PROFILE_POINTS)
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 1, PROFILE_POINTS)
    """
    def __init__(self, dim=64, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # UNet Blocks
        self.inc = ConvBlock(1, dim, time_emb_dim)
        self.down1 = DownBlock(dim, dim * 2, time_emb_dim)
        self.down2 = DownBlock(dim * 2, dim * 4, time_emb_dim)
        
        self.bot = ConvBlock(dim * 4, dim * 8, time_emb_dim)
        
        self.up1 = UpBlock(dim * 8, dim * 4, time_emb_dim)
        self.up2 = UpBlock(dim * 4, dim * 2, time_emb_dim)
        self.outc = nn.Conv1d(dim * 2, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.squeeze(-1))
        original_size = x.shape[-1]
        
        x1 = self.inc(x, t_emb)
        s1, x2 = self.down1(x1, t_emb)
        s2, x3 = self.down2(x2, t_emb)
        
        x_bot = self.bot(x3, t_emb)
        
        x_up = self.up1(x_bot, s2, t_emb)
        x_up = self.up2(x_up, s1, t_emb)
        
        output = self.outc(x_up)
        
        # Ensure output matches input size
        if output.shape[-1] != original_size:
            output = torch.nn.functional.interpolate(output, size=original_size, mode='linear', align_corners=False)
        
        return output


# ===== FNO Architecture =====

class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO"""
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
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
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )

    def forward(self, x, t_emb):
        # Spectral convolution
        x1 = self.conv(x)
        x2 = self.w(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(t_emb).unsqueeze(-1)  # (B, width, 1)
        x = x1 + x2 + time_emb
        
        return F.gelu(x)


class FNO1D(nn.Module):
    """
    Fourier Neural Operator for modeling the vector field v(u, t).
    Input:
        x (u_t): (Batch, 1, PROFILE_POINTS)
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 1, PROFILE_POINTS)
    """
    def __init__(self, modes=16, width=64, time_emb_dim=128):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Input projection
        self.fc0 = nn.Linear(1, self.width)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(modes, width, time_emb_dim) for _ in range(4)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, t):
        # x: (B, 1, L)
        # t: (B, 1)
        
        t_emb = self.time_mlp(t.squeeze(-1))  # (B, time_emb_dim)
        
        # Transpose for FNO: (B, L, 1) -> (B, L, width)
        x = x.transpose(1, 2)  # (B, L, 1)
        x = self.fc0(x)  # (B, L, width)
        
        # Transpose back for spectral conv: (B, width, L)
        x = x.transpose(1, 2)  # (B, width, L)
        
        # Apply FNO layers
        for fno_layer in self.fno_layers:
            x = fno_layer(x, t_emb)
        
        # Transpose for final projection: (B, L, width)
        x = x.transpose(1, 2)  # (B, L, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (B, L, 1)
        
        # Transpose back to output format: (B, 1, L)
        x = x.transpose(1, 2)  # (B, 1, L)
        
        return x


# ===== Model Factory =====

def create_model(model_type: str, config) -> nn.Module:
    """Factory function to create model based on config."""
    if model_type.lower() == "unet":
        return UNet1D(
            dim=config.unet_dim,
            time_emb_dim=config.time_emb_dim
        )
    elif model_type.lower() == "fno":
        return FNO1D(
            modes=config.fno_modes,
            width=config.fno_width,
            time_emb_dim=config.time_emb_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'unet' or 'fno'.")


if __name__ == "__main__":
    # Test both models
    cfg = config.cfg
    
    # Test UNet
    print("Testing UNet1D...")
    unet = UNet1D(dim=cfg.unet_dim, time_emb_dim=cfg.time_emb_dim)
    x = torch.randn(2, 1, 128)
    t = torch.randn(2, 1)
    out = unet(x, t)
    print(f"UNet input: {x.shape}, output: {out.shape}")
    
    # Test FNO
    print("Testing FNO1D...")
    fno = FNO1D(modes=cfg.fno_modes, width=cfg.fno_width, time_emb_dim=cfg.time_emb_dim)
    out = fno(x, t)
    print(f"FNO input: {x.shape}, output: {out.shape}")
    
    # Test factory
    print("Testing factory...")
    model = create_model("unet", cfg)
    print(f"Factory UNet output: {model(x, t).shape}")
    
    model = create_model("fno", cfg)
    print(f"Factory FNO output: {model(x, t).shape}")
    
    print("All tests passed!")
