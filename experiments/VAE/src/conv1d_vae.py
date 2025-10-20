import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual block with optional downsampling."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if stride > 1 or in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        
        return F.relu(out + residual)


class Conv1DVAE(nn.Module):
    """1D Convolutional VAE for Vs profiles."""
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 128, 256], kernel_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_channels = 1  # Treat as 1D signal
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(ResidualBlock1D(in_channels, hidden_dim, kernel_size, stride=2))
            in_channels = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_dim)
            encoded = self.encoder(dummy)
            self.encoded_dim = encoded.shape[1] * encoded.shape[2]
            
        self.fc_mu = nn.Linear(self.encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoded_dim)
        
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        
        # First decoder layer should match encoder output channels
        decoder_layers.append(ResidualBlock1D(hidden_dims_reversed[0], hidden_dims_reversed[0], kernel_size, stride=1))
        
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            if i == 0:
                continue  # Skip first iteration as we handled it above
            out_channels = hidden_dims_reversed[i] if i < len(hidden_dims_reversed) else 1
            stride = 2 if i < len(hidden_dims_reversed) - 1 else 1
            decoder_layers.append(ResidualBlock1D(hidden_dim, out_channels, kernel_size, stride=stride))
            
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final layer to match input dimension
        # Calculate the expected output size after decoder
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            encoded = self.encoder(dummy_input)
            decoded = self.decoder(encoded)
            expected_size = decoded.shape[2]
        
        # Use adaptive pooling to ensure correct output size
        self.final_adaptive = nn.AdaptiveAvgPool1d(input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # Reshape to (batch, 1, length) for conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Encode
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        mu, logvar = self.fc_mu(encoded_flat), self.fc_logvar(encoded_flat)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded_flat = self.fc_decode(z)
        decoded = decoded_flat.view(encoded.size(0), encoded.size(1), encoded.size(2))
        
        recon = self.decoder(decoded)
        recon = self.final_adaptive(recon)
        
        # Reshape back to (batch, length)
        if recon.dim() == 3:
            recon = recon.squeeze(1)
            
        return recon, mu, logvar


def conv1d_vae_loss_function(reconstruction, x, mu, logvar, beta=1.0, layer_weights=None, tv_weight=0.01):
    """
    Enhanced VAE loss with weighted reconstruction, TV regularization, and configurable beta.
    """
    # Weighted L1 reconstruction loss
    if layer_weights is not None:
        weights = layer_weights.unsqueeze(0).expand_as(x)
        recon_loss = torch.mean(weights * torch.abs(reconstruction - x))
    else:
        recon_loss = F.l1_loss(reconstruction, x, reduction='mean')
    
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total variation regularization for smoothness
    tv_loss = torch.mean(torch.abs(reconstruction[:, 1:] - reconstruction[:, :-1]))
    
    total_loss = recon_loss + beta * kld + tv_weight * tv_loss
    
    return total_loss, {
        'recon_loss': recon_loss.item(),
        'kld_loss': kld.item(),
        'tv_loss': tv_loss.item(),
        'total_loss': total_loss.item()
    }
