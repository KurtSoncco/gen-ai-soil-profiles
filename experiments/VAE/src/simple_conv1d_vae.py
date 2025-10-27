import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv1DVAE(nn.Module):
    """Simplified 1D Convolutional VAE for Vs profiles."""

    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: Simple conv layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate encoder output size
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_dim)
            encoded = self.encoder(dummy)
            self.encoded_dim = encoded.shape[1] * encoded.shape[2]

        self.fc_mu = nn.Linear(self.encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_dim, latent_dim)

        # Decoder - simplified approach
        self.fc_decode = nn.Linear(latent_dim, self.encoded_dim)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(input_dim),  # Ensure output matches input_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Handle the case where data comes as [batch, features, 1] instead of [batch, features]
        if x.dim() == 3:
            if x.shape[2] == 1:
                x = x.squeeze(2)  # Remove the last dimension
            elif x.shape[1] == 1:
                x = x.squeeze(1)  # Remove the middle dimension
            else:
                # If it's [batch, features, length], transpose to [batch, length, features]
                # then treat as [batch, features]
                x = x.transpose(1, 2).contiguous().view(x.shape[0], -1)

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
        # Reshape to match encoder output dimensions
        batch_size = z.size(0)
        decoded = decoded_flat.view(batch_size, encoded.size(1), encoded.size(2))

        recon = self.decoder(decoded)

        # Reshape back to (batch, length)
        if recon.dim() == 3:
            recon = recon.squeeze(1)

        return recon, mu, logvar


def simple_conv1d_vae_loss_function(
    reconstruction,
    x,
    mu,
    log_var,
    beta: float = 1.0,
    layer_weights=None,
    tv_weight=0.01,
):
    """
    Enhanced VAE loss with weighted reconstruction, TV regularization, and configurable beta.
    """
    # Weighted L1 reconstruction loss
    if layer_weights is not None:
        weights = layer_weights.unsqueeze(0).expand_as(x)
        recon_loss = torch.mean(weights * torch.abs(reconstruction - x))
    else:
        recon_loss = F.l1_loss(reconstruction, x, reduction="mean")

    # KL divergence
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Total variation regularization for smoothness
    tv_loss = torch.mean(torch.abs(reconstruction[:, 1:] - reconstruction[:, :-1]))

    total_loss = recon_loss + beta * kld + tv_weight * tv_loss

    return total_loss, {
        "recon_loss": recon_loss.item(),
        "kld_loss": kld.item(),
        "tv_loss": tv_loss.item(),
        "total_loss": total_loss.item(),
    }
