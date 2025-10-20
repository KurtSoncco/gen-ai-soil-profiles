import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE. Based on the original VQ-VAE paper [link](https://arxiv.org/abs/1711.00937).
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Softplus(),  # Ensure output is positive
        )
        self.input_dim = input_dim

    def forward(self, x):
        # The vq layer expects an image-like tensor, so we reshape it
        # The latent_dim becomes the channel dimension
        z = self.encoder(x)
        z = z.unsqueeze(-1).unsqueeze(-1)  # Add dummy spatial dimensions

        vq_loss, quantized, perplexity, _ = self.vq_layer(z)
        quantized = quantized.squeeze(-1).squeeze(-1)  # Remove dummy dimensions

        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity


def vq_vae_loss_function(reconstruction, x, vq_loss):
    recon_error = F.mse_loss(reconstruction, x, reduction="mean")
    return recon_error + vq_loss


def evaluate_model(model, data_loader, device):
    """Evaluates the model's reconstruction loss on a given dataset."""
    model.eval()
    total_recon_error = 0
    total_vq_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)
            reconstruction, vq_loss, _ = model(data)
            recon_error = F.mse_loss(reconstruction, data, reduction="mean")
            total_recon_error += recon_error.item()
            total_vq_loss += vq_loss.item()
    return total_recon_error + total_vq_loss
