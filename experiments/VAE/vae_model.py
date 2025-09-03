import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.z_mean_layer = nn.Linear(64, latent_dim)
        self.z_log_var_layer = nn.Linear(64, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.encoder_fc1(x))
        h = torch.relu(self.encoder_fc2(h))
        return self.z_mean_layer(h), self.z_log_var_layer(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h = torch.relu(self.decoder_fc1(z))
        h = torch.relu(self.decoder_fc2(h))
        return self.decoder_fc3(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var


def vae_loss_function(reconstruction, x, mu, log_var):
    BCE = nn.functional.mse_loss(reconstruction, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def evaluate_model(model, loss_function, data_loader, device):
    """Evaluates the model's reconstruction loss on a given dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)
            reconstruction, mu, log_var = model(data)
            loss = loss_function(reconstruction, data, mu, log_var)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)
