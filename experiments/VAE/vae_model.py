import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Softplus(),  # Ensure output is positive
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, self.encoder[0].in_features))
        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


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
