import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from typing import Optional


class LatentGMMSampler:
    """Gaussian Mixture Model sampler for VAE latent space."""

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.is_fitted = False

    def fit(self, latent_samples: np.ndarray) -> None:
        """Fit GMM to latent samples."""
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type="full",
        )
        self.gmm.fit(latent_samples)
        self.is_fitted = True

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from fitted GMM."""
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before sampling")
        return self.gmm.sample(n_samples)[0]

    def log_likelihood(self, samples: np.ndarray) -> np.ndarray:
        """Compute log likelihood of samples."""
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before computing likelihood")
        return self.gmm.score_samples(samples)


def extract_latent_samples(
    model, data_loader, device, max_samples: Optional[int] = None
):
    """Extract latent samples from VAE encoder."""
    model.eval()
    latent_samples = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if max_samples and len(latent_samples) >= max_samples:
                break

            # Handle both tuple (DAE) and single tensor (baseline) cases
            if isinstance(batch_data, tuple):
                data = batch_data[0]
            else:
                data = batch_data

            # Ensure data is a tensor (handle list case)
            if isinstance(data, list):
                data = torch.stack(data)
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data)

            data = data.to(device)

            if model.__class__.__name__ in ["Conv1DVAE", "SimpleConv1DVAE"]:
                # Conv1D VAE
                if data.dim() == 2:
                    data = data.unsqueeze(1)
                encoded = model.encoder(data)
                encoded_flat = encoded.view(encoded.size(0), -1)
                mu, _ = model.fc_mu(encoded_flat), model.fc_logvar(encoded_flat)
            else:
                # MLP VAE
                h = model.encoder(data.view(-1, model.encoder[0].in_features))
                mu, _ = model.fc_mu(h), model.fc_logvar(h)

            latent_samples.append(mu.cpu().numpy())

    return np.vstack(latent_samples)


def generate_with_gmm(model, gmm_sampler, n_samples: int, device, input_dim: int):
    """Generate samples using GMM prior."""
    model.eval()

    # Sample from GMM
    z_samples = gmm_sampler.sample(n_samples)
    z_tensor = torch.from_numpy(z_samples).float().to(device)

    with torch.no_grad():
        if model.__class__.__name__ in ["Conv1DVAE", "SimpleConv1DVAE"]:
            # Conv1D VAE
            decoded_flat = model.fc_decode(z_tensor)
            # Reshape to match encoder output
            batch_size = z_tensor.size(0)
            hidden_dim = 256  # Last hidden dim from Conv1D VAE
            decoded = decoded_flat.view(batch_size, hidden_dim, -1)
            generated = model.decoder(decoded)
            generated = model.final_conv(generated)
            if generated.dim() == 3:
                generated = generated.squeeze(1)
        else:
            # MLP VAE
            generated = model.decoder(z_tensor)

    return generated.cpu().numpy()


def compute_layer_weights(depths: np.ndarray) -> torch.Tensor:
    """Compute layer weights proportional to layer thickness."""
    layer_thickness = np.diff(depths)
    # Normalize weights
    weights = layer_thickness / np.sum(layer_thickness)
    return torch.from_numpy(weights).float()


def log_vs_transform(vs_profiles: np.ndarray, vs_min: float = 50.0) -> np.ndarray:
    """Transform Vs profiles to log space."""
    return np.log(np.maximum(vs_profiles, vs_min))


def exp_vs_transform(log_vs_profiles: np.ndarray) -> np.ndarray:
    """Transform log Vs profiles back to linear space."""
    return np.exp(log_vs_profiles)


def vs_to_log_vs_profiles(
    vs_profiles: np.ndarray, depths: np.ndarray, vs_min: float = 50.0
) -> np.ndarray:
    """Convert Vs profiles to log(Vs) representation."""
    return log_vs_transform(vs_profiles, vs_min)


def log_vs_to_vs_profiles(log_vs_profiles: np.ndarray) -> np.ndarray:
    """Convert log(Vs) profiles back to Vs."""
    return exp_vs_transform(log_vs_profiles)
