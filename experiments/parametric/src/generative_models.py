"""
Generative Models for Parametric Profile Parameters

This module implements generative models (GMM and MLP) to learn the distribution
of parametric profile parameters and generate new ones.
"""

import logging
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ParameterGMM:
    """
    Gaussian Mixture Model for generating parametric profile parameters.
    """

    def __init__(self, n_components: int = 8, random_state: int = 42):
        """
        Initialize GMM for parameter generation.

        Args:
            n_components: Number of Gaussian components
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=n_components, random_state=random_state, covariance_type="full"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, parameters: np.ndarray) -> None:
        """
        Fit GMM to parameter data.

        Args:
            parameters: Array of shape (n_samples, n_params) containing fitted parameters
        """
        logging.info(
            f"Fitting GMM with {self.n_components} components to {parameters.shape[0]} samples"
        )

        # Standardize parameters
        parameters_scaled = self.scaler.fit_transform(parameters)

        # Fit GMM
        self.gmm.fit(parameters_scaled)
        self.is_fitted = True

        logging.info(
            f"GMM fitted successfully. Log-likelihood: {self.gmm.score(parameters_scaled):.4f}"
        )

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new parameters using the fitted GMM.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            Array of shape (n_samples, n_params) containing generated parameters
        """
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before generating samples")

        # Generate samples from GMM
        parameters_scaled, _ = self.gmm.sample(n_samples)

        # Transform back to original scale
        parameters = self.scaler.inverse_transform(parameters_scaled)

        return parameters

    def save(self, filepath: str) -> None:
        """Save the fitted GMM model."""
        model_data = {
            "gmm": self.gmm,
            "scaler": self.scaler,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logging.info(f"GMM model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load a fitted GMM model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.gmm = model_data["gmm"]
        self.scaler = model_data["scaler"]
        self.n_components = model_data["n_components"]
        self.random_state = model_data["random_state"]
        self.is_fitted = model_data["is_fitted"]

        logging.info(f"GMM model loaded from {filepath}")


class ParameterMLP(nn.Module):
    """
    Multi-Layer Perceptron for generating parametric profile parameters.
    Uses a VAE-like architecture to learn parameter distributions.
    """

    def __init__(
        self, input_dim: int, latent_dim: int = 16, hidden_dims: List[int] = [64, 32]
    ):
        """
        Initialize MLP for parameter generation.

        Args:
            input_dim: Number of input parameters
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            prev_dim = hidden_dim

        encoder_layers.extend(
            [
                nn.Linear(prev_dim, latent_dim * 2)  # mu and logvar
            ]
        )

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode parameters to latent space."""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to parameters."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new parameters from latent space."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            generated = self.decode(z)
        return generated


class ParameterVAETrainer:
    """
    Trainer for the Parameter VAE model.
    """

    def __init__(
        self, model: ParameterMLP, device: torch.device, learning_rate: float = 1e-3
    ):
        """
        Initialize the trainer.

        Args:
            model: ParameterMLP model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.5
        )

    def vae_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss.

        Args:
            recon: Reconstructed parameters
            x: Original parameters
            mu: Latent mean
            logvar: Latent log variance
            beta: KL divergence weight

        Returns:
            Total loss and loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        total_loss = recon_loss + beta * kld_loss

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kld_loss": kld_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_epoch(
        self, train_loader: torch.utils.data.DataLoader, beta: float = 1.0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(x)
            loss, loss_dict = self.vae_loss(recon, x, mu, logvar, beta)

            loss.backward()
            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            total_recon_loss += loss_dict["recon_loss"]
            total_kld_loss += loss_dict["kld_loss"]

        return {
            "total_loss": total_loss / len(train_loader),
            "recon_loss": total_recon_loss / len(train_loader),
            "kld_loss": total_kld_loss / len(train_loader),
        }

    def validate(
        self, val_loader: torch.utils.data.DataLoader, beta: float = 1.0
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)

                recon, mu, logvar = self.model(x)
                loss, loss_dict = self.vae_loss(recon, x, mu, logvar, beta)

                total_loss += loss_dict["total_loss"]
                total_recon_loss += loss_dict["recon_loss"]
                total_kld_loss += loss_dict["kld_loss"]

        return {
            "total_loss": total_loss / len(val_loader),
            "recon_loss": total_recon_loss / len(val_loader),
            "kld_loss": total_kld_loss / len(val_loader),
        }

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        beta: float = 1.0,
        early_stopping_patience: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            beta: KL divergence weight
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": [],
            "train_kld_loss": [],
            "val_kld_loss": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, beta)

            # Validation
            val_metrics = self.validate(val_loader, beta)

            # Update learning rate
            self.scheduler.step(val_metrics["total_loss"])

            # Store history
            for key in history:
                if key.startswith("train_"):
                    metric_key = key.split("_", 1)[1]
                    if metric_key == "loss":
                        metric_key = "total_loss"
                    history[key].append(train_metrics[metric_key])
                else:
                    metric_key = key.split("_", 1)[1]
                    if metric_key == "loss":
                        metric_key = "total_loss"
                    history[key].append(val_metrics[metric_key])

            # Early stopping
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['total_loss']:.4f}"
                )

        return history

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            filepath,
        )
        logging.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info(f"Model loaded from {filepath}")


class ParameterGenerator:
    """
    Main class for generating parametric profile parameters using either GMM or MLP.
    """

    def __init__(self, model_type: str = "gmm", **kwargs):
        """
        Initialize the parameter generator.

        Args:
            model_type: Type of generative model ('gmm' or 'mlp')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.model: ParameterGMM | ParameterMLP
        self.scaler: StandardScaler | None = None
        self.is_fitted = False

        if model_type == "gmm":
            self.model = ParameterGMM(**kwargs)
        elif model_type == "mlp":
            self.model = ParameterMLP(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        assert self.model is not None

    def fit(self, parameters: np.ndarray, **kwargs) -> None:
        """
        Fit the generative model to parameter data.

        Args:
            parameters: Array of shape (n_samples, n_params) containing fitted parameters
            **kwargs: Additional arguments for training
        """
        if self.model_type == "gmm":
            assert isinstance(self.model, ParameterGMM)
            self.model.fit(parameters)
        elif self.model_type == "mlp":
            assert isinstance(self.model, ParameterMLP)
            # For MLP, we need to set up training
            device = kwargs.get(
                "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            learning_rate = kwargs.get("learning_rate", 1e-3)
            epochs = kwargs.get("epochs", 100)

            # Normalize parameters
            self.scaler = StandardScaler()
            parameters_scaled = self.scaler.fit_transform(parameters)

            # Split data
            X_train, X_val = train_test_split(
                parameters_scaled, test_size=0.2, random_state=42
            )

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float()
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_val).float()
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=32, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )

            # Train model
            trainer = ParameterVAETrainer(self.model, device, learning_rate)
            trainer.train(train_loader, val_loader, epochs=epochs)

            self.model = trainer.model  # Update model reference

        self.is_fitted = True

    def generate(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate new parameters.

        Args:
            n_samples: Number of parameter sets to generate
            **kwargs: Additional arguments for generation

        Returns:
            Array of shape (n_samples, n_params) containing generated parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating samples")

        if self.model_type == "gmm":
            assert isinstance(self.model, ParameterGMM)
            return self.model.generate(n_samples)
        elif self.model_type == "mlp":
            assert isinstance(self.model, ParameterMLP)
            assert self.scaler is not None
            device = kwargs.get(
                "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            generated_scaled = self.model.generate(n_samples, device).cpu().numpy()
            return self.scaler.inverse_transform(generated_scaled)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def save(self, filepath: str) -> None:
        """Save the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")

        if self.model_type == "gmm":
            assert isinstance(self.model, ParameterGMM)
            self.model.save(filepath)
        elif self.model_type == "mlp":
            assert isinstance(self.model, ParameterMLP)
            assert self.scaler is not None
            # Save model and scaler separately
            model_path = filepath.replace(".pkl", "_model.pth")
            scaler_path = filepath.replace(".pkl", "_scaler.pkl")

            torch.save(self.model.state_dict(), model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            logging.info(f"MLP model saved to {model_path} and scaler to {scaler_path}")

    def load(self, filepath: str) -> None:
        """Load a fitted model."""
        if self.model_type == "gmm":
            assert isinstance(self.model, ParameterGMM)
            self.model.load(filepath)
        elif self.model_type == "mlp":
            assert isinstance(self.model, ParameterMLP)
            # Load model and scaler separately
            model_path = filepath.replace(".pkl", "_model.pth")
            scaler_path = filepath.replace(".pkl", "_scaler.pkl")

            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            logging.info(
                f"MLP model loaded from {model_path} and scaler from {scaler_path}"
            )

        self.is_fitted = True
