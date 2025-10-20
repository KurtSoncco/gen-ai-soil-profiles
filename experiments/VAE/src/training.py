import logging

import torch
from tqdm import tqdm
from .vae_model import vae_loss_function
from .vq_vae_model import vq_vae_loss_function

import wandb


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    epochs,
    device,
    model_type="VAE",
    early_stop_patience: int | None = None,
):
    """
    Main training loop for VAE models.
    """
    model.to(device)
    train_losses = []
    test_losses = []

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        for i, data in progress_bar:
            inputs = data[0].to(device)
            optimizer.zero_grad()

            if model_type == "VAE":
                reconstruction, mu, log_var = model(inputs)
                loss = vae_loss_function(reconstruction, inputs, mu, log_var)
            else:  # VQ-VAE
                reconstruction, vq_loss, perplexity = model(inputs)
                loss = vq_vae_loss_function(reconstruction, inputs, vq_loss)
                wandb.log({"perplexity": perplexity.item()})

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                if model_type == "VAE":
                    reconstruction, mu, log_var = model(inputs)
                    loss = vae_loss_function(reconstruction, inputs, mu, log_var)
                else:  # VQ-VAE
                    reconstruction, vq_loss, _ = model(inputs)
                    loss = vq_vae_loss_function(reconstruction, inputs, vq_loss)
                test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        logging.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )
        wandb.log(
            {"train_loss": train_loss, "test_loss": test_loss, "epoch": epoch + 1}
        )

        # Early stopping
        if early_stop_patience is not None:
            if test_loss + 1e-7 < best_val:
                best_val = test_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    logging.info("Early stopping baseline training")
                    break

    return model, train_losses, test_losses
