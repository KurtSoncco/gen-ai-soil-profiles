import logging
from pathlib import Path

import torch
import tqdm
from vae_model import evaluate_model, vae_loss_function


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    EPOCHS: int,
    device: str = "cpu",
):
    logging.info("\nTraining the VAE...")
    model.train()
    train_losses = []
    test_losses = []
    for epoch in range(EPOCHS):
        train_loss = 0
        with tqdm.tqdm(
            total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}"
        ) as pbar:
            for data in train_loader:
                data = data[0].to(device)
                optimizer.zero_grad()
                reconstruction, mu, log_var = model(data)
                loss = vae_loss_function(reconstruction, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if epoch % 10 == 0:
            logging.info(f"Epoch: {epoch}/{EPOCHS}, Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        model.eval()
        test_loss = evaluate_model(model, vae_loss_function, test_loader, device)
        model.train()
        logging.info(f"Test set loss: {test_loss:.4f}")

        test_losses.append(test_loss)

    # Save the model
    torch.save(model.state_dict(), Path(__file__).parent / "vae_model.pth")

    return model, train_losses, test_losses
