import logging
from pathlib import Path

import torch
import tqdm
import wandb
from vae_model import evaluate_model, vae_loss_function


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    EPOCHS: int,
    device: str = "cpu",
):
    logging.info("\nTraining the VAE...")
    model.train()
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    epochs_no_improve = 0
    patience = 10

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
                pbar.set_postfix(
                    {
                        "Train Loss": f"{loss.item():.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        test_loss = evaluate_model(model, vae_loss_function, test_loader, device)
        model.train()
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        logging.info(
            f"Epoch: {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), Path(__file__).parent / "vae_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(
                f"Early stopping triggered after {epoch + 1} epochs due to no improvement in test loss."
            )
            break

    return model, train_losses, test_losses
