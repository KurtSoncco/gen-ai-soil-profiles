import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.optim as optim
from dotenv import load_dotenv
from preprocessing import preprocess_data, standardize_and_create_tts_profiles
from sklearn.model_selection import train_test_split
from training import train
from utils import (
    Vs30_calc,
    calculate_vs30,
    # evaluate_generation,
    plot_stair_profiles,
    tts_to_Vs,
)
from vae_model import VAE, evaluate_model, vae_loss_function
from vq_vae_model import VQVAE  # vq_vae_loss_function
from vq_vae_model import evaluate_model as evaluate_vq_model

import wandb
from soilgen_ai.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

# Basic configuration
setup_logging()
sns.set_palette("colorblind")

# Load the data
file_path = Path(__file__).parent.parent.parent / "data" / "vspdb_vs_profiles.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()
logging.info(f"Data loaded with shape: {df.shape}")
logging.info(f"Data loaded with columns: {df.columns.tolist()}")

# Convert df into profiles dict
profiles_dict = {
    metadata_id: group.sort_values("depth")
    .drop(columns=["velocity_metadata_id"])
    .reset_index(drop=True)
    for metadata_id, group in df.groupby("velocity_metadata_id")
}

# Preprocess the data
profiles_dict, dropped_ids = preprocess_data(profiles_dict)
logging.info(f"Data shape after preprocessing: {df.shape}")
logging.info(f"Number of profiles dropped: {len(dropped_ids)}")

# Compute the Vs30 of the real profiles
real_vs30 = [Vs30_calc(p["depth"], p["vs_value"]) for _, p in profiles_dict.items()]
# Print statistics
logging.info(
    f"Real Vs30 statistics: mean={np.nanmean(real_vs30):.2f}, std={np.nanstd(real_vs30):.2f}, min={np.nanmin(real_vs30):.2f}, max={np.nanmax(real_vs30):.2f}"
)

# --- Standardize and prepare for VAE ---
NUM_LAYERS = 100
MAX_DEPTH = 2000
VS_MAX = 2500  # Used for visualization limits
LATENT_DIM = 32  # Reduced latent dimension
NUM_NEW_PROFILES = 1000  # Number of new profiles to generate

tts_profiles_normalized, standard_depths = standardize_and_create_tts_profiles(
    profiles_dict, NUM_LAYERS, MAX_DEPTH
)
INPUT_DIM = tts_profiles_normalized.shape[1]
logging.info(f"Final standardized TTS data shape: {tts_profiles_normalized.shape}")

## Do some checks before training
# Extract the max tts of each profile and plot distribution
max_tts = np.max(tts_profiles_normalized, axis=1)
plt.figure(figsize=(8, 5))
sns.histplot(max_tts, bins=30, kde=True)
plt.title("Distribution of Maximum Normalized TTS in Profiles")
plt.xlabel("Maximum Normalized TTS")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Training and evaluation part
if True:
    # Split the data
    X_train_tensor, X_test_tensor = train_test_split(
        torch.from_numpy(tts_profiles_normalized).float(),
        test_size=0.2,
        random_state=42,
    )
    logging.info(f"Training data shape: {X_train_tensor.shape}")
    logging.info(f"Testing data shape: {X_test_tensor.shape}")

    # Plot some profiles for check
    def plot_profiles(profiles, depths, title):
        plt.figure(figsize=(10, 6))
        for profile in profiles:
            # `profiles` are now d_tts, need to convert to cumulative for plotting
            tts_cumulative = np.insert(np.cumsum(profile), 0, 0)
            plt.plot(tts_cumulative, depths, alpha=0.5)
        plt.title(title)
        plt.xlabel("Cumulative Travel Time")
        plt.ylabel("Depth")
        plt.ylim(MAX_DEPTH, 0)
        plt.grid()
        plt.show()

    plot_profiles(
        tts_profiles_normalized, standard_depths, "Standardized TTS Profiles (d_tts)"
    )

    # --- Model setup and training ---
    EPOCHS = 500
    BATCH_SIZE = 32  # Increased batch size
    LEARNING_RATE = 1e-4  # Increased learning rate
    MODEL = "VQ-VAE"  # Options: "VAE" or "VQ-VAE"
    WEIGHT_DECAY = 1e-5  # Weight decay for optimizer
    BETAS = (0.6, 0.9)  # Betas for Adam optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if MODEL == "VAE":
        model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
        model_type = "VAE"
    else:
        model = VQVAE(
            input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_embeddings=512
        ).to(device)
        model_type = "VQ-VAE"
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5, betas=(0.6, 0.9)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # --- W&B Initialization ---
    wandb.init(
        project=os.getenv("W_B_PROJECT"),
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "model_type": model_type,
            "num_layers": NUM_LAYERS,
            "max_depth": MAX_DEPTH,
            "input_dim": INPUT_DIM,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
        },
    )
    wandb.watch(model, log="all")
    logging.info("Starting training...")
    # --- Training ---
    model, train_losses, test_losses = train(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        EPOCHS,
        str(device),
        model_type=MODEL,
    )

    # Plot training losses
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Losses")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    plt.savefig(Path(__file__).parent / "training_losses.png", bbox_inches="tight")
    plt.close()

    # --- Evaluation on test set ---
    if MODEL == "VQ-VAE":
        reconstruction_loss = evaluate_vq_model(model, test_loader, device)
    else:
        reconstruction_loss = evaluate_model(
            model, vae_loss_function, test_loader, device
        )
    logging.info(f"Test set reconstruction loss: {reconstruction_loss:.4f}")

    # --- Generation and visualization ---
    model.eval()
    with torch.no_grad():
        if MODEL == "VAE":
            # For VAE, sample from a standard normal distribution in the latent space
            z = torch.randn(NUM_NEW_PROFILES, LATENT_DIM).to(device)
            generated_tts_normalized = model.decoder(z).cpu().numpy()
        else:  # For VQ-VAE
            # For VQ-VAE, we sample discrete latent codes from the codebook and decode them.
            # This is a simple generation method; more advanced methods might train a prior over the codes.
            codebook_indices = torch.randint(
                0, model.vq_layer._num_embeddings, (NUM_NEW_PROFILES,)
            ).to(device)
            z_quantized = model.vq_layer._embedding(codebook_indices)
            generated_tts_normalized = model.decoder(z_quantized).cpu().numpy()

    # Denormalize and convert to Vs
    generated_d_tts_denorm = np.expm1(generated_tts_normalized)
    dz = np.diff(standard_depths)  # This is now an array of layer thicknesses
    generated_vs_profiles = tts_to_Vs(generated_d_tts_denorm, dz)

    # Also convert real test data to Vs for comparison
    real_tts_normalized = X_test_tensor.cpu().numpy()
    real_d_tts_denorm = np.expm1(real_tts_normalized)
    real_vs_profiles = tts_to_Vs(real_d_tts_denorm, dz)

    # Calculate Vs30 for generated and real profiles
    generated_vs30 = [calculate_vs30(p, standard_depths) for p in generated_vs_profiles]
    real_vs30_test = [calculate_vs30(p, standard_depths) for p in real_vs_profiles]

    # Print statistics
    logging.info(
        f"Generated Vs30 statistics: mean={np.nanmean(generated_vs30):.2f}, std={np.nanstd(generated_vs30):.2f}, min={np.nanmin(generated_vs30):.2f}, max={np.nanmax(generated_vs30):.2f}"
    )
    logging.info(
        f"Real (test set) Vs30 statistics: mean={np.nanmean(real_vs30_test):.2f}, std={np.nanstd(real_vs30_test):.2f}, min={np.nanmin(real_vs30_test):.2f}, max={np.nanmax(real_vs30_test):.2f}"
    )

    logging.info("Plotting generated Vs profiles...")
    plot_stair_profiles(
        generated_vs_profiles,
        standard_depths,
        "Generated Shear Wave Velocity Profiles",
        MAX_DEPTH,
        VS_MAX,
    )

    # --- Quantitative Evaluation ---
    logging.info("Evaluating generated profiles...")
    #
    # evaluate_generation(real_vs_profiles, generated_vs_profiles, standard_depths)
    wandb.finish()
    logging.info("VAE script finished.")
