# main.py
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.optim as optim

# Add src to path to import logging_config
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from preprocessing import preprocess_data, standardize_and_create_tts_profiles
from sklearn.model_selection import train_test_split
from training import train
from utils import evaluate_generation, plot_stair_profiles, tts_to_Vs
from vae_model import VAE, evaluate_model, vae_loss_function

from logging_config import setup_logging

# Basic configuration
setup_logging()
sns.set_palette("colorblind")

# Load the data
file_path = Path(__file__).parent.parent.parent / "data" / "vspdb_vs_profiles.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()
logging.info(f"Data loaded with shape: {df.shape}")

# Preprocess the data
df = preprocess_data(df)
logging.info(f"Data shape after preprocessing: {df.shape}")

# --- Standardize and prepare for VAE ---
NUM_LAYERS = 10
MAX_DEPTH = 2000
VS_MAX = 2500  # Used for visualization limits
LATENT_DIM = 32  # Reduced latent dimension

tts_profiles_normalized, standard_depths = standardize_and_create_tts_profiles(
    df, NUM_LAYERS, MAX_DEPTH
)
INPUT_DIM = tts_profiles_normalized.shape[1]
logging.info(f"Final standardized TTS data shape: {tts_profiles_normalized.shape}")


# Split the data
X_train_tensor, X_test_tensor = train_test_split(
    torch.from_numpy(tts_profiles_normalized).float(), test_size=0.2, random_state=42
)
logging.info(f"Training data shape: {X_train_tensor.shape}")
logging.info(f"Testing data shape: {X_test_tensor.shape}")


# Plot some profiles for check
def plot_profiles(profiles, depths, title):
    plt.figure(figsize=(10, 6))
    for profile in profiles:
        plt.plot(profile, depths, alpha=0.5)
    plt.title(title)
    plt.xlabel("Cumulative Travel Time")
    plt.ylabel("Depth")
    plt.ylim(MAX_DEPTH, 0)
    plt.grid()
    plt.show()


plot_profiles(tts_profiles_normalized, standard_depths, "Standardized TTS Profiles")

# --- Model setup and training ---
EPOCHS = 500
BATCH_SIZE = 32  # Increased batch size
LEARNING_RATE = 1e-4  # Increased learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

model, train_losses, test_losses = train(
    model, optimizer, scheduler, train_loader, test_loader, EPOCHS, str(device)
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
reconstruction_loss = evaluate_model(model, vae_loss_function, test_loader, device)
logging.info(f"Test set reconstruction loss: {reconstruction_loss:.4f}")

# --- Generation and visualization ---
model.eval()
NUM_NEW_PROFILES = X_test_tensor.shape[0]  # Generate as many as in test set
with torch.no_grad():
    z = torch.randn(NUM_NEW_PROFILES, LATENT_DIM).to(device)
    # VAE outputs normalized TT profiles
    generated_tts_normalized = model.decoder(z).cpu().numpy()  # type: ignore

# Denormalize and convert to Vs
generated_tts_denorm = np.expm1(generated_tts_normalized)
dz = standard_depths[1] - standard_depths[0]
generated_vs_profiles = tts_to_Vs(generated_tts_denorm, dz)

# Also convert real test data to Vs for comparison
real_tts_normalized = X_test_tensor.cpu().numpy()
real_tts_denorm = np.expm1(real_tts_normalized)
real_vs_profiles = tts_to_Vs(real_tts_denorm, dz)


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
evaluate_generation(real_vs_profiles, generated_vs_profiles, standard_depths)

logging.info("VAE script finished.")
