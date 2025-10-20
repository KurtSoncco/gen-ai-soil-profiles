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
from .preprocessing import preprocess_data, standardize_and_create_tts_profiles
from sklearn.model_selection import train_test_split
from .training import train
from .datasets import TTSDataset
from .dae_trainer import TrainConfig, train_dae, train_vae
from .utils import (
    Vs30_calc,
    calculate_vs30,
    evaluate_generation,
    plot_stair_profiles,
    tts_to_Vs,
)
from .vae_model import VAE, evaluate_model, vae_loss_function
from .vq_vae_model import VQVAE  # vq_vae_loss_function
from .vq_vae_model import evaluate_model as evaluate_vq_model
from .conv1d_vae import Conv1DVAE, conv1d_vae_loss_function
from .simple_conv1d_vae import SimpleConv1DVAE, simple_conv1d_vae_loss_function
from .gmm_sampling import LatentGMMSampler, extract_latent_samples, generate_with_gmm, compute_layer_weights
from .gmm_sampling import vs_to_log_vs_profiles, log_vs_to_vs_profiles
from .enhanced_metrics import compute_weighted_metrics, compute_vs30_metrics, plot_comprehensive_evaluation

import wandb
from soilgen_ai.logging_config import setup_logging


def train_conv1d_vae(model, optimizer, scheduler, train_loader, test_loader, device, 
                    epochs, layer_weights, beta_end, beta_warmup_epochs, tv_weight):
    """Custom training loop for Conv1D VAE with enhanced loss."""
    from tqdm import tqdm
    
    model.to(device)
    train_losses = []
    test_losses = []
    
    def compute_beta(epoch):
        if epoch >= beta_warmup_epochs:
            return beta_end
        fraction = max(0.0, float(epoch) / max(1, beta_warmup_epochs))
        return beta_end * fraction
    
    for epoch in range(epochs):
        beta = compute_beta(epoch)
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (β={beta:.3f})")
        for batch_idx, batch_data in enumerate(pbar):
            # Handle both tuple (DAE) and single tensor (baseline) cases
            if isinstance(batch_data, tuple):
                data, _ = batch_data
            else:
                data = batch_data
            
            # Ensure data is a tensor (handle list case)
            if isinstance(data, list):
                data = torch.stack(data)
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
                
            data = data.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data)
            loss, loss_dict = conv1d_vae_loss_function(
                recon, data, mu, logvar, beta, layer_weights.to(device), tv_weight
            )
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': loss_dict['recon_loss'],
                'kld': loss_dict['kld_loss'],
                'tv': loss_dict['tv_loss']
            })
            
            # Log detailed losses to W&B
            if batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_recon': loss_dict['recon_loss'],
                    'train/batch_kld': loss_dict['kld_loss'],
                    'train/batch_tv': loss_dict['tv_loss'],
                    'train/beta': beta,
                    'epoch': epoch
                })
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in test_loader:
                # Handle both tuple (DAE) and single tensor (baseline) cases
                if isinstance(batch_data, tuple):
                    data, _ = batch_data
                else:
                    data = batch_data
                
                # Ensure data is a tensor (handle list case)
                if isinstance(data, list):
                    data = torch.stack(data)
                elif not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                    
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, _ = conv1d_vae_loss_function(
                    recon, data, mu, logvar, beta, layer_weights.to(device), tv_weight
                )
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        test_losses.append(val_loss)
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, β: {beta:.3f}")
        wandb.log({
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
            'train/beta': beta,
            'epoch': epoch + 1
        })
    
    return train_losses, test_losses


def train_simple_conv1d_vae(model, optimizer, scheduler, train_loader, test_loader, device, 
                           epochs, layer_weights, beta_end, beta_warmup_epochs, tv_weight):
    """Custom training loop for Simple Conv1D VAE with enhanced loss."""
    from tqdm import tqdm
    
    model.to(device)
    train_losses = []
    test_losses = []
    
    def compute_beta(epoch):
        if epoch >= beta_warmup_epochs:
            return beta_end
        fraction = max(0.0, float(epoch) / max(1, beta_warmup_epochs))
        return beta_end * fraction
    
    for epoch in range(epochs):
        beta = compute_beta(epoch)
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (β={beta:.3f})")
        for batch_idx, batch_data in enumerate(pbar):
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
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data)
            loss, loss_dict = simple_conv1d_vae_loss_function(
                recon, data, mu, logvar, beta, layer_weights.to(device), tv_weight
            )
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': loss_dict['recon_loss'],
                'kld': loss_dict['kld_loss'],
                'tv': loss_dict['tv_loss']
            })
            
            # Log detailed losses to W&B
            if batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_recon': loss_dict['recon_loss'],
                    'train/batch_kld': loss_dict['kld_loss'],
                    'train/batch_tv': loss_dict['tv_loss'],
                    'train/beta': beta,
                    'epoch': epoch
                })
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in test_loader:
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
                recon, mu, logvar = model(data)
                loss, _ = simple_conv1d_vae_loss_function(
                    recon, data, mu, logvar, beta, layer_weights.to(device), tv_weight
                )
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        test_losses.append(val_loss)
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, β: {beta:.3f}")
        wandb.log({
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
            'train/beta': beta,
            'epoch': epoch + 1
        })
    
    return train_losses, test_losses

# Load environment variables from .env file
load_dotenv()

# Basic configuration
setup_logging()
sns.set_palette("colorblind")

# Load the data (robust path resolution)
data_candidates = [
    Path(__file__).parent.parent.parent.parent / "data" / "vspdb_vs_profiles.parquet",
    Path(__file__).parent.parent.parent.parent / "data" / "vspdb_tts_profiles.parquet",
    Path.cwd() / "data" / "vspdb_vs_profiles.parquet",
    Path.cwd() / "data" / "vspdb_tts_profiles.parquet",
]
data_file = None
for cand in data_candidates:
    if cand.exists():
        data_file = cand
        break
if data_file is None:
    raise FileNotFoundError(str(data_candidates[0]))

table = pq.read_table(data_file)
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

# --- Configuration ---
NUM_LAYERS = 100
MAX_DEPTH = 2000
VS_MAX = 2500  # Used for visualization limits
LATENT_DIM = 32  # Reduced latent dimension
NUM_NEW_PROFILES = 1000  # Number of new profiles to generate

# Enhanced training options
USE_CONV1D = True  # Use Conv1D VAE instead of MLP
USE_LOG_VS = True  # Train in log(Vs) domain
USE_GMM_SAMPLING = True  # Use GMM for generation
USE_WEIGHTED_LOSS = True  # Use weighted loss functions
USE_DAE = False  # Set True to use DAE+VAE two-stage training

# Training hyperparameters
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MODEL = "VAE"  # Options: "VAE" or "VQ-VAE"
WEIGHT_DECAY = 1e-5
BETAS = (0.6, 0.9)
BETA_END = 0.1  # Lower beta for better generation
BETA_WARMUP_EPOCHS = 100
TV_WEIGHT = 0.01  # Total variation regularization weight

tts_profiles_normalized, standard_depths = standardize_and_create_tts_profiles(
    profiles_dict, NUM_LAYERS, MAX_DEPTH
)

# Convert to Vs profiles for log(Vs) training if enabled
if USE_LOG_VS:
    # Convert TTS to Vs profiles
    dz = np.diff(standard_depths)
    vs_profiles = tts_to_Vs(tts_profiles_normalized, dz)
    
    # Transform to log(Vs) domain
    log_vs_profiles = vs_to_log_vs_profiles(vs_profiles, standard_depths)
    training_data = log_vs_profiles
    logging.info(f"Using log(Vs) domain. Data shape: {training_data.shape}")
else:
    training_data = tts_profiles_normalized
    logging.info(f"Using TTS domain. Data shape: {training_data.shape}")

INPUT_DIM = training_data.shape[1]
logging.info(f"Final training data shape: {training_data.shape}")

# Compute layer weights for weighted loss
layer_weights = compute_layer_weights(standard_depths)

## Do some checks before training
# Extract the max tts of each profile and plot distribution
max_tts = np.max(tts_profiles_normalized, axis=1)
plt.figure(figsize=(8, 5))
sns.histplot(max_tts, bins=30, kde=True)
plt.title("Distribution of Maximum Normalized TTS in Profiles")
plt.xlabel("Maximum Normalized TTS")
plt.ylabel("Frequency")
plt.grid()
plt.close()

# Training and evaluation part
if True:
    # Split the data
    X_train_np, X_val_np = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42,
    )
    logging.info(f"Training data shape: {X_train_np.shape}")
    logging.info(f"Testing data shape: {X_val_np.shape}")

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
        plt.close()

    plot_profiles(
        tts_profiles_normalized, standard_depths, "Standardized TTS Profiles (d_tts)"
    )

    # --- Model setup and training ---
    EPOCHS = 2000
    BATCH_SIZE = 64  # Increased batch size
    LEARNING_RATE = 1e-4  # Increased learning rate
    MODEL = "VAE"  # Options: "VAE" or "VQ-VAE"
    WEIGHT_DECAY = 1e-5  # Weight decay for optimizer
    BETAS = (0.6, 0.9)  # Betas for Adam optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model selection
    if MODEL == "VAE":
        # Use MLP VAE for now - more reliable with this data format
        model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
        model_type = "MLP-VAE"
    else:
        model = VQVAE(
            input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_embeddings=512
        ).to(device)
        model_type = "VQ-VAE"
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # Data loaders
    if USE_DAE:
        train_dataset = TTSDataset(X_train_np, corruption_noise_std=0.05)
        val_dataset = TTSDataset(X_val_np, corruption_noise_std=0.05)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
    else:
        X_train_tensor = torch.from_numpy(X_train_np).float()
        X_val_tensor = torch.from_numpy(X_val_np).float()
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor)
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

    # --- W&B Initialization ---
    # Build a stable run name and ensure single init
    run_name = f"{model_type}-layers{NUM_LAYERS}-lat{LATENT_DIM}-bs{BATCH_SIZE}"
    if wandb.run is None:
        wandb.init(
            project=os.getenv("W_B_PROJECT", "soilgen-vae"),
            name=run_name,
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
            "use_conv1d": USE_CONV1D,
            "use_log_vs": USE_LOG_VS,
            "use_gmm_sampling": USE_GMM_SAMPLING,
            "use_weighted_loss": USE_WEIGHTED_LOSS,
            "use_dae": USE_DAE,
            "beta_end": BETA_END,
            "beta_warmup_epochs": BETA_WARMUP_EPOCHS,
            "tv_weight": TV_WEIGHT,
                 "run_name": run_name,
            },
            reinit=False,
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        # Update name and config if run already exists (avoid re-init)
        try:
            wandb.run.name = run_name
            wandb.config.update({"run_name": run_name}, allow_val_change=True)
        except Exception:
            pass
    wandb.watch(model, log="all")
    if USE_DAE:
        logging.info("Starting DAE pretraining then VAE fine-tuning...")
        cfg = TrainConfig(
            epochs_dae=100,
            epochs_vae=EPOCHS,
            beta_start=0.0,
            beta_end=BETA_END,
            beta_warmup_epochs=BETA_WARMUP_EPOCHS,
            grad_clip_norm=1.0,
            amp=True,
            early_stop_patience=20,
        )
        checkpoint_path = str(Path(__file__).parent / "vae_checkpoint.pt")
        train_dae(
            model,
            optimizer,
            train_loader,
            test_loader,
            device,
            cfg,
            checkpoint_path=checkpoint_path,
        )
        train_vae(
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            device,
            cfg,
            checkpoint_path=checkpoint_path,
        )
        train_losses, test_losses = None, None
    else:
        logging.info("Starting baseline VAE training...")
        
        # Use standard training for MLP VAE
        _, train_losses, test_losses = train(
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            EPOCHS,
            str(device),
            model_type=MODEL,
            early_stop_patience=20,
        )

    # Plot training losses if available (baseline path)
    if train_losses is not None and test_losses is not None:
        plt.plot(train_losses, label="Training Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses (see W&B for details)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

    plt.savefig(Path(__file__).parent / "training_losses.png", dpi=300, bbox_inches="tight")
    logging.info("Training progress plot saved as training_losses.png")
    plt.close()

    # --- Evaluation on test set ---
    if MODEL == "VQ-VAE":
        reconstruction_loss = evaluate_vq_model(model, test_loader, device)
    else:
        reconstruction_loss = evaluate_model(
            model, vae_loss_function, test_loader, device
        )
    logging.info(f"Test set reconstruction loss: {reconstruction_loss:.4f}")

    # Save final model and log as W&B artifact
    final_path = Path(__file__).parent / "vae_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dict(wandb.config),
        "reconstruction_loss": reconstruction_loss,
    }, final_path)
    artifact = wandb.Artifact("vae_model", type="model")
    artifact.add_file(str(final_path))
    wandb.log_artifact(artifact)

    # --- Generation and visualization ---
    model.eval()
    
    # Extract latent samples for GMM fitting if enabled
    gmm_sampler = None
    if USE_GMM_SAMPLING and MODEL == "VAE":
        logging.info("Fitting GMM to latent space...")
        latent_samples = extract_latent_samples(model, train_loader, device, max_samples=5000)
        gmm_sampler = LatentGMMSampler(n_components=8)
        gmm_sampler.fit(latent_samples)
        logging.info(f"GMM fitted with {gmm_sampler.n_components} components")
    
    with torch.no_grad():
        if MODEL == "VAE":
            if USE_GMM_SAMPLING and gmm_sampler is not None:
                # Use GMM sampling
                generated_data = generate_with_gmm(model, gmm_sampler, NUM_NEW_PROFILES, device, INPUT_DIM)
            else:
                # Standard VAE sampling
                z = torch.randn(NUM_NEW_PROFILES, LATENT_DIM).to(device)
                if USE_CONV1D:
                    generated_data = model.decoder(z).cpu().numpy()
                else:
                    generated_data = model.decoder(z).cpu().numpy()
        else:  # For VQ-VAE
            num_embeddings = model.vq_layer._num_embeddings
            codebook_indices = torch.randint(0, num_embeddings, (NUM_NEW_PROFILES,)).to(device)
            z_quantized = model.vq_layer._embedding(codebook_indices)
            generated_data = model.decoder(z_quantized).cpu().numpy()

    # Convert generated data back to Vs profiles
    if USE_LOG_VS:
        # Convert from log(Vs) back to Vs
        generated_vs_profiles = log_vs_to_vs_profiles(generated_data)
    else:
        # Convert from TTS to Vs
        generated_d_tts_denorm = np.expm1(generated_data)
        dz = np.diff(standard_depths)
        generated_vs_profiles = tts_to_Vs(generated_d_tts_denorm, dz)

    # Also convert real test data to Vs for comparison
    if USE_LOG_VS:
        real_vs_profiles = log_vs_to_vs_profiles(X_val_np)
    else:
        real_d_tts_denorm = np.expm1(X_val_np)
        dz = np.diff(standard_depths)
        real_vs_profiles = tts_to_Vs(real_d_tts_denorm, dz)

    # Calculate Vs30 for generated and real profiles
    generated_vs30 = [calculate_vs30(p, standard_depths) for p in generated_vs_profiles]
    real_vs30_test = [calculate_vs30(p, standard_depths) for p in real_vs_profiles]

    # --- Enhanced Evaluation ---
    logging.info("Computing enhanced metrics...")
    
    # Compute weighted metrics
    if USE_WEIGHTED_LOSS:
        weighted_metrics = compute_weighted_metrics(
            real_vs_profiles, generated_vs_profiles, standard_depths, layer_weights.numpy()
        )
        logging.info(f"Weighted MSE: {weighted_metrics['weighted_mse']:.4f}")
        logging.info(f"Weighted MAE: {weighted_metrics['weighted_mae']:.4f}")
        logging.info(f"TV Ratio: {weighted_metrics['tv_ratio']:.4f}")
        
        # Log to W&B
        wandb.log({
            "eval/weighted_mse": weighted_metrics['weighted_mse'],
            "eval/weighted_mae": weighted_metrics['weighted_mae'],
            "eval/tv_ratio": weighted_metrics['tv_ratio'],
        })
    
    # Compute Vs30 metrics
    vs30_metrics = compute_vs30_metrics(real_vs30_test, generated_vs30)
    logging.info(f"Vs30 KS statistic: {vs30_metrics['ks_statistic']:.4f}")
    logging.info(f"Vs30 mean ratio: {vs30_metrics['mean_ratio']:.4f}")
    logging.info(f"Vs30 std ratio: {vs30_metrics['std_ratio']:.4f}")
    
    # Log to W&B
    wandb.log({
        "eval/vs30_ks_statistic": vs30_metrics['ks_statistic'],
        "eval/vs30_mean_ratio": vs30_metrics['mean_ratio'],
        "eval/vs30_std_ratio": vs30_metrics['std_ratio'],
    })
    
    # Create comprehensive evaluation plots
    plot_comprehensive_evaluation(
        real_vs_profiles, generated_vs_profiles, standard_depths, vs30_metrics,
        save_path=str(Path(__file__).parent / "comprehensive_evaluation.png")
    )

    logging.info("Plotting generated Vs profiles...")
    plot_stair_profiles(
        generated_vs_profiles,
        standard_depths,
        "Generated Shear Wave Velocity Profiles",
        MAX_DEPTH,
        VS_MAX,
    )
    plt.savefig(Path(__file__).parent / "generated_profiles.png", dpi=300, bbox_inches="tight")
    logging.info("Generated profiles plot saved as generated_profiles.png")
    plt.close()

    # --- Quantitative Evaluation ---
    logging.info("Evaluating generated profiles...")
    
    evaluate_generation(real_vs_profiles, generated_vs_profiles, standard_depths)
    wandb.finish()
    logging.info("VAE script finished.")
