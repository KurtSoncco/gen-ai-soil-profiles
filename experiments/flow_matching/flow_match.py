import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
class Config:
    PROFILE_POINTS = 128  # Number of points in each Vs profile (after interpolation)
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100 # Increase for a real dataset
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_MOCK_PROFILES = 1000 # Number of mock profiles to generate
    N_SAMPLE_STEPS = 100 # Number of steps for ODE solver during sampling

torch.manual_seed(42)
np.random.seed(42)

# --- 1. Mock Data Generation ---
# This function creates a single mock soil profile.
# You will replace this with your real data loader.
def generate_mock_profile(points=Config.PROFILE_POINTS):
    """
    Generates a mock Vs profile that generally increases with depth.
    """
    depth = np.linspace(0, 1, points)
    
    # Base trend: Vs increases with depth (e.g., a power law or quadratic)
    base_trend = 150 + 400 * (depth ** 0.8)
    
    # Add some "layers" (sharp changes)
    num_layers = np.random.randint(1, 4)
    layers = np.zeros(points)
    for _ in range(num_layers):
        pos = np.random.randint(int(points * 0.2), int(points * 0.9))
        magnitude = np.random.randn() * 80
        width = int(points * 0.1)
        layers[pos:pos+width] += magnitude
    
    # Add noise
    noise = np.random.randn(points) * 15
    
    profile = base_trend + layers + noise
    
    # Ensure profile is non-negative
    profile = np.maximum(profile, 50)
    
    return profile

class SoilProfileDataset(Dataset):
    """
    A PyTorch Dataset to hold the soil profiles.
    This is where you'll load your 2000 real profiles.
    """
    def __init__(self, num_samples=Config.N_MOCK_PROFILES, points=Config.PROFILE_POINTS):
        self.points = points
        print(f"Generating {num_samples} mock soil profiles...")
        # Generate mock data
        self.profiles = [generate_mock_profile(points) for _ in range(num_samples)]
        
        # --- Normalization ---
        # Normalize data to [0, 1] range for stable training
        all_data = np.stack(self.profiles)
        self.min_val = np.min(all_data)
        self.max_val = np.max(all_data)
        self.profiles_normalized = (all_data - self.min_val) / (self.max_val - self.min_val)
        print(f"Data normalized (min: {self.min_val:.2f}, max: {self.max_val:.2f})")

    def __len__(self):
        return len(self.profiles_normalized)

    def __getitem__(self, idx):
        # Return as (Channels, Length) tensor
        profile = self.profiles_normalized[idx]
        return torch.tensor(profile, dtype=torch.float32).unsqueeze(0)
    
    def denormalize(self, profiles_tensor):
        """Helper to convert normalized [0, 1] data back to physical Vs values"""
        profiles = profiles_tensor.cpu().numpy()
        return (profiles * (self.max_val - self.min_val)) + self.min_val

# --- 2. Model (1D UNet) ---
# This will be our vector field v_theta(u, t)

class SinusoidalTimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """Standard Conv1d block: Conv -> GroupNorm -> SiLU"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.silu1 = nn.SiLU()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.silu2 = nn.SiLU()

    def forward(self, x, t):
        # First conv
        h = self.silu1(self.norm1(self.conv1(x)))
        
        # Add time embedding
        time_emb = self.silu1(self.time_mlp(t))
        # (Batch, Emb) -> (Batch, Emb, 1) to broadcast
        h = h + time_emb.unsqueeze(-1) 
        
        # Second conv
        h = self.silu2(self.norm2(self.conv2(h)))
        return h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, t):
        h = self.conv(x, t)
        p = self.pool(h)
        return h, p # Return skip connection

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, time_emb_dim) # For skip connection

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # Concatenate skip connection
        x = self.conv(x, t)
        return x

class UNet1D(nn.Module):
    """
    A 1D UNet for modeling the vector field v(u, t).
    Input:
        x (u_t): (Batch, 1, PROFILE_POINTS)
        t: (Batch, 1)
    Output:
        v_pred: (Batch, 1, PROFILE_POINTS)
    """
    def __init__(self, dim=64, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # UNet Blocks
        self.inc = ConvBlock(1, dim, time_emb_dim)
        self.down1 = DownBlock(dim, dim * 2, time_emb_dim)
        self.down2 = DownBlock(dim * 2, dim * 4, time_emb_dim)
        
        self.bot = ConvBlock(dim * 4, dim * 8, time_emb_dim)
        
        self.up1 = UpBlock(dim * 8, dim * 4, time_emb_dim)
        self.up2 = UpBlock(dim * 4, dim * 2, time_emb_dim)
        self.outc = nn.Conv1d(dim * 2, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.squeeze(-1))
        
        x1 = self.inc(x, t_emb)
        s1, x2 = self.down1(x1, t_emb)
        s2, x3 = self.down2(x2, t_emb)
        
        x_bot = self.bot(x3, t_emb)
        
        x_up = self.up1(x_bot, s2, t_emb)
        x_up = self.up2(x_up, s1, t_emb)
        
        output = self.outc(x_up)
        return output

# --- 3. FFM Training Loop ---
def train_ffm(model, dataloader, config):
    print(f"Starting training on {config.DEVICE}...")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        for batch in dataloader:
            # u1: Real data profile (t=1)
            u1 = batch.to(config.DEVICE)
            
            # u0: Noise profile (t=0)
            u0 = torch.randn_like(u1).to(config.DEVICE)
            
            # t: Random time from [0, 1]
            t = torch.rand(u1.shape[0], 1).to(config.DEVICE)
            
            # Broadcast t for interpolation
            # (Batch, 1) -> (Batch, 1, 1)
            t_broadcast = t.view(-1, 1, 1).expand(-1, 1, config.PROFILE_POINTS)
            
            # ut: Interpolated profile at time t
            # ut = (1-t)*u0 + t*u1
            ut = (1 - t_broadcast) * u0 + t_broadcast * u1
            
            # target_v: The target vector field (u1 - u0)
            target_v = u1 - u0
            
            # --- Forward pass ---
            predicted_v = model(ut, t)
            
            # --- Loss calculation ---
            loss = loss_fn(predicted_v, target_v)
            
            # --- Backward pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss:.6f}")
            
    print("Training complete.")
    return model, losses

# --- 4. Inference (Sampling) ---
@torch.no_grad()
def sample(model, num_samples=4, steps=Config.N_SAMPLE_STEPS):
    """
    Generate new profiles by solving the ODE from t=0 to t=1
    using the forward Euler method.
    """
    model.eval()
    
    # u: Start with pure noise (u_0)
    u = torch.randn((num_samples, 1, Config.PROFILE_POINTS)).to(Config.DEVICE)
    dt = 1.0 / steps
    
    trajectory = [u.cpu().numpy()] # Store trajectory for visualization

    for i in range(steps):
        # t: Current time step
        t_val = i * dt
        t = torch.full((num_samples, 1), t_val).to(Config.DEVICE)
        
        # v_pred = v_theta(u_t, t)
        v_pred = model(u, t)
        
        # Euler step: u_{t+dt} = u_t + v_pred * dt
        u = u + v_pred * dt
        
        if i % (steps // 10) == 0:
            trajectory.append(u.cpu().numpy())
            
    trajectory.append(u.cpu().numpy()) # Final sample
    return u, np.stack(trajectory)

# --- 5. Main Execution & Visualization ---
if __name__ == "__main__":
    config = Config()
    
    # 1. Load Data
    dataset = SoilProfileDataset()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 2. Init Model
    model = UNet1D().to(config.DEVICE)
    
    # 3. Train Model
    model, losses = train_ffm(model, dataloader, config)
    
    # 4. Generate Samples
    print("Generating new samples...")
    generated_samples_normalized, trajectory = sample(model, num_samples=4)
    
    # Denormalize for plotting
    generated_samples = dataset.denormalize(generated_samples_normalized)
    
    # Get some "real" data for comparison
    real_samples_normalized = torch.stack([dataset[i] for i in range(4)])
    real_samples = dataset.denormalize(real_samples_normalized)
    
    # 5. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(20, 16))
    depth_axis = np.linspace(0, 100, config.PROFILE_POINTS) # Mock depth in meters
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Real (Mock) Profiles
    plt.subplot(2, 2, 2)
    for i in range(4):
        plt.plot(real_samples[i, 0, :], depth_axis, label=f"Real Sample {i+1}")
    plt.gca().invert_yaxis() # Put 0m at the top
    plt.title("Real (Mock) Soil Profiles")
    plt.xlabel("Vs (m/s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Generated Profiles
    plt.subplot(2, 2, 3)
    for i in range(4):
        plt.plot(generated_samples[i, 0, :], depth_axis, label=f"Generated Sample {i+1}")
    plt.gca().invert_yaxis()
    plt.title("Generated Soil Profiles")
    plt.xlabel("Vs (m/s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Trajectory (Evolution from Noise)
    plt.subplot(2, 2, 4)
    # trajectory shape: (Steps, Batch, 1, Points)
    # We plot the evolution of the first sample in the batch
    sample_trajectory = trajectory[:, 0, 0, :]
    # Denormalize trajectory
    sample_trajectory = (sample_trajectory * (dataset.max_val - dataset.min_val)) + dataset.min_val
    plt.imshow(sample_trajectory, aspect='auto', origin='lower',
               extent=(0, int(config.PROFILE_POINTS), 0, int(config.N_SAMPLE_STEPS)),
               cmap='viridis')
    plt.title("Generation Trajectory (Sample 0)")
    plt.xlabel("Profile Points")
    plt.ylabel("Sampling Step (t=0 to t=1)")
    plt.colorbar(label="Normalized Vs")

    plt.tight_layout()
    plt.show()
