import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.optim as optim
from preprocessing import preprocessing
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from vae_model import VAE

# Basic configuration
logging.basicConfig(level=logging.INFO)
sns.set_palette("colorblind")

# Load the data
file_path = Path(__file__).parent.parent.parent / "data" / "vspdb_vs_profiles.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()
logging.info(f"Data shape before preprocessing: {df.shape}")

# Preprocess the data
df = df.dropna(subset=["vs_value", "depth"])
logging.info(f"Data shape after dropping NA values: {df.shape}")

# Filter profiles based on Vs and depth constraints
df = preprocessing(df)
logging.info(f"Data shape after filtering: {df.shape}")

# Group data by profile ID
profiles_dict = {
    metadata_id: group.sort_values("depth")
    for metadata_id, group in df.groupby("velocity_metadata_id")
}
logging.info(f"Created {len(profiles_dict)} profiles.")

# --- Key Improvement: Standardize profiles to a fixed size ---
NUM_LAYERS = 100  # Standardize all profiles to 100 layers
MAX_DEPTH = 2000  # The maximum depth for standardization
VS_MAX = 2500  # The maximum Vs value for normalization
LATENT_DIM = 10  # Latent dimension for the VAE

# Standardize and collect the profiles
vs_profiles_list = []
standard_depths = np.linspace(0, MAX_DEPTH, NUM_LAYERS)

for profile_id, profile_data in profiles_dict.items():
    depths = profile_data["depth"].values
    vs_values = profile_data["vs_value"].values

    # Check for monotonic increase in depth (essential for interpolation)
    if not np.all(np.diff(depths) > 0):
        continue

    # Use linear interpolation to resample the profile to the standard depths
    f_interp = interp1d(
        depths,
        vs_values,
        kind="linear",
        fill_value=(vs_values[0], vs_values[-1]),  # type: ignore
        bounds_error=False,
    )
    resampled_vs = f_interp(standard_depths)

    # Ensure the profile is monotonically increasing after resampling
    resampled_vs = np.maximum.accumulate(resampled_vs)

    vs_profiles_list.append(resampled_vs)

# Convert to a single NumPy array
X_train_vs = np.array(vs_profiles_list)

# Normalize the data
X_train_normalized = X_train_vs / VS_MAX
INPUT_DIM = X_train_normalized.shape[1]

# Split the data into training and testing sets
X_train_tensor, X_test_tensor = train_test_split(
    torch.from_numpy(X_train_normalized).float(), test_size=0.2, random_state=42
)
logging.info(f"Final training data shape: {X_train_tensor.shape}")


# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Data Loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
