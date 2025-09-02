from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data
file_path = Path(__file__).parent.parent / "data" / "vae_soil_profiles.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()

# Preprocess the data
df = df.dropna()
df = df[df["profile_id"].notnull()]
df["profile_id"] = df["profile_id"].astype(int)

# Print the shape of the DataFrame
print("DataFrame shape:", df.shape)
