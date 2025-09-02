from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

# Load the data
file_path = Path(__file__).parent.parent / "data" / "vae_soil_profiles.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()
