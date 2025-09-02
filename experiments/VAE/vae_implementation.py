from pathlib import Path

# import matplotlib.pyplot as plt
import pyarrow.parquet as pq

# Load the data
file_path = Path(__file__).parent.parent.parent / "data" / "vspdb_data.parquet"
table = pq.read_table(file_path)
df = table.to_pandas()

# Preprocess the data
df = df.dropna()
print("DataFrame size: ", df.shape)
