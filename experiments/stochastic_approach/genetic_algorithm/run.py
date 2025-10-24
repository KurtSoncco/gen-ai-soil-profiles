import os
from pathlib import Path

from genetic_algorithm import GeneticAlgorithm
from plot_utils import plot_best_profile
from read_data import load_or_generate_data

from soilgen_ai.logging_config import setup_logging

logger = setup_logging()

# Define paths
CWD = Path(__file__).cwd()
DATA_PATH = CWD / "data" / "vspdb_tts_profiles.parquet"
CACHE_PATH = CWD / "data" / "ga_cache.npz"


GA_CONFIG = {
    "seed": 42,  # Random seed for reproducibility
    "pop_size": 5000,  # Population size
    "generations": 200,  # Number of generations to run
    "max_layers": 30,  # Maximum number of layers in the profile
    "min_layer_thickness": 1.0,  # in meters
    "max_allowable_depth": 2000.0,  # in meters, cached to avoid unphysical solutions
    "num_elites": 10,  # Number of best individuals to keep
    "tournament_size": 200,  # Number of individuals in each tournament
    "mutation_rate": 0.5,  # Probability of mutating a gene
    "mutation_sigma": 0.5,  # Std dev of Gaussian noise for mutation
    "fitness_weight_shape": 0.9,  # Weight for shape penalty
    "fitness_weight_dist": 0.9,  # Weight for data misfit
    "fitness_weight_velocity": 0.9,  # New weight for velocity penalty
    "fitness_weight_thickness": 0.9,  # New weight for thickness penalty
    "cache_path": CACHE_PATH,  # Path to cache preprocessed data
}

# Ensure the data directory exists
os.makedirs(DATA_PATH.parent, exist_ok=True)

# Load data (or generate if not found)
real_data = load_or_generate_data(DATA_PATH)

# Initialize and run the algorithm
ga = GeneticAlgorithm(config=GA_CONFIG, real_data_dict=real_data)
best_profile = ga.run()

# Plot the best profile
plot_best_profile(best_profile)
