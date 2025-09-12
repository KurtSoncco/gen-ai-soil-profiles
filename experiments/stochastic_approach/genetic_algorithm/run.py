import os
from pathlib import Path

from genetic_algorithm import GeneticAlgorithm
from plot_utils import plot_best_profile
from read_data import load_or_generate_data

from soilgen_ai.logging_config import setup_logging

logger = setup_logging()

# Load real data for initialization guidance
DATA_PATH = Path(__file__).cwd() / "data" / "vspdb_tts_profiles.parquet"


GA_CONFIG = {
    "seed": 42,  # Random seed for reproducibility
    "pop_size": 5000,  # Population size
    "generations": 200,  # Number of generations to run
    "max_layers": 25,  # Maximum number of layers in the profile
    "max_allowable_depth": 2000.0,  # in meters
    "num_elites": 2,  # Number of best individuals to keep
    "tournament_size": 10,  # Number of individuals in each tournament
    "mutation_rate": 0.15,  # Probability of mutating a gene
    "mutation_sigma": 0.05,  # Std dev of Gaussian noise for mutation
    "fitness_weight_shape": 0.6,
    "fitness_weight_dist": 0.4,
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
