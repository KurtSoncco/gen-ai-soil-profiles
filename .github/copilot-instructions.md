# Gen-AI Soil Profiles Project Guide for AI Agents

This document provides essential guidance for AI agents working on the `gen-ai-soil-profiles` repository.

## 1. Project Overview & Architecture

This project uses generative AI to create and analyze spatially-correlated soil profiles. The goal is to generate realistic synthetic soil data for geotechnical analysis.

The project is organized into several key directories:
- **`soilgen_ai/`**: The core Python package containing base utilities and data processing logic.
- **`data/`**: Stores the primary datasets in Parquet format (e.g., `vspdb_data.parquet`). These are the inputs for the models.
- **`experiments/`**: Contains the different generative modeling approaches. This is where most of the active research and development happens.
  - **`experiments/VAE/`**: Implements a Variational Autoencoder. Key files are `vae_model.py` (the model architecture), `training.py` (the training loop), and `preprocessing.py`.
  - **`experiments/stochastic_approach/`**: Implements other generative methods like a genetic algorithm.
- **`notebooks/`**: Jupyter notebooks for Exploratory Data Analysis (EDA) and experimental work. See `notebooks/VSPDB/EDA.ipynb` for an example of how the raw data is explored.
- **`tests/`**: Unit tests for the project. The structure mirrors the main packages.

## 2. Development Workflow

### Environment and Dependencies
The project uses `uv` for managing the Python environment and dependencies, which are defined in `pyproject.toml`.

**To set up the development environment:**
1. Create the virtual environment: `uv venv`
2. Activate it: `source .venv/bin/activate`
3. Install all dependencies (including dev): `uv sync --extra dev`

### Running Tests
The project uses `pytest` for testing. Tests are located in the `tests/` directory.

**To run the full test suite:**
```bash
pytest
```
When adding new features, please include corresponding tests. For example, a new calculation in `soilgen_ai/vs_profiles/vs_calculation.py` should have tests in `tests/vs_profiles/test_vs_calculation.py`.

## 3. Key Patterns & Conventions

### Data Handling
- Data is primarily handled using `pandas` DataFrames and stored in Parquet files.
- Data loading and preprocessing for the VAE model can be found in `experiments/VAE/preprocessing.py`.

### Machine Learning Models
- The main deep learning framework is `torch`.
- The VAE model implementation is in `experiments/VAE/vae_model.py`.
- Experiment tracking is done using `wandb`. The training script in `experiments/VAE/training.py` shows how `wandb` is initialized and used to log metrics.

### Code Style
- The project uses `ruff` for linting. Ensure your contributions are compliant.
- Follow standard Python conventions (PEP 8).
