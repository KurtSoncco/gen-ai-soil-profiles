#!/usr/bin/env python3
"""
Parametric Profile Modeling Experiment

This script implements the parametric approach for soil profile generation:
1. Fits parametric models (exponential, power law, layered) to Vs profiles
2. Trains generative models (GMM or MLP) on the fitted parameters
3. Generates new profiles by sampling parameters and reconstructing profiles

Usage:
    python run_parametric_experiment.py
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import wandb

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.VAE.src.preprocessing import preprocess_data, standardize_and_create_tts_profiles
from experiments.VAE.src.utils import tts_to_Vs, calculate_vs30, Vs30_calc
from experiments.parametric.src.parametric_models import ParametricProfileFitter
from experiments.parametric.src.generative_models import ParameterGenerator
from soilgen_ai.logging_config import setup_logging


def load_and_preprocess_data():
    """Load and preprocess the soil profile data."""
    logging.info("Loading and preprocessing data...")
    
    # Load the data (robust path resolution)
    data_candidates = [
        project_root / "data" / "vspdb_vs_profiles.parquet",
        project_root / "data" / "vspdb_tts_profiles.parquet",
        Path.cwd() / "data" / "vspdb_vs_profiles.parquet",
        Path.cwd() / "data" / "vspdb_tts_profiles.parquet",
    ]
    
    data_file = None
    for cand in data_candidates:
        if cand.exists():
            data_file = cand
            break
    
    if data_file is None:
        raise FileNotFoundError(f"Data file not found. Tried: {data_candidates}")
    
    # Load data
    table = pq.read_table(data_file)
    df = table.to_pandas()
    logging.info(f"Data loaded with shape: {df.shape}")
    
    # Convert df into profiles dict
    profiles_dict = {
        metadata_id: group.sort_values("depth")
        .drop(columns=["velocity_metadata_id"])
        .reset_index(drop=True)
        for metadata_id, group in df.groupby("velocity_metadata_id")
    }
    
    # Preprocess the data
    profiles_dict, dropped_ids = preprocess_data(profiles_dict)
    logging.info(f"Number of profiles after preprocessing: {len(profiles_dict)}")
    logging.info(f"Number of profiles dropped: {len(dropped_ids)}")
    
    return profiles_dict


def convert_to_vs_profiles(profiles_dict, num_layers=100, max_depth=2000):
    """Convert profiles to standardized Vs profiles."""
    logging.info("Converting profiles to standardized Vs format...")
    
    # Create standardized TTS profiles first
    tts_profiles_normalized, standard_depths = standardize_and_create_tts_profiles(
        profiles_dict, num_layers, max_depth
    )
    
    # Convert TTS to Vs profiles
    dz = np.diff(standard_depths)
    vs_profiles = tts_to_Vs(tts_profiles_normalized, dz)
    
    logging.info(f"Converted to {vs_profiles.shape[0]} Vs profiles with {vs_profiles.shape[1]} layers")
    
    return vs_profiles, standard_depths


def fit_parametric_models(vs_profiles, standard_depths, model_types=['exponential', 'power_law', 'layered']):
    """Fit parametric models to Vs profiles."""
    logging.info("Fitting parametric models to Vs profiles...")
    
    fitted_parameters = {}
    fitters = {}
    
    for model_type in model_types:
        logging.info(f"Fitting {model_type} model...")
        
        # Create fitter
        fitter = ParametricProfileFitter(standard_depths, model_type)
        
        # Fit parameters
        parameters = fitter.fit_profiles(vs_profiles)
        
        # Store results
        fitted_parameters[model_type] = parameters
        fitters[model_type] = fitter
        
        logging.info(f"{model_type} model fitted. Parameter shape: {parameters.shape}")
        logging.info(f"Parameter names: {fitter.get_parameter_names()}")
        
        # Log parameter statistics
        param_names = fitter.get_parameter_names()
        for i, name in enumerate(param_names):
            param_values = parameters[:, i]
            logging.info(f"  {name}: mean={np.mean(param_values):.2f}, "
                        f"std={np.std(param_values):.2f}, "
                        f"min={np.min(param_values):.2f}, "
                        f"max={np.max(param_values):.2f}")
    
    return fitted_parameters, fitters


def train_generative_models(fitted_parameters, generative_model_types=['gmm', 'mlp']):
    """Train generative models on fitted parameters."""
    logging.info("Training generative models...")
    
    generators = {}
    
    for model_type in fitted_parameters.keys():
        logging.info(f"Training generative models for {model_type} parameters...")
        
        generators[model_type] = {}
        
        for gen_type in generative_model_types:
            logging.info(f"  Training {gen_type} generator...")
            
            # Create generator
            if gen_type == 'gmm':
                generator = ParameterGenerator(model_type='gmm', n_components=8)
            elif gen_type == 'mlp':
                generator = ParameterGenerator(
                    model_type='mlp',
                    input_dim=fitted_parameters[model_type].shape[1],
                    latent_dim=16,
                    hidden_dims=[64, 32]
                )
            
            # Train generator
            generator.fit(fitted_parameters[model_type], epochs=100, learning_rate=1e-3)
            
            # Store generator
            generators[model_type][gen_type] = generator
            
            logging.info(f"  {gen_type} generator trained successfully")
    
    return generators


def evaluate_parametric_models(fitters, fitted_parameters, vs_profiles, standard_depths):
    """Evaluate the quality of parametric model fits."""
    logging.info("Evaluating parametric model fits...")
    
    evaluation_results = {}
    
    for model_type, fitter in fitters.items():
        logging.info(f"Evaluating {model_type} model...")
        
        # Reconstruct profiles from fitted parameters
        reconstructed_profiles = fitter.generate_profiles(fitted_parameters[model_type])
        
        # Calculate reconstruction errors
        mse = np.mean((vs_profiles - reconstructed_profiles) ** 2)
        mae = np.mean(np.abs(vs_profiles - reconstructed_profiles))
        
        # Calculate Vs30 errors
        real_vs30 = [calculate_vs30(p, standard_depths) for p in vs_profiles]
        recon_vs30 = [calculate_vs30(p, standard_depths) for p in reconstructed_profiles]
        
        vs30_mse = np.mean((np.array(real_vs30) - np.array(recon_vs30)) ** 2)
        vs30_mae = np.mean(np.abs(np.array(real_vs30) - np.array(recon_vs30)))
        
        evaluation_results[model_type] = {
            'mse': mse,
            'mae': mae,
            'vs30_mse': vs30_mse,
            'vs30_mae': vs30_mae,
            'reconstructed_profiles': reconstructed_profiles
        }
        
        logging.info(f"  {model_type} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        logging.info(f"  {model_type} - Vs30 MSE: {vs30_mse:.4f}, Vs30 MAE: {vs30_mae:.4f}")
    
    return evaluation_results


def generate_new_profiles(generators, fitters, n_new_profiles=1000):
    """Generate new profiles using the trained generative models."""
    logging.info(f"Generating {n_new_profiles} new profiles...")
    
    generated_profiles = {}
    
    for model_type, model_generators in generators.items():
        logging.info(f"Generating profiles with {model_type} model...")
        
        generated_profiles[model_type] = {}
        
        for gen_type, generator in model_generators.items():
            logging.info(f"  Using {gen_type} generator...")
            
            # Generate parameters
            generated_params = generator.generate(n_new_profiles)
            
            # Generate profiles from parameters
            generated_vs_profiles = fitters[model_type].generate_profiles(generated_params)
            
            generated_profiles[model_type][gen_type] = {
                'parameters': generated_params,
                'profiles': generated_vs_profiles
            }
            
            logging.info(f"  Generated {generated_vs_profiles.shape[0]} profiles")
    
    return generated_profiles


def plot_results(vs_profiles, standard_depths, evaluation_results, generated_profiles, fitters, save_dir):
    """Create visualization plots."""
    logging.info("Creating visualization plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("colorblind")
    
    # 1. Original vs Reconstructed profiles comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (model_type, results) in enumerate(evaluation_results.items()):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Plot original profiles
        for j in range(min(10, vs_profiles.shape[0])):
            ax.plot(vs_profiles[j], standard_depths[1:], 'b-', alpha=0.3, linewidth=0.5)
        
        # Plot reconstructed profiles
        recon_profiles = results['reconstructed_profiles']
        for j in range(min(10, recon_profiles.shape[0])):
            ax.plot(recon_profiles[j], standard_depths[1:], 'r-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'{model_type.title()} Model\nMSE: {results["mse"]:.4f}')
        ax.set_xlabel('Vs (m/s)')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(max(standard_depths), 0)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'parametric_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Generated profiles
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for model_type, model_generators in generated_profiles.items():
        for gen_type, results in model_generators.items():
            if plot_idx >= 4:
                break
                
            ax = axes[plot_idx]
            
            # Plot generated profiles
            profiles = results['profiles']
            for j in range(min(20, profiles.shape[0])):
                ax.plot(profiles[j], standard_depths[1:], alpha=0.3, linewidth=0.5)
            
            ax.set_title(f'{model_type.title()} + {gen_type.upper()}\nGenerated Profiles')
            ax.set_xlabel('Vs (m/s)')
            ax.set_ylabel('Depth (m)')
            ax.set_ylim(max(standard_depths), 0)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_dir / 'generated_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parameter distributions
    for model_type, model_generators in generated_profiles.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for gen_type, results in model_generators.items():
            if plot_idx >= 2:
                break
                
            ax = axes[plot_idx]
            
            # Plot parameter distributions
            parameters = results['parameters']
            param_names = fitters[model_type].get_parameter_names()
            
            for i, name in enumerate(param_names):
                ax.hist(parameters[:, i], bins=30, alpha=0.7, label=name)
            
            ax.set_title(f'{model_type.title()} + {gen_type.upper()} Parameters')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{model_type}_parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info("Visualization plots saved")


def main():
    """Main experiment function."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logging.info("Starting Parametric Profile Modeling Experiment")
    
    # Configuration
    NUM_LAYERS = 100
    MAX_DEPTH = 2000
    N_NEW_PROFILES = 1000
    MODEL_TYPES = ['exponential', 'power_law', 'layered']
    GENERATIVE_MODEL_TYPES = ['gmm', 'mlp']
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize W&B
    if wandb.run is None:
        wandb.init(
            project=os.getenv("W_B_PROJECT", "soilgen-parametric"),
            name="parametric-profile-modeling",
            config={
                "num_layers": NUM_LAYERS,
                "max_depth": MAX_DEPTH,
                "n_new_profiles": N_NEW_PROFILES,
                "model_types": MODEL_TYPES,
                "generative_model_types": GENERATIVE_MODEL_TYPES,
            }
        )
    
    try:
        # 1. Load and preprocess data
        profiles_dict = load_and_preprocess_data()
        
        # 2. Convert to standardized Vs profiles
        vs_profiles, standard_depths = convert_to_vs_profiles(profiles_dict, NUM_LAYERS, MAX_DEPTH)
        
        # 3. Fit parametric models
        fitted_parameters, fitters = fit_parametric_models(vs_profiles, standard_depths, MODEL_TYPES)
        
        # 4. Evaluate parametric models
        evaluation_results = evaluate_parametric_models(fitters, fitted_parameters, vs_profiles, standard_depths)
        
        # 5. Train generative models
        generators = train_generative_models(fitted_parameters, GENERATIVE_MODEL_TYPES)
        
        # 6. Generate new profiles
        generated_profiles = generate_new_profiles(generators, fitters, N_NEW_PROFILES)
        
        # 7. Create visualizations
        plot_results(vs_profiles, standard_depths, evaluation_results, generated_profiles, fitters, output_dir)
        
        # 8. Log results to W&B
        for model_type, results in evaluation_results.items():
            wandb.log({
                f"eval/{model_type}_mse": results['mse'],
                f"eval/{model_type}_mae": results['mae'],
                f"eval/{model_type}_vs30_mse": results['vs30_mse'],
                f"eval/{model_type}_vs30_mae": results['vs30_mae'],
            })
        
        # 9. Save models
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_type, model_generators in generators.items():
            for gen_type, generator in model_generators.items():
                model_path = models_dir / f"{model_type}_{gen_type}_generator.pkl"
                generator.save(str(model_path))
        
        logging.info("Parametric Profile Modeling Experiment completed successfully!")
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
