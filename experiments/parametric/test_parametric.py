#!/usr/bin/env python3
"""
Test script for parametric profile modeling components.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from parametric_models import ParametricProfileFitter

# Try to import generative models (requires PyTorch)
try:
    from generative_models import ParameterGenerator
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available, skipping generative model tests")
    HAS_TORCH = False


def test_parametric_models():
    """Test the parametric model implementations."""
    print("Testing parametric models...")
    
    # Create test data
    depths = np.linspace(0, 2000, 101)
    n_profiles = 50
    n_layers = 100
    
    # Create synthetic Vs profiles with exponential-like behavior
    vs_profiles = np.zeros((n_profiles, n_layers))
    layer_centers = (depths[:-1] + depths[1:]) / 2
    
    for i in range(n_profiles):
        vs_shallow = np.random.uniform(200, 400)
        vs_deep = np.random.uniform(1500, 2500)
        z_transition = np.random.uniform(100, 500)
        
        vs_profiles[i] = vs_shallow + (vs_deep - vs_shallow) * (1 - np.exp(-layer_centers / z_transition))
    
    # Test each parametric model
    model_types = ['exponential', 'power_law', 'layered']
    
    for model_type in model_types:
        print(f"  Testing {model_type} model...")
        
        # Create fitter
        fitter = ParametricProfileFitter(depths, model_type)
        
        # Fit parameters
        fitted_params = fitter.fit_profiles(vs_profiles)
        print(f"    Fitted parameters shape: {fitted_params.shape}")
        
        # Generate profiles from parameters
        generated_profiles = fitter.generate_profiles(fitted_params)
        print(f"    Generated profiles shape: {generated_profiles.shape}")
        
        # Calculate reconstruction error
        mse = np.mean((vs_profiles - generated_profiles) ** 2)
        print(f"    Reconstruction MSE: {mse:.4f}")
    
    print("Parametric models test completed!")


def test_generative_models():
    """Test the generative model implementations."""
    if not HAS_TORCH:
        print("Skipping generative models test (PyTorch not available)")
        return
        
    print("Testing generative models...")
    
    # Create synthetic parameter data
    n_samples = 200
    n_params = 3  # For exponential model
    
    # Create parameters with some structure
    parameters = np.random.multivariate_normal(
        mean=[300, 2000, 200],
        cov=[[100, 50, 20], [50, 400, 30], [20, 30, 25]],
        size=n_samples
    )
    
    # Ensure parameters are within reasonable bounds
    parameters[:, 0] = np.clip(parameters[:, 0], 100, 1000)  # vs_shallow
    parameters[:, 1] = np.clip(parameters[:, 1], 500, 2500)  # vs_deep
    parameters[:, 2] = np.clip(parameters[:, 2], 50, 1000)   # z_transition
    
    # Test GMM
    print("  Testing GMM...")
    gmm_generator = ParameterGenerator(model_type='gmm', n_components=4)
    gmm_generator.fit(parameters)
    
    generated_gmm = gmm_generator.generate(100)
    print(f"    Generated GMM parameters shape: {generated_gmm.shape}")
    
    # Test MLP VAE
    print("  Testing MLP VAE...")
    mlp_generator = ParameterGenerator(
        model_type='mlp',
        input_dim=n_params,
        latent_dim=8,
        hidden_dims=[32, 16]
    )
    
    mlp_generator.fit(parameters, epochs=20, learning_rate=1e-3)
    
    generated_mlp = mlp_generator.generate(100)
    print(f"    Generated MLP parameters shape: {generated_mlp.shape}")
    
    print("Generative models test completed!")


def test_integration():
    """Test the integration of parametric and generative models."""
    if not HAS_TORCH:
        print("Skipping integration test (PyTorch not available)")
        return
        
    print("Testing integration...")
    
    # Create test data
    depths = np.linspace(0, 2000, 101)
    n_profiles = 100
    n_layers = 100
    
    # Create synthetic Vs profiles
    vs_profiles = np.zeros((n_profiles, n_layers))
    layer_centers = (depths[:-1] + depths[1:]) / 2
    
    for i in range(n_profiles):
        vs_shallow = np.random.uniform(200, 400)
        vs_deep = np.random.uniform(1500, 2500)
        z_transition = np.random.uniform(100, 500)
        
        vs_profiles[i] = vs_shallow + (vs_deep - vs_shallow) * (1 - np.exp(-layer_centers / z_transition))
    
    # Fit parametric model
    fitter = ParametricProfileFitter(depths, 'exponential')
    fitted_params = fitter.fit_profiles(vs_profiles)
    
    # Train generative model
    generator = ParameterGenerator(model_type='gmm', n_components=6)
    generator.fit(fitted_params)
    
    # Generate new parameters
    new_params = generator.generate(50)
    
    # Generate new profiles
    new_profiles = fitter.generate_profiles(new_params)
    
    print(f"  Original profiles shape: {vs_profiles.shape}")
    print(f"  Fitted parameters shape: {fitted_params.shape}")
    print(f"  Generated parameters shape: {new_params.shape}")
    print(f"  Generated profiles shape: {new_profiles.shape}")
    
    # Create a simple visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original profiles
    plt.subplot(2, 2, 1)
    for i in range(min(10, vs_profiles.shape[0])):
        plt.plot(vs_profiles[i], layer_centers, 'b-', alpha=0.3)
    plt.title('Original Profiles')
    plt.xlabel('Vs (m/s)')
    plt.ylabel('Depth (m)')
    plt.ylim(max(depths), 0)
    
    # Plot generated profiles
    plt.subplot(2, 2, 2)
    for i in range(min(10, new_profiles.shape[0])):
        plt.plot(new_profiles[i], layer_centers, 'r-', alpha=0.3)
    plt.title('Generated Profiles')
    plt.xlabel('Vs (m/s)')
    plt.ylabel('Depth (m)')
    plt.ylim(max(depths), 0)
    
    # Plot parameter distributions
    plt.subplot(2, 2, 3)
    plt.hist(fitted_params[:, 0], bins=20, alpha=0.7, label='Fitted', density=True)
    plt.hist(new_params[:, 0], bins=20, alpha=0.7, label='Generated', density=True)
    plt.title('Vs_shallow Distribution')
    plt.xlabel('Vs_shallow (m/s)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.hist(fitted_params[:, 2], bins=20, alpha=0.7, label='Fitted', density=True)
    plt.hist(new_params[:, 2], bins=20, alpha=0.7, label='Generated', density=True)
    plt.title('z_transition Distribution')
    plt.xlabel('z_transition (m)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_integration.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Integration test completed! Plot saved as 'test_integration.png'")


def main():
    """Run all tests."""
    print("Running parametric profile modeling tests...")
    print("=" * 50)
    
    try:
        test_parametric_models()
        print()
        
        test_generative_models()
        print()
        
        test_integration()
        print()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
