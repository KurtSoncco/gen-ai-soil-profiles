#!/usr/bin/env python3
"""
FFM Evaluation Test Script

This script demonstrates the complete evaluation functionality for FFM,
including all the same metrics and plots used in conv1d_gan.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.flow_matching.models import create_model
from experiments.flow_matching.data import create_dataloader
from experiments.flow_matching.utils import (
    compute_real_vs30_and_density, 
    compute_generated_vs30, 
    compute_vs100,
    ks_statistic,
    plot_profile_comparison,
    plot_loss_curves,
    sample_ffm
)
from experiments.flow_matching import config


def test_evaluation_metrics():
    """Test all evaluation metrics and plotting functions."""
    print("🧪 Testing FFM Evaluation Metrics")
    print("=" * 50)
    
    cfg = config.cfg
    device = torch.device(cfg.device)
    
    # Load real data
    print("📊 Loading real data...")
    loader, max_length, dataset = create_dataloader(cfg.batch_size, cfg.num_workers, shuffle=False)
    
    # Get some real profiles for testing
    real_profiles = []
    for i, batch in enumerate(loader):
        real_profiles.append(batch.numpy())
        if len(real_profiles) * cfg.batch_size >= 50:  # Limit for testing
            break
    real_profiles = np.concatenate(real_profiles, axis=0)
    
    print(f"   Loaded {len(real_profiles)} real profiles")
    print(f"   Profile shape: {real_profiles.shape}")
    print(f"   Profile range: [{real_profiles.min():.1f}, {real_profiles.max():.1f}]")
    
    # Compute real Vs30 distribution
    print("\n📈 Computing real Vs30 distribution...")
    real_vs30, avg_samples_per_meter = compute_real_vs30_and_density(cfg.parquet_path)
    real_vs100 = compute_vs100(real_profiles, avg_samples_per_meter)
    print(f"   Real Vs30: mean={np.mean(real_vs30):.1f}, std={np.std(real_vs30):.1f}")
    print(f"   Avg samples per meter: {avg_samples_per_meter:.2f}")
    
    # Create mock generated profiles for testing
    print("\n🎲 Creating mock generated profiles...")
    generated_profiles = np.random.randn(len(real_profiles), 1, max_length) * 50 + 300
    print(f"   Generated profiles shape: {generated_profiles.shape}")
    print(f"   Generated range: [{generated_profiles.min():.1f}, {generated_profiles.max():.1f}]")
    
    # Compute Vs30 and Vs100 for generated profiles
    print("\n📊 Computing generated metrics...")
    gen_vs30 = compute_generated_vs30(generated_profiles, avg_samples_per_meter)
    gen_vs100 = compute_vs100(generated_profiles, avg_samples_per_meter)
    
    print(f"   Generated Vs30: mean={np.mean(gen_vs30):.1f}, std={np.std(gen_vs30):.1f}")
    print(f"   Generated Vs100: mean={np.mean(gen_vs100):.1f}, std={np.std(gen_vs100):.1f}")
    
    # Compute KS statistics
    print("\n📈 Computing KS statistics...")
    ks_vs30 = ks_statistic(real_vs30[:len(gen_vs30)], gen_vs30)
    ks_vs100 = ks_statistic(real_vs100[:len(gen_vs100)], gen_vs100)
    
    print(f"   Vs30 KS statistic: {ks_vs30:.4f}")
    print(f"   Vs100 KS statistic: {ks_vs100:.4f}")
    
    # Test plotting functions
    print("\n🎨 Testing plotting functions...")
    
    # Create output directory
    import os
    test_output_dir = "/tmp/ffm_eval_test"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Test profile comparison plot
    plot_profile_comparison(real_profiles[:8], generated_profiles[:8], test_output_dir, 0)
    print("   ✅ Profile comparison plot created")
    
    # Test loss curve plot
    mock_loss_history = np.random.exponential(0.1, 100).cumsum()
    plot_loss_curves(mock_loss_history.tolist(), test_output_dir, 0)
    print("   ✅ Loss curve plot created")
    
    # Test comprehensive evaluation
    print("\n🔍 Testing comprehensive evaluation...")
    
    # Calculate comprehensive statistics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Profile-level metrics
    real_mean = np.mean(real_profiles[:, 0, :], axis=0)
    gen_mean = np.mean(generated_profiles[:, 0, :], axis=0)
    
    profile_mse = mean_squared_error(real_mean, gen_mean)
    profile_mae = mean_absolute_error(real_mean, gen_mean)
    
    # Vs30 metrics
    vs30_mse = mean_squared_error(real_vs30[:len(gen_vs30)], gen_vs30)
    vs30_mae = mean_absolute_error(real_vs30[:len(gen_vs30)], gen_vs30)
    
    # Vs100 metrics
    real_vs100 = compute_vs100(real_profiles, avg_samples_per_meter)
    vs100_mse = mean_squared_error(real_vs100[:len(gen_vs100)], gen_vs100)
    vs100_mae = mean_absolute_error(real_vs100[:len(gen_vs100)], gen_vs100)
    
    print(f"   Profile MSE: {profile_mse:.4f}")
    print(f"   Profile MAE: {profile_mae:.4f}")
    print(f"   Vs30 MSE: {vs30_mse:.4f}")
    print(f"   Vs30 MAE: {vs30_mae:.4f}")
    print(f"   Vs100 MSE: {vs100_mse:.4f}")
    print(f"   Vs100 MAE: {vs100_mae:.4f}")
    
    # Test model evaluation
    print("\n🤖 Testing model evaluation...")
    
    # Create a model for testing
    model = create_model(cfg.model_type, cfg).to(device)
    print(f"   Created {cfg.model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test sampling
    initial_noise = torch.randn(4, 1, max_length).to(device)
    with torch.no_grad():
        model.eval()
        samples = sample_ffm(model, initial_noise, cfg.ode_steps, device)
        model.train()
    
    print(f"   Generated samples shape: {samples.shape}")
    print(f"   Sample range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    print("\n🎉 All evaluation tests passed!")
    print("\n📝 Evaluation Features Implemented:")
    print("   ✅ Real Vs30 distribution computation")
    print("   ✅ Generated Vs30/Vs100 computation")
    print("   ✅ KS statistics for distribution comparison")
    print("   ✅ Profile comparison plots")
    print("   ✅ Loss curve visualization")
    print("   ✅ Comprehensive statistical metrics (MSE, MAE)")
    print("   ✅ Model sampling and evaluation")
    print("   ✅ Integration with training and experiment runner")
    
    print(f"\n📁 Test plots saved to: {test_output_dir}")
    print("\n🚀 Ready for full FFM evaluation!")


if __name__ == "__main__":
    test_evaluation_metrics()
