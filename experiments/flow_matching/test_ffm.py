#!/usr/bin/env python3
"""
FFM Implementation Test Script

This script demonstrates the complete FFM implementation with both UNet and FNO architectures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.flow_matching.models import create_model
from experiments.flow_matching.data import create_dataloader
from experiments.flow_matching.train import train_ffm_step, sample_ffm
from experiments.flow_matching import config


def test_ffm_implementation():
    """Test the complete FFM implementation."""
    print("🚀 Testing FFM Implementation")
    print("=" * 50)
    
    cfg = config.cfg
    device = torch.device(cfg.device)
    
    # Load data
    print("📊 Loading data...")
    loader, max_length, dataset = create_dataloader(cfg.batch_size, cfg.num_workers, shuffle=False)
    print(f"   Dataset size: {len(dataset)} profiles")
    print(f"   Max length: {max_length}")
    print(f"   Data range: [{dataset.min_val:.2f}, {dataset.max_val:.2f}]")
    
    # Test both architectures
    architectures = ['unet', 'fno']
    
    for arch in architectures:
        print(f"\n🏗️  Testing {arch.upper()} architecture...")
        
        # Create model
        model = create_model(arch, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test training step
        batch = next(iter(loader))
        loss = train_ffm_step(model, optimizer, batch, cfg)
        print(f"   Training step loss: {loss:.6f}")
        
        # Test sampling
        initial_noise = torch.randn(4, 1, max_length).to(device)
        samples = sample_ffm(model, initial_noise, cfg.ode_steps, device)
        print(f"   Generated samples shape: {samples.shape}")
        print(f"   Sample range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
        
        # Denormalize samples for comparison
        samples_denorm = dataset.denormalize_batch(samples)
        print(f"   Denormalized range: [{samples_denorm.min().item():.1f}, {samples_denorm.max().item():.1f}] m/s")
        
        print(f"   ✅ {arch.upper()} test passed!")
    
    print("\n🎉 All tests passed! FFM implementation is ready.")
    print("\n📝 Next steps:")
    print("   1. Run training: python -m experiments.flow_matching.train")
    print("   2. Generate samples: python -m experiments.flow_matching.sample")
    print("   3. Run full experiment: python -m experiments.flow_matching.run_experiment")


if __name__ == "__main__":
    test_ffm_implementation()
