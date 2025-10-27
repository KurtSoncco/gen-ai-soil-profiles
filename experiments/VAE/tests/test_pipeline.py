#!/usr/bin/env python3
"""
Simple test script to verify the VAE pipeline works correctly.
Tests each component step by step.
"""

import numpy as np
import torch
import logging

from experiments.VAE.src import (
    simple_conv1d_vae,
    gmm_sampling,
    datasets,
    enhanced_metrics,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_conv1d_vae():
    """Test Conv1D VAE architecture."""
    logger.info("Testing Conv1D VAE architecture...")

    SimpleConv1DVAE = simple_conv1d_vae.SimpleConv1DVAE
    simple_conv1d_vae_loss_function = simple_conv1d_vae.simple_conv1d_vae_loss_function

    # Test parameters
    input_dim = 100
    latent_dim = 16
    batch_size = 8

    # Create model
    model = SimpleConv1DVAE(input_dim=input_dim, latent_dim=latent_dim)
    logger.info(
        f"✓ Simple Conv1D VAE created successfully. Parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    recon, mu, logvar = model(x)

    assert recon.shape == x.shape, (
        f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"
    )
    assert mu.shape == (batch_size, latent_dim), f"Mu shape mismatch: {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), (
        f"Logvar shape mismatch: {logvar.shape}"
    )
    logger.info("✓ Forward pass successful")

    # Test loss function
    layer_weights = torch.ones(input_dim) / input_dim
    loss, loss_dict = simple_conv1d_vae_loss_function(
        recon, x, mu, logvar, beta=0.1, layer_weights=layer_weights, tv_weight=0.01
    )

    assert loss.item() > 0, "Loss should be positive"
    assert "recon_loss" in loss_dict, "Missing recon_loss in loss_dict"
    assert "kld_loss" in loss_dict, "Missing kld_loss in loss_dict"
    assert "tv_loss" in loss_dict, "Missing tv_loss in loss_dict"
    logger.info("✓ Loss function successful")


def test_gmm_sampling():
    """Test GMM sampling functionality."""
    logger.info("Testing GMM sampling...")

    LatentGMMSampler = gmm_sampling.LatentGMMSampler
    compute_layer_weights = gmm_sampling.compute_layer_weights
    vs_to_log_vs_profiles = gmm_sampling.vs_to_log_vs_profiles
    log_vs_to_vs_profiles = gmm_sampling.log_vs_to_vs_profiles

    # Test layer weights
    depths = np.linspace(0, 100, 11)  # 10 layers
    weights = compute_layer_weights(depths)
    assert len(weights) == 10, f"Expected 10 weights, got {len(weights)}"
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6), (
        "Weights should sum to 1"
    )
    logger.info("✓ Layer weights computation successful")

    # Test log(Vs) transformations
    vs_profiles = np.random.uniform(100, 1000, (5, 10))  # 5 profiles, 10 layers
    log_vs = vs_to_log_vs_profiles(vs_profiles, depths)
    vs_back = log_vs_to_vs_profiles(log_vs)

    assert np.allclose(vs_profiles, vs_back, rtol=1e-6), "Log(Vs) round-trip failed"
    logger.info("✓ Log(Vs) transformations successful")

    # Test GMM sampler
    latent_samples = np.random.randn(100, 8)  # 100 samples, 8D latent
    sampler = LatentGMMSampler(n_components=3)
    sampler.fit(latent_samples)

    new_samples = sampler.sample(10)
    assert new_samples.shape == (10, 8), (
        f"GMM sampling shape mismatch: {new_samples.shape}"
    )
    logger.info("✓ GMM sampling successful")


def test_datasets():
    """Test dataset functionality."""
    logger.info("Testing datasets...")

    TTSDataset = datasets.TTSDataset

    # Test TTSDataset
    data = np.random.randn(20, 10)  # 20 samples, 10 features
    dataset = TTSDataset(data, corruption_noise_std=0.1)

    assert len(dataset) == 20, f"Dataset length mismatch: {len(dataset)}"

    # Test __getitem__
    noisy, clean = dataset[0]
    assert noisy.shape == clean.shape, "Noisy and clean shapes should match"
    assert not torch.allclose(noisy, clean), "Noisy and clean should be different"
    logger.info("✓ TTSDataset successful")

    # Test TensorDataset
    tensor_data = torch.randn(20, 10)
    tensor_dataset = torch.utils.data.TensorDataset(tensor_data)

    sample = tensor_dataset[0]
    assert isinstance(sample, tuple), "TensorDataset should return tuple"
    assert len(sample) == 1, "TensorDataset tuple should have 1 element"
    logger.info("✓ TensorDataset successful")


def test_enhanced_metrics():
    """Test enhanced metrics functionality."""
    logger.info("Testing enhanced metrics...")

    compute_weighted_metrics = enhanced_metrics.compute_weighted_metrics
    compute_vs30_metrics = enhanced_metrics.compute_vs30_metrics

    # Test weighted metrics
    real_profiles = np.random.uniform(100, 1000, (10, 5))  # 10 profiles, 5 layers
    gen_profiles = real_profiles + np.random.normal(0, 50, (10, 5))  # Add noise
    depths = np.linspace(0, 50, 6)  # 5 layers
    weights = np.ones(5) / 5

    metrics = compute_weighted_metrics(real_profiles, gen_profiles, depths, weights)
    assert "weighted_mse" in metrics, "Missing weighted_mse"
    assert "weighted_mae" in metrics, "Missing weighted_mae"
    assert "tv_ratio" in metrics, "Missing tv_ratio"
    logger.info("✓ Weighted metrics successful")

    # Test Vs30 metrics
    real_vs30 = [300, 400, 500, 350, 450]
    gen_vs30 = [320, 380, 520, 330, 470]

    vs30_metrics = compute_vs30_metrics(list(real_vs30), list(gen_vs30))
    assert "ks_statistic" in vs30_metrics, "Missing ks_statistic"
    assert "mean_ratio" in vs30_metrics, "Missing mean_ratio"
    logger.info("✓ Vs30 metrics successful")


def main():
    """Run all tests."""
    logger.info("Starting VAE pipeline tests...")

    tests = [
        ("Conv1D VAE", test_conv1d_vae),
        ("GMM Sampling", test_gmm_sampling),
        ("Datasets", test_datasets),
        ("Enhanced Metrics", test_enhanced_metrics),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("TEST SUMMARY:")
    logger.info(f"{'=' * 50}")

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:20} : {status}")
        if success:
            passed += 1

    logger.info(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        logger.info("🎉 All tests passed! Pipeline is ready to use.")
    else:
        logger.error("❌ Some tests failed. Check the errors above.")

    return passed == len(results)


if __name__ == "__main__":
    main()
