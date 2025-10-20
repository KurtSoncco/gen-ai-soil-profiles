# VAE Experiments for Soil Profile Generation

This directory contains the complete implementation of Variational Autoencoder (VAE) experiments for generating realistic shear wave velocity (Vs) profiles from real geotechnical data.

## 📁 Directory Structure

```
VAE/
├── README.md                    # This file
├── src/                        # Source code
│   ├── vae_implementation.py   # Main implementation script
│   ├── vae_model.py           # Basic MLP VAE model
│   ├── conv1d_vae.py          # Conv1D VAE architecture
│   ├── simple_conv1d_vae.py   # Simplified Conv1D VAE
│   ├── vq_vae_model.py        # Vector Quantized VAE
│   ├── dae_trainer.py         # DAE pretraining and VAE fine-tuning
│   ├── datasets.py            # Dataset handling and corruption
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── enhanced_metrics.py    # Comprehensive evaluation metrics
│   ├── gmm_sampling.py        # GMM-based latent space sampling
│   ├── training.py            # Training utilities
│   └── utils.py               # General utilities and plotting
├── models/                     # Saved model weights
│   ├── vae_model.pth          # Trained VAE model
│   ├── smoke_model.pth        # Smoke test model
│   └── smoke_checkpoint.pt    # Smoke test checkpoint
├── plots/                      # Generated visualizations
│   ├── training_losses.png    # Training progress plot
│   ├── generated_profiles.png # Generated Vs profiles
│   ├── generated_vs_profiles.png # Detailed profile visualization
│   ├── comprehensive_evaluation.png # Complete evaluation metrics
│   ├── vs30_comparison.png    # Vs30 distribution comparison
│   └── vs_dist_comparison.png # Vs distribution comparison by depth
├── configs/                    # Configuration files
│   └── wandb_sweep.yaml       # Weights & Biases sweep configuration
├── tests/                      # Test scripts
│   ├── smoke_test.py          # Quick end-to-end test
│   └── test_pipeline.py       # Comprehensive pipeline tests
└── results/                    # Experimental results (empty, for future use)
```

## 🚀 Quick Start

### 1. Run the Main Experiment
```bash
cd src/
python vae_implementation.py
```

### 2. Run Tests
```bash
cd tests/
python smoke_test.py          # Quick smoke test
python test_pipeline.py       # Comprehensive tests
```

## 📊 Key Features

### Models Implemented
- **MLP VAE**: Basic multi-layer perceptron VAE
- **Conv1D VAE**: 1D convolutional VAE for sequential data
- **Simple Conv1D VAE**: Simplified Conv1D architecture
- **VQ-VAE**: Vector Quantized VAE (available but not used in main pipeline)

### Training Approaches
- **Baseline Training**: Standard VAE training
- **DAE Pretraining**: Denoising Autoencoder pretraining followed by VAE fine-tuning

### Evaluation Metrics
- **Weighted MSE/MAE**: Layer-thickness weighted reconstruction errors
- **Total Variation Ratio**: Smoothness comparison
- **Vs30 Statistics**: Mean, std, and KS test comparisons
- **Depth-wise Analysis**: Vs distribution comparison at different depths

### Advanced Features
- **GMM Sampling**: Gaussian Mixture Model for latent space sampling
- **Enhanced Metrics**: Comprehensive evaluation beyond basic reconstruction
- **Wandb Integration**: Experiment tracking and visualization
- **Automatic Plot Saving**: All plots saved instead of displayed

## 📈 Recent Results

The latest run achieved:
- **Final Loss**: Train: 0.0026, Test: 0.0030
- **Vs30 Comparison**: Real (mean=219.28, std=132.96) vs Generated (mean=221.84, std=115.83)
- **KS Statistic**: 0.0959 (good distribution match)
- **Mean Ratio**: 1.0117 (excellent mean preservation)
- **Std Ratio**: 0.8712 (good variance preservation)

## 🔧 Configuration

### Key Parameters (in `vae_implementation.py`)
- `LATENT_DIM = 64`: Latent space dimensionality
- `EPOCHS = 500`: Training epochs
- `BATCH_SIZE = 64`: Batch size
- `LEARNING_RATE = 0.001`: Learning rate
- `MAX_DEPTH = 100`: Maximum profile depth (meters)
- `VS_MAX = 2000`: Maximum Vs value (m/s)

### Wandb Sweep Configuration
Located in `configs/wandb_sweep.yaml` for hyperparameter optimization.

## 📝 File Descriptions

### Core Implementation
- `vae_implementation.py`: Main script orchestrating the entire pipeline
- `vae_model.py`: Basic MLP VAE implementation
- `conv1d_vae.py`: Conv1D VAE for sequential data
- `simple_conv1d_vae.py`: Simplified Conv1D VAE for debugging

### Training & Evaluation
- `dae_trainer.py`: DAE pretraining and VAE fine-tuning
- `enhanced_metrics.py`: Comprehensive evaluation metrics
- `gmm_sampling.py`: GMM-based sampling for better generation
- `training.py`: Training utilities and loss functions

### Data Handling
- `datasets.py`: Dataset classes and data corruption for DAE
- `preprocessing.py`: Data preprocessing and standardization
- `utils.py`: General utilities, plotting, and Vs30 calculations

### Testing
- `smoke_test.py`: Quick end-to-end functionality test
- `test_pipeline.py`: Comprehensive pipeline testing

## 🎯 Usage Examples

### Run Complete Pipeline
```bash
cd src/
python vae_implementation.py
```

### Test Individual Components
```bash
cd tests/
python test_pipeline.py
```

### Run Wandb Sweep
```bash
cd configs/
wandb sweep wandb_sweep.yaml
```

## 📊 Output Files

After running the pipeline, you'll find:
- **Models**: Trained weights in `models/`
- **Plots**: All visualizations in `plots/`
- **Logs**: Wandb experiment tracking
- **Results**: Evaluation metrics and statistics

## 🔍 Troubleshooting

### Common Issues
1. **CUDA/GPU Issues**: The code automatically detects and uses available devices
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Plot Display**: All plots are saved automatically (no display required)

### Dependencies
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas
- Wandb

## 📚 References

This implementation is based on:
- Variational Autoencoders (Kingma & Welling, 2013)
- Denoising Autoencoders (Vincent et al., 2008)
- GMM-based sampling techniques
- Geotechnical Vs profile analysis

---

**Last Updated**: October 17, 2025  
**Status**: ✅ Fully functional and tested
