# Parametric Profile Modeling

This directory contains the implementation of parametric profile modeling for soil shear wave velocity (Vs) profiles. Instead of generating full profiles as sequences of points, this approach uses mathematical functions to describe Vs profiles with a small set of parameters.

## Approach Overview

The parametric approach consists of two main components:

1. **Parametric Profile Models**: Mathematical functions that describe Vs profiles using a small set of parameters
2. **Generative Models**: Machine learning models that learn to generate the parameters of these functions

## Parametric Models

### 1. Exponential Model
Describes Vs profiles using an exponential function:
```
Vs(z) = Vs_shallow + (Vs_deep - Vs_shallow) * (1 - exp(-z/z_transition))
```

**Parameters:**
- `vs_shallow`: Shear wave velocity at surface (m/s)
- `vs_deep`: Asymptotic shear wave velocity at depth (m/s)
- `z_transition`: Characteristic transition depth (m)

### 2. Power Law Model
Describes Vs profiles using a power law function:
```
Vs(z) = vs_shallow * (1 + z/z_ref)^alpha
```

**Parameters:**
- `vs_shallow`: Shear wave velocity at surface (m/s)
- `z_ref`: Reference depth (m)
- `alpha`: Power law exponent

### 3. Layered Model
Describes Vs profiles using distinct layers with constant velocities:
```
Vs(z) = vs_layer_i for z_i <= z < z_{i+1}
```

**Parameters:**
- `vs_layer_i`: Shear wave velocity for layer i (m/s)

## Generative Models

### 1. Gaussian Mixture Model (GMM)
- Learns the distribution of parametric model parameters
- Uses multiple Gaussian components to capture complex parameter distributions
- Fast generation and good for capturing multi-modal distributions

### 2. Multi-Layer Perceptron (MLP) VAE
- Uses a Variational Autoencoder architecture
- Learns a latent representation of parameter distributions
- Can capture more complex relationships between parameters

## File Structure

```
parametric/
├── src/
│   ├── __init__.py
│   ├── parametric_models.py      # Parametric profile model implementations
│   └── generative_models.py      # Generative model implementations
├── run_parametric_experiment.py  # Main experiment script
├── evaluate_parametric.py       # Evaluation and visualization script
├── run_experiment.py            # Simple runner script
└── README.md                   # This file
```

## Usage

### Running the Experiment

```bash
# Run the complete experiment
python run_experiment.py

# Or run individual components
python run_parametric_experiment.py
python evaluate_parametric.py
```

### Key Features

1. **Multiple Parametric Models**: Supports exponential, power law, and layered models
2. **Multiple Generative Models**: Supports both GMM and MLP VAE approaches
3. **Comprehensive Evaluation**: Includes statistical comparisons and visualizations
4. **Flexible Configuration**: Easy to modify model parameters and experiment settings

## Configuration

The experiment can be configured by modifying the constants in `run_parametric_experiment.py`:

```python
NUM_LAYERS = 100              # Number of layers in standardized profiles
MAX_DEPTH = 2000              # Maximum depth (m)
N_NEW_PROFILES = 1000         # Number of new profiles to generate
MODEL_TYPES = ['exponential', 'power_law', 'layered']
GENERATIVE_MODEL_TYPES = ['gmm', 'mlp']
```

## Output

The experiment generates:

1. **Model Files**: Trained generative models saved as pickle files
2. **Visualization Plots**: 
   - Parametric model comparison plots
   - Generated profile visualizations
   - Parameter distribution plots
3. **Evaluation Results**: Statistical comparison metrics
4. **W&B Logs**: Training metrics and evaluation results

## Advantages of Parametric Approach

1. **Interpretability**: Parameters have physical meaning
2. **Efficiency**: Small number of parameters compared to full profile sequences
3. **Flexibility**: Easy to incorporate domain knowledge and constraints
4. **Robustness**: Less prone to overfitting due to reduced parameter space
5. **Scalability**: Fast generation and evaluation

## Comparison with VAE Approach

| Aspect | VAE | Parametric |
|--------|-----|------------|
| Parameter Count | ~100 (full profile) | 3-5 (model parameters) |
| Interpretability | Low | High |
| Physical Meaning | Limited | High |
| Generation Speed | Moderate | Fast |
| Domain Knowledge | Hard to incorporate | Easy to incorporate |
| Overfitting Risk | Higher | Lower |

## Future Extensions

1. **Hybrid Models**: Combine multiple parametric models
2. **Hierarchical Models**: Model different depth ranges with different functions
3. **Uncertainty Quantification**: Add uncertainty estimates to generated profiles
4. **Conditional Generation**: Generate profiles conditioned on site characteristics
5. **Physics-Informed Models**: Incorporate geophysical constraints
