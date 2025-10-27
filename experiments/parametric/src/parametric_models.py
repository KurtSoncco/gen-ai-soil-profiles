"""
Parametric Profile Models for Soil Shear Wave Velocity Profiles

This module implements mathematical functions to describe Vs profiles using
a small set of parameters, making the generation more interpretable and
computationally efficient.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize


class ParametricProfileModel:
    """
    Base class for parametric profile models.
    Defines the interface for fitting and generating Vs profiles.
    """

    def __init__(self, depths: np.ndarray):
        """
        Initialize the parametric model.

        Args:
            depths: Array of depth values (in meters) for the profile
        """
        self.depths = depths
        self.num_layers = len(depths) - 1

    def fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """
        Fit the parametric model to a given Vs profile.

        Args:
            vs_profile: Array of Vs values for each layer

        Returns:
            Dictionary containing the fitted parameters
        """
        raise NotImplementedError

    def generate(self, params: Dict[str, float]) -> np.ndarray:
        """
        Generate a Vs profile from parameters.

        Args:
            params: Dictionary containing the model parameters

        Returns:
            Array of Vs values for each layer
        """
        raise NotImplementedError

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for each parameter during fitting.

        Returns:
            Dictionary mapping parameter names to (min, max) bounds
        """
        raise NotImplementedError


class ExponentialModel(ParametricProfileModel):
    """
    Exponential Vs profile model with shallow Vs, deep Vs, and transition depth.

    Vs(z) = Vs_shallow + (Vs_deep - Vs_shallow) * (1 - exp(-z/z_transition))

    Parameters:
        - vs_shallow: Shear wave velocity at surface (m/s)
        - vs_deep: Asymptotic shear wave velocity at depth (m/s)
        - z_transition: Characteristic transition depth (m)
    """

    def __init__(self, depths: np.ndarray):
        super().__init__(depths)
        self.layer_centers = (depths[:-1] + depths[1:]) / 2

    def fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """Fit exponential model to Vs profile."""

        def objective(params):
            vs_shallow, vs_deep, z_transition = params
            predicted = self._exponential_function(
                self.layer_centers, vs_shallow, vs_deep, z_transition
            )
            return np.sum((vs_profile - predicted) ** 2)

        # Initial guess based on profile statistics
        vs_min = np.min(vs_profile)
        vs_max = np.max(vs_profile)
        z_max = np.max(self.depths)

        initial_guess = [vs_min, vs_max, z_max / 3]
        bounds = [(100, 1000), (500, 2500), (50, 1000)]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if not result.success:
            logging.warning(f"Exponential model fitting failed: {result.message}")
            # Fallback to simple linear model
            return self._fallback_linear_fit(vs_profile)

        return {
            "vs_shallow": result.x[0],
            "vs_deep": result.x[1],
            "z_transition": result.x[2],
        }

    def generate(self, params: Dict[str, float]) -> np.ndarray:
        """Generate Vs profile from exponential parameters."""
        vs_shallow = params["vs_shallow"]
        vs_deep = params["vs_deep"]
        z_transition = params["z_transition"]

        return self._exponential_function(
            self.layer_centers, vs_shallow, vs_deep, z_transition
        )

    def _exponential_function(
        self, z: np.ndarray, vs_shallow: float, vs_deep: float, z_transition: float
    ) -> np.ndarray:
        """Exponential Vs function."""
        return vs_shallow + (vs_deep - vs_shallow) * (1 - np.exp(-z / z_transition))

    def _fallback_linear_fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """Fallback to linear model if exponential fitting fails."""
        vs_min = np.min(vs_profile)
        vs_max = np.max(vs_profile)
        z_max = np.max(self.depths)

        return {"vs_shallow": vs_min, "vs_deep": vs_max, "z_transition": z_max / 2}

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for exponential model."""
        return {
            "vs_shallow": (100, 1000),
            "vs_deep": (500, 2500),
            "z_transition": (50, 1000),
        }


class PowerLawModel(ParametricProfileModel):
    """
    Power law Vs profile model.

    Vs(z) = vs_shallow * (1 + z/z_ref)^alpha

    Parameters:
        - vs_shallow: Shear wave velocity at surface (m/s)
        - z_ref: Reference depth (m)
        - alpha: Power law exponent
    """

    def __init__(self, depths: np.ndarray):
        super().__init__(depths)
        self.layer_centers = (depths[:-1] + depths[1:]) / 2

    def fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """Fit power law model to Vs profile."""

        def objective(params):
            vs_shallow, z_ref, alpha = params
            predicted = self._power_law_function(
                self.layer_centers, vs_shallow, z_ref, alpha
            )
            return np.sum((vs_profile - predicted) ** 2)

        # Initial guess
        vs_min = np.min(vs_profile)
        z_max = np.max(self.depths)

        initial_guess = [vs_min, z_max / 2, 0.3]
        bounds = [(100, 1000), (50, 1000), (0.1, 1.0)]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if not result.success:
            logging.warning(f"Power law model fitting failed: {result.message}")
            return self._fallback_linear_fit(vs_profile)

        return {"vs_shallow": result.x[0], "z_ref": result.x[1], "alpha": result.x[2]}

    def generate(self, params: Dict[str, float]) -> np.ndarray:
        """Generate Vs profile from power law parameters."""
        vs_shallow = params["vs_shallow"]
        z_ref = params["z_ref"]
        alpha = params["alpha"]

        return self._power_law_function(self.layer_centers, vs_shallow, z_ref, alpha)

    def _power_law_function(
        self, z: np.ndarray, vs_shallow: float, z_ref: float, alpha: float
    ) -> np.ndarray:
        """Power law Vs function."""
        return vs_shallow * (1 + z / z_ref) ** alpha

    def _fallback_linear_fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """Fallback to linear model if power law fitting fails."""
        vs_min = np.min(vs_profile)
        z_max = np.max(self.depths)

        return {"vs_shallow": vs_min, "z_ref": z_max / 2, "alpha": 0.3}

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for power law model."""
        return {"vs_shallow": (100, 1000), "z_ref": (50, 1000), "alpha": (0.1, 1.0)}


class LayeredModel(ParametricProfileModel):
    """
    Layered Vs profile model with distinct layers.

    Parameters:
        - vs_layers: List of Vs values for each layer
        - layer_depths: List of depths for layer boundaries
    """

    def __init__(self, depths: np.ndarray, num_layers: int = 5):
        super().__init__(depths)
        self.num_model_layers = num_layers
        self.layer_boundaries = np.linspace(0, np.max(depths), num_layers + 1)
        self.layer_centers = (depths[:-1] + depths[1:]) / 2

    def fit(self, vs_profile: np.ndarray) -> Dict[str, float]:
        """Fit layered model to Vs profile."""
        params = {}

        for i in range(self.num_model_layers):
            # Find which original layers belong to this model layer
            layer_mask = (self.layer_centers >= self.layer_boundaries[i]) & (
                self.layer_centers < self.layer_boundaries[i + 1]
            )

            if np.any(layer_mask):
                params[f"vs_layer_{i}"] = np.mean(vs_profile[layer_mask])
            else:
                # Fallback if no layers in this range
                params[f"vs_layer_{i}"] = np.mean(vs_profile)

        return params

    def generate(self, params: Dict[str, float]) -> np.ndarray:
        """Generate Vs profile from layered parameters."""
        vs_profile = np.zeros(self.num_layers)

        for i in range(self.num_model_layers):
            layer_mask = (self.layer_centers >= self.layer_boundaries[i]) & (
                self.layer_centers < self.layer_boundaries[i + 1]
            )
            vs_profile[layer_mask] = params[f"vs_layer_{i}"]

        return vs_profile

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for layered model."""
        bounds = {}
        for i in range(self.num_model_layers):
            bounds[f"vs_layer_{i}"] = (100, 2500)
        return bounds


class ParametricProfileFitter:
    """
    Fits parametric models to Vs profiles and extracts parameters.
    """

    def __init__(self, depths: np.ndarray, model_type: str = "exponential"):
        """
        Initialize the parametric profile fitter.

        Args:
            depths: Array of depth values
            model_type: Type of parametric model ('exponential', 'power_law', 'layered')
        """
        self.depths = depths

        if model_type == "exponential":
            self.model = ExponentialModel(depths)
        elif model_type == "power_law":
            self.model = PowerLawModel(depths)
        elif model_type == "layered":
            self.model = LayeredModel(depths)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit_profiles(self, vs_profiles: np.ndarray) -> np.ndarray:
        """
        Fit parametric models to multiple Vs profiles.

        Args:
            vs_profiles: Array of shape (n_profiles, n_layers) containing Vs profiles

        Returns:
            Array of shape (n_profiles, n_params) containing fitted parameters
        """
        n_profiles = vs_profiles.shape[0]
        param_names = list(self.model.get_parameter_bounds().keys())
        n_params = len(param_names)

        fitted_params = np.zeros((n_profiles, n_params))

        for i in range(n_profiles):
            try:
                params_dict = self.model.fit(vs_profiles[i])
                fitted_params[i] = [params_dict[name] for name in param_names]
            except Exception as e:
                logging.warning(f"Failed to fit profile {i}: {e}")
                # Use fallback parameters
                bounds = self.model.get_parameter_bounds()
                fitted_params[i] = [np.mean(bounds[name]) for name in param_names]

        return fitted_params

    def generate_profiles(self, params: np.ndarray) -> np.ndarray:
        """
        Generate Vs profiles from parameters.

        Args:
            params: Array of shape (n_profiles, n_params) containing parameters

        Returns:
            Array of shape (n_profiles, n_layers) containing generated Vs profiles
        """
        n_profiles = params.shape[0]
        param_names = list(self.model.get_parameter_bounds().keys())

        generated_profiles = np.zeros((n_profiles, self.model.num_layers))

        for i in range(n_profiles):
            params_dict = {name: params[i, j] for j, name in enumerate(param_names)}
            generated_profiles[i] = self.model.generate(params_dict)

        return generated_profiles

    def get_parameter_names(self) -> List[str]:
        """Get the names of the parameters."""
        return list(self.model.get_parameter_bounds().keys())
