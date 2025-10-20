import os
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, random, vmap
from scipy.stats import gaussian_kde

from soilgen_ai.logging_config import setup_logging

logger = setup_logging()


class GeneticAlgorithm:
    """
    Encapsulates a genetic algorithm for optimization problems to evolve velocity-time profiles.
    """

    def __init__(self, config: Dict, real_data_dict: Dict[str, pd.DataFrame]):
        self.config = config
        self.key = random.PRNGKey(config["seed"])

        # Unpack configuration for easier access
        self.pop_size = config["pop_size"]
        self.max_layers = config["max_layers"]
        self.max_allowable_depth = config["max_allowable_depth"]
        self.chromosome_len = self.max_layers * 2
        self.common_depths = jnp.linspace(0, self.max_allowable_depth, 100)
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.cache_path = config.get("cache_path")

        # Common depth axis for comparing profiles
        self.common_depths = jnp.linspace(0, self.max_allowable_depth, 100)

        # Preprocess real data to get distributions and reference profiles
        self._preprocess_real_data(real_data_dict)

        # Vectorize core functions for population-wide operations
        self._initialize_population = jit(vmap(self._create_individual))
        self.calculate_population_fitness = jit(vmap(self._calculate_fitness))
        self._crossover_vmap = jit(vmap(self._crossover, in_axes=(0, 0, 0)))
        self._mutation_vmap = jit(vmap(self._mutation, in_axes=(0, 0)))

    def _preprocess_real_data(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Calculates KDEs and interpolated profiles from real data, using caching.

        Args:
            data_dict (Dict[str, pd.DataFrame]): Dictionary of real profiles.

        """
        # If a cache path is provided and the file exists, load from cache
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading preprocessed data from cache: {self.cache_path}")
            cached_data = jnp.load(self.cache_path)
            self.real_max_depths = cached_data["real_max_depths"]
            self.real_num_layers = cached_data["real_num_layers"]
            self.presampled_max_depths = cached_data["presampled_max_depths"]
            self.presampled_num_layers = cached_data["presampled_num_layers"]
            self.real_profiles_interp = cached_data["real_profiles_interp"]
            logger.info("Successfully loaded data from cache.")
            return

        logger.info("Preprocessing real data...")
        # Filter out empty or invalid profiles
        valid_profiles = [
            p for p in data_dict.values() if not p.empty and p["tts"].max() > 0
        ]

        self.real_max_depths = jnp.array([p["depth"].max() for p in valid_profiles])

        def count_layers(profile: pd.DataFrame, threshold: int = 50) -> int:
            # A simple way to estimate layers based on velocity changes if available,
            # or just use number of points as a proxy if not.
            if "vs_value" in profile.columns:
                velocities = profile["vs_value"].to_numpy()
                changes = np.abs(np.diff(velocities)) > threshold
                return int(np.sum(changes) + 1)
            return len(profile)

        self.real_num_layers = jnp.array([count_layers(p) for p in valid_profiles])

        # Scott's rule for bandwidth selection
        depth_bw = self.real_max_depths.shape[0] ** (-1.0 / 5.0)
        layers_bw = self.real_num_layers.shape[0] ** (-1.0 / 5.0)

        # Create SciPy KDEs for sampling
        depth_kde = gaussian_kde(self.real_max_depths, bw_method=depth_bw)
        layers_kde = gaussian_kde(self.real_num_layers, bw_method=layers_bw)

        # ✨ IMPROVEMENT: Pre-sample from KDEs to avoid host_callback in the GA loop.
        # This is a major performance optimization.
        num_presamples = 2000
        np.random.seed(self.config["seed"])  # for reproducibility of sampling
        self.presampled_max_depths = jnp.array(depth_kde.resample(num_presamples)[0])
        self.presampled_num_layers = jnp.array(layers_kde.resample(num_presamples)[0])

        # Interpolate real profiles onto a common depth axis for MSE comparison
        self.real_profiles_interp = jnp.array(
            [
                jnp.interp(
                    self.common_depths,
                    jnp.array(p["depth"].values),
                    jnp.array(p["tts"].values),
                    left=0,
                )
                for p in valid_profiles
            ]
        )
        logger.info("Preprocessing complete.")

        # If a cache path is provided, save the processed data
        if self.cache_path:
            logger.info(f"Saving preprocessed data to cache: {self.cache_path}")
            # Ensure the directory exists
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            jnp.savez(
                self.cache_path,
                real_max_depths=self.real_max_depths,
                real_num_layers=self.real_num_layers,
                presampled_max_depths=self.presampled_max_depths,
                presampled_num_layers=self.presampled_num_layers,
                real_profiles_interp=self.real_profiles_interp,
            )
            logger.info("Successfully saved data to cache.")

    @staticmethod
    @jit
    def _get_profile_properties(
        chromosome: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts number of layers and max depth from a chromosome.

        Args:
            chromosome (jnp.ndarray): The chromosome array representing a profile.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Number of layers and maximum depth.
        """
        depths = chromosome[::2]
        mask = depths > 0
        num_layers = jnp.sum(mask).astype(jnp.int32)
        max_depth = jnp.max(depths)

        return (num_layers, max_depth)

    @staticmethod
    @jit
    def _is_profile_valid(chromosome: jnp.ndarray) -> jnp.ndarray:
        """
        Checks if a profile is valid (monotonically increasing depth and time).

        Args:
            chromosome (jnp.ndarray): The chromosome array representing a profile.

        Returns:
            jnp.ndarray: A boolean array indicating the validity of the profile.
        """
        depths = chromosome[::2]
        times = chromosome[1::2]
        mask = depths > 0
        num_layers = jnp.sum(mask)

        def check_monotonicity(arr: jnp.ndarray) -> jnp.ndarray:
            """Checks if the active part of an array is monotonically increasing."""
            diffs = jnp.diff(arr)
            # Create a mask for only the differences that matter (between active layers)
            valid_diff_mask = jnp.arange(len(diffs)) < (num_layers - 1)
            # Check if all relevant differences are positive
            return jnp.all(jnp.where(valid_diff_mask, diffs > 0, True))

        # A profile is valid if it has 0 or 1 layer, or if it has >1 and is monotonic.
        is_monotonic = check_monotonicity(depths) & check_monotonicity(times)
        return (num_layers <= 1) | is_monotonic

    @staticmethod
    @jit
    def _calculate_velocity_penalty(
        chromosome: jnp.ndarray, epsilon: float
    ) -> jnp.ndarray:
        """
        Calculates a penalty score based on the number of velocity reversals.
        The idea is to penalize profiles where the interval velocity decreases with depth.

        Args:
            chromosome (jnp.ndarray): The chromosome array representing a profile.
            epsilon (float): A small value to prevent division by zero.

        Returns:
            jnp.ndarray: The penalty score (number of velocity reversals) as a scalar array.
        """
        depths = chromosome[::2]
        times = chromosome[1::2]  # These are two-way times

        # Filter for active layers and add the origin for diff calculation
        mask = depths > 0
        num_layers = jnp.sum(mask)

        # --- Sort active layers to the front to avoid dynamic slicing ---
        # To handle active/inactive layers without changing array sizes, we sort them.
        # Inactive layers are given an infinite sort key to push them to the end.
        sort_keys = jnp.where(mask, depths, jnp.inf)
        sorted_indices = jnp.argsort(sort_keys)

        # Apply the sorting to depths and times
        sorted_depths = depths[sorted_indices]
        sorted_times = times[sorted_indices]

        # --- Calculate properties using fixed-size arrays ---
        # Prepend the origin (0, 0) for the first layer's calculation.
        # `jnp.pad` maintains a static array size.
        padded_depths = jnp.pad(sorted_depths, (1, 0))
        padded_times = jnp.pad(sorted_times, (1, 0))

        # Calculate properties for each layer
        thicknesses = jnp.diff(padded_depths)
        # Convert two-way travel time (TTS) to one-way for velocity calculation
        one_way_times = jnp.diff(padded_times) / 2.0

        velocities = thicknesses / (one_way_times + epsilon)

        # Find the differences between adjacent layer velocities
        velocity_diffs = jnp.diff(velocities)

        # --- Mask results to count reversals only for active layers ---
        # A velocity difference is valid only if it's between two active layers.
        # The number of valid velocity differences is `num_layers - 1`.
        # This mask is correct even if num_layers is 0 or 1 (it becomes all False).
        diff_mask = jnp.arange(velocity_diffs.shape[0]) < num_layers - 1

        # Count reversals where velocity decreases (diff < 0), using the mask
        # to only consider valid differences.
        num_reversals = jnp.sum(jnp.where(diff_mask, velocity_diffs < 0, 0.0))

        return num_reversals.astype(float)

    @staticmethod
    @jit
    def _calculate_thickness_penalty(
        chromosome: jnp.ndarray, min_thickness: float = 5.00
    ) -> jnp.ndarray:
        """
        Calculates a JAX-compatible penalty for layers thinner than a minimum threshold.

        This version is JIT-compatible because it avoids Python control flow (`if`)
        and dynamic slicing that would create arrays of non-static size.

        Args:
            chromosome (jnp.ndarray): The chromosome array [d1, t1, d2, t2, ...].
            min_thickness (float): Minimum allowable thickness for a layer.

        Returns:
            jnp.ndarray: The penalty score (a scalar array).
        """
        depths = chromosome[::2]

        # Create a mask for active layers (where depth > 0)
        mask = depths > 0
        num_layers = jnp.sum(mask)

        # --- Sort active layers to the front to avoid dynamic slicing ---
        # Inactive layers are given an infinite sort key to push them to the end.
        sort_keys = jnp.where(mask, depths, jnp.inf)
        sorted_indices = jnp.argsort(sort_keys)
        sorted_depths = depths[sorted_indices]

        # --- Calculate thicknesses using fixed-size arrays ---
        # Prepend the origin (0) to calculate the first layer's thickness.
        # `jnp.pad` maintains a static array size.
        padded_depths = jnp.pad(sorted_depths, (1, 0))

        # Calculate thicknesses for all potential layers
        thicknesses = jnp.diff(padded_depths)

        # --- Mask results to penalize only active layers ---
        # A thickness is valid only if it corresponds to an active layer.
        # The number of valid thicknesses is `num_layers`.
        thickness_mask = jnp.arange(thicknesses.shape[0]) < num_layers

        # A layer is penalized if it's both active AND thinner than the minimum.
        is_thin = thicknesses < min_thickness

        # We use jnp.logical_and to combine the conditions before summing.
        num_thin_layers = jnp.sum(jnp.logical_and(thickness_mask, is_thin))

        return num_thin_layers.astype(float)

    def _create_individual(self, key: random.PRNGKey) -> jnp.ndarray:  # type: ignore
        """Creates a single random, valid individual chromosome."""
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

        # Sample from pre-generated KDE samples to guide creation
        num_layers_sampled = random.choice(subkey1, self.presampled_num_layers)
        num_layers = jnp.clip(
            jnp.round(num_layers_sampled).astype(jnp.int32), 2, self.max_layers
        )

        max_depth_sampled = random.choice(subkey2, self.presampled_max_depths)
        max_depth = jnp.clip(max_depth_sampled, 100, self.max_allowable_depth)

        # Generate monotonically increasing depths and times
        depth_deltas = random.uniform(subkey3, (self.max_layers,), minval=0.1)
        depths = jnp.cumsum(depth_deltas)
        time_deltas = random.uniform(
            subkey4, (self.max_layers,), minval=0.001, maxval=0.10
        )
        times = jnp.cumsum(time_deltas)

        # Create a mask for the active number of layers
        mask = jnp.arange(self.max_layers) < num_layers

        # Rescale depths to match the sampled max_depth
        last_valid_depth = jnp.max(jnp.where(mask, depths, 0.0))
        scaling_factor = max_depth / (last_valid_depth + self.epsilon)
        scaled_depths = depths * scaling_factor

        # Apply mask to zero out unused elements
        final_depths = jnp.where(mask, scaled_depths, 0.0)
        final_times = jnp.where(mask, times, 0.0)

        # Interleave depths and times to form the chromosome
        return jnp.stack([final_depths, final_times], axis=1).flatten()

    def _calculate_fitness(self, chromosome: jnp.ndarray) -> jnp.ndarray:
        """Calculates the fitness score for a single chromosome."""
        # 1. Feasibility Check
        is_valid = self._is_profile_valid(chromosome)

        # 2. Shape Similarity (MSE vs real profiles)
        depths = chromosome[::2]
        times = chromosome[1::2]
        gen_times_interp = jnp.interp(self.common_depths, depths, times, left=0)
        mse_vs_real = jnp.mean(
            (gen_times_interp - self.real_profiles_interp) ** 2, axis=1
        )
        min_mse = jnp.min(mse_vs_real)
        shape_fitness = 1.0 / (1.0 + min_mse)

        # 3. Distribution Similarity (KDE score)
        num_layers, max_depth = self._get_profile_properties(chromosome)

        # This JAX-native KDE is only used for fitness evaluation, not sampling
        depth_score = self._kde_jax(
            max_depth, self.real_max_depths, 0.5
        )  # Using fixed bw for simplicity
        layers_score = self._kde_jax(num_layers, self.real_num_layers, 0.5)
        dist_fitness = jnp.mean(jnp.array([depth_score, layers_score]))

        # ✨ 4. NEW: Calculate physical penalties ✨
        velocity_penalty = self._calculate_velocity_penalty(chromosome, self.epsilon)
        thickness_penalty = self._calculate_thickness_penalty(
            chromosome, self.config["min_layer_thickness"]
        )

        # Final weighted fitness, gated by validity
        w_shape = self.config["fitness_weight_shape"]
        w_dist = self.config["fitness_weight_dist"]
        w_vel = self.config["fitness_weight_velocity"]  # New weight
        w_thick = self.config["fitness_weight_thickness"]  # New weight

        # Combine penalties into a single penalty factor
        total_fitness = (
            w_shape * shape_fitness
            + w_dist * dist_fitness
            - w_vel * velocity_penalty
            - w_thick * thickness_penalty
        )

        # Return 0 fitness if the profile is not valid
        return jnp.array(jnp.where(is_valid, total_fitness, 0.0))

    def _selection(
        self, population: jnp.ndarray, fitness_scores: jnp.ndarray, key: jnp.ndarray
    ) -> jnp.ndarray:
        """Performs tournament selection to choose parents."""

        def tournament(subkey: jnp.ndarray) -> jnp.ndarray:
            contender_indices = random.choice(
                subkey,
                self.pop_size,
                shape=(self.config["tournament_size"],),
                replace=False,
            )
            contender_fitness = fitness_scores[contender_indices]
            winner_idx = contender_indices[jnp.argmax(contender_fitness)]
            return population[winner_idx]

        keys = random.split(key, self.pop_size)
        return vmap(tournament)(keys)

    def _crossover(
        self, key: jnp.ndarray, parent1: jnp.ndarray, parent2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Performs single-point crossover on a pair of parents using a mask.
        This method avoids dynamic slicing to be compatible with jit and vmap.
        """
        # Crossover point must be even to not split a (depth, time) pair
        crossover_point = random.randint(key, (), 1, self.max_layers) * 2

        # Create a boolean mask instead of slicing
        mask = jnp.arange(self.chromosome_len) < crossover_point

        # Use jnp.where to select genes from parents based on the mask
        child1 = jnp.where(mask, parent1, parent2)
        child2 = jnp.where(mask, parent2, parent1)

        return child1, child2

    def _mutation(self, key: jnp.ndarray, chromosome: jnp.ndarray) -> jnp.ndarray:
        """Applies Gaussian noise mutation to a chromosome."""
        key, subkey = random.split(key)
        mutation_mask = random.bernoulli(
            key, self.config["mutation_rate"], shape=chromosome.shape
        )
        noise = (
            random.normal(subkey, shape=chromosome.shape)
            * self.config["mutation_sigma"]
        )
        mutated_chromosome = chromosome + mutation_mask * noise
        # Ensure values don't become negative after mutation
        return jnp.maximum(mutated_chromosome, 0)

    @staticmethod
    @jit
    def _kde_jax(x: jnp.ndarray, points: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
        """JAX-native implementation of a Gaussian Kernel Density Estimator."""
        n = points.shape[0]
        diffs = x - points
        norm_constant = 1.0 / (n * bandwidth * jnp.sqrt(2 * jnp.pi))
        kernel_vals = jnp.exp(-0.5 * (diffs / bandwidth) ** 2)
        return norm_constant * jnp.sum(kernel_vals)

    def run(self):
        """Executes the main genetic algorithm evolutionary loop."""
        self.key, pop_key = random.split(self.key)
        population_keys = random.split(pop_key, self.pop_size)
        population = self._initialize_population(population_keys)

        logger.info("\n--- Starting Genetic Algorithm Evolution ---")
        for generation in range(self.config["generations"]):
            # Calculate fitness for the entire population
            fitness_scores = self.calculate_population_fitness(population)

            # --- Elitism: Carry over the best individuals ---
            num_elites = self.config["num_elites"]
            elite_indices = jnp.argsort(fitness_scores)[-num_elites:]
            elites = population[elite_indices]

            # --- Selection ---
            self.key, select_key = random.split(self.key)
            parents = self._selection(population, fitness_scores, select_key)

            # --- Crossover ---
            self.key, crossover_key, shuffle_key = random.split(self.key, 3)
            shuffled_parents = random.permutation(shuffle_key, parents, axis=0)

            num_pairs = (self.pop_size - num_elites) // 2
            parent_pairs1 = shuffled_parents[:num_pairs]
            parent_pairs2 = shuffled_parents[num_pairs : num_pairs * 2]

            crossover_keys = random.split(crossover_key, num_pairs)
            children1, children2 = self._crossover_vmap(
                crossover_keys, parent_pairs1, parent_pairs2
            )
            offspring = jnp.concatenate([children1, children2])

            # --- Mutation ---
            self.key, mutation_key = random.split(self.key)
            mutation_keys = random.split(mutation_key, offspring.shape[0])
            mutated_offspring = self._mutation_vmap(mutation_keys, offspring)

            # --- Create New Population ---
            population = jnp.concatenate([elites, mutated_offspring])

            # --- Logging ---
            best_fitness = jnp.max(fitness_scores)
            avg_fitness = jnp.mean(fitness_scores)
            if generation % 10 == 0:  # Log less frequently to avoid clutter
                logger.info(
                    f"Generation {generation:03d} | "
                    f"Best Fitness: {best_fitness:.4f} | "
                    f"Avg Fitness: {avg_fitness:.4f}"
                )

        # --- Final Results ---
        logger.info("--- Evolution Finished ---")
        final_fitness = self.calculate_population_fitness(population)
        best_idx = jnp.argmax(final_fitness)
        best_individual = population[best_idx]
        best_fitness_val = final_fitness[best_idx]

        logger.info(f"\nBest individual fitness: {best_fitness_val:.4f}")
        num_layers, max_depth = self._get_profile_properties(best_individual)
        logger.info(
            f"Best individual properties: {num_layers} layers, Max Depth: {max_depth:.2f}"
        )

        return best_individual
