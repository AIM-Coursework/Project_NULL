"""
Shared Base Class for Metaheuristic Optimisation
================================================
Provides common functionality for all metaheuristics:
- Stratified subsampling of training data
- Encoding/decoding of solutions
- Fitness evaluation

A solution is represented as a 1D NumPy array of continuous values in [0, 1].
- The first `n_features` elements determine feature selection (mask > 0.5).
- The remaining elements are linearly mapped to hyperparameter bounds.
"""

import os
import sys
import time
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split

# Import fitness_function from base_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import fitness_function, ModelConfig

class MetaheuristicBase:
    def __init__(self, model_type, X_train, y_train, class_weights,
                 n_features, hyperparam_bounds, cfg: Optional[ModelConfig]=None,
                 subsample_ratio=0.05, max_generations=15, pop_size=20):
        """
        Initialize the base metaheuristic attributes and prepare data.
        """
        self.model_type = model_type
        self.class_weights = class_weights
        self.n_features = n_features
        self.hyperparam_bounds = hyperparam_bounds
        self.cfg = cfg
        self.seed = cfg.seed if cfg else 42
        self.max_generations = max_generations
        self.pop_size = pop_size
        
        # Set seeds
        np.random.seed(self.seed)
        
        # Extract hyperparameter ordered keys
        self.hp_keys = list(self.hyperparam_bounds.keys())
        self.n_hyperparams = len(self.hp_keys)
        self.solution_dim = self.n_features + self.n_hyperparams
        
        # Stratified subsampling for fitness evaluations
        # We only need fitness rankings to be correct, so 20% is sufficient and much faster.
        if subsample_ratio < 1.0:
            print(f"Creating {subsample_ratio*100:.0f}% stratified subsample for fitness evaluations...")
            # We don't need the other 80% here, we only use the subsample for fitness
            self.X_sub, _, self.y_sub, _ = train_test_split(
                X_train, y_train, train_size=subsample_ratio,
                stratify=y_train, random_state=self.seed
            )
        else:
            self.X_sub, self.y_sub = X_train, y_train

        # Cast to float32 to halve memory footprint (float64 -> float32)
        if hasattr(self.X_sub, "values"):
            self.X_sub = self.X_sub.values.astype(np.float32)
        else:
            self.X_sub = np.asarray(self.X_sub, dtype=np.float32)

        if hasattr(self.y_sub, "values"):
            self.y_sub = self.y_sub.values.ravel()
        else:
            self.y_sub = np.ravel(self.y_sub)

        print(f"Subsample size: {len(self.y_sub):,} | Normal: {(self.y_sub == 0).sum():,} | Attack: {(self.y_sub == 1).sum():,}")
        
        # Cache for fitness evaluations to prevent redundant re-evaluations
        self._fitness_cache = {}

    def _decode_solution(self, solution):
        """
        Decode a continuous [0, 1] solution vector into feature_mask and hyperparams.
        """
        # 1. Feature Mask (first n_features elements)
        feature_vector = solution[:self.n_features]
        feature_mask = (feature_vector >= 0.5).astype(bool)
        
        # Ensure at least one feature is selected (fallback to feature 0)
        if not np.any(feature_mask):
            feature_mask[0] = True
            
        # 2. Hyperparameters (remaining elements)
        hp_vector = solution[self.n_features:]
        hyperparams = {}
        for i, key in enumerate(self.hp_keys):
            lower, upper = self.hyperparam_bounds[key]
            val = lower + hp_vector[i] * (upper - lower)
            
            # If bounds are purely integers, round and cast
            # (e.g. n_estimators, max_depth)
            if isinstance(lower, int) and isinstance(upper, int):
                val = int(round(val))
            hyperparams[key] = val
            
        return feature_mask, hyperparams

    def _evaluate(self, solution):
        """
        Decode solution and call the external fitness_function.
        Returns the weighted F1-score (higher is better).
        """
        feature_mask, hyperparams = self._decode_solution(solution)
        
        # Create a hashable cache key from decoded features and hyperparams
        mask_tuple = tuple(feature_mask.tolist())
        hp_tuple = tuple(sorted(hyperparams.items()))
        cache_key = (mask_tuple, hp_tuple)
        
        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]
        
        score = fitness_function(
            model_type=self.model_type,
            X_train=self.X_sub,
            y_train=self.y_sub,
            feature_mask=feature_mask,
            hyperparams=hyperparams,
            class_weights=self.class_weights,
            cfg=self.cfg,
            n_jobs=1  # Single-threaded during search to prevent OOM from worker forking
        )
        self._fitness_cache[cache_key] = score
        return score

    # def _has_converged(self, convergence_history):
    #     """
    #     Check if the search has converged (no improvement for `patience` generations).

    #     Parameters:
    #         convergence_history (list): Best fitness per generation.

    #     Returns:
    #         bool: True if the last `convergence_patience` entries are identical.
    #     """
    #     p = self.convergence_patience
    #     if len(convergence_history) < p:
    #         return False
    #     recent = convergence_history[-p:]
    #     return all(abs(recent[i] - recent[0]) < 1e-6 for i in range(1, len(recent)))
                
    #             # Code to all to all metaheuristics if implementing early stopping/convergence
    #             if self._has_converged(convergence_history):
    #                 print(f"\n  Converged at generation {gen + 1} (no improvement for {self.convergence_patience} gens)")
    #                 break

    def run(self):
        """
        To be implemented by subclasses.
        Must return a dict containing:
        {
            "best_solution": [...]
            "best_feature_mask": [...]
            "best_hyperparams": {...}
            "best_fitness": float
            "convergence_history": [history of best fitness per gen]
            "runtime": float
        }
        """
        raise NotImplementedError("run() must be implemented by the subclass.")
