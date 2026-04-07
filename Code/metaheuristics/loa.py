"""
Lion Optimisation Algorithm (LOA)
=================================
Implements the Lion Optimisation Algorithm.
- 80% Pride (exploitation / hunting around the best solution)
- 20% Nomad (exploration / random roaming)
"""

import time
import gc
import numpy as np
from tqdm import tqdm
from . import MetaheuristicBase

class LionOptimisationAlgorithm(MetaheuristicBase):
    def __init__(self, *args, pride_ratio=0.8, roaming_step=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pride_ratio = pride_ratio
        self.roaming_step = roaming_step

    def run(self):
        start_time = time.time()
        
        pride_size = int(self.pop_size * self.pride_ratio)
        nomad_size = self.pop_size - pride_size
        
        # Initialize population
        positions = np.random.rand(self.pop_size, self.solution_dim)
        fitnesses = np.zeros(self.pop_size)
        
        gbest_fitness = -np.inf
        gbest_position = None
        
        # Evaluate initial population
        print(f"\nEvaluating initial population (size: {self.pop_size})...")
        for i in tqdm(range(self.pop_size), desc="Initial Population", dynamic_ncols=True):
            fit = self._evaluate(positions[i])
            fitnesses[i] = fit
            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
                
        convergence_history = []
        print("\nStarting Lion Optimisation Algorithm (LOA)...")
        
        pbar = tqdm(range(self.max_generations), desc="LOA Generations", dynamic_ncols=True)
        for gen in pbar:
            gen_start = time.time()
            
            # Sort population to separate pride and nomads
            sorted_idx = np.argsort(fitnesses)[::-1]
            positions = positions[sorted_idx]
            fitnesses = fitnesses[sorted_idx]
            
            new_positions = np.zeros_like(positions)
            
            # Elitism: keep the absolute best (alpha)
            new_positions[0] = positions[0].copy()
            
            # Pride behavior (hunting / exploitation)
            current_roaming_step = self.roaming_step * (1.0 - (gen / self.max_generations))
            
            for i in range(1, pride_size):
                # Move towards the alpha lion (gbest)
                step = np.random.rand(self.solution_dim) * (gbest_position - positions[i])
                noise = np.random.normal(0, current_roaming_step, self.solution_dim)
                
                pos = positions[i] + step + noise
                new_positions[i] = np.clip(pos, 0.0, 1.0)
                
            # Nomad behavior (roaming / exploration)
            for i in range(pride_size, self.pop_size):
                # Random roaming in the search space
                new_positions[i] = np.random.rand(self.solution_dim)
                
            positions = new_positions
            
            # Evaluate new population
            # We skip index 0 as it's the elite alpha
            for i in range(1, self.pop_size):
                fit = self._evaluate(positions[i])
                fitnesses[i] = fit
                
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()
            
            # Explicit garbage collection to prevent Random Forest / Joblib memory leaks
            gc.collect()
                    
            convergence_history.append(gbest_fitness)
            pbar.set_postfix(best_f1=f"{gbest_fitness:.4f}")
            
        runtime = time.time() - start_time
        feature_mask, hyperparams = self._decode_solution(gbest_position)
        
        return {
            "algorithm": "LOA",
            "best_solution": gbest_position.tolist(),
            "best_feature_mask": feature_mask.tolist(),
            "best_hyperparams": hyperparams,
            "best_fitness": gbest_fitness,
            "convergence_history": convergence_history,
            "runtime": runtime
        }
