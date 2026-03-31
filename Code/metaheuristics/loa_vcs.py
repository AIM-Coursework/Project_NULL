"""
LOA-VCS Hybrid Algorithm
========================
Combines Lion Optimisation Algorithm (LOA) exploration with
Virus Colony Search (VCS) exploitation.
- First half of generations: LOA phase (exploration)
- Second half of generations: VCS phase (exploitation)
"""

import time
import gc
import numpy as np
from tqdm import tqdm
from . import MetaheuristicBase

class HybridLOAVCS(MetaheuristicBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # LOA parameters
        self.pride_ratio = 0.8
        self.roaming_step = 0.1
        
        # VCS parameters
        self.immune_rate = 0.2

    def run(self):
        start_time = time.time()
        
        pride_size = int(self.pop_size * self.pride_ratio)
        
        # Initialize population
        positions = np.random.rand(self.pop_size, self.solution_dim)
        fitnesses = np.zeros(self.pop_size)
        
        gbest_fitness = -np.inf
        gbest_position = None
        
        # Evaluate initial population
        print(f"\nEvaluating initial population (size: {self.pop_size})...")
        for i in tqdm(range(self.pop_size), desc="Initial Population"):
            fit = self._evaluate(positions[i])
            fitnesses[i] = fit
            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
                
        convergence_history = []
        print("\nStarting LOA-VCS Hybrid Algorithm...")
        
        pbar = tqdm(range(self.max_generations), desc="LOA-VCS Generations")
        for gen in pbar:
            gen_start = time.time()
            new_positions = np.zeros_like(positions)
            
            # Sort population
            sorted_idx = np.argsort(fitnesses)[::-1]
            positions = positions[sorted_idx]
            fitnesses = fitnesses[sorted_idx]
            
            # Elitism: keep best
            new_positions[0] = positions[0].copy()
            
            if gen < self.max_generations // 2:
                # ==========================
                # LOA Phase (Exploration)
                # ==========================
                current_roaming_step = self.roaming_step * (1.0 - (gen / self.max_generations))
                
                # Pride
                for i in range(1, pride_size):
                    step = np.random.rand(self.solution_dim) * (gbest_position - positions[i])
                    noise = np.random.normal(0, current_roaming_step, self.solution_dim)
                    new_positions[i] = np.clip(positions[i] + step + noise, 0.0, 1.0)
                    
                # Nomads
                for i in range(pride_size, self.pop_size):
                    new_positions[i] = np.random.rand(self.solution_dim)
            else:
                # ==========================
                # VCS Phase (Exploitation)
                # ==========================
                sigma = 1.0 - (gen / self.max_generations)
                
                # Diffusion and Infection
                for i in range(1, self.pop_size):
                    # Infection (move to best)
                    if np.random.rand() < 0.5:
                        step = np.random.rand(self.solution_dim) * (gbest_position - positions[i])
                        new_positions[i] = np.clip(positions[i] + step, 0.0, 1.0)
                    # Diffusion (Gaussian noise)
                    else:
                        noise = np.random.normal(0, sigma * 0.1, self.solution_dim)
                        new_positions[i] = np.clip(positions[i] + noise, 0.0, 1.0)
                
                # Apply VCS immune response directly here
                num_immune = int(self.pop_size * self.immune_rate)
                # The weakest elements will be at the end of the sorted positions array 
                # (but they haven't been evaluated yet for the current gen). 
                # We will just replace them blindly for next evaluation.
                for i in range(self.pop_size - num_immune, self.pop_size):
                    new_positions[i] = np.random.rand(self.solution_dim)
                    
            positions = new_positions
            
            # Evaluate new population
            for i in range(1, self.pop_size):
                fit = self._evaluate(positions[i])
                fitnesses[i] = fit
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()
                    
            # Explicit garbage collection to prevent joblib OOMs 
            gc.collect()
            
            convergence_history.append(gbest_fitness)
            pbar.set_postfix(best_f1=f"{gbest_fitness:.4f}")
            
        runtime = time.time() - start_time
        feature_mask, hyperparams = self._decode_solution(gbest_position)
        
        return {
            "algorithm": "LOA-VCS",
            "best_solution": gbest_position.tolist(),
            "best_feature_mask": feature_mask.tolist(),
            "best_hyperparams": hyperparams,
            "best_fitness": gbest_fitness,
            "convergence_history": convergence_history,
            "runtime": runtime
        }
