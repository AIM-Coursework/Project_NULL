"""
Virus Colony Search (VCS)
=========================
Implements the Virus Colony Search algorithm.
- Gaussian diffusion (exploration)
- Infection of best hosts (exploitation)
- Immune replacement of weakest (random replacement)
"""

import time
import numpy as np
from tqdm import tqdm
from . import MetaheuristicBase

class VirusColonySearch(MetaheuristicBase):
    def __init__(self, *args, infection_rate=0.5, immune_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.infection_rate = infection_rate
        self.immune_rate = immune_rate

    def run(self):
        start_time = time.time()
        
        # Initialize viruses
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
        print("\nStarting Virus Colony Search (VCS)...")
        
        pbar = tqdm(range(self.max_generations), desc="VCS Generations")
        for gen in pbar:
            gen_start = time.time()
            
            new_positions = np.zeros_like(positions)
            
            # Diffusion and Infection
            sigma = 1.0 - (gen / self.max_generations)  # Decreasing variance
            
            for i in range(self.pop_size):
                if i == 0:
                    # Elitism: keep the best virus intact
                    new_positions[i] = gbest_position.copy()
                    continue
                
                # Infection phase: move towards the best host cells
                if np.random.rand() < self.infection_rate:
                    step = np.random.rand(self.solution_dim) * (gbest_position - positions[i])
                    new_positions[i] = positions[i] + step
                # Diffusion phase: Gaussian random walk
                else:
                    noise = np.random.normal(0, sigma * 0.1, self.solution_dim)
                    new_positions[i] = positions[i] + noise
                    
                new_positions[i] = np.clip(new_positions[i], 0.0, 1.0)
                
            positions = new_positions
            
            # Evaluate after infection/diffusion
            for i in range(1, self.pop_size):
                fit = self._evaluate(positions[i])
                fitnesses[i] = fit
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()
                    
            # Immune Response (replacement of weakest viruses)
            num_immune = int(self.pop_size * self.immune_rate)
            if num_immune > 0:
                # Sort descending
                sorted_idx = np.argsort(fitnesses)[::-1]
                positions = positions[sorted_idx]
                fitnesses = fitnesses[sorted_idx]
                
                # Replace the weakest `num_immune` viruses
                for i in range(self.pop_size - num_immune, self.pop_size):
                    positions[i] = np.random.rand(self.solution_dim)
                    fit = self._evaluate(positions[i])
                    fitnesses[i] = fit
                    if fit > gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()
                        
            convergence_history.append(gbest_fitness)
            pbar.set_postfix(best_f1=f"{gbest_fitness:.4f}")
            
        runtime = time.time() - start_time
        feature_mask, hyperparams = self._decode_solution(gbest_position)
        
        return {
            "algorithm": "VCS",
            "best_solution": gbest_position.tolist(),
            "best_feature_mask": feature_mask.tolist(),
            "best_hyperparams": hyperparams,
            "best_fitness": gbest_fitness,
            "convergence_history": convergence_history,
            "runtime": runtime
        }
