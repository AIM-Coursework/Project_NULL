"""
Particle Swarm Optimisation (PSO)
=================================
Implements continuous Particle Swarm Optimisation for feature selection and hyperparameter tuning.
- v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
- x = x + v (clipped to [0, 1])
- v is clamped to [-0.5, 0.5] to prevent explosion
- w decays linearly from 0.9 to 0.4
"""

import time
import numpy as np
from tqdm import tqdm
from . import MetaheuristicBase

class ParticleSwarmOptimisation(MetaheuristicBase):
    def __init__(self, *args, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, v_max=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max

    def run(self):
        start_time = time.time()
        
        # Initialize swarm
        positions = np.random.rand(self.pop_size, self.solution_dim)
        # Velocities in [-v_max, v_max]
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.solution_dim))
        
        pbest_positions = positions.copy()
        pbest_fitnesses = np.zeros(self.pop_size)
        
        gbest_position = None
        gbest_fitness = -np.inf
        
        # Evaluate initial swarm
        print(f"\nEvaluating initial swarm (size: {self.pop_size})...")
        for i in tqdm(range(self.pop_size), desc="Initial Swarm"):
            fit = self._evaluate(positions[i])
            pbest_fitnesses[i] = fit
            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
                
        convergence_history = []
        print("\nStarting Particle Swarm Optimisation...")
        
        pbar = tqdm(range(self.max_generations), desc="PSO Generations")
        for gen in pbar:
            gen_start = time.time()
            
            # Linear inertia weight decay
            w = self.w_max - (self.w_max - self.w_min) * (gen / self.max_generations)
            
            for i in range(self.pop_size):
                r1 = np.random.rand(self.solution_dim)
                r2 = np.random.rand(self.solution_dim)
                
                # Update velocity
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social    = self.c2 * r2 * (gbest_position - positions[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0.0, 1.0)
                
                # Evaluate new position
                fit = self._evaluate(positions[i])
                
                # Update personal best
                if fit > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if fit > gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()
                        
            convergence_history.append(gbest_fitness)
            pbar.set_postfix(best_f1=f"{gbest_fitness:.4f}")
            
        runtime = time.time() - start_time
        feature_mask, hyperparams = self._decode_solution(gbest_position)
        
        return {
            "algorithm": "PSO",
            "best_solution": gbest_position.tolist(),
            "best_feature_mask": feature_mask.tolist(),
            "best_hyperparams": hyperparams,
            "best_fitness": gbest_fitness,
            "convergence_history": convergence_history,
            "runtime": runtime
        }
