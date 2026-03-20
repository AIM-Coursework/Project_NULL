"""
Genetic Algorithm (GA)
======================
Implements a real-coded Genetic Algorithm for feature selection and hyperparameter tuning.
- Selection: Tournament (k=3)
- Crossover: Two-point (rate 0.8)
- Mutation: Random reset (rate 1/n_dims)
- Elitism: Top 2
"""

import time
import numpy as np
from tqdm import tqdm
from . import MetaheuristicBase

class GeneticAlgorithm(MetaheuristicBase):
    def __init__(self, *args, crossover_rate=0.8, tournament_size=3, elitism_count=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.mutation_rate = 1.0 / self.solution_dim

    def _tournament_selection(self, population, fitnesses):
        # Select k random individuals
        idx = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        # Return the one with the best fitness
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            # Two-point crossover
            pt1, pt2 = np.sort(np.random.choice(self.solution_dim, 2, replace=False))
            child1, child2 = parent1.copy(), parent2.copy()
            
            # Swap the middle segment
            child1[pt1:pt2], child2[pt1:pt2] = parent2[pt1:pt2].copy(), parent1[pt1:pt2].copy()
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, child):
        # Random reset mutation for continuous genes [0, 1]
        mask = np.random.rand(self.solution_dim) < self.mutation_rate
        child[mask] = np.random.rand(np.sum(mask))
        return child

    def run(self):
        start_time = time.time()
        
        # Initialize population randomly
        population = np.random.rand(self.pop_size, self.solution_dim)
        fitnesses = np.zeros(self.pop_size)
        
        # Evaluate initial population
        print(f"\nEvaluating initial population (size: {self.pop_size})...")
        for i in tqdm(range(self.pop_size), desc="Initial Population"):
            fitnesses[i] = self._evaluate(population[i])
            
        gbest_fitness = -np.inf
        gbest_solution = None
        convergence_history = []
        
        print("\nStarting Genetic Algorithm...")
        
        pbar = tqdm(range(self.max_generations), desc="GA Generations")
        for gen in pbar:
            gen_start = time.time()
            
            # Find best in current generation
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > gbest_fitness:
                gbest_fitness = fitnesses[current_best_idx]
                gbest_solution = population[current_best_idx].copy()
                
            convergence_history.append(gbest_fitness)
            
            # Create next generation
            new_population = np.zeros_like(population)
            
            # Elitism
            sorted_idx = np.argsort(fitnesses)[::-1]
            for i in range(self.elitism_count):
                new_population[i] = population[sorted_idx[i]].copy()
                
            # Generate offspring
            for i in range(self.elitism_count, self.pop_size, 2):
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                
                c1, c2 = self._crossover(p1, p2)
                
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population[i] = c1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = c2
                    
            population = new_population
            
            # Evaluate new population (skip elites to save time)
            new_fitnesses = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                if i < self.elitism_count:
                    # Elites keep their fitness
                    new_fitnesses[i] = fitnesses[sorted_idx[i]]
                else:
                    new_fitnesses[i] = self._evaluate(population[i])
                    
            fitnesses = new_fitnesses
            
            pbar.set_postfix(best_f1=f"{gbest_fitness:.4f}")
            
        runtime = time.time() - start_time
        feature_mask, hyperparams = self._decode_solution(gbest_solution)
        
        return {
            "algorithm": "GA",
            "best_solution": gbest_solution.tolist(),
            "best_feature_mask": feature_mask.tolist(),
            "best_hyperparams": hyperparams,
            "best_fitness": gbest_fitness,
            "convergence_history": convergence_history,
            "runtime": runtime
        }
