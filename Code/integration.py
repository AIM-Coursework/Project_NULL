"""
Integration Module
==================
Main orchestrator CLI for running IDS metaheuristic experiments.
Connects preprocessing, base models, and metaheuristic searches.

Usage:
    python Code/integration.py
"""

import os
import sys
import time
import json
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import shared modules
from preprocessing import load_processed_data, set_seed
from base_model import train_and_predict, ModelConfig

# Import Metaheuristics
from metaheuristics.ga import GeneticAlgorithm
from metaheuristics.pso import ParticleSwarmOptimisation
from metaheuristics.loa import LionOptimisationAlgorithm
from metaheuristics.vcs import VirusColonySearch
from metaheuristics.loa_vcs import HybridLOAVCS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

METAHEURISTIC_MAP = {
    1: ("GA", GeneticAlgorithm),
    2: ("PSO", ParticleSwarmOptimisation),
    3: ("LOA", LionOptimisationAlgorithm),
    4: ("VCS", VirusColonySearch),
    5: ("LOA-VCS", HybridLOAVCS)
}

MODEL_MAP = {
    1: "rf",
    2: "xgboost"
}

def run_experiment(model_type, meta_choice, X_train, y_train, X_test, class_weights, cfg):
    """
    Run a specific experiment: metaheuristic search -> retrain full -> save results.
    If meta_choice == 0 (Baseline), skips search and trains with defaults + all features.
    """
    seed = cfg.seed
    set_seed(seed)
    
    n_features = X_train.shape[1]
    is_baseline = (meta_choice == 0)
    
    meta_name = "Baseline" if is_baseline else METAHEURISTIC_MAP[meta_choice][0]
    timestamp = datetime.now().strftime("%d%m%y_%H-%M")
    run_name = f"{model_type.upper()}_{meta_name}_{timestamp}"
    out_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {run_name}")
    print(f"{'='*70}")
    
    best_feature_mask = None
    best_hyperparams = cfg.default_hyperparams[model_type].copy()
    meta_results = {}
    
    # Run Metaheuristic Search if not baseline
    if not is_baseline:
        AlgorithmClass = METAHEURISTIC_MAP[meta_choice][1]
        bounds = cfg.hyperparam_bounds[model_type]
        
        optimizer = AlgorithmClass(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            class_weights=class_weights,
            n_features=n_features,
            hyperparam_bounds=bounds,
            cfg=cfg,
            subsample_ratio=0.05,   # Use 5% for fitness
            max_generations=15,    # Default as per project scope
            pop_size=20
        )
        
        meta_results = optimizer.run()
        best_feature_mask = meta_results["best_feature_mask"]
        best_hyperparams = meta_results["best_hyperparams"]
        
        print("\nSearch Complete.")
        print(f"Best Fitness (F1): {meta_results['best_fitness']:.4f}")
        print(f"Selected Features: {sum(best_feature_mask)} / {n_features}")
        
    print("\nTraining final model on FULL training set...")
    
    # Train on full data wrapped in a tqdm timer for consistency
    for _ in tqdm(range(1), desc=f"Final Training ({model_type.upper()})", dynamic_ncols=True):
        final_results = train_and_predict(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            feature_mask=best_feature_mask,
            hyperparams=best_hyperparams,
            class_weights=class_weights,
            seed=seed
        )
    
    # Save meta_results
    results_json = {
        "run_name": run_name,
        "model_type": model_type,
        "metaheuristic": meta_name,
        "n_features_selected": final_results["n_features"],
        "total_features": n_features,
        "best_hyperparams": best_hyperparams,
        "best_feature_mask": best_feature_mask if is_baseline else best_feature_mask,
        "metaheuristic_results": meta_results, # History, time, fitness
        "final_training_time": final_results["training_time"],
        "timestamp": timestamp
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
        
    # Save model
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(final_results["model"], f)
        
    # Save predictions
    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        y_pred=final_results["y_pred"],
        y_proba=final_results["y_proba"] if final_results["y_proba"] is not None else np.array([])
    )
    
    print(f"\nExperiment saved to: {out_dir}")
    print(f"{'='*70}\n")
    return out_dir


def main():
    print(f"\n{'='*70}")
    print(f"  IDS Metaheuristic Optimisation CLI")
    print(f"{'='*70}")

    # -------------------------------------------------------------------------------------
    
    print("\n[INFO] Loading processed dataset (>1.7M rows). This may take up to 30 seconds...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_processed_data()
    except FileNotFoundError:
        print("\n[ERROR] Processed data not found. Please run preprocessing.py first.\n")
        sys.exit(1)

    # -------------------------------------------------------------------------------------
        
    print("\nSelect base model:")
    print("  [1] Random Forest (RF)")
    print("  [2] XGBoost")
    
    try:
        model_choice = int(input("Enter choice [1-2]: "))
        if model_choice not in MODEL_MAP:
            raise ValueError
        model_type = MODEL_MAP[model_choice]
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid choice or aborted. Exiting.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------------------
        
    print("\nSelect metaheuristic algorithm:")
    print("  [0] Baseline (All features + Default params)")
    print("  [1] Genetic Algorithm (GA)")
    print("  [2] Particle Swarm Optimisation (PSO)")
    print("  [3] Lion Optimisation Algorithm (LOA)")
    print("  [4] Virus Colony Search (VCS)")
    print("  [5] LOA-VCS Hybrid")
    print("  [6] Run ALL (Baseline + 1 through 5)")
    
    try:
        meta_choice = int(input("Enter choice [0-6]: "))
        if meta_choice not in list(range(7)):
            raise ValueError
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid choice or aborted. Exiting.")
        sys.exit(1)
        
    cfg = ModelConfig(model_type=model_type, seed=42)
        
    # For now we evaluate on the Test set here just to save time since it's an end-to-end framework
    # In actual usage we should evaluate separately, but the prompt says 
    # train on full training dataset, so we pass X_test as eval.
    if meta_choice == 6:
        # Run all
        for choice in tqdm(range(0, 6), desc="Running All Metaheuristics", dynamic_ncols=True):
            run_experiment(model_type, choice, X_train, y_train, X_test, class_weights, cfg)
    else:
        run_experiment(model_type, meta_choice, X_train, y_train, X_test, class_weights, cfg)
    
    # -------------------------------------------------------------------------------------
        
    print("All tasks completed successfully!\n")

if __name__ == "__main__":
    main()
