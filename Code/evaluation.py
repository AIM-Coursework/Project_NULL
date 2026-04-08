"""
Evaluation & Visualisation Module
Evaluates the predictions of all IDS experiments, generates standard metrics,
and creates comparison visualisations (Confusion Matrices, Metric Plots).

Usage:
    python Code/evaluation.py
"""

# Python Standard Libraries
import os
import sys
import json
import glob

# External Libraries
import numpy as np                                      # For numerical operations  | Used to load predictions
import pandas as pd                                     # For data manipulation     | Used to create summary table
import matplotlib.pyplot as plt                         # For plotting              | Used to create plots
import seaborn as sns                                   # For enhanced plotting     | Used to create plots

# Scikit-learn Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Import shared functions to load test labels
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_processed_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
EVAL_OUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_reports")

def extract_metrics(y_true, y_pred, y_proba):
    """
    Computes key classification metrics.
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Binary classification expected: 0=Normal, 1=Attack
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Fallback if somehow perfectly predicted one class or multi-class
        TN, FP, FN, TP = 0, 0, 0, 0
        if cm.size == 1:
            if y_true[0] == 0:
                TN = cm[0, 0]
            else:
                TP = cm[0, 0]
    
    # Core sklearn metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted average accounting for class imbalance
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Binary metrics (pos_label=1 for Attack)
    precision_b = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall_b = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1_b = f1_score(y_true, y_pred, average="binary", zero_division=0)
    
    # Macro metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    auc = roc_auc_score(y_true, y_proba) if len(y_proba) > 0 and len(np.unique(y_true)) > 1 else np.nan
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    # Return metrics
    return {
        "Accuracy": accuracy,
        "Precision (Binary)": precision_b,
        "Recall (Binary)": recall_b,
        "F1 (Binary)": f1_b,
        "F1 (Weighted)": f1_w,
        "F1 (Macro)": f1_macro,
        "Precision (Weighted)": precision_w,
        "Recall (Weighted)": recall_w,
        "AUC": auc,
        "FPR": fpr,
        "CM": cm
    }

def plot_confusion_matrix(cm, labels, title, out_path):
    """Plots and saves a single confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_comparisons(df_results, out_dir):
    """Generates comparison bar plots for key metrics."""
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_to_plot = ["F1 (Weighted)", "F1 (Binary)", "Accuracy", "FPR", "n_features_selected", "final_training_time"]
    titles = [
        "Weighted F1-Score", 
        "Binary F1-Score", 
        "Accuracy", 
        "False Positive Rate", 
        "Number of Features Selected", 
        "Training Time (s)"
    ]
    
    # Check if there are any results
    if len(df_results) == 0:
        return
        
    # Plot each metric
    for metric, title in zip(metrics_to_plot, titles):
        if metric not in df_results.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x="metaheuristic", y=metric, hue="model_type", data=df_results)
        plt.title(f"Comparison of {title}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"compare_{metric}.png"), dpi=300)
        plt.close()

def load_run_results(run_dir, y_test):
    """Loads predictions and computes metrics for a single run directory."""
    results_json_path = os.path.join(run_dir, "results.json")
    preds_npz_path = os.path.join(run_dir, "predictions.npz")
    
    # Check if both files exist
    if not (os.path.exists(results_json_path) and os.path.exists(preds_npz_path)):
        return None
        
    # Load results
    with open(results_json_path, "r") as f:
        meta = json.load(f)
        
    # Load predictions
    preds = np.load(preds_npz_path)
    y_pred = preds["y_pred"]
    y_proba = preds["y_proba"] if "y_proba" in preds.files and len(preds["y_proba"]) > 0 else []
    
    # Check if lengths match
    if len(y_pred) != len(y_test):
        print(f"  [WARN] Length mismatch in {os.path.basename(run_dir)}. Expected {len(y_test)}, got {len(y_pred)}. Skipping.")
        return None
        
    # Extract metrics
    metrics = extract_metrics(y_test, y_pred, y_proba)
    
    # Create row
    row = {
        "run_name": meta.get("run_name", os.path.basename(run_dir)),
        "model_type": meta.get("model_type", "unknown").upper(),
        "metaheuristic": meta.get("metaheuristic", "unknown"),
        "n_features_selected": meta.get("n_features_selected", 0),
        "total_features": meta.get("total_features", 0),
        "final_training_time": meta.get("final_training_time", 0.0),
        "search_time": meta.get("metaheuristic_results", {}).get("runtime", 0.0) if isinstance(meta.get("metaheuristic_results"), dict) else 0.0,
    }
    
    # Calculate total time
    row["total_time"] = row["final_training_time"] + row["search_time"]
    
    # Add metrics to row
    for k, v in metrics.items():
        if k != "CM":
            row[k] = v
            
    return row, metrics["CM"]

def main():
    print(f"\n{'='*70}")
    print(f"  IDS System Evaluation & Visualisation Module")
    print(f"{'='*70}")

    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print("\nError: No results directory found. Run integration.py first to generate results.")
        sys.exit(1)
        
    print("\nLoading true test labels...")
    try:
        # Load preprocessed data; ignore unused splits
        _, _, _, _, _, y_test_df, _ = load_processed_data()
        y_test = y_test_df.values.ravel()
    except Exception as e:
        print(f"\n[ERROR] Failed to load processed data: {e}")
        sys.exit(1)
        
    # Create evaluation directory, if it exists already then retain it
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    
    # Get all run directories
    run_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    
    # Check if there are any runs
    if not run_dirs:
        print("\n[INFO] No experiment runs found in results directory.")
        sys.exit(0)
        
    print(f"Found {len(run_dirs)} experiment runs. Evaluating...\n")
    
    rows = []
    
    # Evaluate each run
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"  Evaluating: {run_name}")
        
        # Load results
        result = load_run_results(run_dir, y_test)
        if result:
            row, cm = result
            rows.append(row)
            cm_out_path = os.path.join(EVAL_OUT_DIR, f"cm_{run_name}.png")
            title = f"{row['model_type']} - {row['metaheuristic']}"
            plot_confusion_matrix(cm, labels=["Normal", "Attack"], title=title, out_path=cm_out_path)
            
    if not rows:
        print("\n[INFO] No valid results could be processed.")
        sys.exit(0)
        
    # Create DataFrame
    df_results = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(EVAL_OUT_DIR, "all_results_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV to: {csv_path}")
    
    # Print terminal table
    print("\n" + "="*150)
    print(f"Performance Summary")
    print("-" * 150)
    print(f"{'Model':<10} {'Metaheuristic':<15} {'Features':<10} {'Accuracy':<10} {'Prec(Bin)':<10} {'Rec(Bin)':<10} {'F1(Bin)':<10} {'F1(Wght)':<10} {'F1(Mac)':<10} {'FPR':<10} {'Search(s)':<12} {'Train(s)':<10} {'Total(s)':<12}")
    print("-" * 150)
    for _, row in df_results.iterrows():
        print(f"{row['model_type']:<10} "
              f"{row['metaheuristic']:<15} "
              f"{row['n_features_selected']:<10} "
              f"{row['Accuracy']:<10.4f} "
              f"{row.get('Precision (Binary)', 0):<10.4f} "
              f"{row.get('Recall (Binary)', 0):<10.4f} "
              f"{row.get('F1 (Binary)', 0):<10.4f} "
              f"{row.get('F1 (Weighted)', 0):<10.4f} "
              f"{row.get('F1 (Macro)', 0):<10.4f} "
              f"{row['FPR']:<10.4f} "
              f"{row.get('search_time', 0.0):<12.2f} "
              f"{row['final_training_time']:<10.2f} "
              f"{row.get('total_time', 0.0):<12.2f}")
    print("="*150 + "\n")
    
    # Plot comparisons
    plot_comparisons(df_results, EVAL_OUT_DIR)
    print(f"Generated comparison plots and confusion matrices in: {EVAL_OUT_DIR}\n")

if __name__ == "__main__":
    main()
