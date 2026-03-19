"""
Base Model Module — RF & XGBoost
=======================================
Provides a model-agnostic interface for creating, training, and scoring
IDS classifiers. Supports switching between Random Forest and XGBoost
via a configurable model_type parameter.

Design:
  - Factory pattern: create_model() returns the correct sklearn-compatible estimator
  - train_and_predict(): single train/predict cycle returning raw outputs
  - fitness_function(): 5-fold stratified CV for metaheuristic evaluation
  - Metrics computation and evaluation display deferred to evaluation.py

Usage:
    from base_model import create_model, train_and_predict, fitness_function
"""

# ============================================================
# Cell 1: Imports & Seed
# ============================================================

import os                                               # Used for path manipulation and file operations
import sys                                              # Used for system-specific parameters and functions
import time                                             # Used for timing the execution
import numpy as np                                      # Used for numerical operations
import pandas as pd                                     # Used for data manipulation and analysis
from dataclasses import dataclass                       # Used for creating the configuration class

from sklearn.ensemble import RandomForestClassifier     # Used for Random Forest classification
from sklearn.model_selection import StratifiedKFold     # Used for stratified cross-validation
from sklearn.metrics import f1_score                    # Only metric needed internally (fitness function)
from xgboost import XGBClassifier                       # Used for XGBoost classification

# Import shared utilities from preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import set_seed, load_processed_data

set_seed(42)


# ============================================================
# Cell 2: Configuration
# ============================================================

@dataclass
class ModelConfig:
    """
    Base model configuration with model type switch.

    Attributes:
        model_type (str): Which model to use — "rf" or "xgboost".
        seed (int): Random seed for reproducibility.
        cv_folds (int): Number of folds for cross-validation in fitness function.
    """
    model_type  : str = "rf"
    seed        : int = 42
    cv_folds    : int = 5


# Default hyperparameters for each model type (sklearn defaults)
DEFAULT_HYPERPARAMS = {
    "rf": {
        "n_estimators"          : 100,
        "max_depth"             : None,           # Unlimited depth
        "min_samples_split"     : 2,
        "min_samples_leaf"      : 1,
        "max_features"          : "sqrt",
    },
    "xgboost": {
        "n_estimators"          : 100,
        "max_depth"             : 6,
        "learning_rate"         : 0.3,
        "subsample"             : 1.0,
        "colsample_bytree"      : 1.0,
        "gamma"                 : 0.0,
        "reg_alpha"             : 0.0,
        "reg_lambda"            : 1.0,
    },
}

# Hyperparameter search space bounds for each model (used by metaheuristics)
HYPERPARAM_BOUNDS = {
    "rf": {
        "n_estimators"          : (50, 500),
        "max_depth"             : (3, 50),
        "min_samples_split"     : (2, 20),
        "min_samples_leaf"      : (1, 10),
        "max_features"          : (0.1, 1.0),     # Fraction of features
    },
    "xgboost": {
        "n_estimators"          : (50, 500),
        "max_depth"             : (3, 15),
        "learning_rate"         : (0.01, 0.3),
        "subsample"             : (0.5, 1.0),
        "colsample_bytree"      : (0.3, 1.0),
        "gamma"                 : (0.0, 5.0),
        "reg_alpha"             : (0.0, 10.0),
        "reg_lambda"            : (0.0, 10.0),
    },
}

# Human-readable model names
MODEL_NAMES = {
    "rf"        : "Random Forest",
    "xgboost"   : "XGBoost",
}


# ============================================================
# Cell 3: Helpers
# ============================================================

def _apply_feature_mask(X, feature_mask):
    """
    Apply a boolean feature mask to select columns from a feature matrix.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        feature_mask (np.ndarray or None): Boolean mask. If None, all features are used.

    Returns:
        tuple: (selected feature array, number of selected features)
    """
    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=bool)
        if isinstance(X, pd.DataFrame):
            X_sel = X.iloc[:, feature_mask].values
        else:
            X_sel = X[:, feature_mask]
        return X_sel, int(feature_mask.sum())
    else:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return X_arr, X_arr.shape[1]


# ============================================================
# Cell 4: Model Factory
# ============================================================

def create_model(model_type, hyperparams=None, class_weights=None, seed=42):
    """
    Factory function that creates and returns a sklearn-compatible classifier.

    Parameters:
        model_type (str): One of "rf" or "xgboost".
        hyperparams (dict, optional): Model hyperparameters. Defaults to DEFAULT_HYPERPARAMS.
        class_weights (dict, optional): {class_label: weight} for imbalance handling.
        seed (int): Random seed.

    Returns:
        sklearn-compatible classifier instance.

    Raises:
        ValueError: If model_type is not one of "rf" or "xgboost".
    """
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS[model_type].copy()

    
    if model_type == "rf":
        # Random Forest — class_weight passed directly
        return RandomForestClassifier(
            n_estimators        = hyperparams.get("n_estimators", 100),
            max_depth           = hyperparams.get("max_depth", None),
            min_samples_split   = hyperparams.get("min_samples_split", 2),
            min_samples_leaf    = hyperparams.get("min_samples_leaf", 1),
            max_features        = hyperparams.get("max_features", "sqrt"),
            class_weight        = class_weights,
            random_state        = seed,
            n_jobs              = -1,               # Use all CPU cores
        )

    elif model_type == "xgboost":
        # XGBoost — uses scale_pos_weight for binary imbalance
        scale_pos_weight = 1.0
        if class_weights and 1 in class_weights and 0 in class_weights:
            scale_pos_weight = class_weights[1] / class_weights[0]

        return XGBClassifier(
            n_estimators        = hyperparams.get("n_estimators", 100),
            max_depth           = hyperparams.get("max_depth", 6),
            learning_rate       = hyperparams.get("learning_rate", 0.3),
            subsample           = hyperparams.get("subsample", 1.0),
            colsample_bytree    = hyperparams.get("colsample_bytree", 1.0),
            gamma               = hyperparams.get("gamma", 0.0),
            reg_alpha           = hyperparams.get("reg_alpha", 0.0),
            reg_lambda          = hyperparams.get("reg_lambda", 1.0),
            scale_pos_weight    = scale_pos_weight,
            random_state        = seed,
            n_jobs              = -1,
            eval_metric         = "logloss",
            verbosity           = 0,            # Suppress XGBoost warnings
        )

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from: 'rf', 'xgboost'.")


# ============================================================
# Cell 5: Train & Predict
# ============================================================

def train_and_predict(model_type, X_train, y_train, X_test,
                      feature_mask=None, hyperparams=None, class_weights=None, seed=42):
    """
    Train a model and return raw predictions (no metrics computation).

    Parameters:
        model_type (str): "rf" or "xgboost".
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        X_test (pd.DataFrame or np.ndarray): Test/validation features.
        feature_mask (np.ndarray, optional): Boolean mask for feature selection.
            If None, all features are used.
        hyperparams (dict, optional): Model hyperparameters. Defaults per model type.
        class_weights (dict, optional): {class_label: weight} for imbalance handling.
        seed (int): Random seed.

    Returns:
        dict: Raw outputs for downstream evaluation:
            - y_pred: predicted labels (np.ndarray)
            - y_proba: predicted probabilities for class 1 (np.ndarray or None)
            - model: trained model instance
            - training_time: seconds taken to train
            - n_features: number of features used
    """
    # Apply feature mask
    X_train_sel, n_features = _apply_feature_mask(X_train, feature_mask)
    X_test_sel, _           = _apply_feature_mask(X_test, feature_mask)

    y_train_arr = y_train.values.ravel() if isinstance(y_train, (pd.DataFrame, pd.Series)) else np.ravel(y_train)

    # Create and train model
    model = create_model(model_type, hyperparams, class_weights, seed)

    start_time = time.time()
    
    # Train the model
    model.fit(X_train_sel, y_train_arr)

    training_time = time.time() - start_time

    # Predict
    y_pred  = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "y_pred"        : y_pred,
        "y_proba"       : y_proba,
        "model"         : model,
        "training_time" : round(training_time, 2),
        "n_features"    : n_features,
    }


# ============================================================
# Cell 6: Fitness Function (5-Fold CV)
# ============================================================

def fitness_function(model_type, X_train, y_train, feature_mask=None,
                     hyperparams=None, class_weights=None, n_folds=5, seed=42):
    """
    Evaluate a candidate solution using stratified k-fold cross-validation.

    This is the function metaheuristics call to score each candidate.
    Uses the TRAINING set only — val/test are never seen.

    Parameters:
        model_type (str): "rf" or "xgboost".
        X_train (pd.DataFrame or np.ndarray): Training features (full training set).
        y_train (pd.Series or np.ndarray): Training labels.
        feature_mask (np.ndarray, optional): Boolean mask for feature selection.
        hyperparams (dict, optional): Model hyperparameters.
        class_weights (dict, optional): {class_label: weight}.
        n_folds (int): Number of CV folds.
        seed (int): Random seed.

    Returns:
        float: Mean weighted F1-score across all folds (higher = better).
    """
    # Apply feature mask (with early exit for empty mask)
    X_sel, n_features = _apply_feature_mask(X_train, feature_mask)
    if n_features == 0:
        return 0.0  # No features selected → worst fitness

    y = y_train.values.ravel() if isinstance(y_train, (pd.DataFrame, pd.Series)) else np.ravel(y_train)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_train_idx, fold_val_idx in skf.split(X_sel, y):
        X_fold_train, X_fold_val = X_sel[fold_train_idx], X_sel[fold_val_idx]
        y_fold_train, y_fold_val = y[fold_train_idx], y[fold_val_idx]

        model = create_model(model_type, hyperparams, class_weights, seed)
        model.fit(X_fold_train, y_fold_train)

        y_pred = model.predict(X_fold_val)
        score  = f1_score(y_fold_val, y_pred, average="weighted")
        fold_scores.append(score)

    return float(np.mean(fold_scores))
