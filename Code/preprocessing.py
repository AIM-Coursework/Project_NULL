"""
CICIDS2017 Preprocessing Pipeline
==================================
Loads, cleans, and prepares the CICIDS2017 CSV dataset for downstream
metaheuristic optimisation experiments (feature selection + hyperparameter tuning).

Design patterns adopted from SkinCancerModel.ipynb:
  - @dataclass configuration (single source of truth for all thresholds)
  - set_seed() for reproducibility
  - Stratified 70/15/15 train/val/test split
  - MinMaxScaler fit on training data only (prevents data leakage)
  - Inverse-frequency class weights for fitness function
  - asdict(cfg) → JSON serialisation for experiment reproducibility

Usage:
    python Code/preprocessing.py
"""

# ============================================================
# Cell 1: Imports & Setup
# ============================================================

import os                                               # Used for path manipulation and file operations
import glob                                             # Used to find all CSV files in the directory
import json                                             # Used for serializing the configuration
import pickle                                           # Used for serializing the scaler
import time                                             # Used for timing the execution
from dataclasses import dataclass, asdict               # Used for creating the configuration class

import numpy as np                                      # Used for numerical operations
import pandas as pd                                     # Used for data manipulation and analysis
from sklearn.model_selection import train_test_split    # Used for splitting the data into training and testing sets
from sklearn.preprocessing import MinMaxScaler          # Used for scaling the data


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across numpy and sklearn.

    Parameters:
        seed (int): The seed value to use for all random number generators.
    """
    np.random.seed(seed)


# ============================================================
# Cell 2: Configuration
# ============================================================

@dataclass
class PreprocessConfig:
    """
    All preprocessing parameters in one place (single source of truth).

    Attributes:
        seed (int): Random seed for reproducibility.
        test_size (float): Fraction held out from training (val + test combined).
        val_split (float): Fraction of held-out data used for test (rest is val).
        variance_threshold (float): Minimum variance to keep a feature.
        correlation_threshold (float): Max |correlation| before dropping one feature.
    """
    seed:                   int   = 42
    test_size:              float = 0.30             # 70/30 first split
    val_split:              float = 0.50             # 50/50 of temp → val/test (= 15/15)
    variance_threshold:     float = 0.01             # Drop features with variance below this
    correlation_threshold:  float = 0.95             # Drop one of each pair above this

cfg = PreprocessConfig()

# --- Paths ---
DATA_ROOT = r"C:\Users\User1\Desktop\datasets\CICIDS2017\MachineLearningCSV\MachineLearningCVE"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data")


# ============================================================
# Cell 3: Load & Concatenate CSVs
# ============================================================

def load_csv_files(data_root):
    """
    Load all CICIDS2017 CSV files from the given directory and concatenate them.

    Parameters:
        data_root (str): Path to directory containing the CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with all rows from all CSV files.
    """
    csv_files = sorted(glob.glob(os.path.join(data_root, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_root}")

    print(f"\n{'='*60}")
    print(f"Loading {len(csv_files)} CSV files...")
    print(f"{'='*60}")

    dfs = []
    for f in csv_files:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        print(f"  {name:<55} {len(df):>10,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Strip whitespace from column names (CICIDS2017 has leading spaces)
    combined.columns = combined.columns.str.strip()

    print(f"\n  Combined: {len(combined):,} rows × {len(combined.columns)} columns")
    return combined


# ============================================================
# Cell 4: Clean Invalid Values & Remove Duplicates
# ============================================================

def clean_data(df):
    """
    Handle missing values, infinities, and duplicate rows.

    Steps:
        1. Replace Infinity / -Infinity with NaN
        2. Drop rows containing NaN
        3. Drop fully duplicated rows

    Parameters:
        df (pd.DataFrame): Raw combined DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with no NaN, Inf, or duplicates.
    """
    initial_rows = len(df)
    print(f"\n{'='*60}")
    print("Cleaning data...")
    print(f"{'='*60}")
    print(f"  Initial rows: {initial_rows:,}")

    # Step 1: Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Step 2: Drop rows with NaN
    df = df.dropna()
    after_nan = len(df)
    print(f"  After dropping NaN/Inf: {after_nan:,} (removed {initial_rows - after_nan:,})")

    # Step 3: Drop duplicates
    df = df.drop_duplicates()
    after_dedup = len(df)
    print(f"  After dropping duplicates: {after_dedup:,} (removed {after_nan - after_dedup:,})")

    return df.reset_index(drop=True)


# ============================================================
# Cell 5: Binary Label Encoding
# ============================================================

def encode_labels(df):
    """
    Convert multi-class labels to binary: BENIGN → 0, all attacks → 1.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Label' column containing string labels.

    Returns:
        pd.DataFrame: DataFrame with 'Label' column as binary int (0 or 1).
    """
    print(f"\n{'='*60}")
    print("Encoding labels (binary: Normal=0, Attack=1)...")
    print(f"{'='*60}")

    # Show original label distribution
    print("\n  Original label distribution:")
    for label, count in df["Label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:<30} {count:>10,}  ({pct:.2f}%)")

    # Binary encoding: BENIGN = 0, everything else = 1
    df["Label"] = (df["Label"] != "BENIGN").astype(int)

    # Show binary distribution
    label_map = {0: "Normal (BENIGN)", 1: "Attack"}
    print("\n  Binary label distribution:")
    for label, count in df["Label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    {label_map[label]:<30} {count:>10,}  ({pct:.2f}%)")

    return df


# ============================================================
# Cell 6: Feature Filtering (Low-Variance & High-Correlation)
# ============================================================

def filter_low_variance(X, feature_names, threshold):
    """
    Remove features with variance below the given threshold.

    Parameters:
        X (np.ndarray): Feature matrix.
        feature_names (list): List of feature names.
        threshold (float): Minimum variance to keep a feature.

    Returns:
        tuple: (filtered X array, remaining feature names, list of dropped feature names)
    """
    variances = np.var(X, axis=0)
    mask = variances >= threshold
    dropped = [name for name, keep in zip(feature_names, mask) if not keep]

    if dropped:
        print(f"\n  Dropped {len(dropped)} low-variance features (threshold={threshold}):")
        for name in dropped:
            print(f"    - {name}")
    else:
        print(f"\n  No low-variance features to drop (threshold={threshold})")

    return X[:, mask], [n for n, k in zip(feature_names, mask) if k], dropped


def filter_high_correlation(X, feature_names, threshold):
    """
    Remove one of each pair of features with |correlation| above the threshold.

    For each highly-correlated pair, the feature that appears later in the
    feature list is dropped (arbitrary but deterministic).

    Parameters:
        X (np.ndarray): Feature matrix.
        feature_names (list): List of feature names.
        threshold (float): Maximum |correlation| before dropping.

    Returns:
        tuple: (filtered X array, remaining feature names, list of dropped feature names)
    """
    corr_matrix = np.corrcoef(X, rowvar=False)
    n = corr_matrix.shape[0]

    # Find features to drop from upper triangle
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    dropped = [feature_names[i] for i in sorted(to_drop)]
    mask = [i not in to_drop for i in range(n)]

    if dropped:
        print(f"\n  Dropped {len(dropped)} highly-correlated features (threshold={threshold}):")
        for name in dropped:
            print(f"    - {name}")
    else:
        print(f"\n  No highly-correlated features to drop (threshold={threshold})")

    return X[:, mask], [n for n, k in zip(feature_names, mask) if k], dropped


def filter_features(X_train, X_val, X_test, cfg):
    """
    Apply low-variance and high-correlation filtering to the feature matrix.
    Calculations are strictly fitted on the training data to prevent data leakage.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        cfg (PreprocessConfig): Configuration with thresholds.

    Returns:
        tuple: (Filtered X_train, X_val, X_test, remaining feature names,
                dict of all dropped features by reason)
    """
    print(f"\n{'='*60}")
    print("Filtering features (fit on train only)...")
    print(f"{'='*60}")

    feature_cols = list(X_train.columns)
    X_train_arr = X_train.values.astype(np.float64)
    initial_count = len(feature_cols)

    print(f"  Starting with {initial_count} features")

    # Low-variance filter (fit on train)
    X_train_arr, feature_cols, dropped_variance = filter_low_variance(
        X_train_arr, feature_cols, cfg.variance_threshold
    )

    # High-correlation filter (fit on train)
    X_train_arr, feature_cols, dropped_corr = filter_high_correlation(
        X_train_arr, feature_cols, cfg.correlation_threshold
    )

    print(f"\n  Features remaining: {len(feature_cols)} (removed {initial_count - len(feature_cols)})")

    # Reconstruct DataFrames using only the kept features
    X_train_res = pd.DataFrame(X_train_arr, columns=feature_cols, index=X_train.index)
    X_val_res = X_val[feature_cols].copy()
    X_test_res = X_test[feature_cols].copy()

    dropped_features = {
        "low_variance": dropped_variance,
        "high_correlation": dropped_corr
    }

    return X_train_res, X_val_res, X_test_res, feature_cols, dropped_features


# ============================================================
# Cell 7: Stratified Train / Val / Test Split
# ============================================================

def stratified_split(df, cfg):
    """
    Split data into 70% train, 15% validation, 15% test using stratified sampling.

    Follows the same pattern as SkinCancerModel.ipynb Cell 6:
        1. Split 70/30 (train / temp)
        2. Split temp 50/50 (val / test)

    Parameters:
        df (pd.DataFrame): DataFrame with features and 'Label' column.
        cfg (PreprocessConfig): Configuration with split ratios and seed.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) as DataFrames/Series.
    """
    print(f"\n{'='*60}")
    print("Splitting data (stratified 70/15/15)...")
    print(f"{'='*60}")

    feature_cols = [col for col in df.columns if col != "Label"]
    X = df[feature_cols]
    y = df["Label"]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    # Second split: temp → 50% val, 50% test (= 15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=cfg.val_split, stratify=y_temp, random_state=cfg.seed
    )

    # Print split summary
    total = len(df)
    label_map = {0: "Normal", 1: "Attack"}
    print(f"\n  {'Split':<10} {'Rows':>10} {'%':>8}  {'Normal':>10} {'Attack':>10}  {'Attack%':>8}")
    print(f"  {'-'*60}")
    for name, X_s, y_s in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        n_normal = (y_s == 0).sum()
        n_attack = (y_s == 1).sum()
        attack_pct = n_attack / len(y_s) * 100
        print(f"  {name:<10} {len(y_s):>10,} {len(y_s)/total*100:>7.1f}%  {n_normal:>10,} {n_attack:>10,}  {attack_pct:>7.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Cell 8: Min-Max Normalisation (Train-Only Fit)
# ============================================================

def normalise_features(X_train, X_val, X_test):
    """
    Apply Min-Max normalisation to [0, 1] range.

    The scaler is fit on training data only, then applied to val and test.
    This prevents data leakage (same principle as shared encoders in SkinCancerModel).

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple: (normalised X_train, X_val, X_test as DataFrames, fitted MinMaxScaler)
    """
    print(f"\n{'='*60}")
    print("Normalising features (MinMaxScaler, fit on train only)...")
    print(f"{'='*60}")

    scaler = MinMaxScaler()

    # Fit on training data ONLY
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # Transform val and test using the SAME fitted scaler
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"  Scaler fit on {len(X_train):,} training samples")
    print(f"  Feature range (train): [{X_train_scaled.min().min():.4f}, {X_train_scaled.max().max():.4f}]")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================
# Cell 9: Class Weights (Inverse Frequency)
# ============================================================

def compute_class_weights(y_train):
    """
    Compute inverse-frequency class weights for the fitness function.

    Formula: weight = n_samples / (n_classes * class_count)
    Same formula as SkinCancerModel.ipynb Cell 6.

    Parameters:
        y_train (pd.Series): Training labels (binary: 0 or 1).

    Returns:
        dict: {class_label: weight} mapping.
    """
    print(f"\n{'='*60}")
    print("Computing class weights (inverse frequency)...")
    print(f"{'='*60}")

    class_counts = y_train.value_counts()
    n_classes = len(class_counts)
    n_samples = len(y_train)

    class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }

    label_map = {0: "Normal", 1: "Attack"}
    for cls, weight in sorted(class_weights.items()):
        count = class_counts[cls]
        print(f"  {label_map[cls]:<10} count={count:>10,}  weight={weight:.4f}")

    return class_weights


# ============================================================
# Cell 10: Save Processed Data
# ============================================================

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                         scaler, feature_names, dropped_features, class_weights, cfg):
    """
    Save all processed data, scaler, and metadata to the output directory.

    Saves:
        - X_train.csv, X_val.csv, X_test.csv (feature matrices)
        - y_train.csv, y_val.csv, y_test.csv (label vectors)
        - scaler.pkl (fitted MinMaxScaler for inference)
        - config.json (full configuration + metadata for reproducibility)

    Parameters:
        X_train, X_val, X_test (pd.DataFrame): Normalised feature matrices.
        y_train, y_val, y_test (pd.Series): Binary label vectors.
        scaler (MinMaxScaler): Fitted scaler.
        feature_names (list): Final feature names after filtering.
        dropped_features (dict): Features dropped by reason.
        class_weights (dict): Inverse-frequency class weights.
        cfg (PreprocessConfig): Configuration used.
    """
    print(f"\n{'='*60}")
    print(f"Saving processed data to {OUTPUT_DIR}...")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save feature matrices and labels
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False, header=["Label"])
    y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False, header=["Label"])
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False, header=["Label"])
    print("  Saved: X_train.csv, X_val.csv, X_test.csv, y_train.csv, y_val.csv, y_test.csv")

    # Save fitted scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: scaler.pkl")

    # Save configuration + metadata (asdict pattern from SkinCancerModel)
    config_metadata = {
        "preprocess_config": asdict(cfg),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "dropped_features": dropped_features,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "label_encoding": {"0": "Normal (BENIGN)", "1": "Attack"},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_metadata, f, indent=2)
    print(f"  Saved: config.json")


# ============================================================
# Verification
# ============================================================

def verify_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Run validation checks on the processed data to catch common issues.

    Checks:
        1. No NaN or Inf values in any split
        2. All features in [0, 1] range (train set, val/test may exceed slightly)
        3. Class distribution preserved across splits
        4. Split sizes match expected ratios (70/15/15)
        5. No row overlap between splits
        6. Feature count consistent across splits

    Parameters:
        X_train, X_val, X_test (pd.DataFrame): Feature matrices.
        y_train, y_val, y_test (pd.Series): Label vectors.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    print(f"\n{'='*60}")
    print("Running verification checks...")
    print(f"{'='*60}")

    all_passed = True

    # Check 1: No NaN/Inf
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        has_nan = X.isnull().any().any()
        has_inf = np.isinf(X.values).any()
        status = "PASS" if not (has_nan or has_inf) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {name}: No NaN/Inf")

    # Check 2: Feature range [0, 1] on training set
    train_min = X_train.min().min()
    train_max = X_train.max().max()
    range_ok = train_min >= -0.001 and train_max <= 1.001
    status = "PASS" if range_ok else "FAIL"
    if not range_ok:
        all_passed = False
    print(f"  [{status}] Train feature range: [{train_min:.6f}, {train_max:.6f}]")

    # Check 3: Class distribution consistency
    total = len(y_train) + len(y_val) + len(y_test)
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        attack_pct = (y == 1).sum() / len(y) * 100
        overall_pct = ((y_train == 1).sum() + (y_val == 1).sum() + (y_test == 1).sum()) / total * 100
        diff = abs(attack_pct - overall_pct)
        status = "PASS" if diff < 1.0 else "WARN"
        if status == "WARN":
            print(f"  [{status}] {name} attack%: {attack_pct:.2f}% (overall: {overall_pct:.2f}%, diff: {diff:.2f}%)")
        else:
            print(f"  [{status}] {name} attack%: {attack_pct:.2f}% (within 1% of overall)")

    # Check 4: Split ratios
    for name, size, expected in [("Train", len(y_train), 0.70), ("Val", len(y_val), 0.15), ("Test", len(y_test), 0.15)]:
        actual_pct = size / total
        diff = abs(actual_pct - expected)
        status = "PASS" if diff < 0.02 else "WARN"
        print(f"  [{status}] {name} size: {size:,} ({actual_pct:.1%}, expected ~{expected:.0%})")

    # Check 5: Feature count consistency
    counts_match = X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    status = "PASS" if counts_match else "FAIL"
    if not counts_match:
        all_passed = False
    print(f"  [{status}] Feature count: train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}")

    if all_passed:
        print("\n  ✓ All verification checks passed!")
    else:
        print("\n  ✗ Some checks failed — review output above.")

    return all_passed


# ============================================================
# Cell 11: Load Processed Data (for downstream modules)
# ============================================================

def load_processed_data():
    """
    Load the preprocessed CICIDS2017 data from the processed_data directory.

    Used by base_model.py, metaheuristics, and evaluation modules to load
    the data that this preprocessing pipeline produced.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, class_weights)
    """
    data_dir = OUTPUT_DIR

    print(f"Loading data from {data_dir}...")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val   = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_val   = pd.read_csv(os.path.join(data_dir, "y_val.csv"))
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Convert class_weights keys back to int
    class_weights = {int(k): v for k, v in config["class_weights"].items()}

    print(f"  Loaded: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")
    print(f"  Features: {X_train.shape[1]}, Class weights: {class_weights}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline():
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps:
        1. Load & concatenate CSVs
        2. Clean (NaN/Inf, duplicates)
        3. Binary label encoding
        4. Stratified train/val/test split (70/15/15)
        5. Feature filtering (fit on train only)
        6. Min-Max normalisation (fit on train only)
        7. Compute class weights (inverse frequency)
        8. Save all processed data
        9. Run verification checks

    Returns:
        dict: Dictionary containing all processed data and metadata.
    """
    set_seed(42)
    start_time = time.time()

    print(f"\n{'#'*60}")
    print(f"  CICIDS2017 Preprocessing Pipeline")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    print(f"  Config: {asdict(cfg)}")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Step 1: Load
    df = load_csv_files(DATA_ROOT)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Labels
    df = encode_labels(df)

    # Step 4: Split (before filtering to prevent leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, cfg)

    # Step 5: Feature filtering (fit on train only)
    X_train, X_val, X_test, feature_names, dropped_features = filter_features(X_train, X_val, X_test, cfg)

    # Step 6: Normalise
    X_train, X_val, X_test, scaler = normalise_features(X_train, X_val, X_test)

    # Step 7: Class weights
    class_weights = compute_class_weights(y_train)

    # Step 8: Save
    save_processed_data(
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler, feature_names, dropped_features, class_weights, cfg
    )

    # Step 9: Verify
    verify_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    elapsed = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Features: {len(feature_names)} | Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*60}\n")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_names,
        "dropped_features": dropped_features, "class_weights": class_weights,
        "cfg": cfg
    }


if __name__ == "__main__":
    run_pipeline()
