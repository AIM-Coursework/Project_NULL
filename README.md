# Project_NULL -  IDS Optimisation Framework

> **Task:** Intrusion Detection System (IDS) Optimisation using Metaheuristic Algorithms.  
> **Dataset:** CICIDS2017 (Canadian Institute for Cybersecurity) — Tabular Flow Features, CSV Files.  
> **Classification:** Binary (Normal vs Attack).


## Project Directory Structure


```text
Project_NULL/
├── Code/
│   ├── preprocessing.py        # Run 1st
│   ├── base_model.py           # XGBoost & RF Baseline Models
│   ├── integration.py          # Run 2nd
│   ├── evaluation.py           # Run 3rd & Final!
│   └── metaheuristics/        
│       ├── __init__.py         # MetaheuristicBase Class (shared mechanics)
│       ├── ga.py               # Genetic Algorithm
│       ├── pso.py              # Particle Swarm Optimisation
│       ├── loa.py              # Lion Optimisation Algorithm
│       ├── vcs.py              # Virus Colony Search
│       └── loa_vcs.py          # Hybrid LOA-VCS
│
├── requirements.txt            # Used for virtual environments & missing modules
├── MachineLearningCSV.zip      # Dataset
└── README.md                   # Project overview (this file)
```
- The project code is extensively decoupled to ensure that data preparation, model training, metaheuristic search, and final evaluation operate completely independently and are modular.

---

## Pipeline Sequence & Directions

Running an evaluation requires executing the modules sequentially. Each step produces artefacts that the subsequent step demands, follow the steps in this order:

### Step 1: Environment Setup
You can set up the required dependencies using either a virtual environment with `requirements.txt`, or by installing the packages directly.

**Option A: Virtual Environment (Recommended)**

```
# Create a virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate

# Install prerequisites
pip install -r requirements.txt

# If it still doesn't work, install manually while in venv using pip install [missing module].
```

**Option B: Direct Pip Install**

- Another option is to directly download the modules :D

    ```
    pip install numpy pandas scikit-learn xgboost tqdm matplotlib seaborn
    ```

### Step 2: Preprocessing
```
python Code/preprocessing.py
```
* **File Role:** 
        
    Applies all the preprocessing steps after reading the raw CSV drops:
    
    - Cleans NaNs, Infinities, Duplicates & Missing labels. 
    - Binary label encoding for categorical variables (`Normal=0, Attack=1`).
    - Feature filtering via dropping low-variance and highly correlated features.
    - Separating the clean data into a 70/15/15 stratified test split.
    - scales numerical values to `[0,1]` using `MinMaxScaler`.
    - Imposes Inverse frequency class weightings to prioritize attacks.

* **Output:** Cleaned datasets and configurations are dumped safely into `Code/processed_data/`.

### Step 3: Metaheuristic Optimisation & Training
```
python Code/integration.py
```
* **File Role:** 

    Provides a command-line interface which allows you to select the base model (RF or XGBoost) and the Metaheuristic to optimise it. It executes the algorithm (using a rapid 5% subsample for fitness approximations), finds the perfect parameters, and then trains the finalized model against the *entire* 1.7M dataset.  
* **Output:** Dumps the final model state, predictions (`predictions.npz`), and execution times inside `Code/results/{RUN_ID}/`.

### Step 4: Evaluation & Visualisation
```
python Code/evaluation.py
```
* **File Role:** 

    Sweeps inside the `results/` directory, extracts actual `y_test` metrics, and dynamically computes tables scoring `Accuracy`, `Precision`, `F1-score` across Binary/Macro/Weighted classes alongside `FPR`, `Training Time`, and `Search Time`.
* **Output:** Generates `evaluation_reports/all_results_summary.csv` and visual overlays (Confusion Matrices and Bar plots).


