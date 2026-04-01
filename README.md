# CS3244 Group Project: Market Liquidity Filter

This repository handles the initial data pipeline, filtering our raw dataset of 8,400 US stocks down to the top 1,000 most liquid assets based on Average Daily Dollar Volume (ADDV). This ensures our trading models are trained on practically viable assets without extreme slippage.

## Local Setup Instructions

1. **Clone the repository:**
   `git clone <paste-repo-url-here>`

2. **Set up the virtual environment:**
   `python3 -m venv venv`
   `source venv/bin/activate` (Mac) OR `venv\Scripts\activate` (Windows)

3. **Install dependencies:**
   `pip install -r requirements.txt`

## How to Generate the Data

1. Download the "Price Volume Data for All US Stocks & ETFs" dataset from Kaggle.
2. Extract the `Stocks` folder into the root of this project.
3. Run `python filter_stocks.py` to calculate ADDV and generate `top_1000_liquid_stocks.csv`.
4. Run `python isolate_top_1000.py` to extract those specific 1,000 `.txt` files into a clean `Top1000_Stocks` directory for our models to use.

## ARMA Workflow (Fixed + Tuning)

This project now supports an expanding-window ARMA workflow with explicit train/validation/test splits.

### 1) Build Pair Datasets

Run the pair dataset builder first so each window folder contains:
- `train_pair_dataset.csv`
- `val_pair_dataset.csv` (expanding folds)
- `test_pair_dataset.csv` (holdout)

Command:
```bash
python -m src.models.pair_dataset_builder
```

### 2) Fixed-Parameter ARMA (Baseline Execution)

Run ARMA with fixed `(p, q)` across windows/pairs:
```bash
python -m src.models.arma
```

Useful options:
```bash
python -m src.models.arma --window 2010_2012 --p 2 --q 1 --eval_split both
python -m src.models.arma --window 2010_2012 --pair aapl-msft --spread_col spread_kalman
```

Default roots:
- input: `data/processed/pair_datasets`
- output: `data/processed/arma_outputs`

### 3) ARMA Hyperparameter Tuning

Tune `(p, q)` per window using validation RMSE as the primary criterion:
```bash
python -m src.models.arma_tuning
```

Useful options:
```bash
python -m src.models.arma_tuning --window 2010_2012 --p_values 1,2,3 --q_values 1,2,3
python -m src.models.arma_tuning --pair aapl-msft --spread_col spread_kalman
```

Default output:
- `data/processed/arma_tuning_outputs`

Produced tuning files:
- `all_validation_results.csv`
- `best_validation_results.csv`
- `test_results.csv` (when test split exists)

### 4) Visualization Notebook

Open and run:
- `tests/visualize_arma_results.ipynb`

The notebook:
- keeps validation and test summaries separate
- ranks by RMSE (MAE secondary)
- plots inline for report writing
- optionally saves report artifacts to `outputs/arma_report`
