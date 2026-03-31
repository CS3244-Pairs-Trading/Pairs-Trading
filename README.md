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
5. Run `eda_before.ipynb` to load full EDA on raw cleaned prices: Plots 1–6, ADF stationarity check, outlier investigation.
6. Run `handle_outliers.py` to handle outliers (suspicious rows, genuine extreme rows) in `data/interim/prices_clean.csv` .
7. Run `eda_after.ipynb` to re-run EDA plots affected by outlier handling (plots 1, 3, 4, 5, ADF) and confirm outlier handling worked as expected.