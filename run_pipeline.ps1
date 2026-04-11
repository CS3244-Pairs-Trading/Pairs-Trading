# To get help, run: .\run_pipeline.ps1 -? or Get-Help .\run_pipeline.ps1 or .\run_pipeline.ps1 -h or .\run_pipeline.ps1 -help

<#
.SYNOPSIS
    Runs the Pairs Trading Pipeline.
.EXAMPLE
    .\run_pipeline.ps1 -horizon 5 -SkipPrep
#>

# Configuration & Defaults
param (
    [Parameter(Position=0)]
    [int]$horizon = 10,

    [Parameter(Position=1)]
    [int]$top_k = 20,

    [Parameter(Position=2)]
    [string]$model = "all",

    [switch]$SkipDownload,
    [switch]$SkipPrep,
    [switch]$SkipPairDiscovery,
    [switch]$SkipPairSelection,
    [switch]$SkipDatasetBuilding,
    [switch]$SkipModelTraining,
    [switch]$SkipBacktesting,

    [Alias("h", "help")]
    [switch]$ShowHelp
)

# Valid Models List
$validModels = @("ou", "arma", "linear", "xgboost", "lstm", "all")

# Usage / Help Logic
if ($ShowHelp) {
    Write-Host "Usage: .\run_pipeline.ps1 -horizon <int> -top_k <int> -model <string> [-SkipDownload] [-SkipPrep] [-SkipPairDiscovery] [-SkipPairSelection] [-SkipDatasetBuilding] [-SkipModelTraining] [-SkipBacktesting]" -ForegroundColor Cyan
    Write-Host "`nOptions:"
    Write-Host "  -horizon              Prediction horizon (default: 10)"
    Write-Host "  -top_k                Number of top pairs to select (default: 20)"
    Write-Host "  -model                Model: ou, arma, linear, xgboost, lstm, all (default: all)"
    Write-Host "  -SkipDownload         Skip checking/downloading the Kaggle raw data"
    Write-Host "  -SkipPrep             Skip Step 1 (python prepare_data.py)"
    Write-Host "  -SkipPairDiscovery    Skip Step 2 (PCA -> OPTICS -> Ranking)"
    Write-Host "  -SkipPairSelection    Skip Step 3 (Pair Selection)"
    Write-Host "  -SkipDatasetBuilding  Skip Step 4 (Dataset Building)"
    Write-Host "  -SkipModelTraining    Skip Step 5 (Model Training & Prediction)"
    Write-Host "  -SkipBacktesting      Skip Step 6 (Backtesting Engine)"
    exit
}

# Validate Model Input
if ($validModels -notcontains $model) {
    Write-Error "Invalid model '$model'. Choose from: $($validModels -join ', ')"
    exit 1
}

$DATA_DIR = "data/raw"
$DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
$py = "python"

Write-Host "-------------------------------------------------------" -ForegroundColor Yellow
Write-Host "PHASE: Initialization"
Write-Host "Config: Horizon=$horizon, Top_K=$top_k, Model=$model"
Write-Host "-------------------------------------------------------" -ForegroundColor Yellow

# --- Step 0: Data Acquisition ---
if ($SkipDownload) {
    Write-Host "==> [0/6] SkipDownload flag detected. Skipping data check." -ForegroundColor Gray
} else {
    Write-Host "==> [0/6] Checking Data Source..." -ForegroundColor Green
    if (Test-Path $DATA_DIR -PathType Container -and (Get-ChildItem $DATA_DIR).Count -gt 0) {
        Write-Host "Data directory '$DATA_DIR' exists and is not empty. Skipping download."
    } else {
        Write-Host "Data not found. Downloading..."
        New-Item -ItemType Directory -Force -Path $DATA_DIR
        Invoke-WebRequest -Uri $DATA_URL -OutFile "$DATA_DIR\archive.zip"
        Write-Host "Unzipping data..."
        Expand-Archive -Path "$DATA_DIR\archive.zip" -DestinationPath $DATA_DIR -Force
        Remove-Item "$DATA_DIR\archive.zip"
    }
}

# --- Step 1: Data Preparation ---
if ($SkipPrep) {
    Write-Host "==> [1/6] SkipPrep flag detected. Skipping preparation." -ForegroundColor Gray
} else {
    Write-Host "==> [1/6] Data Preparation" -ForegroundColor Green
    & $py prepare_data.py
}

# --- Steps 2-4: Core Pipeline ---
if ($SkipPairDiscovery) {
    Write-Host "==> [2/6] SkipPairDiscovery flag detected. Skipping pair discovery." -ForegroundColor Gray
} else {
    Write-Host "==> [2/6] Pair Discovery (PCA -> OPTICS -> Ranking)" -ForegroundColor Green
    & $py -m src.clustering.pca
    & $py -m src.clustering.optics
    & $py -m src.pairs_discovery.rank_pairs
}

if ($SkipPairSelection) {
    Write-Host "==> [3/6] SkipPairSelection flag detected. Skipping pair selection." -ForegroundColor Gray
} else {
    Write-Host "==> [3/6] Pair Selection" -ForegroundColor Green
    & $py -m src.pairs_discovery.pairs_selection --top_k $top_k
}

if ($SkipDatasetBuilding) {
    Write-Host "==> [4/6] SkipDatasetBuilding flag detected. Skipping dataset building." -ForegroundColor Gray
} else {
    Write-Host "==> [4/6] Dataset Building" -ForegroundColor Green
    & $py -m src.models.pair_dataset_builder
}

# --- Step 5: Model Training & Prediction ---
if ($SkipModelTraining) {
    Write-Host "==> [5/6] SkipModelTraining flag detected. Skipping model training." -ForegroundColor Gray
} else {
    Write-Host "==> [5/6] Model Training & Prediction ($model)" -ForegroundColor Green
    switch ($model) {
        "ou"      { & $py -m src.models.ou }
        "arma"    { 
            & $py -m src.models.arma_tuning --spread_col spread_ols --horizon $horizon --eval_split val
            & $py -m src.models.arma_tuning --spread_col spread_kalman --horizon $horizon --eval_split val
        }
        "linear"  { & $py -m src.models.linear_regression }
        "xgboost" { & $py -m src.models.xgboost_model }
        "lstm"    { & $py -m src.models.lstm }
        "all"     {
            & $py -m src.models.ou
            & $py -m src.models.arma_tuning --spread_col spread_ols --horizon $horizon --eval_split val
            & $py -m src.models.linear_regression
            & $py -m src.models.xgboost_model
            & $py -m src.models.lstm
        }
    }
}

# --- Step 6: Backtesting ---
if ($SkipBacktesting) {
    Write-Host "==> [6/6] SkipBacktesting flag detected. Skipping backtesting." -ForegroundColor Gray
} else {
    Write-Host "==> [6/6] Backtesting Engine" -ForegroundColor Green
    & $py -m src.backtest.backtest_engine
    & $py -m src.backtest.backtest_engine --holdout
}

Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host "SUCCESS: Pipeline completed."
Write-Host "-------------------------------------------------------" -ForegroundColor Cyan