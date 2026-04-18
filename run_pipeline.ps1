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
    [switch]$SkipBestParamRuns,
    [switch]$SkipBacktesting,

    [Alias("h", "help")]
    [switch]$ShowHelp
)

function Measure-Time {
    param([string]$Name, [scriptblock]$Script)
    Write-Host "Starting: $Name at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    $start = Get-Date
    Invoke-Command -ScriptBlock $Script
    $end = Get-Date
    $duration = $end - $start
    Write-Host "Completed: $Name in $($duration.TotalSeconds) seconds." -ForegroundColor Green
}

# Valid Models List
$validModels = @("ou", "arma", "linear", "xgboost", "lstm", "lstm_encdec", "all")

# Usage / Help Logic
if ($ShowHelp) {
    Write-Host "Usage: .\run_pipeline.ps1 -horizon <int> -top_k <int> -model <string> [-SkipDownload] [-SkipPrep] [-SkipPairDiscovery] [-SkipPairSelection] [-SkipDatasetBuilding] [-SkipModelTraining] [-SkipBestParamRuns] [-SkipBacktesting]" -ForegroundColor Cyan
    Write-Host "`nOptions:"
    Write-Host "  -horizon              Prediction horizon (default: 10)"
    Write-Host "  -top_k                Number of top pairs to select (default: 20)"
    Write-Host "  -model                Model: ou, arma, linear, xgboost, lstm, lstm_encdec, all (default: all)"
    Write-Host "  -SkipDownload         Skip checking/downloading the Kaggle raw data"
    Write-Host "  -SkipPrep             Skip Step 1 (python prepare_data.py)"
    Write-Host "  -SkipPairDiscovery    Skip Step 2 (PCA -> OPTICS -> Ranking)"
    Write-Host "  -SkipPairSelection    Skip Step 3 (Pair Selection)"
    Write-Host "  -SkipDatasetBuilding  Skip Step 4 (Dataset Building)"
    Write-Host "  -SkipModelTraining    Skip Step 5 (Model Training & Prediction)"
    Write-Host "  -SkipBestParamRuns   Skip Step 6 (Frozen Best-Params Runs)"
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
$globalStartTime = Get-Date

Write-Host "-------------------------------------------------------" -ForegroundColor Yellow
Write-Host "PHASE: Initialization"
Write-Host "Config: Horizon=$horizon, Top_K=$top_k, Model=$model"
Write-Host "-------------------------------------------------------" -ForegroundColor Yellow

# --- Step 0: Data Acquisition ---
if ($SkipDownload) {
    Write-Host "==> [0/6] SkipDownload flag detected. Skipping data check." -ForegroundColor Gray
} else {
    Measure-Time -Name "Data Source Check & Download" -Script {
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
}

# --- Step 1: Data Preparation ---
if ($SkipPrep) {
    Write-Host "==> [1/7] SkipPrep flag detected. Skipping preparation." -ForegroundColor Gray
} else {
    Measure-Time -Name "Data Preparation" -Script {
        Write-Host "==> [1/7] Data Preparation" -ForegroundColor Green
        & $py prepare_data.py
    }
}

# --- Steps 2-4: Core Pipeline ---
if ($SkipPairDiscovery) {
    Write-Host "==> [2/7] SkipPairDiscovery flag detected. Skipping pair discovery." -ForegroundColor Gray
} else {
    Measure-Time -Name "Pair Discovery (PCA -> OPTICS -> Ranking)" -Script {
        Write-Host "==> [2/7] Pair Discovery (PCA -> OPTICS -> Ranking)" -ForegroundColor Green
        & $py -m src.clustering.pca
        & $py -m src.clustering.optics
        & $py -m src.pairs_discovery.rank_pairs
    }
}

if ($SkipPairSelection) {
    Write-Host "==> [3/7] SkipPairSelection flag detected. Skipping pair selection." -ForegroundColor Gray
} else {
    Measure-Time -Name "Pair Selection" -Script {
        Write-Host "==> [3/7] Pair Selection" -ForegroundColor Green
        & $py -m src.pairs_discovery.pairs_selection --top_k $top_k
    }
}

if ($SkipDatasetBuilding) {
    Write-Host "==> [4/7] SkipDatasetBuilding flag detected. Skipping dataset building." -ForegroundColor Gray
} else {
    Measure-Time -Name "Dataset Building" -Script {
        Write-Host "==> [4/7] Dataset Building" -ForegroundColor Green
        & $py -m src.models.pair_dataset_builder
    }
}

# --- Step 5: Model Training & Prediction ---
if ($SkipModelTraining) {
    Write-Host "==> [5/7] SkipModelTraining flag detected. Skipping model training." -ForegroundColor Gray
} else {
    Measure-Time -Name "Model Training & Prediction" -Script {
        Write-Host "==> [5/7] Model Training & Prediction ($model)" -ForegroundColor Green
        switch ($model) {
            "ou"      { & $py -m src.models.ou }
            "arma"    { 
                # & $py -m src.models.arma_tuning --spread_col spread_ols --horizon $horizon
                & $py -m src.models.arma_tuning --spread_col spread_kalman --horizon $horizon
            }
            "linear"  { & $py -m src.models.linear_regression }
            "xgboost" { & $py -m src.models.xgboost_model --spread_type kalman }
            "lstm"    { & $py -m src.models.lstm }
            "lstm_encdec" { & $py -m src.models.lstm_encoder_decoder --spread kalman }
            "all"     {
                & $py -m src.models.ou
                # & $py -m src.models.arma_tuning --spread_col spread_ols --horizon $horizon
                & $py -m src.models.arma_tuning --spread_col spread_kalman --horizon $horizon
                & $py -m src.models.linear_regression
                & $py -m src.models.xgboost_model --spread_type kalman
                & $py -m src.models.lstm
                & $py -m src.models.lstm_encoder_decoder --spread kalman
            }
        }
        & Write-Host "Clearing Python memory..." -ForegroundColor Yellow
        & $py -c "import gc; gc.collect()"
    }
}

# --- Step 6: Frozen Best-Params Runs ---
if ($SkipBestParamRuns) {
    Write-Host "==> [6/7] SkipBestParamRuns flag detected. Skipping frozen-parameter runs." -ForegroundColor Gray
} else {
    Measure-Time -Name "Frozen Best-Params Runs" -Script {
        Write-Host "==> [6/7] Frozen Best-Params Runs ($model)" -ForegroundColor Green
        switch ($model) {
            "ou"      { & $py -m src.models.ou }
            "arma"    {
                # & $py -m src.models.arma --spread_col spread_ols --p 9 --q 8 --horizon $horizon --eval_split val
                & $py -m src.models.arma --spread_col spread_kalman --p 6 --q 2 --horizon $horizon --eval_split val
            }
            "linear"  { & $py -m src.models.linear_regression }
            "xgboost" {
                & $py -m src.models.xgboost_model --spread_type kalman --no_tune
                # better version after editing xgboost_model.py:
                # & $py -m src.models.xgboost_model --spread_type kalman --no_tune --max_depth 4 --n_estimators 200 --learning_rate 0.05
            }
            "lstm"    {
                & $py -m src.models.lstm --spread kalman --hidden 64 --window_size 20 --lr 0.001 --no_tune
            }
            "lstm_encdec" {
                & $py -m src.models.lstm_encoder_decoder --spread kalman --hidden 64 --window_size 20 --lr 0.001 --no_tune
            }
            "all"     {
                & $py -m src.models.ou
                # & $py -m src.models.arma --spread_col spread_ols --p 9 --q 8 --horizon $horizon --eval_split val
                & $py -m src.models.arma --spread_col spread_kalman --p 7 --q 2 --horizon $horizon --eval_split val
                & $py -m src.models.linear_regression
                # & $py -m src.models.xgboost_model --spread_type ols --no_tune --max_depth 3 --n_estimators 100 --learning_rate 0.01
                & $py -m src.models.xgboost_model --spread_type kalman --no_tune --max_depth 3 --n_estimators 200 --learning_rate 0.01
                # & $py -m src.models.lstm --spread ols --hidden 32 --window_size 20 --lr 0.001 --no_tune
                & $py -m src.models.lstm --spread kalman --hidden 32 --window_size 20 --lr 0.001 --no_tune
                # & $py -m src.models.lstm_encoder_decoder --spread ols --hidden 32 --window_size 20 --lr 0.0005 --no_tune
                & $py -m src.models.lstm_encoder_decoder --spread kalman --hidden 64 --window_size 20 --lr 0.0005 --no_tune
            }
        }
        & Write-Host "Clearing Python memory..." -ForegroundColor Yellow
        & $py -c "import gc; gc.collect()"
    }
}

# --- Step 7: Backtesting ---
if ($SkipBacktesting) {
    Write-Host "==> [7/7] SkipBacktesting flag detected. Skipping backtesting." -ForegroundColor Gray
} else {
    Measure-Time -Name "Backtesting Engine" -Script {
        Write-Host "==> [7/7] Backtesting Engine" -ForegroundColor Green
        & $py -m src.backtest.backtest_engine
        & $py -m src.backtest.backtest_engine --holdout
    }
}

$globalEndTime = Get-Date
$totalDuration = $globalEndTime - $globalStartTime
Write-Host "Total Pipeline Duration: $($totalDuration.TotalMinutes) minutes." -ForegroundColor Green

Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host "SUCCESS: Pipeline completed."
Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
