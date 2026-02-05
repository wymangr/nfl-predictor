# NFL Prediction Tool

A machine learning command-line application built to predict NFL game outcomes for Yahoo Pick'em leagues. This tool uses XGBoost regression with engineered features and hyperparameter optimization to generate predictions for NFL games.

## Overview

This project provides a comprehensive NFL prediction system that:
- Fetches and manages NFL game data from multiple sources
- Engineers features based on team performance, quarterback stats, and historical trends
- Trains an XGBoost model with optimized hyperparameters
- Generates predictions for past and future games
- Produces detailed HTML reports with accuracy metrics and power rankings
- Supports hyperparameter tuning and feature selection

## Features

- **Data Management**: Automated data collection and backup from NFL sources
- **Model Training**: XGBoost regression with configurable hyperparameters
- **Predictions**: Generate predictions for past games (validation) and future games
- **Hyperparameter Optimization**: Random search with early stopping and validation
- **Feature Selection**: Automated feature importance analysis
- **Bucket Analysis**: Deep dive into prediction performance across 70+ metric buckets to identify strengths and weaknesses
- **Confidence Calibration**: Historical bucket-based accuracy calculation with 29% better calibration than raw model confidence
- **Bucket-Based Predictions**: Optional `--bucket` flag shows historical accuracy for similar past games (e.g., 30.3% vs 93% model confidence)
- **Reporting**: Comprehensive HTML reports with accuracy metrics, power rankings, QB changes, and configuration comparisons

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   cd nfl-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Set up the CLI alias** (optional but recommended):
   
   On Windows (PowerShell):
   ```bash
   echo 'Set-Alias nfl "python nfl.py"' >> .venv/Scripts/Activate.ps1
   ```
   
   On Windows (Git Bash):
   ```bash
   echo 'alias nfl="python nfl.py"' >> .venv/Scripts/activate
   ```
   
   On macOS/Linux:
   ```bash
   echo 'alias nfl="python nfl.py"' >> .venv/bin/activate
   ```

4. **Activate the virtual environment**:
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

   On Windows (Git Bash):
   ```bash
   source .venv/Scripts/activate
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After setup, you can use the `nfl` command (if you set up the alias) or `python nfl.py` to access all functionality.

### Quick Start

1. **Refresh NFL data**:
   ```bash
   nfl data refresh
   ```

2. **Train the model**:
   ```bash
   nfl model train
   ```

3. **Generate predictions for upcoming games**:
   ```bash
   nfl model predict future
   ```

4. **Show predictions for past games**:
   ```bash
   nfl model predict past
   ```

### Command Reference

#### Data Management

```bash
# Refresh NFL data from sources
nfl data refresh [--backup] [--backfill-date YEAR] [--spreads]
```

**Options:**
- `--backup`: Create a database backup before refreshing data
- `--backfill-date YEAR`: Specify the starting year for data backfill (default: 2003)
- `--spreads`: Update current week spread data only (faster, for weekly updates)

#### Model Operations

```bash
# Train the prediction model
nfl model train [--spread-line]

# Generate predictions for past games (validation)
nfl model predict past [--year YEAR|YEARS|all] [--report] [--spread-line]

# Generate predictions for future games
nfl model predict future [--report] [--spread-line] [--bucket]
```

**Options:**
- `--year YEAR|YEARS|all`: Specify year(s) for predictions. Can be a single year (2025), multiple comma-separated years (2024,2025), or 'all' for all available seasons (default: 2025)
- `--report`: Generate an HTML report after predictions
- `--spread-line`: Use the nflreadpy spread line for predictions instead of Yahoo spread
- `--bucket`: Show bucket-based historical accuracy alongside model confidence for each prediction

#### Configuration & Optimization

```bash
# Optimize the confidence formula to maximize confidence points
nfl config optimize-confidence [OPTIONS]

# Run feature selection analysis
nfl config feature-selection [--spread-line]

# Perform hyperparameter optimization via random search
nfl config random-search [OPTIONS]
```

**Optimize Confidence Options:**
- `--granularity LEVEL`: Search density - 'coarse' (fast, default), 'fine' (balanced), or 'ultra' (slow but thorough)
- `--min-features N`: Minimum number of features in each combination (default: 3)
- `--max-features N`: Maximum number of features in each combination (default: 7)
- `--workers N`: Number of parallel workers (default: auto-detect CPU count). Use 1 to disable multiprocessing
- `--verify`: Run verification check before optimization to ensure calculations work correctly
- `--top-features N`: Only consider top N features by correlation with correct predictions (default: 12)

**Random Search Options:**
- `--iterations N`: Number of hyperparameter combinations to test (default: 100)
- `--threshold N`: Minimum required 2025 prediction score (default: 55.0)
- `--resume`: Resume from a previous random search run
- `--min-train-r2 N`: Minimum required training R² (default: 0.27)
- `--min-iterations N`: Minimum required early stopping iterations (default: 20)
- `--spread-line`: Use the nflreadpy spread line for predictions instead of Yahoo spread

#### Reports

```bash
# Generate past predictions report
nfl report past-predictions

# Identify quarterback changes for the current week
nfl report qb-change

# Generate past predictions report
nfl report past-predictions

# Generate future predictions report
nfl report future-predictions

# Generate power rankings report
nfl report power-rankings [--season YEAR]

# Compare model configurations
nfl report compare-configs [OPTIONS]

# Analyze prediction performance across 70+ metric buckets
nfl report bucket-analysis [--future] [--past]
```

**Compare Configs Options:**
- `--log-file PATH`: Path to random_search_results.txt (default: random_search_results.txt)
- `--top-n N`: Only evaluate top N configs by original score
- `--output-file PATH`: Output HTML file (default: config_comparison.html)
- `--spread-line`: Use the nflreadpy spread line for predictions instead of Yahoo spread

**Bucket Analysis Options:**
- `--future`: Analyze future predictions using historical bucket performance
- `--past`: Analyze all past predictions with bucket-based confidence validation
- *(no flags)*: Show detailed bucket performance analysis across all metrics


## Model Details

### Algorithm
- **Model**: XGBoost Regressor
- **Target Variable**: Point differential (home team score - away team score)
- **Features**: Engineered features including:
  - Team performance metrics (win rates, point differentials)
  - Home/away splits
  - Quarterback statistics
  - Recent form (rolling averages)
  - Head-to-head history
  - Conference matchup indicators

### Bucket-Based Confidence System
The bucket analysis system provides calibrated confidence metrics by:
- Analyzing 70+ buckets across 14 key metrics (confidence score, spread, EPA, YPA, YPC, QB changes, etc.)
- Calculating historical accuracy for each bucket based on 290+ past predictions
- Using weighted averaging to compute bucket-based confidence for new predictions
- Providing 1% interval accuracy brackets (e.g., 59-60% bucket confidence → 30.3% actual accuracy)

**Key Insights:**
- **Model Calibration**: Raw model confidence averages 91% but actual accuracy is 61% (30% error)
- **Bucket Calibration**: Bucket confidence averages 61.11% vs 61.03% actual (0.08% error)
- **Brier Score**: Bucket system scores 0.2324 vs model's 0.3288 (29% better)
- **Critical Threshold**: Predictions with 62%+ bucket confidence achieve 70-87% actual accuracy, while <61% only achieves 30-50%

Use `nfl model predict future --bucket` to see both metrics side-by-side for better decision-making.

### Feature Engineering
The model uses a comprehensive set of engineered features that capture:
- Team strength and recent performance
- Home field advantage
- Quarterback impact and changes
- Divisional and conference matchups
- Historical trends

### Optimization
- Hyperparameter optimization via random search with validation
- Early stopping to prevent overfitting
- Feature selection based on importance scores
- Model persistence for consistent predictions

## Output Files

The application generates several output files:

- **nfl-prediction.db**: SQLite database containing all game data and predictions
- **nfl-prediction.pkl**: Serialized trained model
- **best_hyperparameters.json**: Optimal hyperparameters from random search
- **nfl_past_prediction_report.html**: Validation metrics and past game accuracy
- **nfl_future_prediction_report.html**: Upcoming game predictions
- **nfl_power_rankings_2025.html**: Team power rankings
- **config_comparison.html**: Hyperparameter configuration comparison
- **random_search_results.txt**: Log of hyperparameter search results

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- **nflreadpy**: NFL data fetching
- **pandas**: Data manipulation
- **xgboost**: Machine learning model
- **scikit-learn**: ML utilities and metrics
- **click**: Command-line interface
- **SQLAlchemy**: Database ORM
- **beautifulsoup4**: Web scraping utilities

## Tips

- Always backup your database before refreshing data (`--backup` flag)
- Run hyperparameter optimization periodically to maintain model performance
- Review past prediction reports to validate model accuracy before using future predictions
- Generate power rankings to understand team strength trends
- Use the `--resume` flag with random search to continue optimization runs
- **Use `--bucket` flag** for predictions to see historical accuracy - when model confidence is 93% but bucket accuracy is 30%, exercise caution!
- Run `nfl report bucket-analysis` to identify which metrics are most predictive
- Check bucket confidence for risky predictions: if bucket confidence is below 61%, historical data suggests the prediction is unreliable

## Troubleshooting

### Database Issues
If you encounter database errors, try:
```bash
nfl data refresh --backup --backfill-date 2003
```

### Model Performance Issues
If predictions seem inaccurate:
1. Refresh data to ensure it's current
2. Run hyperparameter optimization
3. Retrain the model
4. Review the past predictions report for validation metrics

### Missing Dependencies
If you get import errors:
```bash
pip install -r requirements.txt --upgrade
```

## License

This project is for personal use and educational purposes.

## Contributing

This is a personal project. Feel free to fork and modify for your own use.
