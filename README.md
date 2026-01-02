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
- **QB Change Detection**: Identifies quarterback changes that may impact predictions
- **Model Training**: XGBoost regression with configurable hyperparameters
- **Predictions**: Generate predictions for past games (validation) and future games
- **Hyperparameter Optimization**: Random search with early stopping and validation
- **Feature Selection**: Automated feature importance analysis
- **Reporting**: Comprehensive HTML reports with accuracy metrics, power rankings, and configuration comparisons

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
nfl data refresh [--backup] [--backfill-date YEAR]

# Identify quarterback changes for the current week
nfl data qb-change
```

**Options:**
- `--backup`: Create a database backup before refreshing data
- `--backfill-date YEAR`: Specify the starting year for data backfill (default: 2003)

#### Model Operations

```bash
# Train the prediction model
nfl model train

# Generate predictions for past games (validation)
nfl model predict past [--year YEAR] [--report]

# Generate predictions for future games
nfl model predict future [--report] [--year YEAR]
```

**Options:**
- `--year YEAR`: Specify the year for predictions (default: 2025)
- `--report`: Generate an HTML report after predictions

#### Configuration & Optimization

```bash
# Run feature selection analysis
nfl config feature-selection

# Perform hyperparameter optimization via random search
nfl config random-search [OPTIONS]
```

**Random Search Options:**
- `--iterations N`: Number of hyperparameter combinations to test (default: 100)
- `--threshold N`: Minimum required 2025 prediction score (default: 55.0)
- `--resume`: Resume from a previous random search run
- `--min-train-r2 N`: Minimum required training R² (default: 0.27)
- `--min-iterations N`: Minimum required early stopping iterations (default: 20)

#### Reports

```bash
# Generate past predictions report
nfl report past-predictions

# Generate future predictions report
nfl report future-predictions

# Generate power rankings report
nfl report power-rankings [--season YEAR]

# Compare model configurations
nfl report compare-configs [OPTIONS]
```

**Compare Configs Options:**
- `--log-file PATH`: Path to random_search_results.txt (default: random_search_results.txt)
- `--top-n N`: Only evaluate top N configs by original score
- `--output-file PATH`: Output HTML file (default: config_comparison.html)


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
