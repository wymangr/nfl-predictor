#!/usr/bin/env python3
"""
NFL Prediction CLI Tool

A command-line interface for managing NFL data and predictions.
"""

import click
import nflreadpy as nfl

from src.model.train import train_model
from src.model.predict import get_future_predictions, get_past_predictions
from src.config.feature_selection import run_feature_selection
from src.config.random_search import run_random_search
from src.config.optimize_confidence import run_optimization
from src.data.data import backfil_data
from src.data.update_spreads import update_current_spreads
from src.data.qb_changes import get_qb_change
from src.data.backup import backup_database
from src.reports.nfl_past_prediction_report import (
    generate_past_prediction_report,
    load_data,
    save_accuracy_metrics_to_db,
)
from src.reports.compare_configs_report import generate_compare_configs_report
from src.reports.nfl_power_ranking_report import generate_power_rankings_report
from src.reports.nfl_future_prediction_report import generate_future_predictions_report


@click.group()
def cli():
    """NFL prediction and data management tool."""
    pass


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option(
    "--backup",
    is_flag=True,
    help="Create a backup of the database before refreshing data.",
)
@click.option(
    "--backfill-date",
    type=int,
    default=2003,
    help="Date to backfill data from (default: 2003).",
)
@click.option(
    "--spreads",
    is_flag=True,
    help="Update current spread data only.",
)
def refresh(backfill_date, backup, spreads):
    """Refresh NFL data from sources."""
    print("Running: nfl data refresh")
    if backup:
        backup_database()
    if spreads:
        update_current_spreads()
    else:
        backfil_data(backfill_date)


@data.command()
def qb_change():
    """Identify quarterback changes for the current week."""
    print("Running: nfl data qb-change")
    get_qb_change()


@cli.group()
def model():
    """Model training, optimization, and prediction commands."""
    pass


@model.group()
def predict():
    """Prediction commands."""
    pass


@predict.command()
@click.option(
    "--year",
    type=str,
    default="2025",
    help="Year(s) to generate predictions for. Can be a single year (2025), multiple years (2024,2025), or 'all' (default: 2025).",
)
@click.option(
    "--report",
    is_flag=True,
    help="Generate prediction report after predictions are made.",
)
@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
def past(year, report, spread_line):
    """Generate predictions for past games."""
    print("Running: nfl model predict past")
    get_past_predictions(year, spread_line)
    if report:
        df = load_data()
        generate_past_prediction_report(df)
        save_accuracy_metrics_to_db(df)


@predict.command()
@click.option(
    "--report",
    is_flag=True,
    help="Generate prediction report after predictions are made.",
)
@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
def future(report, spread_line):
    """Generate predictions for future games."""
    print("Running: nfl model predict future")
    get_future_predictions(spread_line)
    if report:
        df = load_data()
        save_accuracy_metrics_to_db(df)
        generate_future_predictions_report()


@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
@model.command()
def train(spread_line):
    """Train the prediction model."""
    print("Running: nfl model train")
    train_model(spread_line)


@cli.group()
def config():
    """Configuration commands."""
    pass


@config.command()
@click.option(
    "--granularity",
    type=click.Choice(["coarse", "fine", "ultra"]),
    default="coarse",
    help="Search granularity: coarse (fast, default), fine (balanced), ultra (slow but thorough).",
)
@click.option(
    "--min-features",
    type=int,
    default=3,
    help="Minimum number of features in each combination. Default: 3",
)
@click.option(
    "--max-features",
    type=int,
    default=7,
    help="Maximum number of features in each combination. Default: 7",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: auto-detect CPU count). Use 1 to disable multiprocessing.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Run verification check before optimization to ensure calculations work correctly.",
)
@click.option(
    "--top-features",
    type=int,
    default=12,
    help="Only consider top N features by correlation with correct predictions. Default: 12",
)
def optimize_confidence(
    granularity, min_features, max_features, workers, verify, top_features
):
    """Optimize the confidence formula to maximize confidence points for correct predictions."""
    print("Running: nfl config optimize-confidence")
    run_optimization(
        granularity, min_features, max_features, workers, verify, top_features
    )


@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
@config.command()
def feature_selection(spread_line):
    """Perform feature selection for the model."""
    print("Running: nfl model feature-selection")
    run_feature_selection(spread_line)


@config.command()
@click.option(
    "--iterations",
    type=int,
    default=100,
    help="Number of hyperparameter combinations to test (default: 100).",
)
@click.option(
    "--threshold",
    type=float,
    default=55.0,
    help="Minimum required 2025 prediction score (default: 55.0).",
)
@click.option("--resume", is_flag=True, help="Resume from previous random search run.")
@click.option(
    "--min-train-r2",
    type=float,
    default=0.27,
    help="Minimum required training R² (default: 0.27)",
)
@click.option(
    "--min-iterations",
    type=int,
    default=20,
    help="Minimum required early stopping iterations (default: 20)",
)
@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
def random_search(
    iterations, threshold, resume, min_train_r2, min_iterations, spread_line
):
    """Perform random search for hyperparameter optimization."""
    print("Running: nfl model random-search")
    print(
        f"Parameters: iterations={iterations}, threshold={threshold}, resume={resume}, min_train_r2={min_train_r2}, min_iterations={min_iterations}"
    )
    run_random_search(
        iterations, threshold, resume, min_train_r2, min_iterations, spread_line
    )


@cli.group()
def report():
    """Report generation commands."""
    pass


@report.command()
def past_predictions():
    """Generate predictions report."""
    df = load_data()
    generate_past_prediction_report(df)
    save_accuracy_metrics_to_db(df)


@report.command()
def future_predictions():
    """Generate outlier report for future predictions."""
    generate_future_predictions_report()


@report.command()
@click.option(
    "--log-file",
    type=str,
    default="random_search_results.txt",
    help="Path to random_search_results.txt (default: random_search_results.txt)",
)
@click.option(
    "--top-n",
    type=int,
    default=None,
    help="Only evaluate top N configs by original score (default: None)",
)
@click.option(
    "--output-file",
    type=str,
    default="config_comparison.html",
    help="Output HTML file (default: config_comparison.html)",
)
@click.option(
    "--spread-line",
    is_flag=True,
    help="Use the nflreadpy spread line for predictions instead of Yahoo spread.",
)
def compare_configs(log_file, top_n, output_file, spread_line):
    """Generate compare configs report."""
    generate_compare_configs_report(log_file, top_n, output_file, spread_line)


@click.option(
    "--season",
    type=int,
    default=2025,
    help="Season year (default: 2025)",
)
@report.command()
def power_rankings(season):
    """Generate power rankings report."""
    generate_power_rankings_report(season=season)


if __name__ == "__main__":
    cli()
