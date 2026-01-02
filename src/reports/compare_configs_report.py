import ast
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
import json
import re

from src.model.features import (
    engineer_features,
    FEATURE_SELECTION_BASE_FEATURES as base_features,
    TARGET as target,
)
from src.model.train import run_query
from src.model.overfitting import check_overfitting, regression_metrics
from src.model.predict import get_past_predictions_model


def parse_config_log(log_file: str) -> list:
    """
    Parses the log file by first cleaning NumPy objects for ast.literal_eval,
    then using json.dumps to ensure JSON compatibility.
    """
    configs = []

    # 1. Regex to find ALL NumPy data type constructors and constants
    # It covers np.float64(...), np.True_, np.float32(...), etc.
    numpy_pattern = re.compile(
        # Matches np.float64(xxx) or np.float32(xxx)
        r"np\.(?:float\d+)\(([^)]+)\)"
        # OR matches np.True_ or np.False_ or np.True or np.False
        r"|np\.(?:True|False)_?"
    )

    # Function to temporarily replace NumPy expressions with clean Python literals
    def replace_numpy_for_ast(match):
        # Group 1 captures the inner value for np.float64(...) format
        if match.group(1):
            return match.group(1)  # Return the number (e.g., 9.598439992162815)
        # For constants, return the Python literal (True/False)
        elif match.group(0).startswith("np.True"):
            return "True"
        elif match.group(0).startswith("np.False"):
            return "False"
        return match.group(0)

    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith("{"):

                    # Step A: Clean up NumPy types, allowing ast.literal_eval to succeed
                    temp_line = numpy_pattern.sub(replace_numpy_for_ast, line)

                    try:
                        # Step B: Use ast.literal_eval to safely convert the Python string to a dict
                        py_dict = ast.literal_eval(temp_line)

                        # Step C: Convert the Python dict to a clean JSON string,
                        # automatically handling 'True' -> 'true', 'None' -> 'null', and proper quoting.
                        json_string = json.dumps(py_dict)

                        # Step D: Convert the clean JSON string back to the final dictionary
                        config = json.loads(json_string)
                        configs.append(config)

                    except ValueError as e:
                        # Catches ast.literal_eval errors
                        print(f"⚠️ Error parsing Python literal: {e}")
                    except json.JSONDecodeError as e:
                        # Catches json.loads errors (less likely now)
                        print(f"⚠️ Error parsing JSON from line: {e}")
                    except Exception as e:
                        print(f"⚠️ Unexpected error: {e}")

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{log_file}'")

    return configs


def train_and_evaluate_config(
    config: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    best_features,
    base_features,
) -> dict:
    """Train a model with given config and return evaluation metrics."""

    # Extract hyperparameters from config
    params = config["params"]

    # Fixed parameters (same as random_search.py)
    fixed_params = {
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mae",
        "early_stopping_rounds": 50,
    }

    # Train model with evaluation history
    model = XGBRegressor(**params, **fixed_params)

    # Fit with evaluation sets to capture training history
    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False
    )

    # Get evaluation history
    evals_result = model.evals_result()

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = regression_metrics(y_train, y_train_pred)
    val_metrics = regression_metrics(y_val, y_val_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Check overfitting
    overfitting_report = check_overfitting(
        train_mae=train_metrics["mae"],
        val_mae=val_metrics["mae"],
        train_rmse=train_metrics["rmse"],
        val_rmse=val_metrics["rmse"],
        train_r2=train_metrics["r2"],
        val_r2=val_metrics["r2"],
    )

    # Evaluate on 2025 with week-by-week analysis
    model_dict = {
        "model": model,
        "features": best_features,
        "base_features": base_features,
    }
    # week_by_week_2025 = get_2025_week_by_week_results(model_dict)
    week_by_week_2025 = get_past_predictions_model(model_dict)
    score_2025 = week_by_week_2025["overall_accuracy"]

    # Get feature importance
    feature_importance = model.feature_importances_
    spread_idx = (
        best_features.index("spread_line") if "spread_line" in best_features else -1
    )
    spread_importance = feature_importance[spread_idx] if spread_idx >= 0 else 0.0

    # Get top 10 features
    feature_importance_pairs = list(zip(best_features, feature_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    top_10_features = [
        {"feature": name, "importance": float(importance)}
        for name, importance in feature_importance_pairs[:10]
    ]

    # Calculate within X points
    test_errors = y_test - y_test_pred
    val_errors = y_val - y_val_pred
    within_3_val = (np.abs(val_errors) <= 3).mean() * 100
    within_7_val = (np.abs(val_errors) <= 7).mean() * 100
    within_3_test = (np.abs(test_errors) <= 3).mean() * 100
    within_7_test = (np.abs(test_errors) <= 7).mean() * 100

    return {
        "config_iteration": config.get("iteration", 0),
        "params": params,
        "train_mae": train_metrics["mae"],
        "val_mae": val_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "val_r2": val_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "val_rmse": val_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "overfitting_gap": val_metrics["mae"] - train_metrics["mae"],
        "overfitting_gap_pct": (
            (val_metrics["mae"] - train_metrics["mae"]) / train_metrics["mae"]
        )
        * 100,
        "is_overfitting": overfitting_report["overfit"],
        "overfitting_severity": overfitting_report["severity"],
        "best_iteration": model.best_iteration,
        "n_estimators": params["n_estimators"],
        "early_stop_ratio": model.best_iteration / params["n_estimators"],
        "spread_importance": spread_importance,
        "spread_importance_pct": spread_importance * 100,
        "top_10_features": top_10_features,
        "score_2025": score_2025,
        "consistency_std": week_by_week_2025["consistency_std"],
        "weeks_above_50": week_by_week_2025["weeks_above_50"],
        "total_weeks": week_by_week_2025["total_weeks"],
        "min_week_accuracy": week_by_week_2025["min_week_accuracy"],
        "weekly_accuracies": week_by_week_2025["weekly_accuracies"],
        "within_3_val": within_3_val,
        "within_7_val": within_7_val,
        "within_3_test": within_3_test,
        "within_7_test": within_7_test,
        "evals_result": evals_result,  # Store training history
    }


def rank_results(results: list) -> list:
    """Rank results by overall quality for pick'em predictions, heavily favoring consistency and weeks above 50%.

    Scoring system is designed so that maximum points represent theoretical perfection:
    - Every single week above 50% accuracy
    - Overall 2025 accuracy well above 60%
    - Extremely low MAE and minimal overfitting
    - Perfect consistency across all weeks

    Total possible: 100 points (Weeks >50%: 25, Min Week: 25, Consistency: 10, 2025 Acc: 20, Test MAE: 10, Overfitting: 10)
    Realistic top scores: 60-80 points for good configs, 80-95 for excellent ones
    """
    if not results:
        return []

    # Score each result
    scored_results = []
    for result in results:
        score = 0

        # Weeks above 50% (higher is better) - 25 points max
        # Only award points proportional to the PERCENTAGE of weeks above 50%
        # Perfect score requires 100% of weeks above 50%
        weeks_percentage = (
            result["weeks_above_50"] / result["total_weeks"]
            if result["total_weeks"] > 0
            else 0
        )
        weeks_above_50_score = 25 * weeks_percentage
        score += weeks_above_50_score

        # Consistency (lower std is better) - 10 points max
        # Use exponential penalty for inconsistency - perfect consistency = 0 std
        # Typical std is 10-20%, so scale accordingly
        consistency_penalty = min(
            result["consistency_std"] / 20.0, 1.0
        )  # Normalize to 0-1
        consistency_score = 10 * (1 - consistency_penalty)
        score += consistency_score

        # Minimum week accuracy - 25 points max
        # Award points based on absolute minimum week performance
        # 50% = full points, 0% = no points
        min_week_score = 25 * (result["min_week_accuracy"] / 50.0)
        min_week_score = min(min_week_score, 25)  # Cap at 25
        score += min_week_score

        # Overall 2025 accuracy - 20 points max
        # 65% = full points, anything less is proportional
        accuracy_score = 20 * (result["score_2025"] / 65.0)
        accuracy_score = min(accuracy_score, 20)  # Cap at 20
        score += accuracy_score

        # Test MAE - 10 points max
        # Award points based on absolute performance
        # 12.0 MAE = 0 points, 8.0 MAE = full points (linear scale, more sensitive range)
        mae = result["test_mae"]
        if mae >= 12.0:
            test_score = 0
        elif mae <= 8.0:
            test_score = 10
        else:
            test_score = 10 * (1 - (mae - 8.0) / 4.0)
        score += test_score

        # Overfitting - 10 points max
        # Award points based on absolute overfitting gap
        # 0% gap = full points, 15%+ gap = no points (tightened threshold)
        gap_pct = result["overfitting_gap_pct"]
        if gap_pct >= 15:
            gap_score = 0
        else:
            gap_score = 10 * (1 - gap_pct / 15.0)
        score += gap_score

        scored_results.append(
            {
                **result,
                "overall_score": score,
                "score_breakdown": {
                    "weeks_above_50": weeks_above_50_score,
                    "consistency": consistency_score,
                    "min_week": min_week_score,
                    "accuracy_2025": accuracy_score,
                    "test_mae": test_score,
                    "overfitting": gap_score,
                },
            }
        )

    # Sort by score
    scored_results.sort(key=lambda x: x["overall_score"], reverse=True)
    return scored_results


def get_line_color(index):
    """Get a distinct color for line graphs."""
    colors = [
        "#667eea",  # Purple-blue
        "#10b981",  # Green
        "#ef4444",  # Red
        "#f59e0b",  # Orange
        "#06b6d4",  # Cyan
        "#8b5cf6",  # Purple
        "#ec4899",  # Pink
        "#14b8a6",  # Teal
        "#f97316",  # Dark orange
        "#6366f1",  # Indigo
        "#84cc16",  # Lime
        "#a855f7",  # Violet
        "#22d3ee",  # Light cyan
        "#fb923c",  # Light orange
        "#4ade80",  # Light green
    ]
    return colors[index % len(colors)]


def generate_html_report(ranked_results: list, output_file: str):
    """Generate HTML comparison report."""

    html = []
    html.append(
        """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Hyperparameter Config Comparison</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        color: #333;
                    }
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        overflow: hidden;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }
                    .header h1 { font-size: 2.5em; margin-bottom: 10px; }
                    .content { padding: 40px; }
                    .section-title {
                        font-size: 1.8em;
                        color: #667eea;
                        margin: 30px 0 20px 0;
                        padding-bottom: 10px;
                        border-bottom: 3px solid #667eea;
                    }
                    .comparison-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .comparison-table thead {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }
                    .comparison-table th {
                        padding: 15px;
                        text-align: left;
                        font-weight: 600;
                        font-size: 0.9em;
                        text-transform: uppercase;
                    }
                    .comparison-table td {
                        padding: 15px;
                        border-bottom: 1px solid #f0f0f0;
                    }
                    .comparison-table tbody tr:hover { background: #f8f9fa; }
                    .config-card {
                        background: white;
                        border-radius: 12px;
                        padding: 30px;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        border-left: 5px solid #667eea;
                    }
                    .config-card.rank-1 {
                        border-left-color: #10b981;
                        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                    }
                    .rank-badge {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 8px 20px;
                        border-radius: 20px;
                        font-weight: 600;
                        display: inline-block;
                        margin-bottom: 15px;
                    }
                    .rank-badge.rank-1 {
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    }
                    .metrics-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }
                    .metric-box {
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 4px solid #667eea;
                    }
                    .metric-label {
                        font-size: 0.8em;
                        color: #6b7280;
                        text-transform: uppercase;
                        margin-bottom: 5px;
                    }
                    .metric-value {
                        font-size: 1.5em;
                        font-weight: 700;
                        color: #1f2937;
                    }
                    .params-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 10px;
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 15px 0;
                        font-size: 0.9em;
                    }
                    .param-item {
                        padding: 8px;
                        background: white;
                        border-radius: 4px;
                    }
                    .param-name {
                        color: #6b7280;
                        font-size: 0.85em;
                    }
                    .param-value {
                        color: #1f2937;
                        font-weight: 600;
                    }
                    .winner-box {
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 12px;
                        margin: 30px 0;
                    }
                    .winner-box h3 { font-size: 1.5em; margin-bottom: 15px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🏈 Hyperparameter Configuration Comparison</h1>
                        <div style="margin-top: 10px; font-size: 1.1em;">"""
    )

    html.append(
        f"Compared {len(ranked_results)} configurations | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    )
    html.append(
        """</div>
        </div>
        <div class="content">"""
    )

    # Quick comparison table
    html.append(
        """
            <h2 class="section-title">📊 Quick Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Config #</th>
                        <th>2025 Total</th>
                        <th>Weeks >50%</th>
                        <th>Min Week</th>
                        <th>Consistency</th>
                        <th>Test MAE</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>"""
    )

    for rank, result in enumerate(ranked_results, 1):
        html.append(
            f"""
                    <tr>
                        <td><strong>#{rank}</strong></td>
                        <td>Config {result['config_iteration']}</td>
                        <td>{result['score_2025']:.1f}%</td>
                        <td>{result['weeks_above_50']}/{result['total_weeks']}</td>
                        <td>{result['min_week_accuracy']:.1f}%</td>
                        <td>±{result['consistency_std']:.1f}%</td>
                        <td>{result['test_mae']:.3f}</td>
                        <td>{result['overall_score']:.1f}/100</td>
                    </tr>"""
        )

    html.append(
        """
            </tbody>
            </table>"""
    )

    # Score breakdown table
    html.append(
        """
            <h2 class="section-title">🎯 Score Breakdown (100 Points Total)</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Config #</th>
                        <th>Weeks >50%<br/>(25 pts)</th>
                        <th>Consistency<br/>(10 pts)</th>
                        <th>Min Week<br/>(25 pts)</th>
                        <th>2025 Acc<br/>(20 pts)</th>
                        <th>Test MAE<br/>(10 pts)</th>
                        <th>Overfitting<br/>(10 pts)</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>"""
    )

    for rank, result in enumerate(ranked_results, 1):
        breakdown = result["score_breakdown"]
        html.append(
            f"""
                    <tr>
                        <td><strong>#{rank}</strong></td>
                        <td>Config {result['config_iteration']}</td>
                        <td>{breakdown['weeks_above_50']:.1f}</td>
                        <td>{breakdown['consistency']:.1f}</td>
                        <td>{breakdown['min_week']:.1f}</td>
                        <td>{breakdown['accuracy_2025']:.1f}</td>
                        <td>{breakdown['test_mae']:.1f}</td>
                        <td>{breakdown['overfitting']:.1f}</td>
                        <td><strong>{result['overall_score']:.1f}</strong></td>
                    </tr>"""
        )

    html.append(
        """
                </tbody>
            </table>"""
    )

    # Winner recommendation
    best = ranked_results[0]
    html.append(
        f"""
            <div class="winner-box">
                <h3>🏆 Recommended Configuration: #{best['config_iteration']}</h3>
                <p style="margin: 15px 0;">This configuration has the most CONSISTENT week-to-week performance:</p>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 5px 0;">→ Most consistent: ±{best['consistency_std']:.1f}% standard deviation</li>
                    <li style="padding: 5px 0;">→ Reliable weeks: {best['weeks_above_50']}/{best['total_weeks']} weeks above 50%</li>
                    <li style="padding: 5px 0;">→ Worst week: {best['min_week_accuracy']:.1f}% (not catastrophic)</li>
                    <li style="padding: 5px 0;">→ Overall 2025: {best['score_2025']:.1f}%</li>
                    <li style="padding: 5px 0;">→ Quality score: {best['overall_score']:.1f}/100 points</li>
                </ul>
            </div>"""
    )

    # All configurations detailed cards
    html.append(
        """
            <h2 class="section-title">📋 All Configurations</h2>"""
    )

    for rank, result in enumerate(ranked_results, 1):
        card_class = "rank-1" if rank == 1 else ""
        html.append(
            f"""
            <div class="config-card {card_class}">
                <span class="rank-badge rank-{rank}">Rank #{rank} - Config {result['config_iteration']}</span>
                <div style="font-size: 1.2em; margin: 10px 0;">
                    Overall Score: <strong>{result['overall_score']:.1f}/100 points</strong>
                </div>
                
                <h3 style="margin-top: 20px; color: #667eea;">Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-label">2025 Total</div>
                        <div class="metric-value">{result['score_2025']:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Consistency</div>
                        <div class="metric-value">±{result['consistency_std']:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Weeks >50%</div>
                        <div class="metric-value">{result['weeks_above_50']}/{result['total_weeks']}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Min Week</div>
                        <div class="metric-value">{result['min_week_accuracy']:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Test MAE</div>
                        <div class="metric-value">{result['test_mae']:.3f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Test R²</div>
                        <div class="metric-value">{result['test_r2']:.3f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Overfit Gap</div>
                        <div class="metric-value">{result['overfitting_gap_pct']:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Train MAE</div>
                        <div class="metric-value">{result['train_mae']:.3f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Val MAE</div>
                        <div class="metric-value">{result['val_mae']:.3f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Early Stop</div>
                        <div class="metric-value">{result['best_iteration']}/{result['n_estimators']}</div>
                    </div>
                </div>
                
                <h3 style="margin-top: 20px; color: #667eea;">Week-by-Week 2025 Performance</h3>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 10px;">"""
        )

        # Add weekly accuracy breakdown
        for i, accuracy in enumerate(result["weekly_accuracies"], 1):
            if accuracy >= 60:
                color = "#10b981"  # Green (60%+)
            elif accuracy >= 50:
                color = "#FF9900"  # Yellow (50-59.999%)
            else:
                color = "#ef4444"  # Red (0-49.999%)
            html.append(
                f"""
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 6px; border-left: 3px solid {color};">
                            <div style="font-size: 0.8em; color: #6b7280;">Week {i}</div>
                            <div style="font-size: 1.3em; font-weight: 700; color: {color};">{accuracy:.1f}%</div>
                        </div>"""
            )

        html.append(
            """
                    </div>
                </div>
                
                <h3 style="margin-top: 20px; color: #667eea;">Hyperparameters</h3>
                <div class="params-grid">"""
        )

        for param, value in sorted(result["params"].items()):
            if isinstance(value, float):
                html.append(
                    f"""
                    <div class="param-item">
                        <div class="param-name">{param}</div>
                        <div class="param-value">{value:.4f}</div>
                    </div>"""
                )
            else:
                html.append(
                    f"""
                    <div class="param-item">
                        <div class="param-name">{param}</div>
                        <div class="param-value">{value}</div>
                    </div>"""
                )

        html.append(
            """
                </div>
                
                <h3 style="margin-top: 20px; color: #667eea;">Top 10 Most Important Features</h3>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid #667eea;">
                                <th style="padding: 10px; text-align: left; color: #667eea;">Rank</th>
                                <th style="padding: 10px; text-align: left; color: #667eea;">Feature</th>
                                <th style="padding: 10px; text-align: right; color: #667eea;">Importance</th>
                                <th style="padding: 10px; text-align: left; color: #667eea;">Bar</th>
                            </tr>
                        </thead>
                        <tbody>"""
        )

        max_importance = (
            max([f["importance"] for f in result["top_10_features"]])
            if result["top_10_features"]
            else 1
        )
        for rank, feature_info in enumerate(result["top_10_features"], 1):
            bar_width = (feature_info["importance"] / max_importance) * 100
            html.append(
                f"""
                            <tr style="border-bottom: 1px solid #e5e7eb;">
                                <td style="padding: 10px; font-weight: 600;">{rank}</td>
                                <td style="padding: 10px; font-family: monospace; font-size: 0.9em;">{feature_info['feature']}</td>
                                <td style="padding: 10px; text-align: right; font-weight: 600;">{feature_info['importance']:.4f}</td>
                                <td style="padding: 10px;">
                                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: {bar_width}%; height: 20px; border-radius: 4px;"></div>
                                </td>
                            </tr>"""
            )

        html.append(
            """
                        </tbody>
                    </table>
                </div>
            </div>"""
        )

    # Add interactive line graph at the bottom
    html.append(
        """
            <h2 class="section-title">📈 Week-by-Week Performance Comparison</h2>
            <div style="background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <div style="margin-bottom: 20px;">
                    <label style="font-weight: 600; margin-right: 10px;">Filter Configurations:</label>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                        <button onclick="showAllConfigs()" style="padding: 8px 16px; border: 2px solid #667eea; background: #667eea; color: white; border-radius: 6px; cursor: pointer; font-weight: 600;">Show All</button>
                        <button onclick="showTopN(3)" style="padding: 8px 16px; border: 2px solid #667eea; background: white; color: #667eea; border-radius: 6px; cursor: pointer; font-weight: 600;">Top 3</button>
                        <button onclick="showTopN(5)" style="padding: 8px 16px; border: 2px solid #667eea; background: white; color: #667eea; border-radius: 6px; cursor: pointer; font-weight: 600;">Top 5</button>
                        <button onclick="toggleConfig('all')" style="padding: 8px 16px; border: 2px solid #667eea; background: white; color: #667eea; border-radius: 6px; cursor: pointer; font-weight: 600;">Toggle All</button>
                    </div>
                    <div id="configCheckboxes" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 15px; max-height: 200px; overflow-y: auto; padding: 15px; background: #f8f9fa; border-radius: 8px;">
"""
    )

    # Add checkboxes for each config
    for rank, result in enumerate(ranked_results, 1):
        color = get_line_color(rank - 1)
        checked = "checked" if rank <= 5 else ""
        html.append(
            f"""
                        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                            <input type="checkbox" id="config_{result['config_iteration']}" {checked} onchange="updateChart()" style="cursor: pointer;">
                            <span style="width: 20px; height: 3px; background: {color}; border-radius: 2px;"></span>
                            <span style="font-size: 0.9em;">Config {result['config_iteration']} (#{rank})</span>
                        </label>"""
        )

    html.append(
        """
                    </div>
                </div>
                <canvas id="weeklyChart" style="max-height: 500px;"></canvas>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <script>
                // Prepare data for Chart.js
                const configData = ["""
    )

    # Add JavaScript data for each config
    for rank, result in enumerate(ranked_results, 1):
        color = get_line_color(rank - 1)
        weekly_data = result["weekly_accuracies"]
        html.append(
            f"""
                    {{
                        configId: {result['config_iteration']},
                        rank: {rank},
                        label: 'Config {result['config_iteration']} (#{rank})',
                        data: {weekly_data},
                        borderColor: '{color}',
                        backgroundColor: '{color}33',
                        tension: 0.3,
                        borderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }},"""
        )

    html.append(
        """
                ];
                
                // Get number of weeks from first config
                const numWeeks = configData[0].data.length;
                const weekLabels = Array.from({length: numWeeks}, (_, i) => `Week ${i + 1}`);
                
                // Create 50% reference line dataset
                const fiftyPercentLine = {
                    label: '50% Threshold',
                    data: Array(numWeeks).fill(50),
                    borderColor: '#6b7280',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [10, 5],
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    tension: 0
                };
                
                // Create chart
                const ctx = document.getElementById('weeklyChart').getContext('2d');
                let chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: weekLabels,
                        datasets: [fiftyPercentLine, ...configData.filter((_, idx) => idx < 5)] // Show 50% line + top 5 by default
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top',
                                labels: {
                                    usePointStyle: true,
                                    padding: 15,
                                    font: {
                                        size: 12,
                                        weight: '600'
                                    }
                                }
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false,
                                callbacks: {
                                    label: function(context) {
                                        return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    callback: function(value) {
                                        return value + '%';
                                    }
                                },
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                title: {
                                    display: true,
                                    text: 'Accuracy (%)',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                }
                            },
                            x: {
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                title: {
                                    display: true,
                                    text: '2025 Week',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                }
                            }
                        },
                        interaction: {
                            mode: 'nearest',
                            axis: 'x',
                            intersect: false
                        }
                    }
                });
                
                // Update chart based on checkboxes
                function updateChart() {
                    const selectedDatasets = [fiftyPercentLine]; // Always include 50% line
                    configData.forEach(config => {
                        const checkbox = document.getElementById(`config_${config.configId}`);
                        if (checkbox && checkbox.checked) {
                            selectedDatasets.push(config);
                        }
                    });
                    
                    chart.data.datasets = selectedDatasets;
                    chart.update();
                }
                
                // Show all configs
                function showAllConfigs() {
                    configData.forEach(config => {
                        const checkbox = document.getElementById(`config_${config.configId}`);
                        if (checkbox) checkbox.checked = true;
                    });
                    updateChart();
                }
                
                // Show top N configs
                function showTopN(n) {
                    configData.forEach((config, idx) => {
                        const checkbox = document.getElementById(`config_${config.configId}`);
                        if (checkbox) {
                            checkbox.checked = (idx < n);
                        }
                    });
                    updateChart();
                }
                
                // Toggle all configs
                function toggleConfig(action) {
                    const allChecked = configData.every(config => {
                        const checkbox = document.getElementById(`config_${config.configId}`);
                        return checkbox && checkbox.checked;
                    });
                    
                    configData.forEach(config => {
                        const checkbox = document.getElementById(`config_${config.configId}`);
                        if (checkbox) checkbox.checked = !allChecked;
                    });
                    updateChart();
                }
            </script>
        </div>
    </div>
</body>
</html>"""
    )

    # Write file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"\n💾 HTML report saved to: {output_file}")


def generate_training_diagnostics_report(ranked_results: list, output_file: str):
    """Generate comprehensive training diagnostics HTML report with learning curves and analysis."""

    html = []
    html.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Diagnostics Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; }
        .intro-box {
            background: #f0f9ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }
        .intro-box h3 { color: #667eea; margin-bottom: 10px; }
        .intro-box p { line-height: 1.6; margin-bottom: 10px; }
        .config-card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        .config-card.rank-1 {
            border-left-color: #10b981;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }
        .config-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e5e7eb;
        }
        .config-title {
            font-size: 1.5em;
            font-weight: 700;
            color: #1f2937;
        }
        .config-rank {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
        }
        .config-rank.rank-1 {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metric-item {
            text-align: center;
        }
        .metric-label {
            font-size: 0.85em;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.3em;
            font-weight: 700;
            color: #1f2937;
        }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .chart-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .chart-description {
            font-size: 0.9em;
            color: #6b7280;
            margin-bottom: 20px;
            padding: 12px;
            background: #f0f9ff;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }
        .insight-box {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .insight-box strong { color: #92400e; }
        canvas { max-height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Training Diagnostics Report</h1>
            <p>"""
        + f"Detailed training analysis for {len(ranked_results)} configurations | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        + """</p>
        </div>
        <div class="content">
            <div class="intro-box">
                <h3>Understanding Training Diagnostics</h3>
                <p><strong>Learning Curves</strong> show how model error decreases as training progresses. Good models show steady improvement on both training and validation sets.</p>
                <p><strong>Train vs Validation Performance</strong> reveals overfitting. If validation error is much higher than training error, the model is memorizing rather than learning.</p>
                <p><strong>Early Stopping</strong> indicates when the model stopped improving. Models that stop too early may be undertrained, while those that use most iterations learned well.</p>
                <p><strong>Error Distribution</strong> shows prediction accuracy. Errors clustered near zero indicate precise predictions.</p>
            </div>
            """
    )

    # Generate diagnostics for each config
    for rank, result in enumerate(ranked_results, 1):
        card_class = "rank-1" if rank == 1 else ""
        evals = result.get("evals_result", {})

        html.append(
            f"""
            <div class="config-card {card_class}">
                <div class="config-header">
                    <div class="config-title">Config #{result['config_iteration']}</div>
                    <div class="config-rank rank-{rank}">Rank #{rank}</div>
                </div>
                
                <div class="metrics-summary">
                    <div class="metric-item">
                        <div class="metric-label">2025 Accuracy</div>
                        <div class="metric-value">{result['score_2025']:.1f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Train MAE</div>
                        <div class="metric-value">{result['train_mae']:.2f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Val MAE</div>
                        <div class="metric-value">{result['val_mae']:.2f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Test MAE</div>
                        <div class="metric-value">{result['test_mae']:.2f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Stopped At</div>
                        <div class="metric-value">{result['best_iteration']}/{result['n_estimators']}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Overall Score</div>
                        <div class="metric-value">{result['overall_score']:.1f}/100</div>
                    </div>
                </div>
                """
        )

        # Check if we have training history
        if evals and "validation_0" in evals and "mae" in evals["validation_0"]:
            # Extract training history (XGBoost stores as validation_0=train, validation_1=val)
            train_mae = evals["validation_0"]["mae"]
            val_mae = evals["validation_1"]["mae"]

            if len(train_mae) > 0 and len(val_mae) > 0:
                iterations = list(range(1, len(train_mae) + 1))
                best_iter = result["best_iteration"]

                # Learning Curve (MAE)
                html.append(
                    f"""
                <div class="chart-container">
                    <div class="chart-title">📈 Learning Curve (MAE)</div>
                    <div class="chart-description">
                        <strong>What this shows:</strong> Model error (Mean Absolute Error) over training iterations. Lower is better.
                        The red vertical line shows where early stopping occurred (best iteration = {best_iter}).
                        <br><strong>Good signs:</strong> Both lines decreasing together, small gap between train/val, stopping not too early.
                        <br><strong>Warning signs:</strong> Large gap (overfitting), val increasing while train decreasing (memorization), stopping after just a few iterations (undertraining).
                    </div>
                    <canvas id="mae_chart_{result['config_iteration']}"></canvas>
                </div>
                
                <script>
                    (function() {{
                        const ctx = document.getElementById('mae_chart_{result['config_iteration']}');
                        if (!ctx) return;
                        
                        new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: {iterations},
                                datasets: [
                                    {{
                                        label: 'Training MAE',
                                        data: {train_mae},
                                        borderColor: '#3b82f6',
                                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                        tension: 0.3,
                                        borderWidth: 2,
                                        pointRadius: 1,
                                        pointHoverRadius: 4
                                    }},
                                    {{
                                        label: 'Validation MAE',
                                        data: {val_mae},
                                        borderColor: '#ef4444',
                                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                        tension: 0.3,
                                        borderWidth: 2,
                                        pointRadius: 1,
                                        pointHoverRadius: 4
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {{
                                    legend: {{ 
                                        position: 'top',
                                        labels: {{
                                            usePointStyle: true,
                                            padding: 15
                                        }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false
                                    }},
                                    annotation: {{
                                        annotations: {{
                                            earlyStopLine: {{
                                                type: 'line',
                                                xMin: {best_iter},
                                                xMax: {best_iter},
                                                borderColor: 'rgba(220, 38, 38, 0.8)',
                                                borderWidth: 2,
                                                borderDash: [10, 5],
                                                label: {{
                                                    display: true,
                                                    content: 'Early Stop',
                                                    position: 'start',
                                                    backgroundColor: 'rgba(220, 38, 38, 0.8)',
                                                    color: 'white',
                                                    font: {{
                                                        size: 11,
                                                        weight: 'bold'
                                                    }},
                                                    padding: 4
                                                }}
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        title: {{ display: true, text: 'MAE (points)' }},
                                        beginAtZero: false
                                    }},
                                    x: {{
                                        title: {{ display: true, text: 'Iteration' }}
                                    }}
                                }}
                            }}
                        }});
                    }})();
                </script>
                """
                )

                # Overfitting visualization
                train_val_gap = [v - t for t, v in zip(train_mae, val_mae)]
                html.append(
                    f"""
                <div class="chart-container">
                    <div class="chart-title">⚠️ Overfitting Monitor (Val - Train MAE)</div>
                    <div class="chart-description">
                        <strong>What this shows:</strong> The gap between validation and training error. Larger positive values indicate overfitting.
                        <br><strong>Good signs:</strong> Small positive gap (&lt; 1.0 points), stable or decreasing over time.
                        <br><strong>Warning signs:</strong> Large gap (&gt; 1.5 points), increasing gap (model learning noise, not patterns).
                    </div>
                    <canvas id="gap_chart_{result['config_iteration']}"></canvas>
                </div>
                
                <script>
                    (function() {{
                        const ctx = document.getElementById('gap_chart_{result['config_iteration']}');
                        if (!ctx) return;
                        
                        // Create dynamic background colors based on gap values
                        const gapData = {train_val_gap};
                        const backgroundColors = gapData.map(value => {{
                            if (value > 1.5) return 'rgba(239, 68, 68, 0.3)';  // Red for large gap
                            if (value > 1.0) return 'rgba(245, 158, 11, 0.3)'; // Orange for moderate gap
                            return 'rgba(16, 185, 129, 0.3)';  // Green for small gap
                        }});
                        
                        new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: {iterations},
                                datasets: [
                                    {{
                                        label: 'Validation - Training Gap',
                                        data: gapData,
                                        borderColor: '#ec4899',
                                        backgroundColor: backgroundColors,
                                        segment: {{
                                            borderColor: ctx => {{
                                                const value = ctx.p1.parsed.y;
                                                if (value > 1.5) return '#ef4444';  // Red
                                                if (value > 1.0) return '#f59e0b';  // Orange
                                                return '#ec4899';  // Pink
                                            }}
                                        }},
                                        fill: true,
                                        tension: 0.3,
                                        borderWidth: 2,
                                        pointRadius: 1,
                                        pointHoverRadius: 4
                                    }},
                                    {{
                                        label: 'Acceptable Threshold (1.0)',
                                        data: Array({len(iterations)}).fill(1.0),
                                        borderColor: '#10b981',
                                        borderWidth: 2,
                                        borderDash: [5, 5],
                                        pointRadius: 0,
                                        fill: false
                                    }}
                                ]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {{
                                    legend: {{ 
                                        position: 'top',
                                        labels: {{
                                            usePointStyle: true,
                                            padding: 15
                                        }}
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                        callbacks: {{
                                            afterLabel: function(context) {{
                                                const value = context.parsed.y;
                                                if (context.datasetIndex === 0) {{
                                                    if (value > 1.5) return '⚠️ High overfitting';
                                                    if (value > 1.0) return '⚠️ Moderate overfitting';
                                                    return '✅ Good generalization';
                                                }}
                                                return '';
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        title: {{ display: true, text: 'MAE Gap (points)' }},
                                        beginAtZero: true
                                    }},
                                    x: {{
                                        title: {{ display: true, text: 'Iteration' }}
                                    }}
                                }}
                            }}
                        }});
                    }})();
                </script>
                """
                )

                # Insights
                final_gap = result["overfitting_gap"]
                gap_pct = result["overfitting_gap_pct"]

                html.append(
                    f"""
                <div class="insight-box">
                    <strong>Training Insights for Config #{result['config_iteration']}:</strong><br>
                    • Model stopped at iteration {best_iter} out of {result['n_estimators']} ({result['early_stop_ratio']*100:.1f}% of max iterations)<br>
                    • Final training MAE: {result['train_mae']:.3f} | Validation MAE: {result['val_mae']:.3f} | Gap: {final_gap:.3f} ({gap_pct:+.1f}%)<br>
                    • Overfitting status: {'❌ Overfitting detected' if result['is_overfitting'] else '✅ No overfitting'} ({result['overfitting_severity']})<br>
                    • Test MAE: {result['test_mae']:.3f} (final unseen data performance)<br>
                    • 2025 spread prediction accuracy: {result['score_2025']:.1f}% ({result['weeks_above_50']}/{result['total_weeks']} weeks above 50%)
                </div>
                """
                )

                # Error Distribution Chart
                # Calculate prediction errors for test set
                # Note: We can't reconstruct exact predictions from stored metrics,
                # so we'll show error statistics based on within_X metrics
                within_3_pct = result["within_3_test"]
                within_7_pct = result["within_7_test"]
                test_mae = result["test_mae"]

                html.append(
                    f"""
                <div class="chart-container">
                    <div class="chart-title">📊 Test Set Error Analysis</div>
                    <div class="chart-description">
                        <strong>What this shows:</strong> Distribution of how close predictions are to actual values on the test set.
                        Shows percentage of predictions within different error thresholds.
                        <br><strong>Good signs:</strong> High percentage within ±3 and ±7 points, most predictions clustered near zero error.
                        <br><strong>Warning signs:</strong> Low percentages, indicating predictions are frequently far from actual values.
                    </div>
                    <canvas id="error_chart_{result['config_iteration']}"></canvas>
                </div>
                
                <script>
                    (function() {{
                        const ctx = document.getElementById('error_chart_{result['config_iteration']}');
                        if (!ctx) return;
                        
                        new Chart(ctx, {{
                            type: 'bar',
                            data: {{
                                labels: ['Within ±3 pts', 'Within ±7 pts', 'Beyond ±7 pts'],
                                datasets: [{{
                                    label: 'Prediction Accuracy',
                                    data: [{within_3_pct:.1f}, {within_7_pct - within_3_pct:.1f}, {100 - within_7_pct:.1f}],
                                    backgroundColor: [
                                        'rgba(16, 185, 129, 0.8)',  // Green for ±3
                                        'rgba(245, 158, 11, 0.8)',  // Orange for ±7
                                        'rgba(239, 68, 68, 0.8)'    // Red for beyond
                                    ],
                                    borderColor: [
                                        'rgb(16, 185, 129)',
                                        'rgb(245, 158, 11)',
                                        'rgb(239, 68, 68)'
                                    ],
                                    borderWidth: 2
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {{
                                    legend: {{ display: false }},
                                    tooltip: {{
                                        callbacks: {{
                                            label: function(context) {{
                                                return context.parsed.y.toFixed(1) + '% of predictions';
                                            }},
                                            afterLabel: function(context) {{
                                                const idx = context.dataIndex;
                                                if (idx === 0) return '(Very accurate)';
                                                if (idx === 1) return '(Reasonably accurate)';
                                                return '(Needs improvement)';
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: true,
                                        max: 100,
                                        title: {{ 
                                            display: true, 
                                            text: 'Percentage of Predictions (%)',
                                            font: {{
                                                size: 12,
                                                weight: 'bold'
                                            }}
                                        }},
                                        ticks: {{
                                            callback: function(value) {{
                                                return value + '%';
                                            }}
                                        }}
                                    }},
                                    x: {{
                                        title: {{ 
                                            display: true, 
                                            text: 'Error Range',
                                            font: {{
                                                size: 12,
                                                weight: 'bold'
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }})();
                </script>
                
                <div class="insight-box">
                    <strong>Error Analysis for Config #{result['config_iteration']}:</strong><br>
                    • Test MAE: {test_mae:.3f} points (average absolute error)<br>
                    • Within ±3 points: {within_3_pct:.1f}% of predictions (excellent accuracy)<br>
                    • Within ±7 points: {within_7_pct:.1f}% of predictions (acceptable for spread betting)<br>
                    • Beyond ±7 points: {100 - within_7_pct:.1f}% of predictions (may lose against spread)
                </div>
                """
                )
            else:
                html.append(
                    """
                <div class="insight-box">
                    <strong>⚠️ Training history not available (no iterations recorded).</strong><br>
                    The training data was captured but appears to be empty. This might happen if the model converged immediately.
                </div>
                """
                )
        else:
            html.append(
                """
                <div class="insight-box">
                    <strong>⚠️ Training history not available for this configuration.</strong><br>
                    The model was trained but detailed iteration-by-iteration metrics were not captured.
                </div>
                """
            )

        html.append("</div>")  # Close config-card

    html.append(
        """
        </div>
        </div>
        </body>
        </html>"""
    )

    # Write file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"💾 Training diagnostics report saved to: {output_file}")


def get_current_config() -> dict:
    """Retrieve the current hyperparameter configuration from best_hyperparameters.json"""
    with open("best_hyperparameters.json", "r") as f:
        current_params = json.load(f)

    config = {"iteration": 0, "params": current_params}
    return config


def generate_compare_configs_report(log_file: str, top_n: int, output_file: str):

    current_config = get_current_config()
    # Check if log file exists
    if not Path(log_file).exists():
        configs = [current_config]
    else:
        configs = parse_config_log(log_file)
        configs.append(current_config)

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    print(f"Log file: {log_file}")

    # Parse configurations
    print(f"\nParsing configurations...")
    print(f"✓ Found {len(configs)} valid configurations")

    if len(configs) == 0:
        print("❌ No configurations found in log file!")
        return

    # Limit to top N if specified
    if top_n and top_n < len(configs):
        # Sort by original val_mae (lower is better)
        configs.sort(key=lambda x: x.get("val_mae", float("inf")))
        configs = configs[:top_n]
        print(f"📊 Evaluating top {top_n} configurations")

    # Load data (same as train.py)
    print(f"\nLoading data...")
    training_data = run_query(
        "SELECT * FROM training_data WHERE season <= 2025 ORDER BY season ASC, week ASC"
    )
    df = pd.DataFrame(training_data)
    df["point_differential"] = df["home_score"] - df["away_score"]

    print(f"✓ Loaded {len(df)} games")

    # Feature engineering
    print(f"\nEngineering features...")
    model_data, best_features = engineer_features(df)
    model_data = model_data.dropna(subset=best_features + [target])
    model_data = model_data.sort_values(["season", "week"]).reset_index(drop=True)

    print(f"✓ {len(best_features)} features ready")

    # Create splits
    model_data_for_split = model_data.copy()
    # train_idx = int(len(model_data_for_split) * 0.7)
    # val_idx = int(len(model_data_for_split) * 0.85)

    # train_df = model_data_for_split.iloc[:train_idx]
    # val_df = model_data_for_split.iloc[train_idx:val_idx]
    # test_df = model_data_for_split.iloc[val_idx:]

    train_df = model_data_for_split[model_data_for_split["season"] <= 2023]
    val_df = model_data_for_split[model_data_for_split["season"] == 2024]
    test_df = model_data_for_split[model_data_for_split["season"] == 2025]

    X_train = train_df[best_features]
    y_train = train_df[target]
    X_val = val_df[best_features]
    y_val = val_df[target]
    X_test = test_df[best_features]
    y_test = test_df[target]

    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Train and evaluate each configuration
    print(f"\n{'='*80}")
    print(f"TRAINING AND EVALUATING CONFIGURATIONS")
    print(f"{'='*80}\n")

    results = []
    for i, config in enumerate(configs, 1):
        config_num = config.get("iteration", i)
        print(f"[{i}/{len(configs)}] Training Config #{config_num}...", end=" ")

        try:
            result = train_and_evaluate_config(
                config,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                best_features,
                base_features,
            )
            results.append(result)
            print(
                f"✓ Test MAE: {result['test_mae']:.3f} | 2025: {result['score_2025']:.1f}%"
            )
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"❌ Error: {e}")
            continue

    if len(results) == 0:
        print("\n❌ No configurations successfully evaluated!")
        return

    # Rank results
    print(f"\n{'='*80}")
    print(f"RANKING RESULTS")
    print(f"{'='*80}\n")

    ranked_results = rank_results(results)

    # Print top 5
    print("🏆 Top 5 Configurations:")
    print(f"{'='*80}")
    for rank, result in enumerate(ranked_results[:5], 1):
        print(
            f"\n#{rank} - Config {result['config_iteration']} (Score: {result['overall_score']:.1f}/100)"
        )
        print(
            f"  2025: {result['score_2025']:.1f}% | Weeks >50%: {result['weeks_above_50']}/{result['total_weeks']} | Consistency: ±{result['consistency_std']:.1f}%"
        )

    # Generate HTML report
    print(f"\n{'='*80}")
    print(f"GENERATING REPORTS")
    print(f"{'='*80}")
    generate_html_report(ranked_results, output_file)

    # Generate training diagnostics report
    diagnostics_file = output_file.replace(".html", "_diagnostics.html")
    generate_training_diagnostics_report(ranked_results, diagnostics_file)

    # Print recommendation
    best = ranked_results[0]
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\n🏆 Use Configuration #{best['config_iteration']}")
    print(f"   2025 Total: {best['score_2025']:.1f}%")
    print(f"   Consistency: ±{best['consistency_std']:.1f}% (lower is better)")
    print(f"   Weeks >50%: {best['weeks_above_50']}/{best['total_weeks']}")
    print(f"   Minimum Week: {best['min_week_accuracy']:.1f}%")
    print(f"   Overall Score: {best['overall_score']:.1f}/100 points")
    print(f"\n{'='*80}\n")
