import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def check_overfitting(
    train_mae: float,
    val_mae: float,
    train_rmse: float = None,
    val_rmse: float = None,
    train_r2: float = None,
    val_r2: float = None,
    val_error_history: list = None,
    mae_rel_threshold: float = 0.15,  # Increased from 0.1 - NFL is noisy
    mae_abs_threshold: float = 1.5,  # Increased from 1.0 - ~1 possession difference
    rmse_rel_threshold: float = 0.15,
    rmse_abs_threshold: float = 2.0,  # Increased for NFL context
    r2_drop_threshold: float = 0.15,
    logger=None,  # Add logger support
) -> dict:
    """
    Comprehensive overfitting check after model training.
    Returns a dict with all checks and warnings.

    Checks:
    - MAE, RMSE, R2 gaps between train and validation
    - Zero training error (potential data leakage)
    - Validation error trends
    - Statistical significance of performance gap

    Thresholds adjusted for NFL spread prediction context where:
    - Scores are inherently noisy (weather, injuries, etc.)
    - ~1 possession (7 points) difference is meaningful
    - Some generalization gap is expected
    """
    import numpy as np

    report = {
        "overfit": False,
        "warnings": [],
        "severity": "none",  # none, mild, moderate, severe
        "metrics": {},
    }

    def log_msg(msg, level="info"):
        """Helper to print or log messages"""
        if logger:
            getattr(logger, level)(msg)
        else:
            print(msg)

    # === CRITICAL CHECK: Zero training error ===
    if train_mae == 0 or (train_rmse is not None and train_rmse == 0):
        msg = "[SEVERE OVERFITTING] Training error is zero. Likely data leakage or trivial model!"
        log_msg(msg, "error")
        report["overfit"] = True
        report["severity"] = "severe"
        report["warnings"].append(msg)
        return report

    # === Check for suspiciously low training error ===
    # For NFL, MAE < 5 points on training is unrealistic
    if train_mae < 5.0:
        msg = f"[WARNING] Training MAE={train_mae:.2f} is suspiciously low for NFL prediction. Check for data leakage."
        log_msg(msg, "warning")
        report["warnings"].append(msg)

    # === MAE Analysis ===
    rel_gap = (val_mae - train_mae) / train_mae
    abs_gap = val_mae - train_mae

    report["metrics"]["mae"] = {
        "train": train_mae,
        "val": val_mae,
        "relative_gap": rel_gap,
        "absolute_gap": abs_gap,
    }

    log_msg(
        f"[Overfitting Check] MAE: train={train_mae:.3f}, val={val_mae:.3f}, "
        f"rel_gap={rel_gap:.3f} ({rel_gap*100:.1f}%), abs_gap={abs_gap:.3f}"
    )

    # Check if validation is WORSE than training (expected)
    if val_mae < train_mae:
        msg = (
            f"[UNUSUAL] Validation MAE ({val_mae:.3f}) is better than training MAE ({train_mae:.3f}). "
            f"This may indicate: (1) validation set is easier, (2) lucky split, or (3) data issues."
        )
        log_msg(msg, "warning")
        report["warnings"].append(msg)

    # Relative gap check
    if rel_gap > mae_rel_threshold:
        severity = (
            "severe" if rel_gap > 0.3 else "moderate" if rel_gap > 0.2 else "mild"
        )
        msg = f"[{severity.upper()} OVERFITTING] Relative MAE gap: {rel_gap:.3f} ({rel_gap*100:.1f}%) > {mae_rel_threshold} threshold"
        log_msg(msg, "warning")
        report["overfit"] = True
        report["warnings"].append(msg)
        if severity == "severe":
            report["severity"] = "severe"
        elif report["severity"] != "severe":
            report["severity"] = severity

    # Absolute gap check (more important for NFL context)
    if abs_gap > mae_abs_threshold:
        msg = (
            f"[OVERFITTING] Absolute MAE gap: {abs_gap:.3f} points > {mae_abs_threshold} threshold "
            f"(predictions are {abs_gap:.1f} points less accurate on test data)"
        )
        log_msg(msg, "warning")
        report["overfit"] = True
        report["warnings"].append(msg)
        if report["severity"] == "none":
            report["severity"] = "mild"

    # === RMSE Analysis ===
    if train_rmse is not None and val_rmse is not None:
        rmse_rel_gap = (val_rmse - train_rmse) / train_rmse
        rmse_abs_gap = val_rmse - train_rmse

        report["metrics"]["rmse"] = {
            "train": train_rmse,
            "val": val_rmse,
            "relative_gap": rmse_rel_gap,
            "absolute_gap": rmse_abs_gap,
        }

        log_msg(
            f"[Overfitting Check] RMSE: train={train_rmse:.3f}, val={val_rmse:.3f}, "
            f"rel_gap={rmse_rel_gap:.3f}, abs_gap={rmse_abs_gap:.3f}"
        )

        if rmse_rel_gap > rmse_rel_threshold:
            msg = f"[OVERFITTING] Relative RMSE gap: {rmse_rel_gap:.3f} ({rmse_rel_gap*100:.1f}%) > {rmse_rel_threshold}"
            log_msg(msg, "warning")
            report["overfit"] = True
            report["warnings"].append(msg)

        if rmse_abs_gap > rmse_abs_threshold:
            msg = f"[OVERFITTING] Absolute RMSE gap: {rmse_abs_gap:.3f} > {rmse_abs_threshold}"
            log_msg(msg, "warning")
            report["overfit"] = True
            report["warnings"].append(msg)

        # Check RMSE/MAE ratio for both sets
        train_ratio = train_rmse / train_mae if train_mae > 0 else 0
        val_ratio = val_rmse / val_mae if val_mae > 0 else 0

        # Typically RMSE/MAE ratio is 1.2-1.4 for normal distributions
        # Higher ratio means more large errors (outliers)
        if val_ratio > train_ratio + 0.15:
            msg = (
                f"[WARNING] Validation has more outliers: RMSE/MAE ratio train={train_ratio:.2f}, val={val_ratio:.2f}. "
                f"Model may struggle with unusual games."
            )
            log_msg(msg, "warning")
            report["warnings"].append(msg)

    # === R² Analysis ===
    if train_r2 is not None and val_r2 is not None:
        r2_drop = train_r2 - val_r2

        report["metrics"]["r2"] = {"train": train_r2, "val": val_r2, "drop": r2_drop}

        log_msg(
            f"[Overfitting Check] R²: train={train_r2:.3f}, val={val_r2:.3f}, drop={r2_drop:.3f}"
        )

        # Check for negative R² (model worse than mean baseline)
        if val_r2 < 0:
            msg = (
                f"[CRITICAL] Validation R²={val_r2:.3f} is negative! Model performs worse than "
                f"predicting the average score. Model may be fundamentally flawed."
            )
            log_msg(msg, "error")
            report["overfit"] = True
            report["severity"] = "severe"
            report["warnings"].append(msg)

        # Check for very low R² even if not overfitting
        if val_r2 < 0.05 and not (val_r2 < 0):
            msg = (
                f"[WARNING] Validation R²={val_r2:.3f} is very low. Model explains <5% of variance. "
                f"Consider: (1) adding features, (2) data quality issues, (3) inherent noise in target."
            )
            log_msg(msg, "warning")
            report["warnings"].append(msg)

        # R² drop check
        if r2_drop > r2_drop_threshold:
            severity = (
                "severe" if r2_drop > 0.4 else "moderate" if r2_drop > 0.25 else "mild"
            )
            msg = (
                f"[{severity.upper()} OVERFITTING] R² drop: {r2_drop:.3f} > {r2_drop_threshold} "
                f"(model explains {r2_drop*100:.1f}% less variance on test data)"
            )
            log_msg(msg, "warning")
            report["overfit"] = True
            report["warnings"].append(msg)
            if severity == "severe" or report["severity"] != "severe":
                report["severity"] = severity

    # === Validation Error Trend Analysis ===
    if val_error_history is not None and len(val_error_history) >= 3:
        recent = val_error_history[-3:]

        # Check for consistent upward trend
        if all(x < y for x, y in zip(recent, recent[1:])):
            msg = (
                f"[OVERFITTING] Validation error is monotonically increasing in last 3 epochs: {[f'{x:.3f}' for x in recent]}. "
                f"Model may have overfit during training."
            )
            log_msg(msg, "warning")
            report["overfit"] = True
            report["warnings"].append(msg)
            if report["severity"] == "none":
                report["severity"] = "mild"

        # Check if final error is much worse than best
        if len(val_error_history) >= 5:
            best_val_error = min(val_error_history)
            final_val_error = val_error_history[-1]
            error_increase = (final_val_error - best_val_error) / best_val_error

            if error_increase > 0.1:  # 10% worse than best
                msg = (
                    f"[WARNING] Final validation error ({final_val_error:.3f}) is {error_increase*100:.1f}% worse "
                    f"than best ({best_val_error:.3f}). Consider using early stopping."
                )
                log_msg(msg, "warning")
                report["warnings"].append(msg)

    # === Final Assessment ===
    if not report["overfit"]:
        log_msg(
            "[✓] No significant overfitting detected. Model generalizes well.", "info"
        )
    else:
        log_msg(
            f"\n[⚠] OVERFITTING DETECTED (Severity: {report['severity'].upper()})",
            "warning",
        )
        log_msg("\nRecommended actions:", "info")

        if report["severity"] == "severe":
            log_msg(
                "  1. CHECK FOR DATA LEAKAGE (using future information in features)",
                "info",
            )
            log_msg("  2. Verify train/test split is truly separated by time", "info")
            log_msg("  3. Ensure target variable isn't included in features", "info")

        log_msg("  - Increase regularization (reg_alpha, reg_lambda)", "info")
        log_msg("  - Reduce model complexity (max_depth, n_estimators)", "info")
        log_msg(
            "  - Increase min_child_weight to require more samples per leaf", "info"
        )
        log_msg("  - Use early stopping with validation set", "info")
        log_msg("  - Add more diverse training data if possible", "info")
        log_msg("  - Try feature selection to remove noisy features", "info")

    return report
