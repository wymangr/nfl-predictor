"""Registry describing every report the dashboard can render."""

import io
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.helpers.database_helpers import run_query
from src.model.predict import get_future_predictions
from src.reports.analyze_future_bucket_confidence import print_future_analysis
from src.reports.analyze_past_bucket_confidence import print_past_bucket_analysis
from src.reports.compare_configs_report import generate_compare_configs_report
from src.reports.nfl_future_prediction_report import generate_future_predictions_report
from src.reports.nfl_past_prediction_report import (
    generate_past_prediction_report,
    load_data,
    save_accuracy_metrics_to_db,
)
from src.reports.nfl_power_ranking_report import generate_power_rankings_report
from src.reports.past_predictions_analysis import print_bucket_analysis
from src.reports.qb_changes import get_qb_change


@dataclass(frozen=True)
class Report:
    key: str
    label: str
    description: str
    kind: str  # "html" renders into an iframe, "text" into a <pre>
    generate: Callable[..., None]
    output_file: Callable[..., Path] | None = None
    # Slow reports serve their last generated file and only rebuild on request.
    cached: bool = False
    seasons_param: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False)


def _past_predictions() -> None:
    df = load_data()
    generate_past_prediction_report(df)
    save_accuracy_metrics_to_db(df)


def _future_predictions() -> None:
    get_future_predictions()
    generate_future_predictions_report()


def _compare_configs() -> None:
    generate_compare_configs_report(
        "random_search_results.txt", None, "config_comparison.html"
    )


REPORTS: dict[str, Report] = {
    r.key: r
    for r in [
        Report(
            key="future-predictions",
            label="Future Predictions",
            description="Re-runs the model, then shows picks and confidence metrics.",
            kind="html",
            generate=_future_predictions,
            output_file=lambda **_: Path("nfl_future_prediction_report.html"),
        ),
        Report(
            key="past-predictions",
            label="Past Predictions",
            description="Historical accuracy broken down by team, spread, and situation.",
            kind="html",
            generate=_past_predictions,
            output_file=lambda **_: Path("nfl_past_prediction_report.html"),
        ),
        Report(
            key="power-rankings",
            label="Power Rankings",
            description="Weekly power rankings, strength of schedule, and adjusted ranks.",
            kind="html",
            generate=lambda season: generate_power_rankings_report(season=season),
            output_file=lambda season, **_: Path(f"nfl_power_rankings_{season}.html"),
            seasons_param=True,
        ),
        Report(
            key="compare-configs",
            label="Config Comparison",
            description="Ranked hyperparameter configs. Retrains models, so refresh is manual.",
            kind="html",
            generate=_compare_configs,
            output_file=lambda **_: Path("config_comparison.html"),
            cached=True,
        ),
        Report(
            key="compare-configs-diagnostics",
            label="Config Diagnostics",
            description="Training diagnostics produced alongside the config comparison.",
            kind="html",
            generate=_compare_configs,
            output_file=lambda **_: Path("config_comparison_diagnostics.html"),
            cached=True,
        ),
        Report(
            key="qb-changes",
            label="QB Changes",
            description="Quarterback changes for the current week.",
            kind="text",
            generate=get_qb_change,
        ),
        Report(
            key="bucket-analysis",
            label="Bucket Analysis",
            description="Past prediction performance grouped into metric buckets.",
            kind="text",
            generate=print_bucket_analysis,
        ),
        Report(
            key="bucket-analysis-future",
            label="Bucket Analysis (Future)",
            description="Bucket confidence for the upcoming week's predictions.",
            kind="text",
            generate=print_future_analysis,
        ),
        Report(
            key="bucket-analysis-past",
            label="Bucket Analysis (Past)",
            description="Bucket confidence accuracy across completed games.",
            kind="text",
            generate=print_past_bucket_analysis,
        ),
    ]
}


def available_seasons() -> list[int]:
    """Seasons that have power ranking data, newest first."""
    rows = run_query(
        "SELECT DISTINCT season FROM team_power_rankings ORDER BY season DESC"
    )
    return [row["season"] for row in rows]


class ReportNotBuilt(Exception):
    """A cached report has never been generated."""


def render(report: Report, season: int | None = None, force: bool = False) -> str:
    """Produce the report body, regenerating it unless a cached copy will do."""
    kwargs = {"season": season} if report.seasons_param else {}

    if report.kind == "text":
        with report.lock:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                report.generate(**kwargs)
        return buffer.getvalue()

    assert report.output_file is not None
    path = report.output_file(**kwargs)

    with report.lock:
        if report.cached and not force:
            if not path.exists():
                raise ReportNotBuilt(report.key)
        else:
            report.generate(**kwargs)
        return path.read_text(encoding="utf-8")
