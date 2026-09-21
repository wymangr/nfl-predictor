"""Local dashboard that renders the nfl reports on demand."""

import html
import threading
import time
import traceback

from flask import Flask, abort, jsonify, render_template, request

from src.dashboard.reports import (
    REPORTS,
    ReportNotBuilt,
    available_seasons,
    available_weeks,
    render,
)
from src.data.data import backfil_data
from src.data.update_spreads import update_current_spreads
from src.model.train import train_model

app = Flask(__name__)

# Every task rewrites shared state, so only ever run one at a time.
_task_lock = threading.Lock()

_TASKS = {
    "spreads": update_current_spreads,
    "full": backfil_data,
    "train": train_model,
}


def _error_page(title: str, detail: str) -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font:14px/1.5 system-ui,sans-serif;padding:2rem;color:#c92a2a}"
        "pre{white-space:pre-wrap;color:#495057;background:#f1f3f5;padding:1rem;"
        "border-radius:6px}</style>"
        f"<h2>{html.escape(title)}</h2><pre>{html.escape(detail)}</pre>"
    )


@app.get("/")
def index():
    seasons = available_seasons()
    return render_template(
        "index.html",
        reports=list(REPORTS.values()),
        seasons=seasons,
        weeks=available_weeks(seasons[0]) if seasons else [],
    )


@app.get("/report/<key>")
def report(key: str):
    entry = REPORTS.get(key)
    if entry is None:
        abort(404)

    season = None
    if entry.seasons_param:
        seasons = available_seasons()
        if not seasons:
            return _error_page(
                "No season data", "Run `nfl data refresh` to populate power rankings."
            )
        season = request.args.get("season", type=int)
        if season not in seasons:
            season = seasons[0]

    week = None
    if entry.weeks_param and season is not None:
        weeks = available_weeks(season)
        week = request.args.get("week", type=int)
        if week not in weeks:
            week = weeks[0] if weeks else None

    force = request.args.get("force") == "1"

    try:
        body = render(entry, season=season, week=week, force=force)
    except ReportNotBuilt:
        return _error_page(
            "Not generated yet",
            "This report is expensive to build, so it is not generated "
            'automatically. Click "Regenerate" to build it.',
        )
    except Exception:
        return _error_page(f"{entry.label} failed to generate", traceback.format_exc())

    if entry.kind == "text":
        return (
            "<!doctype html><meta charset='utf-8'>"
            "<style>body{margin:0;background:#0b0d10}"
            "pre{font:12px/1.45 ui-monospace,Consolas,monospace;color:#dee2e6;"
            "padding:1.25rem;white-space:pre;overflow:auto}</style>"
            f"<pre>{html.escape(body)}</pre>"
        )
    return body


@app.get("/api/seasons")
def seasons():
    return jsonify(available_seasons())


@app.get("/api/weeks")
def weeks():
    season = request.args.get("season", type=int)
    if season is None:
        return jsonify([])
    return jsonify(available_weeks(season))


@app.post("/api/run/<task>")
def run_task(task: str):
    action = _TASKS.get(task)
    if action is None:
        abort(404)

    # Forces a CORS preflight, so another site cannot trigger a task.
    if request.headers.get("X-Requested-With") != "nfl-dashboard":
        abort(403)

    if not _task_lock.acquire(blocking=False):
        return jsonify(ok=False, error="Another task is already running."), 409

    started = time.monotonic()
    try:
        action()
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=f"{type(exc).__name__}: {exc}"), 500
    finally:
        _task_lock.release()

    return jsonify(ok=True, seconds=round(time.monotonic() - started))


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    app.run(host=host, port=port, debug=False, threaded=True)
