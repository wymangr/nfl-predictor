"""Model accuracy sliced by every spread-related factor.

Answers "should I fade this pick?" by showing how the model has historically
performed for the spread size, the market line, and the gap between the two.
"""

import pandas as pd
from datetime import datetime

from src.helpers.database_helpers import get_db_engine

# Cells built on fewer than this many games are shown muted.
MIN_SAMPLE = 10

SPREAD_BUCKETS = [
    ("Pick'em (≤2.5)", lambda m: m <= 2.5),
    ("Field goal (3-3.5)", lambda m: 3 <= m <= 3.5),
    ("Small (4-6)", lambda m: 4 <= m <= 6),
    ("Mid (6.5-9.5)", lambda m: 6.5 <= m <= 9.5),
    ("Large (10+)", lambda m: m >= 10),
]

DELTA_BUCKETS = [
    ("Line favors dog by 2+", lambda d: d <= -2),
    ("Line favors dog by 1-2", lambda d: -2 < d <= -1),
    ("Line favors dog by <1", lambda d: -1 < d < 0),
    ("Spreads agree", lambda d: d == 0),
    ("Line favors fav by <1", lambda d: 0 < d < 1),
    ("Line favors fav by 1-2", lambda d: 1 <= d < 2),
    ("Line favors fav by 2+", lambda d: d >= 2),
]

# Spreads move in half points, so this walks one step at a time. The tails are
# collapsed because individual half points beyond +/-2 have almost no games.
DELTA_BUCKETS_FINE = [
    ("Line favors dog by 2+", lambda d: d <= -2),
    ("Line favors dog by 1.5", lambda d: d == -1.5),
    ("Line favors dog by 1.0", lambda d: d == -1.0),
    ("Line favors dog by 0.5", lambda d: d == -0.5),
    ("Spreads agree", lambda d: d == 0),
    ("Line favors fav by 0.5", lambda d: d == 0.5),
    ("Line favors fav by 1.0", lambda d: d == 1.0),
    ("Line favors fav by 1.5", lambda d: d == 1.5),
    ("Line favors fav by 2.0", lambda d: d == 2.0),
    ("Line favors fav by 2.5+", lambda d: d >= 2.5),
]

CONFIDENCE_BUCKETS = [
    ("1-4 (least)", lambda c: c <= 4),
    ("5-8", lambda c: 5 <= c <= 8),
    ("9-12", lambda c: 9 <= c <= 12),
    ("13-16 (most)", lambda c: c >= 13),
]

# Compact forms of the delta buckets, for use as a column in the pick tables.
DELTA_SHORT = {
    "Line favors dog by 2+": "Dog 2+",
    "Line favors dog by 1-2": "Dog 1-2",
    "Line favors dog by <1": "Dog <1",
    "Spreads agree": "Agree",
    "Line favors fav by <1": "Fav <1",
    "Line favors fav by 1-2": "Fav 1-2",
    "Line favors fav by 2+": "Fav 2+",
}


def delta_bucket_stats():
    """Historical accuracy per (line-disagreement bucket, pick side), plus the baseline."""
    df = load_spread_data()
    if df.empty:
        return 0.0, {}

    df = df[df["spread_line"].notna()]
    df["delta_bucket"] = df["delta"].apply(lambda v: _bucket(v, DELTA_BUCKETS))
    baseline = df["correct"].mean() * 100
    stats = {
        key: (int(g["correct"].sum()), len(g), g["correct"].mean() * 100)
        for key, g in df.groupby(["delta_bucket", "pick_side"])
    }
    return baseline, stats


def delta_bucket_cell(prediction, baseline, stats):
    """Which Spread Analysis bucket a game lands in, with that bucket's track record."""
    line = prediction.get("spread_line")
    if line is None or pd.isna(line):
        return '<td data-value="">—</td>'

    home_fav = prediction["spread_favorite"] == prediction["home_team"]
    delta = (line if home_fav else -line) - abs(prediction["spread"])
    bucket = _bucket(delta, DELTA_BUCKETS)
    side = (
        "Favorite"
        if prediction["predicted_winner"] == prediction["spread_favorite"]
        else "Underdog"
    )

    correct, total, pct = stats.get((bucket, side), (0, 0, None))
    if total == 0:
        history = '<span class="sub">no history</span>'
        colors = "background-color:#fff;color:#adb5bd"
    else:
        history = (
            f'<span style="display:block;font-size:11px;font-weight:normal;color:#666">'
            f"{pct:.0f}% · {correct}/{total}</span>"
        )
        gap = pct - baseline
        if total < MIN_SAMPLE:
            colors = "background-color:#fff;color:#adb5bd"
        elif gap >= 5:
            colors = "background-color:#d4edda;color:#155724"
        elif gap <= -5:
            colors = "background-color:#f8d7da;color:#721c24"
        else:
            colors = "background-color:#f1f3f5;color:#495057"

    return (
        f'<td data-value="{delta}" style="{colors};font-weight:bold"'
        f' title="{bucket} — model picking the {side.lower()}">'
        f"{DELTA_SHORT.get(bucket, bucket)}{history}</td>"
    )


def load_spread_data():
    """Graded picks joined with the market line. Pushes excluded."""
    query = """
        SELECT pp.season, pp.week, pp.home_team, pp.away_team, pp.spread,
               pp.spread_favorite, pp.predicted_winner, pp.confidence,
               pp.confidence_score, pp.correct, g.spread_line
        FROM past_predictions pp
        JOIN games g ON pp.game_id = g.game_id
        WHERE pp.correct != 'push'
    """
    df = pd.read_sql_query(query, get_db_engine())
    if df.empty:
        return df

    df["correct"] = pd.to_numeric(df["correct"])
    df["yahoo_mag"] = df["spread"].abs()
    df["pick_side"] = df["predicted_winner"].where(
        df["predicted_winner"] != df["spread_favorite"], "Favorite"
    )
    df["pick_side"] = df["pick_side"].where(df["pick_side"] == "Favorite", "Underdog")

    # nflverse is positive when home is favored; re-express from the favorite's side.
    home_fav = df["spread_favorite"] == df["home_team"]
    df["line_toward_fav"] = df["spread_line"].where(home_fav, -df["spread_line"])
    df["line_mag"] = df["line_toward_fav"].abs()
    # Positive means the market line is more favorite-leaning than Yahoo.
    df["delta"] = (df["line_toward_fav"] - df["yahoo_mag"]).round(2)
    return df


def _bucket(value, buckets):
    if pd.isna(value):
        return None
    for label, test in buckets:
        if test(value):
            return label
    return None


def _record(df):
    if df.empty:
        return 0, 0, None
    return int(df["correct"].sum()), len(df), df["correct"].mean() * 100


def _cell(df, baseline):
    """Accuracy cell coloured by how far it sits from the model's overall rate."""
    correct, total, pct = _record(df)
    if total == 0:
        return '<td class="empty">-</td>'

    delta = pct - baseline
    if total < MIN_SAMPLE:
        cls = "low"
    elif delta >= 5:
        cls = "good"
    elif delta <= -5:
        cls = "bad"
    else:
        cls = "neutral"

    return (
        f'<td class="{cls}">{pct:.1f}%'
        f'<span class="sub">{correct}/{total} &middot; {delta:+.1f}</span></td>'
    )


def build_breakdown(df, column, buckets, title, blurb, baseline):
    """One row per bucket, split into all / favorite / underdog picks."""
    rows = []
    for label, _ in buckets:
        subset = df[df[column] == label]
        if subset.empty:
            continue
        rows.append(
            f"<tr><th class='row-label'>{label}</th>"
            f"{_cell(subset, baseline)}"
            f"{_cell(subset[subset.pick_side == 'Favorite'], baseline)}"
            f"{_cell(subset[subset.pick_side == 'Underdog'], baseline)}"
            "</tr>"
        )

    if not rows:
        return ""

    return f"""
    <div class="card">
        <h2>{title}</h2>
        <p class="blurb">{blurb}</p>
        <table>
            <tr><th>Bucket</th><th>All picks</th><th>Model picks favorite</th><th>Model picks underdog</th></tr>
            {''.join(rows)}
        </table>
    </div>
    """


def build_matrix(
    df, row_col, row_buckets, col_col, col_buckets, title, blurb, baseline
):
    """Accuracy grid crossing two bucketed factors."""
    col_labels = [label for label, _ in col_buckets if (df[col_col] == label).any()]
    if not col_labels:
        return ""

    header = "".join(f"<th>{c}</th>" for c in col_labels)
    rows = []
    for label, _ in row_buckets:
        subset = df[df[row_col] == label]
        if subset.empty:
            continue
        cells = "".join(
            _cell(subset[subset[col_col] == c], baseline) for c in col_labels
        )
        rows.append(f"<tr><th class='row-label'>{label}</th>{cells}</tr>")

    if not rows:
        return ""

    return f"""
    <div class="card">
        <h2>{title}</h2>
        <p class="blurb">{blurb}</p>
        <table>
            <tr><th></th>{header}</tr>
            {''.join(rows)}
        </table>
    </div>
    """


def find_edges(df, baseline):
    """Buckets that beat or trail the model's overall accuracy by the widest margin."""
    combos = []
    for side in ("Favorite", "Underdog"):
        for column, buckets, kind in (
            ("yahoo_bucket", SPREAD_BUCKETS, "Yahoo spread"),
            ("line_bucket", SPREAD_BUCKETS, "market line"),
            ("delta_bucket", DELTA_BUCKETS, "line vs Yahoo"),
        ):
            for label, _ in buckets:
                subset = df[(df.pick_side == side) & (df[column] == label)]
                correct, total, pct = _record(subset)
                if total >= MIN_SAMPLE:
                    combos.append(
                        (pct - baseline, pct, correct, total, side, kind, label)
                    )

    if not combos:
        return ""

    combos.sort(reverse=True)
    best, worst = combos[:3], combos[-3:][::-1]

    def render(items, cls):
        return "".join(
            f"<tr><td class='{cls}'>{pct:.1f}%<span class='sub'>{c}/{t} &middot; {d:+.1f}</span></td>"
            f"<td>Model picks <strong>{side.lower()}</strong></td>"
            f"<td>{kind}: {label}</td></tr>"
            for d, pct, c, t, side, kind, label in items
        )

    return f"""
    <div class="card">
        <h2>🎯 Where the model is strongest and weakest</h2>
        <p class="blurb">
            Buckets with at least {MIN_SAMPLE} games, ranked by distance from the
            model's overall {baseline:.1f}% accuracy.
        </p>
        <h3 class="good-head">Trust these</h3>
        <table>{render(best, 'good')}</table>
        <h3 class="bad-head">Consider fading these</h3>
        <table>{render(worst, 'bad')}</table>
    </div>
    """


STYLES = """
    body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto;
           padding: 20px; background-color: #f5f5f5; color: #222; }
    h1 { color: #013369; border-bottom: 3px solid #D50A0A; padding-bottom: 10px; }
    h2 { color: #013369; margin-top: 0; font-size: 20px; }
    h3 { font-size: 15px; margin-bottom: 6px; }
    .good-head { color: #1e7e34; }
    .bad-head { color: #b21f2d; }
    .card { background: #fff; padding: 20px; margin: 20px 0; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .blurb { color: #666; font-size: 13px; margin-top: 0; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; }
    th { background-color: #013369; color: #fff; padding: 9px; text-align: left;
         font-size: 13px; }
    th.row-label { background: #eef1f5; color: #013369; white-space: nowrap; }
    td { padding: 9px; border-bottom: 1px solid #ddd; font-size: 13px;
         font-weight: bold; }
    td .sub { display: block; font-size: 11px; font-weight: normal; color: #666;
              margin-top: 2px; }
    td.good { background-color: #d4edda; color: #155724; }
    td.bad { background-color: #f8d7da; color: #721c24; }
    td.neutral { background-color: #f1f3f5; color: #495057; }
    td.low { background-color: #fff; color: #adb5bd; }
    td.empty { color: #ced4da; font-weight: normal; }
    .summary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color: #fff; padding: 20px; border-radius: 8px; }
    .summary h2 { color: #fff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr));
            gap: 18px; margin-top: 12px; }
    .grid div { text-align: center; }
    .grid span { display: block; font-size: 13px; opacity: .9; }
    .grid strong { font-size: 30px; }
    .legend { font-size: 12px; color: #666; }
    .timestamp { color: #666; font-size: 12px; text-align: right; }
"""


def generate_spread_analysis_report(output_file="nfl_spread_analysis_report.html"):
    """Build the spread factor accuracy report."""
    df = load_spread_data()
    if df.empty:
        print("⚠️  No graded predictions available for spread analysis")
        return

    df["yahoo_bucket"] = df["yahoo_mag"].apply(lambda v: _bucket(v, SPREAD_BUCKETS))
    df["line_bucket"] = df["line_mag"].apply(lambda v: _bucket(v, SPREAD_BUCKETS))
    df["delta_bucket"] = df["delta"].apply(lambda v: _bucket(v, DELTA_BUCKETS))
    df["delta_bucket_fine"] = df["delta"].apply(
        lambda v: _bucket(v, DELTA_BUCKETS_FINE)
    )
    df["confidence_bucket"] = df["confidence"].apply(
        lambda v: _bucket(v, CONFIDENCE_BUCKETS)
    )

    correct, total, baseline = _record(df)
    fav = df[df.pick_side == "Favorite"]
    dog = df[df.pick_side == "Underdog"]
    with_line = df[df["spread_line"].notna()]

    sections = [
        build_breakdown(
            df,
            "yahoo_bucket",
            SPREAD_BUCKETS,
            "📏 Accuracy by Yahoo spread size",
            "How the model does at each spread size, and whether it was taking the favorite or the dog.",
            baseline,
        ),
        build_breakdown(
            with_line,
            "line_bucket",
            SPREAD_BUCKETS,
            "📐 Accuracy by market line size",
            "Same split, bucketed on the nflverse closing line instead of the Yahoo number.",
            baseline,
        ),
        build_breakdown(
            with_line,
            "delta_bucket",
            DELTA_BUCKETS,
            "⚖️ Accuracy by Yahoo vs market line disagreement",
            "Positive buckets mean the closing line is more favorite-leaning than Yahoo had it - "
            "the market moved toward the favorite after Yahoo set its number.",
            baseline,
        ),
        build_breakdown(
            with_line,
            "delta_bucket_fine",
            DELTA_BUCKETS_FINE,
            "🔬 Same disagreement, one half point at a time",
            "The buckets above split into single half-point steps. Spreads only move in "
            "half points, so these are the smallest slices available - which also makes "
            "them the thinnest, so lean on the table above where the counts are larger.",
            baseline,
        ),
        build_matrix(
            df,
            "confidence_bucket",
            CONFIDENCE_BUCKETS,
            "yahoo_bucket",
            SPREAD_BUCKETS,
            "🔢 Confidence rank vs Yahoo spread size",
            "Confidence is the model's within-week rank (16 = most confident).",
            baseline,
        ),
        build_matrix(
            with_line,
            "confidence_bucket",
            CONFIDENCE_BUCKETS,
            "delta_bucket",
            DELTA_BUCKETS,
            "🔀 Confidence rank vs line disagreement",
            "Whether a confident pick still holds up when the market disagrees with Yahoo.",
            baseline,
        ),
        find_edges(with_line, baseline),
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NFL Spread Factor Analysis</title>
    <style>{STYLES}</style>
</head>
<body>
    <h1>📊 Spread Factor Analysis</h1>

    <div class="card summary">
        <h2>Model accuracy against the spread</h2>
        <div class="grid">
            <div><span>Graded picks</span><strong>{total}</strong></div>
            <div><span>Overall accuracy</span><strong>{baseline:.1f}%</strong></div>
            <div><span>Picking favorite</span><strong>{_record(fav)[2] or 0:.1f}%</strong></div>
            <div><span>Picking underdog</span><strong>{_record(dog)[2] or 0:.1f}%</strong></div>
            <div><span>Seasons</span><strong>{df['season'].min()}-{df['season'].max()}</strong></div>
        </div>
    </div>

    <div class="card">
        <p class="legend">
            Every cell shows accuracy, then <em>record</em> and the gap versus the model's
            overall {baseline:.1f}%.
            <span class="good" style="padding:2px 6px;border-radius:3px;">green</span> is 5+ points better,
            <span class="bad" style="padding:2px 6px;border-radius:3px;">red</span> is 5+ points worse,
            grey is within 5 points, and faded cells have fewer than {MIN_SAMPLE} games
            so they are noise as much as signal.
        </p>
    </div>

    {''.join(s for s in sections if s)}

    <p class="timestamp">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Spread analysis report generated: {output_file}")
    print(f"   Graded picks analyzed: {total} (overall {baseline:.1f}%)")
