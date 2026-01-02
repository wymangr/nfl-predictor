import json
from datetime import datetime
import nflreadpy as nfl

from src.helpers.database_helpers import run_query


def generate_power_rankings_report(season=2025):
    """
    Generate an interactive HTML report for NFL team power rankings.

    Args:
        season: Season year to generate report for (default: 2025)
    """
    # Get all data for the season
    current_week = nfl.get_current_week()
    query = f"""
    SELECT * FROM team_power_rankings 
    WHERE season = {season} 
    AND week <= {current_week}
    ORDER BY week, power_ranking DESC
    """
    all_data = run_query(query)

    # Get current week data (latest week)
    max_week = max(row["week"] for row in all_data)
    current_week_data = [row for row in all_data if row["week"] == max_week]
    # Sort by adjusted overall rank (lower is better)
    current_week_data.sort(key=lambda x: x["adj_overall_rank"])

    # Get all teams
    teams = sorted(list(set(row["team"] for row in all_data)))

    # Prepare weekly data for charts
    weekly_data = {}
    for row in all_data:
        team = row["team"]
        if team not in weekly_data:
            weekly_data[team] = []
        weekly_data[team].append(
            {
                "week": row["week"],
                "power_ranking": row["power_ranking"],
                "win_pct": row["win_pct"],
                "offensive_rank": row["offensive_rank"],
                "defensive_rank": row["defensive_rank"],
                "overall_rank": row["overall_rank"],
                "sos": row["sos"],
            }
        )

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Power Rankings Report - {season} Season</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #1e3a8a;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3b82f6;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #bfdbfe;
        }}
        
        .stat-card .label {{
            color: #1e40af;
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .stat-card .value {{
            color: #1e3a8a;
            font-size: 2em;
            font-weight: 700;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        thead {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        tbody tr:hover {{
            background-color: #f0f9ff;
        }}
        
        .rank-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .rank-top {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .rank-mid {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .rank-bottom {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .chart-container {{
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .team-filter {{
            margin: 20px 0;
            padding: 20px;
            background: #f9fafb;
            border-radius: 10px;
            border: 2px solid #e5e7eb;
        }}
        
        .team-filter h3 {{
            margin-bottom: 15px;
            color: #1e3a8a;
        }}
        
        .team-checkboxes {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
        }}
        
        .team-checkbox {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .team-checkbox input {{
            cursor: pointer;
        }}
        
        .team-checkbox label {{
            cursor: pointer;
            font-size: 0.9em;
        }}
        
        .info-box {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .info-box h4 {{
            color: #1e40af;
            margin-bottom: 10px;
        }}
        
        .info-box p {{
            color: #1e3a8a;
            line-height: 1.6;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            margin: 5px;
            transition: transform 0.2s;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        }}
        
        .progress-bar {{
            background: #e5e7eb;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏈 NFL Power Rankings Report</h1>
            <p>{season} Season - Week {max_week} | Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <!-- Summary Statistics -->
            <div class="section">
                <h2 class="section-title">📊 Season Overview</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Current Week</div>
                        <div class="value">{max_week}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Teams Tracked</div>
                        <div class="value">{len(teams)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Top Team</div>
                        <div class="value">{current_week_data[0]['team']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Power Rating</div>
                        <div class="value">{sum(row['power_ranking'] for row in current_week_data) / len(current_week_data):.3f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Current Power Rankings Table -->
            <div class="section">
                <h2 class="section-title">🏆 Current Power Rankings (Week {max_week})</h2>
                <div class="info-box">
                    <h4>📌 About These Rankings</h4>
                    <p>The <strong>Adjusted Rankings</strong> (shown in the main rank column) account for strength of schedule and provide the most accurate assessment of team performance. Original rankings are shown for comparison to illustrate the impact of SOS adjustments.</p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Team</th>
                            <th>Power Rating</th>
                            <th>Win %</th>
                            <th>Adj Off</th>
                            <th>Adj Def</th>
                            <th>Adj Overall</th>
                            <th>SOS</th>
                            <th>Orig Overall</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add table rows
    for idx, row in enumerate(current_week_data, 1):
        rank_class = (
            "rank-top" if idx <= 10 else ("rank-mid" if idx <= 21 else "rank-bottom")
        )
        rank_change = row["overall_rank"] - row["adj_overall_rank"]
        change_icon = "↑" if rank_change > 0 else ("↓" if rank_change < 0 else "–")
        change_color = (
            "#16a34a"
            if rank_change > 0
            else ("#dc2626" if rank_change < 0 else "#6b7280")
        )

        html += f"""
                        <tr>
                            <td><span class="rank-badge {rank_class}">{idx}</span></td>
                            <td><strong>{row['team']}</strong></td>
                            <td>{row['power_ranking']:.4f}</td>
                            <td>{row['win_pct']:.1%}</td>
                            <td>{row['adj_offensive_rank']}</td>
                            <td>{row['adj_defensive_rank']}</td>
                            <td><strong>{row['adj_overall_rank']}</strong></td>
                            <td>{row['sos']:.3f}</td>
                            <td>{row['overall_rank']}</td>
                            <td style="color: {change_color}; font-weight: 600;">{change_icon} {abs(rank_change)}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
            
            <!-- Strength of Schedule Analysis -->
            <div class="section">
                <h2 class="section-title">💪 Strength of Schedule Analysis</h2>
                <div class="info-box">
                    <h4>Understanding SOS (Strength of Schedule)</h4>
                    <p>The Strength of Schedule (SOS) metric represents the average win percentage of a team's opponents. 
                    A higher SOS (closer to 1.0) means the team has faced tougher opponents, while a lower SOS indicates an easier schedule. 
                    This metric helps contextualize a team's record and performance.</p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Team</th>
                            <th>SOS</th>
                            <th>SOS Visual</th>
                            <th>Rank Before Adj</th>
                            <th>Rank After Adj</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Sort by SOS
    sos_sorted = sorted(current_week_data, key=lambda x: x["sos"], reverse=True)
    for row in sos_sorted:
        rank_change = row["overall_rank"] - row["adj_overall_rank"]
        change_icon = "↑" if rank_change > 0 else ("↓" if rank_change < 0 else "→")
        change_color = (
            "green" if rank_change > 0 else ("red" if rank_change < 0 else "gray")
        )

        html += f"""
                        <tr>
                            <td><strong>{row['team']}</strong></td>
                            <td>{row['sos']:.3f}</td>
                            <td>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {row['sos'] * 100}%"></div>
                                </div>
                            </td>
                            <td>{row['overall_rank']}</td>
                            <td>{row['adj_overall_rank']}</td>
                            <td style="color: {change_color}; font-weight: 600;">{change_icon} {abs(rank_change)}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
                <div class="info-box">
                    <h4>How Adjusted Rankings Work</h4>
                    <p>Adjusted rankings account for strength of schedule to provide a more accurate assessment of team performance. 
                    Teams that perform well against tough opponents may see their adjusted rank improve, while teams with easier schedules 
                    may see their rank adjusted downward. The "Change" column shows how many positions a team moved after SOS adjustment.</p>
                </div>
            </div>
            
            <!-- Weekly Trends Chart -->
            <div class="section">
                <h2 class="section-title">📈 Weekly Power Ranking Trends</h2>
                <div class="team-filter">
                    <h3>Filter Teams:</h3>
                    <button class="btn" onclick="selectAllTeams()">Select All</button>
                    <button class="btn" onclick="deselectAllTeams()">Deselect All</button>
                    <button class="btn" onclick="selectTop10()">Top 10 Only</button>
                    <div class="team-checkboxes" id="teamCheckboxes">
"""

    # Add team checkboxes
    for team in teams:
        html += f"""
                        <div class="team-checkbox">
                            <input type="checkbox" id="team_{team}" value="{team}" checked onchange="updateChart()">
                            <label for="team_{team}">{team}</label>
                        </div>
"""

    html += """
                    </div>
                </div>
                <div class="chart-container">
                    <div id="powerRankingChart"></div>
                </div>
            </div>
            
            <!-- Offensive vs Defensive Rankings -->
            <div class="section">
                <h2 class="section-title">⚔️ Offensive vs Defensive Rankings (Adjusted)</h2>
                <div class="info-box">
                    <h4>Understanding the Quadrants</h4>
                    <p><strong>Top-Left:</strong> Elite Offense + Elite Defense = Championship contenders<br>
                    <strong>Top-Right:</strong> Elite Offense + Weak Defense = High-scoring but vulnerable<br>
                    <strong>Bottom-Left:</strong> Weak Offense + Elite Defense = Low-scoring grinders<br>
                    <strong>Bottom-Right:</strong> Needs improvement on both sides</p>
                </div>
                <div class="chart-container">
                    <div id="offDefChart"></div>
                </div>
            </div>
            
            <!-- Rankings Comparison: Original vs Adjusted -->
            <div class="section">
                <h2 class="section-title">🔄 Rankings Comparison: Original vs Adjusted</h2>
                <div class="info-box">
                    <h4>Biggest Movers After SOS Adjustment</h4>
                    <p>This section highlights which teams benefited or suffered most from SOS adjustments across all three categories.</p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Team</th>
                            <th>Offensive (Orig → Adj)</th>
                            <th>Δ</th>
                            <th>Defensive (Orig → Adj)</th>
                            <th>Δ</th>
                            <th>Overall (Orig → Adj)</th>
                            <th>Δ</th>
                            <th>SOS</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Sort by total rank change (sum of absolute changes)
    comparison_data = sorted(
        current_week_data,
        key=lambda x: abs(x["offensive_rank"] - x["adj_offensive_rank"])
        + abs(x["defensive_rank"] - x["adj_defensive_rank"])
        + abs(x["overall_rank"] - x["adj_overall_rank"]),
        reverse=True,
    )

    for row in comparison_data:
        off_change = row["offensive_rank"] - row["adj_offensive_rank"]
        def_change = row["defensive_rank"] - row["adj_defensive_rank"]
        overall_change = row["overall_rank"] - row["adj_overall_rank"]

        off_icon = "↑" if off_change > 0 else ("↓" if off_change < 0 else "–")
        def_icon = "↑" if def_change > 0 else ("↓" if def_change < 0 else "–")
        overall_icon = (
            "↑" if overall_change > 0 else ("↓" if overall_change < 0 else "–")
        )

        off_color = (
            "#16a34a"
            if off_change > 0
            else ("#dc2626" if off_change < 0 else "#6b7280")
        )
        def_color = (
            "#16a34a"
            if def_change > 0
            else ("#dc2626" if def_change < 0 else "#6b7280")
        )
        overall_color = (
            "#16a34a"
            if overall_change > 0
            else ("#dc2626" if overall_change < 0 else "#6b7280")
        )

        html += f"""
                        <tr>
                            <td><strong>{row['team']}</strong></td>
                            <td>{row['offensive_rank']} → {row['adj_offensive_rank']}</td>
                            <td style="color: {off_color}; font-weight: 600;">{off_icon} {abs(off_change)}</td>
                            <td>{row['defensive_rank']} → {row['adj_defensive_rank']}</td>
                            <td style="color: {def_color}; font-weight: 600;">{def_icon} {abs(def_change)}</td>
                            <td>{row['overall_rank']} → {row['adj_overall_rank']}</td>
                            <td style="color: {overall_color}; font-weight: 600;">{overall_icon} {abs(overall_change)}</td>
                            <td>{row['sos']:.3f}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
            
            <!-- Recent Performance (Last 3 Weeks) -->
            <div class="section">
                <h2 class="section-title">🔥 Recent Form (Last 3 Weeks)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Team</th>
                            <th>Current Rating</th>
                            <th>L3 Rating</th>
                            <th>Trend</th>
                            <th>Win % L3</th>
                            <th>Momentum</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Sort by recent form
    form_sorted = sorted(
        current_week_data, key=lambda x: x["power_ranking_l3"], reverse=True
    )
    for row in form_sorted:
        trend = row["power_ranking_l3"] - row["prev_power_ranking_l3"]
        trend_icon = "🔥" if trend > 0.02 else ("❄️" if trend < -0.02 else "➡️")
        momentum = "Hot" if trend > 0.02 else ("Cold" if trend < -0.02 else "Stable")
        momentum_color = (
            "#16a34a" if trend > 0.02 else ("#dc2626" if trend < -0.02 else "#6b7280")
        )

        html += f"""
                        <tr>
                            <td><strong>{row['team']}</strong></td>
                            <td>{row['power_ranking']:.4f}</td>
                            <td>{row['power_ranking_l3']:.4f}</td>
                            <td>{trend_icon} {trend:+.4f}</td>
                            <td>{row['win_pct_l3']:.1%}</td>
                            <td style="color: {momentum_color}; font-weight: 600;">{momentum}</td>
                        </tr>
"""

    html += f"""
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Store weekly data
        const weeklyData = {json.dumps(weekly_data)};
        const teams = {json.dumps(teams)};
        const currentWeekData = {json.dumps(current_week_data)};
        
        // Update power ranking trend chart
        function updateChart() {{
            const selectedTeams = [];
            teams.forEach(team => {{
                const checkbox = document.getElementById(`team_${{team}}`);
                if (checkbox && checkbox.checked) {{
                    selectedTeams.push(team);
                }}
            }});
            
            const traces = selectedTeams.map(team => {{
                const data = weeklyData[team];
                return {{
                    x: data.map(d => d.week),
                    y: data.map(d => d.power_ranking),
                    name: team,
                    mode: 'lines+markers',
                    type: 'scatter'
                }};
            }});
            
            const layout = {{
                title: 'Power Rankings Over Time',
                xaxis: {{ title: 'Week' }},
                yaxis: {{ title: 'Power Ranking' }},
                hovermode: 'closest',
                showlegend: true,
                height: 500
            }};
            
            Plotly.newPlot('powerRankingChart', traces, layout);
        }}
        
        // Team selection functions
        function selectAllTeams() {{
            teams.forEach(team => {{
                document.getElementById(`team_${{team}}`).checked = true;
            }});
            updateChart();
        }}
        
        function deselectAllTeams() {{
            teams.forEach(team => {{
                document.getElementById(`team_${{team}}`).checked = false;
            }});
            updateChart();
        }}
        
        function selectTop10() {{
            deselectAllTeams();
            currentWeekData.slice(0, 10).forEach(row => {{
                document.getElementById(`team_${{row.team}}`).checked = true;
            }});
            updateChart();
        }}
        
        // Offensive vs Defensive scatter plot
        function createOffDefChart() {{
            const trace = {{
                x: currentWeekData.map(d => d.adj_offensive_rank),
                y: currentWeekData.map(d => d.adj_defensive_rank),
                mode: 'markers+text',
                type: 'scatter',
                text: currentWeekData.map(d => d.team),
                textposition: 'top center',
                marker: {{
                    size: currentWeekData.map(d => d.power_ranking * 50),
                    color: currentWeekData.map(d => d.power_ranking),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{
                        title: 'Power<br>Rating'
                    }}
                }}
            }};
            
            const layout = {{
                title: 'Adjusted Offensive Rank vs Adjusted Defensive Rank (Bubble size = Power Rating)',
                xaxis: {{ 
                    title: 'Adjusted Offensive Rank',
                    autorange: 'reversed'
                }},
                yaxis: {{ 
                    title: 'Adjusted Defensive Rank',
                    autorange: 'reversed'
                }},
                hovermode: 'closest',
                height: 600,
                shapes: [
                    {{
                        type: 'line',
                        x0: 16, x1: 16,
                        y0: 0, y1: 32,
                        line: {{
                            color: 'rgba(100, 100, 100, 0.3)',
                            width: 2,
                            dash: 'dash'
                        }}
                    }},
                    {{
                        type: 'line',
                        x0: 0, x1: 32,
                        y0: 16, y1: 16,
                        line: {{
                            color: 'rgba(100, 100, 100, 0.3)',
                            width: 2,
                            dash: 'dash'
                        }}
                    }}
                ]
            }};
            
            Plotly.newPlot('offDefChart', [trace], layout);
        }}
        
        // Initialize charts
        updateChart();
        createOffDefChart();
    </script>
</body>
</html>
"""

    # Write to file
    filename = f"nfl_power_rankings_{season}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report generated successfully: {filename}")
    return filename
