"""Non-interactive runner: predict today's FIFA 2026 matches (option 1)."""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from prediction.engine import PredictionEngine
from display.formatter import render_prediction_panel, render_match_list

console = Console()

with console.status("[cyan]Loading prediction engine...[/cyan]", spinner="dots"):
    engine = PredictionEngine()

console.print(f"  [dim]ELO database: {len(engine.elo_ratings)} teams loaded[/dim]\n")

with console.status("[cyan]Fetching today's FIFA 2026 fixtures...[/cyan]", spinner="dots"):
    fixtures = engine.aggregator.get_today_matches()

if not fixtures:
    console.print("[yellow]No FIFA 2026 matches found for today.[/yellow]")
    sys.exit(0)

console.print(f"\n  [green]Found {len(fixtures)} match(es)[/green]\n")
render_match_list(fixtures)

for i, f in enumerate(fixtures, 1):
    home = f.get("home_team", "")
    away = f.get("away_team", "")
    if not home or not away:
        continue
    odds_badge = " [green][AUTO-ODDS][/green]" if f.get("sofascore_id") else ""
    console.print(f"\n  [bold cyan]Predicting {i}/{len(fixtures)}: {home} vs {away}[/bold cyan]{odds_badge}")
    with console.status("  [cyan]Running models...[/cyan]", spinner="dots"):
        try:
            pred = engine.predict_match(
                home_team=home, away_team=away,
                match_id=str(f.get("id", "")),
                sofascore_id=f.get("sofascore_id"),
                venue_city=f.get("city", "Dallas"),
                match_date=(f.get("date", "")[:10] if f.get("date") else None),
            )
            pred["fixture"] = f
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            continue
    render_prediction_panel(pred)
