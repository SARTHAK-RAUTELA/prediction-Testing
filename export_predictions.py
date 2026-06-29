"""
Non-interactive prediction runner.
Fetches today's FIFA 2026 matches, runs all models, and writes results +
prediction date to website/predictions_export.json.
"""
import json
import os
import sys
from datetime import date

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from prediction.engine import PredictionEngine

WEBSITE_DIR = os.path.join(os.path.dirname(__file__), "website")
OUTPUT_FILE = os.path.join(WEBSITE_DIR, "predictions_export.json")


def _auto_stage(d: date) -> str:
    if d < date(2026, 6, 28):    return "group_stage"
    if d <= date(2026, 7, 3):    return "round_of_32"
    if d <= date(2026, 7, 7):    return "round_of_16"
    if d <= date(2026, 7, 12):   return "quarter_final"
    if d <= date(2026, 7, 16):   return "semi_final"
    return "final"


def _extract_markets(pred: dict) -> dict:
    m = pred.get("markets", {})
    return {
        "home_win":  round(m.get("home_win", 0) * 100, 1),
        "draw":      round(m.get("draw", 0) * 100, 1),
        "away_win":  round(m.get("away_win", 0) * 100, 1),
        "btts_yes":  round(m.get("btts_yes", 0) * 100, 1),
        "over_2_5":  round(m.get("over_2_5", 0) * 100, 1),
        "under_2_5": round(m.get("under_2_5", 0) * 100, 1),
    }


def _extract_value_bets(pred: dict) -> list:
    out = []
    for vb in pred.get("value_bets", []):
        out.append({
            "market":    vb.get("market", ""),
            "label":     vb.get("label", ""),
            "edge_pct":  round(vb.get("edge_pct", 0), 1),
            "odds":      vb.get("odds", None),
            "kelly_pct": round(vb.get("kelly_pct", 0), 1),
        })
    return out


def main():
    today = date.today()
    stage = _auto_stage(today)

    print(f"[export_predictions] Date: {today}  Stage: {stage}")
    print("[export_predictions] Loading prediction engine...")
    engine = PredictionEngine()

    print("[export_predictions] Fetching today's fixtures...")
    fixtures = engine.aggregator.get_today_matches()

    predictions = []

    if fixtures:
        print(f"[export_predictions] {len(fixtures)} fixture(s) found — running models...")
        for f in fixtures:
            home = f.get("home_team", "")
            away = f.get("away_team", "")
            if not home or not away:
                continue
            print(f"  Predicting: {home} vs {away}")
            try:
                pred = engine.predict_match(
                    home_team=home,
                    away_team=away,
                    match_id=str(f.get("id", "")),
                    sofascore_id=f.get("sofascore_id"),
                    venue_city=f.get("city", "Dallas"),
                    match_date=(f.get("date", "")[:10] if f.get("date") else None),
                    stage=stage,
                )
                lambdas = pred.get("lambdas", {})
                predictions.append({
                    "home":           home,
                    "away":           away,
                    "venue":          f.get("venue", f.get("city", "")),
                    "kickoff":        f.get("date", "")[:16] if f.get("date") else "",
                    "stage":          stage,
                    "lambda_home":    round(lambdas.get("home", 0), 3),
                    "lambda_away":    round(lambdas.get("away", 0), 3),
                    "markets":        _extract_markets(pred),
                    "predicted_score": pred.get("predicted_score", {}),
                    "confidence":     pred.get("confidence", ""),
                    "value_bets":     _extract_value_bets(pred),
                })
            except Exception as e:
                print(f"  [ERROR] {home} vs {away}: {e}")
    else:
        print("[export_predictions] No live fixtures found via API — using empty list.")

    export = {
        "prediction_date":   today.isoformat(),
        "stage":             stage,
        "generated_at":      today.strftime("%B %d, %Y"),
        "fixture_count":     len(predictions),
        "predictions":       predictions,
    }

    os.makedirs(WEBSITE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False)

    print(f"\n[export_predictions] Saved {len(predictions)} prediction(s) to:")
    print(f"  {OUTPUT_FILE}")
    print(f"  prediction_date = {today.isoformat()}")


if __name__ == "__main__":
    main()
