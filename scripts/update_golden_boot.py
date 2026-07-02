"""
Fetches current World Cup top-scorers from football-data.org and writes
docs/data/golden_boot.json for the static site to consume same-origin.

football-data.org's CORS policy only allows browser calls from http://localhost,
so this can't be fetched client-side from the deployed GitHub Pages site — it
must run server-side (locally, or via the scheduled GitHub Action) and commit
the resulting JSON.

Usage: python scripts/update_golden_boot.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.football_data_collector import FootballDataCollector

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data", "golden_boot.json")


def main():
    collector = FootballDataCollector()
    if not collector.is_configured:
        print("FOOTBALL_DATA_API_KEY not set — skipping golden boot update.")
        sys.exit(1)

    scorers = collector.get_top_scorers("WC")
    if not scorers:
        print("No scorers returned from football-data.org — leaving existing file untouched.")
        sys.exit(1)

    top10 = sorted(scorers, key=lambda s: (s.get("goals") or 0, s.get("assists") or 0), reverse=True)[:10]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorers": [
            {
                "name": (s.get("player") or {}).get("name", "—"),
                "team": (s.get("team") or {}).get("name", ""),
                "crest": (s.get("team") or {}).get("crest", ""),
                "flag": "",
                "goals": s.get("goals") or 0,
                "assists": s.get("assists") or 0,
                "xg": None,
            }
            for s in top10
        ],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(payload['scorers'])} scorers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
