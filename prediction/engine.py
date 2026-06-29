"""
Main prediction engine.
Orchestrates all data collection and model computation for a match.
"""
import threading
from datetime import date
from typing import Dict, List, Optional, Tuple
from collectors.data_aggregator import DataAggregator
from models.composite_model import compute_lambdas
from models.elo_model import load_elo_ratings, update_elo, save_elo_ratings
from models.goalscorer_model import calculate_goalscorer_probs, first_goalscorer_probs
from prediction.markets import (
    calculate_all_markets, find_value_bets,
    calibrate_lambda_to_totals, calibrate_1x2_to_bookmaker,
)
from prediction.confidence import calculate_confidence
from prediction.stakes_analyzer import analyze_stakes
from config import KO_BOOKMAKER_BLEND, GROUP_BOOKMAKER_BLEND

KO_STAGES = {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final"}


class PredictionEngine:
    def __init__(self):
        self.aggregator = DataAggregator()
        self.elo_ratings = load_elo_ratings()
        self._lock = threading.RLock()

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        match_id: Optional[str] = None,
        sofascore_id: Optional[int] = None,
        venue_city: str = "Dallas",
        match_date: Optional[str] = None,
        bookmaker_odds: Optional[Dict] = None,
        force_refresh: bool = False,
        stage: str = "group_stage",
        bankroll: float = 100.0,
    ) -> Dict:
        """
        Full prediction pipeline for a single match.
        Returns complete prediction dict with all markets + confidence.
        """
        if force_refresh and match_id:
            refreshed = self.aggregator.refresh_lineups(match_id)
            if refreshed:
                pass  # data_aggregator updates cache

        # Collect all data
        data = self.aggregator.get_match_full_data(
            home_team, away_team, match_id, sofascore_id=sofascore_id
        )

        # H2H data
        h2h = self.aggregator.get_h2h_history(home_team, away_team)
        data["h2h"] = h2h

        # Weather for match venue
        m_date = match_date or date.today().isoformat()
        weather = self.aggregator.weather.get_match_weather(venue_city, m_date)
        data["weather"] = weather
        weather_impact = weather.get("impact_factor", 1.0) if weather else 1.0

        # Compute lambdas
        lam_home, lam_away, diagnostics = compute_lambdas(
            home_team=home_team,
            away_team=away_team,
            home_form=data["home_form"],
            away_form=data["away_form"],
            home_lineup=data["lineups"]["home"],
            away_lineup=data["lineups"]["away"],
            home_news=data["home_news"],
            away_news=data["away_news"],
            weather_impact=weather_impact,
            is_neutral=True,  # FIFA 2026 in USA/Canada/Mexico — mostly neutral
            elo_ratings=self.elo_ratings,
            stage=stage,
            venue_city=venue_city,
        )

        # Fetch WC top scorers for goalscorer model
        top_scorers: Dict[str, float] = {}
        try:
            raw_sc = self.aggregator.fd.get_top_scorers("WC")
            for s in raw_sc:
                pname = s.get("player", {}).get("name", "")
                goals = s.get("numberOfGoals") or s.get("goals", 0)
                if pname and goals:
                    top_scorers[pname] = int(goals)
        except Exception:
            pass

        # Compute player goalscorer probabilities
        home_scorers = calculate_goalscorer_probs(
            lineup=data["lineups"]["home"],
            team_lam=lam_home,
            top_scorers=top_scorers,
        )
        away_scorers = calculate_goalscorer_probs(
            lineup=data["lineups"]["away"],
            team_lam=lam_away,
            top_scorers=top_scorers,
        )
        home_1st = first_goalscorer_probs(home_scorers)
        away_1st = first_goalscorer_probs(away_scorers)

        # Calculate all markets (pure Poisson)
        markets = calculate_all_markets(lam_home, lam_away)

        # Use auto-fetched Sofascore odds if no manual odds were provided
        auto_odds = data.get("bookmaker_odds")
        effective_odds = bookmaker_odds or auto_odds

        # Dual-market calibration: use over/under bookmaker odds to recalibrate λ
        if effective_odds:
            cal_lam_h, cal_lam_a = calibrate_lambda_to_totals(
                lam_home, lam_away, effective_odds, line=2.5
            )
            # Only apply if calibration meaningfully changed λ (avoid noise)
            if abs(cal_lam_h - lam_home) > 0.05 or abs(cal_lam_a - lam_away) > 0.05:
                markets = calculate_all_markets(cal_lam_h, cal_lam_a)
                diagnostics["cal_lam_home"] = cal_lam_h
                diagnostics["cal_lam_away"] = cal_lam_a

        # 1x2 calibration: blend Poisson probs with bookmaker-implied probs
        # KO stage: trust bookmaker more (62%), group stage: 55%
        blend_weight = KO_BOOKMAKER_BLEND if stage in KO_STAGES else GROUP_BOOKMAKER_BLEND
        calibrated_markets = markets
        if effective_odds:
            calibrated_markets = calibrate_1x2_to_bookmaker(markets, effective_odds, blend_weight=blend_weight)

        # Value bets: compare pure Poisson model odds vs bookmaker (not calibrated)
        value_bets = []
        if effective_odds:
            value_bets = find_value_bets(markets, effective_odds)

        # Confidence scoring — uses calibrated markets when available
        has_news = bool(
            data["home_news"].get("article_count", 0) or
            data["away_news"].get("article_count", 0)
        )
        confidence = calculate_confidence(
            markets=calibrated_markets,
            diagnostics=diagnostics,
            home_form=data["home_form"],
            away_form=data["away_form"],
            home_lineup=data["lineups"]["home"],
            away_lineup=data["lineups"]["away"],
            h2h=h2h,
            has_news=has_news,
            weather=weather,
            lineup_confirmed=data["lineups"].get("confirmed", False),
            has_bookmaker_odds=bool(effective_odds),
            bookmaker_odds=effective_odds,
        )

        # Stakes analysis (Kelly Criterion)
        stakes = analyze_stakes(
            markets=calibrated_markets,
            value_bets=value_bets,
            bookmaker_odds=effective_odds,
            bankroll=bankroll,
            stage=stage,
        )

        return {
            "home_team": home_team,
            "away_team": away_team,
            "match_id": match_id,
            "sofascore_id": sofascore_id,
            "venue_city": venue_city,
            "date": m_date,
            "stage": stage,
            "markets": calibrated_markets,
            "markets_raw": markets,
            "diagnostics": diagnostics,
            "confidence": confidence,
            "value_bets": value_bets,
            "stakes": stakes,
            "bookmaker_odds_source": "manual" if bookmaker_odds else ("sofascore" if auto_odds else None),
            "data": {
                "home_form_count": len(data["home_form"]),
                "away_form_count": len(data["away_form"]),
                "h2h_count": len(h2h),
                "has_lineups": bool(data["lineups"]["home"]),
                "lineup_confirmed": data["lineups"].get("confirmed", False),
                "home_formation": data["lineups"].get("home_formation", ""),
                "away_formation": data["lineups"].get("away_formation", ""),
                "home_lineup": data["lineups"]["home"],
                "away_lineup": data["lineups"]["away"],
                "weather": weather,
                "home_news": data["home_news"],
                "away_news": data["away_news"],
                "missing_home_players": diagnostics.get("home_player_impact", {}).get("missing_key_players", []),
                "missing_away_players": diagnostics.get("away_player_impact", {}).get("missing_key_players", []),
                "data_sources": data.get("data_sources", []),
                "h2h": h2h,
                "home_scorers": home_scorers,
                "away_scorers": away_scorers,
                "home_1st_scorers": home_1st,
                "away_1st_scorers": away_1st,
                "top_scorers_count": len(top_scorers),
            },
        }

    def predict_today(
        self,
        bookmaker_odds_map: Optional[Dict] = None,
        target_date: Optional[date] = None,
    ) -> List[Dict]:
        """Predict all FIFA 2026 matches for today."""
        fixtures = self.aggregator.get_today_matches(target_date)
        if not fixtures:
            return []

        predictions = []
        for fixture in fixtures:
            home = fixture.get("home_team", "")
            away = fixture.get("away_team", "")
            if not home or not away:
                continue

            match_odds = None
            if bookmaker_odds_map:
                key = f"{home.lower()}_{away.lower()}"
                match_odds = bookmaker_odds_map.get(key)

            try:
                pred = self.predict_match(
                    home_team=home,
                    away_team=away,
                    match_id=str(fixture.get("id", "")),
                    sofascore_id=fixture.get("sofascore_id"),
                    venue_city=fixture.get("city", "Dallas"),
                    match_date=fixture.get("date", "")[:10] if fixture.get("date") else None,
                    bookmaker_odds=match_odds,
                )
                pred["fixture"] = fixture
                predictions.append(pred)
            except Exception as e:
                predictions.append({
                    "home_team": home,
                    "away_team": away,
                    "error": str(e),
                    "fixture": fixture,
                })

        return predictions

    def update_elo_from_result(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        stage: str = "group_stage",
    ) -> None:
        """Update ELO ratings after a match result is known."""
        with self._lock:
            self.elo_ratings = update_elo(
                self.elo_ratings, home_team, away_team,
                home_goals, away_goals, stage
            )
            save_elo_ratings(self.elo_ratings)

    def monitor_lineup_changes(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        previous_prediction: Dict,
    ) -> Optional[Dict]:
        """
        Check for lineup changes and re-predict if they occurred.
        Returns updated prediction if lineups changed, else None.
        """
        new_data = self.aggregator.refresh_lineups(match_id)
        if not new_data:
            return None

        old_lineup_home = {p.get("name") for p in previous_prediction.get("data", {}).get("home_lineup", [])}
        old_lineup_away = {p.get("name") for p in previous_prediction.get("data", {}).get("away_lineup", [])}
        new_lineup_home = {p.get("name") for p in new_data.get("lineups", {}).get("home", [])}
        new_lineup_away = {p.get("name") for p in new_data.get("lineups", {}).get("away", [])}

        if old_lineup_home != new_lineup_home or old_lineup_away != new_lineup_away:
            return self.predict_match(
                home_team=home_team,
                away_team=away_team,
                match_id=match_id,
                force_refresh=False,
            )
        return None
