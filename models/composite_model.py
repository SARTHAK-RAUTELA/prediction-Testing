"""
Composite model combining Poisson, ELO, form, player impact, sentiment, weather,
and FIFA 2026 live tournament context (xG, playing style, momentum).
Produces final expected goals (λ) used for all market calculations.
"""
from typing import Dict, Tuple, Optional, List
from config import (
    MODEL_WEIGHTS, HOME_ADVANTAGE, WC2026_OBSERVED_AVG_GOALS,
    KO_LAMBDA_REDUCTION, altitude_goal_factor,
)
from models.poisson_model import build_score_matrix
from models.elo_model import expected_goals_from_elo, win_probability, get_team_elo, load_elo_ratings
from models.form_analyzer import (
    calculate_expected_goals, calculate_form_score, form_multiplier, recent_goal_avg
)
from models.player_impact import calculate_player_impact


WC_LEAGUE_AVG_GOALS = WC2026_OBSERVED_AVG_GOALS  # 2.83 — updated from 72 group stage games


def _get_tournament_context(home_team: str, away_team: str) -> Tuple[Dict, Dict]:
    """Load WC 2026 live stats for both teams from team_analysis.json."""
    try:
        from collectors.wc_results_collector import get_team_wc_stats
        home_ctx = get_team_wc_stats(home_team)
        away_ctx = get_team_wc_stats(away_team)
        return home_ctx, away_ctx
    except Exception:
        empty = {
            "avg_goals_scored": None, "avg_goals_conceded": None,
            "xg_overperformance": 0.0, "style_attack_bonus": 1.0,
            "style_defense_bonus": 1.0, "momentum_factor": 1.0, "played": 0,
        }
        return empty, empty


def apply_tournament_context(
    lam_h: float, lam_a: float,
    home_ctx: Dict, away_ctx: Dict,
) -> Tuple[float, float]:
    """
    Blend ELO/form-based lambdas with tournament-observed scoring rates.
    Rules:
    - If team has 2+ WC matches, their observed avg_goals_scored gets 30% weight
      blended into lambda (xG regression: overperforming teams regress toward xG).
    - Style bonuses (attack/defense) applied with 15% weight.
    - Momentum factor applied with 10% weight.
    """
    # --- Home team attack ---
    home_played = home_ctx.get("played", 0)
    away_played = away_ctx.get("played", 0)

    if home_played >= 2 and home_ctx.get("avg_goals_scored") is not None:
        obs_lam_h = home_ctx["avg_goals_scored"]
        # xG overperformance regression: if scoring > xG, nudge down slightly
        xg_over_h = home_ctx.get("xg_overperformance", 0.0)
        obs_lam_h = obs_lam_h - (xg_over_h * 0.2)  # 20% regression toward xG
        obs_lam_h = max(0.3, obs_lam_h)
        lam_h = lam_h * 0.70 + obs_lam_h * 0.30
    elif home_played == 1 and home_ctx.get("avg_goals_scored") is not None:
        obs_lam_h = home_ctx["avg_goals_scored"]
        lam_h = lam_h * 0.85 + obs_lam_h * 0.15

    # --- Away team attack ---
    if away_played >= 2 and away_ctx.get("avg_goals_scored") is not None:
        obs_lam_a = away_ctx["avg_goals_scored"]
        xg_over_a = away_ctx.get("xg_overperformance", 0.0)
        obs_lam_a = obs_lam_a - (xg_over_a * 0.2)
        obs_lam_a = max(0.3, obs_lam_a)
        lam_a = lam_a * 0.70 + obs_lam_a * 0.30
    elif away_played == 1 and away_ctx.get("avg_goals_scored") is not None:
        obs_lam_a = away_ctx["avg_goals_scored"]
        lam_a = lam_a * 0.85 + obs_lam_a * 0.15

    # --- Playing style: home attack vs away defense ---
    h_atk_style = home_ctx.get("style_attack_bonus", 1.0)
    h_def_style = home_ctx.get("style_defense_bonus", 1.0)
    a_atk_style = away_ctx.get("style_attack_bonus", 1.0)
    a_def_style = away_ctx.get("style_defense_bonus", 1.0)
    STYLE_WEIGHT = 0.15
    lam_h *= (1.0 + (h_atk_style - 1.0) * STYLE_WEIGHT)
    lam_h /= (1.0 + (a_def_style - 1.0) * STYLE_WEIGHT)
    lam_a *= (1.0 + (a_atk_style - 1.0) * STYLE_WEIGHT)
    lam_a /= (1.0 + (h_def_style - 1.0) * STYLE_WEIGHT)

    # --- Momentum ---
    h_momentum = home_ctx.get("momentum_factor", 1.0)
    a_momentum = away_ctx.get("momentum_factor", 1.0)
    MOMENTUM_WEIGHT = 0.10
    lam_h *= (1.0 + (h_momentum - 1.0) * MOMENTUM_WEIGHT)
    lam_a *= (1.0 + (a_momentum - 1.0) * MOMENTUM_WEIGHT)

    return round(max(0.30, min(4.50, lam_h)), 3), round(max(0.30, min(4.50, lam_a)), 3)


def _ko_stage_adjustment(lam_h: float, lam_a: float, stage: str) -> Tuple[float, float]:
    """
    Knockout rounds are more defensive. Apply reduction factor for KO stages.
    Also model the fact that KO teams are more evenly matched (survivors of group stage).
    """
    ko_stages = {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final"}
    if stage not in ko_stages:
        return lam_h, lam_a

    reduction = KO_LAMBDA_REDUCTION  # 0.90 default
    # Semi-finals and final: even more defensive
    if stage in ("semi_final", "final"):
        reduction = 0.87
    elif stage == "quarter_final":
        reduction = 0.88

    return round(lam_h * reduction, 3), round(lam_a * reduction, 3)


def _golden_boot_boost(team: str, lineup: List[Dict], top_scorers_wc: Dict) -> float:
    """
    Players chasing the Golden Boot score more in KO games.
    Returns a multiplier > 1.0 if a top scorer is in the lineup.
    """
    if not top_scorers_wc or not lineup:
        return 1.0

    # Hardcoded WC 2026 top scorers with goals
    TOP_SCORERS_JUNE29 = {
        "Messi": 6, "Haaland": 6, "Mbappé": 4, "Dembélé": 4,
        "Vinicius": 4, "Sarr": 4, "Kane": 3, "Manzambi": 3,
        "Undav": 3, "Gakpo": 3, "Cunha": 3, "Balogun": 3,
    }

    lineup_names = {p.get("name", "").lower() for p in lineup}
    max_goals = 0
    for name, goals in TOP_SCORERS_JUNE29.items():
        if any(name.lower() in ln or ln in name.lower() for ln in lineup_names):
            max_goals = max(max_goals, goals)

    if max_goals >= 6:
        return 1.08  # Messi / Haaland — major impact
    elif max_goals >= 4:
        return 1.05
    elif max_goals >= 3:
        return 1.03
    return 1.0


def _altitude_adjustment(lam_h: float, lam_a: float, venue_city: str) -> Tuple[float, float]:
    """Boost both lambdas symmetrically at high-altitude venues."""
    factor = altitude_goal_factor(venue_city)
    if factor == 1.0:
        return lam_h, lam_a
    return round(min(4.50, lam_h * factor), 3), round(min(4.50, lam_a * factor), 3)


def compute_lambdas(
    home_team: str,
    away_team: str,
    home_form: List[Dict],
    away_form: List[Dict],
    home_lineup: List[Dict],
    away_lineup: List[Dict],
    home_news: Dict,
    away_news: Dict,
    weather_impact: float = 1.0,
    is_neutral: bool = True,
    elo_ratings: Optional[Dict] = None,
    stage: str = "group_stage",
    top_scorers_wc: Optional[Dict] = None,
    venue_city: str = "",
) -> Tuple[float, float, Dict]:
    """
    Compute composite λ_home and λ_away from all available signals.
    Returns (lam_home, lam_away, diagnostics_dict).
    """
    if elo_ratings is None:
        elo_ratings = load_elo_ratings()

    # --- ELO component ---
    elo_home = get_team_elo(home_team, elo_ratings)
    elo_away = get_team_elo(away_team, elo_ratings)
    elo_lam_h, elo_lam_a = expected_goals_from_elo(
        elo_home, elo_away, WC_LEAGUE_AVG_GOALS, is_neutral
    )
    elo_1x2 = win_probability(elo_home, elo_away, is_neutral)

    # --- Form component ---
    form_lam_h, form_lam_a = calculate_expected_goals(
        home_form, away_form, home_team, away_team,
        league_avg=WC_LEAGUE_AVG_GOALS, is_neutral=is_neutral
    )
    home_form_score = calculate_form_score(home_form, home_team)
    away_form_score = calculate_form_score(away_form, away_team)

    # Form quality multipliers
    home_form_mult = form_multiplier(home_form_score)
    away_form_mult = form_multiplier(away_form_score)

    # --- Player impact component ---
    home_player_impact = calculate_player_impact(
        home_team, home_lineup, home_news.get("injured_players", [])
    )
    away_player_impact = calculate_player_impact(
        away_team, away_lineup, away_news.get("injured_players", [])
    )

    # --- Sentiment component ---
    # Morale: 0.5 = neutral, >0.5 = positive, <0.5 = negative
    home_morale = home_news.get("morale", 0.5)
    away_morale = away_news.get("morale", 0.5)
    # Convert morale to small λ multiplier: [0.95, 1.05]
    home_sentiment_mult = 0.95 + (home_morale * 0.10)
    away_sentiment_mult = 0.95 + (away_morale * 0.10)

    # Injury risk reduces expected goals
    home_injury_risk = home_news.get("injury_risk", 0.0)
    away_injury_risk = away_news.get("injury_risk", 0.0)
    home_injury_mult = 1.0 - (home_injury_risk * 0.10)
    away_injury_mult = 1.0 - (away_injury_risk * 0.10)

    # --- Weighted composite λ ---
    w = MODEL_WEIGHTS

    # Determine how much to trust form data vs ELO
    # If form data is sparse (< 3 matches per team), rely almost entirely on ELO
    home_form_count = len([r for r in home_form if True])
    away_form_count = len([r for r in away_form if True])
    form_reliability = min(1.0, (home_form_count + away_form_count) / 12.0)

    # Dynamic weights: when form is sparse, boost ELO weight
    elo_weight = w["elo"] + w["poisson"] * (1.0 - form_reliability)
    form_weight = w["poisson"] * form_reliability

    # Clamp extreme form λ values toward ELO when form sample is small
    if form_reliability < 0.5:
        blend = 0.3 + 0.4 * form_reliability  # 0.3 to 0.5 form blend
        form_lam_h = form_lam_h * blend + elo_lam_h * (1 - blend)
        form_lam_a = form_lam_a * blend + elo_lam_a * (1 - blend)

    # Base λ: weighted average of ELO-based and form-based
    total_base_weight = elo_weight + form_weight
    lam_h = (form_weight * form_lam_h + elo_weight * elo_lam_h) / total_base_weight
    lam_a = (form_weight * form_lam_a + elo_weight * elo_lam_a) / total_base_weight

    # Apply form quality multiplier
    lam_h *= (1.0 + (home_form_mult - 1.0) * w["form"])
    lam_a *= (1.0 + (away_form_mult - 1.0) * w["form"])

    # Apply player impact
    lam_h *= home_player_impact["attack_multiplier"] ** w["player_impact"]
    lam_a *= away_player_impact["attack_multiplier"] ** w["player_impact"]

    # Opponent defense: if their key defenders are missing, opponent scores more
    lam_h /= away_player_impact["defense_multiplier"] ** w["player_impact"]
    lam_a /= home_player_impact["defense_multiplier"] ** w["player_impact"]

    # Apply sentiment & injury multipliers
    lam_h *= (home_sentiment_mult ** w["sentiment"]) * (home_injury_mult ** w["player_impact"])
    lam_a *= (away_sentiment_mult ** w["sentiment"]) * (away_injury_mult ** w["player_impact"])

    # Apply weather impact (reduces total goals in bad weather)
    lam_h *= weather_impact
    lam_a *= weather_impact

    # Clip to realistic range
    lam_h = round(max(0.30, min(4.50, lam_h)), 3)
    lam_a = round(max(0.30, min(4.50, lam_a)), 3)

    # --- Tournament context: WC 2026 live scoring rates + style + momentum ---
    home_ctx, away_ctx = _get_tournament_context(home_team, away_team)
    lam_h, lam_a = apply_tournament_context(lam_h, lam_a, home_ctx, away_ctx)

    # --- KO stage: apply conservative defensive reduction ---
    lam_h, lam_a = _ko_stage_adjustment(lam_h, lam_a, stage)

    # --- Golden Boot pursuit boost ---
    home_gb = _golden_boot_boost(home_team, home_lineup, top_scorers_wc or {})
    away_gb = _golden_boot_boost(away_team, away_lineup, top_scorers_wc or {})
    lam_h = round(min(4.50, lam_h * home_gb), 3)
    lam_a = round(min(4.50, lam_a * away_gb), 3)

    # --- Altitude adjustment ---
    altitude_factor = altitude_goal_factor(venue_city)
    lam_h, lam_a = _altitude_adjustment(lam_h, lam_a, venue_city)

    diagnostics = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_lam_home": elo_lam_h,
        "elo_lam_away": elo_lam_a,
        "elo_1x2": elo_1x2,
        "form_lam_home": form_lam_h,
        "form_lam_away": form_lam_a,
        "home_form_score": home_form_score,
        "away_form_score": away_form_score,
        "home_form_mult": home_form_mult,
        "away_form_mult": away_form_mult,
        "home_player_impact": home_player_impact,
        "away_player_impact": away_player_impact,
        "home_sentiment": home_sentiment_mult,
        "away_sentiment": away_sentiment_mult,
        "weather_impact": weather_impact,
        "final_lam_home": lam_h,
        "final_lam_away": lam_a,
        "tournament_context_home": home_ctx,
        "tournament_context_away": away_ctx,
        "stage": stage,
        "ko_reduction_applied": stage in {"round_of_32","round_of_16","quarter_final","semi_final","final"},
        "home_golden_boot_boost": home_gb,
        "away_golden_boot_boost": away_gb,
        "venue_city": venue_city,
        "altitude_factor": altitude_factor,
    }

    return lam_h, lam_a, diagnostics
