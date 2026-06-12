"""
Betting market probability calculator.
Computes all major markets from Poisson score matrix.
"""
import math
import numpy as np
from typing import Dict, List, Tuple
from models.poisson_model import (
    build_score_matrix, home_win_prob, draw_prob, away_win_prob,
    btts_yes, btts_no, over_prob, under_prob,
    asian_handicap_probs, correct_score_probs,
    first_goal_probs, halftime_probs,
)
from config import MAX_GOALS

# Goals split per half (empirical across major tournaments)
_ALPHA_1H = 0.45
_ALPHA_2H = 0.55


ASIAN_TOTAL_LINES = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
ASIAN_HANDICAP_LINES = [
    -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5
]


def prob_to_odds(prob: float) -> float:
    """Convert raw probability to decimal odds (no margin)."""
    if prob <= 0:
        return 999.0
    return round(1.0 / prob, 2)


def asian_line_probs(matrix: np.ndarray, line: float) -> Tuple[float, float]:
    """
    Asian lines use quarter-ball splits for .25 and .75 lines.
    e.g. 2.25 = half at 2.0 (push if exact) + half at 2.5
    """
    if line % 0.5 == 0:
        # Full line or half line — standard
        over = over_prob(matrix, line)
        under = under_prob(matrix, line)
        return round(over, 4), round(under, 4)
    else:
        # Quarter line — split between adjacent full/half lines
        lower = math.floor(line * 2) / 2  # e.g. 2.75 -> 2.5
        upper = lower + 0.5               # e.g. 2.75 -> 3.0

        over_l = over_prob(matrix, lower)
        under_l = under_prob(matrix, lower)
        # At lower line (push = remainder)
        push_l = 1.0 - over_l - under_l

        over_u = over_prob(matrix, upper)
        under_u = under_prob(matrix, upper)

        # Quarter-line = average: half stake on lower, half on upper
        # Over: win half if over lower, win full if over upper
        # Under: win half if under upper, win full if under lower
        over_combined = (over_l * 0.5 + push_l * 0.0 + under_l * 0.0) + (over_u * 0.5)
        under_combined = (under_u * 0.5 + push_l * 0.0 + over_u * 0.0) + (under_l * 0.5)

        # Normalize
        total = over_combined + under_combined
        if total > 0:
            over_combined /= total
            under_combined /= total

        return round(over_combined, 4), round(under_combined, 4)


def clean_sheet_probs(matrix: np.ndarray) -> Dict:
    """
    P(team keeps clean sheet) = P(opponent scores 0).
    Home clean sheet → away scores 0 → sum of column 0.
    Away clean sheet → home scores 0 → sum of row 0.
    """
    home_cs = float(matrix[:, 0].sum())
    away_cs = float(matrix[0, :].sum())
    return {
        "home": {"prob": round(home_cs, 4), "odds": prob_to_odds(home_cs)},
        "away": {"prob": round(away_cs, 4), "odds": prob_to_odds(away_cs)},
    }


def total_goals_exact_probs(matrix: np.ndarray) -> List[Dict]:
    """
    P(total goals = N) for N in {0, 1, 2, 3, 4+}.
    P(N) = sum over all (i,j) where i+j=N of matrix[i][j].
    """
    n_max = matrix.shape[0] - 1
    rows = []
    cumulative = 0.0
    for n in range(4):
        p = float(sum(
            matrix[i, n - i]
            for i in range(n + 1)
            if n - i <= n_max
        ))
        cumulative += p
        rows.append({"goals": str(n), "prob": round(p, 4), "odds": prob_to_odds(p)})
    four_plus = max(1.0 - cumulative, 0.0)
    rows.append({"goals": "4+", "prob": round(four_plus, 4), "odds": prob_to_odds(four_plus)})
    return rows


def btts_half_prob(lam_home: float, lam_away: float, fraction: float) -> Dict:
    """BTTS for one half. fraction = fraction of match goals in that half."""
    m = build_score_matrix(lam_home * fraction, lam_away * fraction, MAX_GOALS)
    y = btts_yes(m)
    n = btts_no(m)
    return {
        "yes": {"prob": round(y, 4), "odds": prob_to_odds(y)},
        "no":  {"prob": round(n, 4), "odds": prob_to_odds(n)},
    }


def htft_combo_probs(lam_home: float, lam_away: float) -> Dict:
    """
    9-way HT/FT combination market.
    Computes joint P(HT result, FT result) via independent HT and 2H score matrices.
    Keys: '1/1', '1/X', '1/2', 'X/1', 'X/X', 'X/2', '2/1', '2/X', '2/2'
    """
    MG = min(MAX_GOALS, 7)
    ht_m = build_score_matrix(lam_home * _ALPHA_1H, lam_away * _ALPHA_1H, MG)
    sh_m = build_score_matrix(lam_home * _ALPHA_2H, lam_away * _ALPHA_2H, MG)

    combos: Dict[str, float] = {
        "1/1": 0.0, "1/X": 0.0, "1/2": 0.0,
        "X/1": 0.0, "X/X": 0.0, "X/2": 0.0,
        "2/1": 0.0, "2/X": 0.0, "2/2": 0.0,
    }

    for i in range(MG + 1):
        for j in range(MG + 1):
            p_ht = float(ht_m[i, j])
            if p_ht < 1e-10:
                continue
            ht_r = "1" if i > j else ("X" if i == j else "2")
            for k in range(MG + 1):
                for l in range(MG + 1):
                    p_sh = float(sh_m[k, l])
                    if p_sh < 1e-10:
                        continue
                    ft_h, ft_a = i + k, j + l
                    ft_r = "1" if ft_h > ft_a else ("X" if ft_h == ft_a else "2")
                    combos[f"{ht_r}/{ft_r}"] += p_ht * p_sh

    return {
        k: {"prob": round(v, 4), "odds": prob_to_odds(v) if v > 0.001 else 999.0}
        for k, v in combos.items()
    }


def calculate_all_markets(lam_home: float, lam_away: float) -> Dict:
    """
    Calculate probabilities for all major betting markets.
    Returns a nested dict with raw probabilities and decimal odds.
    """
    matrix = build_score_matrix(lam_home, lam_away, MAX_GOALS)

    # 1x2
    hw = home_win_prob(matrix)
    dr = draw_prob(matrix)
    aw = away_win_prob(matrix)

    # Normalise (should already sum to ~1 but floating point)
    total_1x2 = hw + dr + aw
    hw /= total_1x2
    dr /= total_1x2
    aw /= total_1x2

    # BTTS
    btts_y = btts_yes(matrix)
    btts_n = btts_no(matrix)

    # Asian Total lines
    asian_totals = {}
    for line in ASIAN_TOTAL_LINES:
        over_p, under_p = asian_line_probs(matrix, line)
        asian_totals[line] = {
            "over": {"prob": over_p, "odds": prob_to_odds(over_p)},
            "under": {"prob": under_p, "odds": prob_to_odds(under_p)},
        }

    # Asian Handicap (home perspective)
    asian_handicaps = {}
    for h in ASIAN_HANDICAP_LINES:
        hp, push, ap = asian_handicap_probs(matrix, h)
        total_ah = hp + ap + push
        if total_ah > 0:
            hp /= total_ah; ap /= total_ah; push /= total_ah
        asian_handicaps[h] = {
            "home": {"prob": round(hp, 4), "odds": prob_to_odds(hp)},
            "push": round(push, 4),
            "away": {"prob": round(ap, 4), "odds": prob_to_odds(ap)},
        }

    # Double Chance
    dc_home_draw = round(hw + dr, 4)
    dc_away_draw = round(aw + dr, 4)
    dc_home_away = round(hw + aw, 4)

    # Draw No Bet
    dnb_total = hw + aw
    dnb_home = round(hw / dnb_total, 4) if dnb_total > 0 else 0.5
    dnb_away = round(aw / dnb_total, 4) if dnb_total > 0 else 0.5

    # Correct Scores
    cs = correct_score_probs(matrix, top_n=15)

    # First Goal
    fg = first_goal_probs(lam_home, lam_away)

    # Halftime Result
    ht = halftime_probs(lam_home, lam_away)

    # 1x2 BTTS combined
    combined = {}
    for outcome, ow in [("home", hw), ("draw", dr), ("away", aw)]:
        combined[f"{outcome}_btts_yes"] = round(ow * btts_y, 4)
        combined[f"{outcome}_btts_no"] = round(ow * btts_n, 4)

    return {
        "lam_home": lam_home,
        "lam_away": lam_away,
        "1x2": {
            "home": {"prob": round(hw, 4), "odds": prob_to_odds(hw)},
            "draw": {"prob": round(dr, 4), "odds": prob_to_odds(dr)},
            "away": {"prob": round(aw, 4), "odds": prob_to_odds(aw)},
        },
        "btts": {
            "yes": {"prob": round(btts_y, 4), "odds": prob_to_odds(btts_y)},
            "no": {"prob": round(btts_n, 4), "odds": prob_to_odds(btts_n)},
        },
        "btts_ht": btts_half_prob(lam_home, lam_away, _ALPHA_1H),
        "btts_2h": btts_half_prob(lam_home, lam_away, _ALPHA_2H),
        "asian_total": asian_totals,
        "asian_handicap": asian_handicaps,
        "double_chance": {
            "home_draw": {"prob": dc_home_draw, "odds": prob_to_odds(dc_home_draw)},
            "away_draw": {"prob": dc_away_draw, "odds": prob_to_odds(dc_away_draw)},
            "home_away": {"prob": dc_home_away, "odds": prob_to_odds(dc_home_away)},
        },
        "draw_no_bet": {
            "home": {"prob": dnb_home, "odds": prob_to_odds(dnb_home)},
            "away": {"prob": dnb_away, "odds": prob_to_odds(dnb_away)},
        },
        "clean_sheet": clean_sheet_probs(matrix),
        "total_goals_exact": total_goals_exact_probs(matrix),
        "htft_combo": htft_combo_probs(lam_home, lam_away),
        "correct_score": cs,
        "first_goal": {
            "home": {"prob": fg["home_first"], "odds": prob_to_odds(fg["home_first"])},
            "none": {"prob": fg["no_goal"], "odds": prob_to_odds(fg["no_goal"])},
            "away": {"prob": fg["away_first"], "odds": prob_to_odds(fg["away_first"])},
        },
        "halftime": {
            "home": {"prob": ht["home"], "odds": prob_to_odds(ht["home"])},
            "draw": {"prob": ht["draw"], "odds": prob_to_odds(ht["draw"])},
            "away": {"prob": ht["away"], "odds": prob_to_odds(ht["away"])},
        },
        "combined_1x2_btts": combined,
    }


def find_value_bets(markets: Dict, bookmaker_odds: Dict) -> List[Dict]:
    """
    Compare model odds vs bookmaker odds to identify value bets.
    A value bet is where our fair odds > bookmaker odds (model probability > implied probability).
    """
    value_bets = []

    def check_value(market_name: str, selection: str, model_prob: float, bookie_odds: float):
        implied_prob = 1.0 / bookie_odds if bookie_odds > 0 else 1.0
        edge = model_prob - implied_prob
        if edge > 0.03:  # at least 3% edge
            ev = (model_prob * (bookie_odds - 1)) - (1 - model_prob)
            value_bets.append({
                "market": market_name,
                "selection": selection,
                "model_prob": round(model_prob, 4),
                "model_odds": prob_to_odds(model_prob),
                "bookie_odds": bookie_odds,
                "implied_prob": round(implied_prob, 4),
                "edge_pct": round(edge * 100, 2),
                "expected_value": round(ev, 4),
            })

    # 1X2
    if "1x2" in bookmaker_odds:
        check_value("1X2", "Home", markets["1x2"]["home"]["prob"], bookmaker_odds["1x2"].get("home", 999))
        check_value("1X2", "Draw", markets["1x2"]["draw"]["prob"], bookmaker_odds["1x2"].get("draw", 999))
        check_value("1X2", "Away", markets["1x2"]["away"]["prob"], bookmaker_odds["1x2"].get("away", 999))

    # BTTS
    if "btts" in bookmaker_odds:
        check_value("BTTS", "Yes", markets["btts"]["yes"]["prob"], bookmaker_odds["btts"].get("yes", 999))
        check_value("BTTS", "No", markets["btts"]["no"]["prob"], bookmaker_odds["btts"].get("no", 999))

    # Draw No Bet (Sofascore key: 'dnb')
    dnb_odds = bookmaker_odds.get("dnb") or bookmaker_odds.get("draw_no_bet") or {}
    if dnb_odds:
        check_value("DNB", "Home", markets["draw_no_bet"]["home"]["prob"], dnb_odds.get("home", 999))
        check_value("DNB", "Away", markets["draw_no_bet"]["away"]["prob"], dnb_odds.get("away", 999))

    # Double Chance (Sofascore keys: home_draw / draw_away / home_away)
    dc_odds = bookmaker_odds.get("double_chance") or {}
    if dc_odds:
        check_value("DC", "1X", markets["double_chance"]["home_draw"]["prob"],
                    dc_odds.get("home_draw", 999))
        check_value("DC", "X2", markets["double_chance"]["away_draw"]["prob"],
                    dc_odds.get("draw_away", 999))
        check_value("DC", "12", markets["double_chance"]["home_away"]["prob"],
                    dc_odds.get("home_away", 999))

    # Halftime 1X2 (Sofascore key: 'ht_1x2')
    ht_odds = bookmaker_odds.get("ht_1x2") or bookmaker_odds.get("halftime") or {}
    if ht_odds:
        check_value("HT 1X2", "Home", markets["halftime"]["home"]["prob"], ht_odds.get("home", 999))
        check_value("HT 1X2", "Draw", markets["halftime"]["draw"]["prob"], ht_odds.get("draw", 999))
        check_value("HT 1X2", "Away", markets["halftime"]["away"]["prob"], ht_odds.get("away", 999))

    value_bets.sort(key=lambda x: x["edge_pct"], reverse=True)
    return value_bets
