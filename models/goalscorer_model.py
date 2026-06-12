"""
Anytime Goalscorer Model.
Uses Poisson distribution with player-level goal shares.

Math:
  Each player's expected goals = λ_team × (player_goal_share / total_share) × 0.85
  P(player scores) = 1 - exp(-player_lam)   [Poisson P(X >= 1)]

Goal shares come from:
  1. Real WC 2026 top-scorer data (most accurate)
  2. Sofascore player rating as a proxy (if no goal data)
  3. Position-based defaults (fallback)
"""
import math
from typing import Dict, List, Optional


# Typical fraction of a team's goals by position (per 90 min across competitions)
_POS_SHARES: Dict[str, float] = {
    # Forwards
    "F":  0.28, "FW": 0.27, "CF": 0.30, "ST": 0.30,
    "LW": 0.17, "RW": 0.17, "SS": 0.20,
    # Attacking midfielders
    "AM": 0.14, "CAM": 0.14, "AMF": 0.14,
    # Wide midfielders
    "LM": 0.09, "RM": 0.09,
    # Central midfielders
    "M":  0.07, "CM": 0.07, "MF": 0.07,
    # Defensive midfielders
    "DM": 0.04, "CDM": 0.04, "DMF": 0.04,
    # Defenders
    "D":  0.04, "DF": 0.04,
    "CB": 0.03, "LB": 0.04, "RB": 0.04, "WB": 0.05,
    # Goalkeepers
    "G":  0.005, "GK": 0.005,
}

_DEFAULT_BY_CATEGORY = {
    "F": 0.26,   # forward
    "M": 0.08,   # midfielder
    "D": 0.04,   # defender
    "G": 0.005,  # goalkeeper
}


def _pos_share(position: str) -> float:
    pos = (position or "").upper().strip()
    if pos in _POS_SHARES:
        return _POS_SHARES[pos]
    # Partial match
    for key, val in _POS_SHARES.items():
        if pos.startswith(key) or key.startswith(pos):
            return val
    # Category fallback
    first = pos[:1]
    return _DEFAULT_BY_CATEGORY.get(first, _DEFAULT_BY_CATEGORY["M"])


def _names_match(a: str, b: str) -> bool:
    """Check if two player names refer to the same person (last-name match)."""
    a_l, b_l = a.lower().strip(), b.lower().strip()
    if a_l == b_l:
        return True
    a_parts, b_parts = a_l.split(), b_l.split()
    # Last name match
    if a_parts and b_parts and a_parts[-1] == b_parts[-1]:
        return True
    # One name contains the other
    if a_l in b_l or b_l in a_l:
        return True
    return False


def calculate_goalscorer_probs(
    lineup: List[Dict],
    team_lam: float,
    top_scorers: Optional[Dict[str, float]] = None,
    player_ratings: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Return anytime-goalscorer probabilities for each starting player.

    Args:
        lineup: list of {name, position, ...} dicts (starters only)
        team_lam: team's expected goals for this match
        top_scorers: {player_name: goals_in_competition} from football-data.org
        player_ratings: {player_name: rating_0_to_10} from Sofascore (optional boost)

    Returns:
        list of {name, position, prob, odds, goals_in_comp, is_key_scorer}
        sorted by prob descending
    """
    if not lineup or team_lam <= 0:
        return []

    starters = lineup[:11]
    raw_shares: List[float] = []

    for player in starters:
        name = player.get("name", "") if isinstance(player, dict) else str(player)
        pos  = player.get("position", "") if isinstance(player, dict) else ""

        # Try to find real goal data
        share = None
        goals_scored = 0
        if top_scorers:
            for scorer_name, goals in top_scorers.items():
                if _names_match(name, scorer_name):
                    goals_scored = goals
                    # Scale goals into a share value; 1 goal ≈ 0.15 extra share
                    share = _pos_share(pos) + goals * 0.12
                    break

        # Sofascore rating boost (high-rated players score more)
        if player_ratings and name in player_ratings:
            rating = player_ratings[name]
            rating_boost = max(0.0, (rating - 6.5) * 0.012)
            if share is not None:
                share += rating_boost
            else:
                share = _pos_share(pos) + rating_boost

        if share is None:
            share = _pos_share(pos)

        raw_shares.append(max(share, 0.001))

    # Normalise — starters account for ~85% of team goals
    total = sum(raw_shares)
    results = []

    for player, share in zip(starters, raw_shares):
        name = player.get("name", "") if isinstance(player, dict) else str(player)
        pos  = player.get("position", "") if isinstance(player, dict) else ""

        # Player's expected goals = team × their fraction × 0.85 starter factor
        player_lam = team_lam * (share / total) * 0.85

        # Poisson: P(scores ≥ 1) = 1 - e^(-λ)
        prob = 1.0 - math.exp(-player_lam)
        fair_odds = round(1.0 / prob, 2) if prob > 0.005 else 200.0

        # Look up goals scored in competition
        goals_in_comp = 0
        if top_scorers:
            for sname, g in top_scorers.items():
                if _names_match(name, sname):
                    goals_in_comp = g
                    break

        results.append({
            "name": name,
            "position": pos or "—",
            "player_lam": round(player_lam, 3),
            "prob": round(prob, 4),
            "odds": fair_odds,
            "goals_in_comp": goals_in_comp,
            "is_key_scorer": goals_in_comp >= 2 or prob >= 0.35,
        })

    results.sort(key=lambda x: x["prob"], reverse=True)
    return results


def first_goalscorer_probs(anytime_probs: List[Dict]) -> List[Dict]:
    """
    Estimate 1st goalscorer probabilities from anytime probs.
    P(first) ≈ P(anytime) × K where K is a normalisation constant.
    """
    if not anytime_probs:
        return []

    # First scorer probability is roughly proportional to scoring rate
    # Simple approximation: distribute 1st goal according to player_lam share
    total_lam = sum(p["player_lam"] for p in anytime_probs)
    if total_lam <= 0:
        return []

    results = []
    for p in anytime_probs:
        first_prob = p["player_lam"] / total_lam if total_lam > 0 else 0
        results.append({
            **p,
            "first_prob": round(first_prob, 4),
            "first_odds": round(1.0 / first_prob, 2) if first_prob > 0.005 else 200.0,
        })

    return sorted(results, key=lambda x: x["first_prob"], reverse=True)
