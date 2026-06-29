import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")      # RapidAPI key (shared: API-Football + Sofascore)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
SPORTS_DB_API_KEY = os.getenv("SPORTS_DB_API_KEY", "3")
ALLSPORTS_API_KEY = os.getenv("ALLSPORTS_API_KEY", "")

# Settings
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "93.0"))
DATA_REFRESH_INTERVAL = int(os.getenv("DATA_REFRESH_INTERVAL", "300"))       # seconds
PRE_MATCH_REFRESH_INTERVAL = int(os.getenv("PRE_MATCH_REFRESH_INTERVAL", "60"))
LINEUP_REFRESH_INTERVAL = int(os.getenv("LINEUP_REFRESH_INTERVAL", "120"))
MAX_GOALS = int(os.getenv("MAX_GOALS_PREDICTION", "8"))
HOME_ADVANTAGE = float(os.getenv("HOME_ADVANTAGE_FACTOR", "1.15"))

# API Base URLs
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
API_FOOTBALL_BASE = "https://api-football-v1.p.rapidapi.com/v3"
SPORTS_DB_BASE = "https://www.thesportsdb.com/api/v1/json"
NEWS_API_BASE = "https://newsapi.org/v2"
GNEWS_BASE = "https://gnews.io/api/v4"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SOFASCORE_BASE = "https://sportapi7.p.rapidapi.com"
FOOTBALL_STANDINGS_BASE = "https://api-football-standings.azharimm.site"
ALLSPORTS_BASE = "https://apiv2.allsportsapi.com/football"
WORLDCUP26_BASE = "https://worldcup26.ir/get"

# ESPN league slugs to try for FIFA 2026
ESPN_WORLD_CUP_SLUGS = [
    "fifa.world",
    "fifa.worldq.concacaf",
]

# TheSportsDB World Cup League IDs
SPORTS_DB_WC_LEAGUE_IDS = [4480, 4328]

# FIFA 2026 World Cup competition codes
FIFA_WC_CODES = ["WC", "FIFA2026", "WC2026"]

# Dixon-Coles correction parameter — calibrated on WC 2026 actual low-score frequency
# Raised from -0.13 to -0.15 based on 72 group stage games showing higher 0-0/1-0 rate
DIXON_COLES_RHO = -0.15

# FIFA 2026 host city altitudes in metres above sea level.
# Only cities that meaningfully affect ball physics / stamina are listed;
# all other WC 2026 venues (New York, LA, Dallas, Miami, Toronto, etc.) are ≤ 200 m.
VENUE_ALTITUDES = {
    "mexico city": 2240,   # Estadio Azteca — significant
    "azteca":      2240,
    "guadalajara": 1566,   # Estadio Akron — notable
    "denver":      1609,   # Empower Field at Mile High — notable
    "monterrey":    538,   # Estadio BBVA — marginal
    "kansas city":  325,   # Arrowhead Stadium — negligible but listed for completeness
}


def altitude_goal_factor(city: str) -> float:
    """
    Lambda multiplier for altitude effect — applied symmetrically to both teams.
    At altitude: thinner air makes the ball travel faster and curve more unpredictably,
    goalkeeper reactions are harder to calibrate, and both defences fatigue faster
    in the second half → net effect is more goals per game.
    Empirical basis: South American WC qualifiers + WC 2010 (Johannesburg 1,753 m).
    """
    city_lower = (city or "").lower()
    altitude_m = 0
    for name, alt in VENUE_ALTITUDES.items():
        if name in city_lower:
            altitude_m = alt
            break
    if altitude_m >= 2000:
        return 1.10   # +10 % — Mexico City (Azteca)
    if altitude_m >= 1500:
        return 1.06   # +6 %  — Denver / Guadalajara
    if altitude_m >= 1000:
        return 1.03   # +3 %
    if altitude_m >= 500:
        return 1.01   # +1 %  — Monterrey
    return 1.0        # sea-level — no adjustment

# Model weights for composite prediction
MODEL_WEIGHTS = {
    "poisson": 0.40,
    "elo": 0.25,
    "form": 0.20,
    "player_impact": 0.10,
    "sentiment": 0.05,
}

# WC 2026 observed avg goals per match (updated after 72 group stage games)
WC2026_OBSERVED_AVG_GOALS = 2.83

# Knockout stage lambda reduction — teams play more defensively in KO rounds
# Reduces both lambdas by this fraction to model conservative KO tactics
KO_LAMBDA_REDUCTION = 0.90

# Bookmaker 1x2 blend weight for KO stage (higher trust in bookmaker odds when more data available)
KO_BOOKMAKER_BLEND = 0.62   # was 0.50 for group stage
GROUP_BOOKMAKER_BLEND = 0.55  # slightly raised from 0.50

# ELO initial ratings for FIFA 2026 teams (based on FIFA rankings + WC qualifying performance)
FIFA_2026_ELO_RATINGS = {
    # South America (CONMEBOL)
    "Argentina": 2095, "Brazil": 2045, "Uruguay": 1900, "Colombia": 1885,
    "Ecuador": 1785, "Chile": 1760, "Paraguay": 1750, "Bolivia": 1700,
    "Peru": 1760, "Venezuela": 1695,

    # Europe (UEFA)
    "France": 2065, "England": 2025, "Spain": 2015, "Germany": 2000,
    "Portugal": 1980, "Netherlands": 1965, "Belgium": 1950, "Italy": 1935,
    "Croatia": 1920, "Denmark": 1905, "Switzerland": 1900, "Poland": 1870,
    "Serbia": 1865, "Austria": 1855, "Hungary": 1845, "Scotland": 1840,
    "Ukraine": 1835, "Turkey": 1830, "Slovakia": 1820, "Wales": 1815,
    "Czech Republic": 1810, "Norway": 1805, "Sweden": 1800, "Greece": 1795,
    "Slovenia": 1790, "Albania": 1780,

    # North/Central America & Caribbean (CONCACAF)
    "USA": 1845, "Mexico": 1880, "Canada": 1780, "Panama": 1730,
    "Costa Rica": 1725, "Honduras": 1715, "Jamaica": 1670, "El Salvador": 1685,
    "Guatemala": 1665, "Trinidad and Tobago": 1660,

    # Africa (CAF)
    "Morocco": 1840, "Senegal": 1790, "Egypt": 1750, "Nigeria": 1770,
    "Ghana": 1760, "Cameroon": 1755, "Algeria": 1745, "Ivory Coast": 1740,
    "South Africa": 1720, "Mali": 1715, "Tunisia": 1765,
    "DR Congo": 1710,

    # Asia (AFC)
    "Japan": 1830, "South Korea": 1805, "Australia": 1790, "Iran": 1730,
    "Saudi Arabia": 1735, "Qatar": 1695, "China": 1705, "Iraq": 1720,
    "Jordan": 1700, "UAE": 1685, "Oman": 1670, "Uzbekistan": 1715,
    "Bahrain": 1665,

    # Oceania (OFC)
    "New Zealand": 1650,
}

# Cache TTL (seconds)
CACHE_TTL = {
    "fixtures": 3600,
    "lineups": 120,
    "form": 7200,
    "h2h": 86400,
    "news": 1800,
    "weather": 3600,
    "standings": 7200,
    "player_stats": 14400,
}

# Leagues tracked for form data
TRACKED_LEAGUES = {
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "PD": "La Liga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "WC": "FIFA World Cup",
    "CL": "Champions League",
}

# Prediction market labels
MARKETS = [
    "1x2",
    "btts",
    "over_under",
    "asian_handicap",
    "double_chance",
    "draw_no_bet",
    "correct_score",
    "first_goal",
    "halftime_result",
]
