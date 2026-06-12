"""
FIFA 2026 Match Prediction - Web Dashboard
Stake-style dark UI with full market coverage.
Run: streamlit run app.py  |  Or double-click FIFA_Web.bat
"""
import sys
import math
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="FIFA 2026 Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Stake-style dark theme ───────────────────────────────────
st.markdown("""
<style>
/* Base */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.25rem; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #7fa8d1; }

/* Odds button */
.ob { background:#1a2637; border:1px solid #2d4a6e; border-radius:8px;
      padding:12px 8px; text-align:center; }
.ob.vb { border:2px solid #00e676; box-shadow:0 0 10px rgba(0,230,118,.22); }
.ob .lbl { color:#7fa8d1; font-size:.76rem; margin-bottom:5px; white-space:nowrap;
           overflow:hidden; text-overflow:ellipsis; }
.ob .val { color:#00e5ff; font-size:1.2rem; font-weight:700; }
.ob .pct { color:#8899aa; font-size:.7rem; margin-top:3px; }
.ob .edg { color:#00e676; font-size:.65rem; font-weight:700; margin-top:2px; }

/* Market section header */
.mhdr { background:#131d30; border-left:3px solid #00e5ff; padding:8px 14px;
        border-radius:4px; margin-bottom:10px; font-size:.87rem;
        font-weight:600; color:#cdd9e5; letter-spacing:.4px; }

/* Match header */
.mhead { background:linear-gradient(135deg,#0d1b2a,#142138);
         padding:20px 28px; border-radius:12px; margin-bottom:16px;
         border:1px solid #1e3351; }

/* Scorer card */
.sc { background:#1a2637; border:1px solid #2d4a6e; border-radius:8px;
      padding:10px 8px; text-align:center; }
.sc.key { border:2px solid #f9a825; box-shadow:0 0 8px rgba(249,168,37,.25); }
.sc .sn { color:#cdd9e5; font-size:.8rem; font-weight:600; margin-bottom:3px; }
.sc .sp { color:#6b8099; font-size:.68rem; margin-bottom:6px; }
.sc .so { color:#00e5ff; font-size:1.1rem; font-weight:700; }
.sc .spct { color:#8899aa; font-size:.68rem; margin-top:2px; }

/* HT/FT grid cell */
.htft { background:#1a2637; border:1px solid #2d4a6e; border-radius:6px;
        padding:8px 6px; text-align:center; }
.htft .hl { color:#7fa8d1; font-size:.68rem; margin-bottom:4px; }
.htft .hv { color:#00e5ff; font-size:1.05rem; font-weight:700; }
.htft .hp { color:#8899aa; font-size:.66rem; margin-top:2px; }

/* Confidence banner */
.conf-hi { background:#1b5e20; }
.conf-md { background:#e65100; }
.conf-lo { background:#7f1010; }

/* Divider spacing */
.mkt-divider { margin: 16px 0; border-top: 1px solid #1e3351; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# Cached resources
# ════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading prediction engine...")
def load_engine():
    from prediction.engine import PredictionEngine
    return PredictionEngine()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fixtures(_engine, date_str: str) -> List[Dict]:
    d = date.fromisoformat(date_str)
    return _engine.aggregator.get_today_matches(d)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prediction(
    _engine, home: str, away: str,
    match_id: str, sofascore_id: Optional[int],
    city: str, match_date: str,
) -> Dict:
    return _engine.predict_match(
        home_team=home, away_team=away,
        match_id=match_id or None, sofascore_id=sofascore_id,
        venue_city=city or "Dallas", match_date=match_date,
    )


# ════════════════════════════════════════════════════════════════════════
# HTML helpers — Stake-style components
# ════════════════════════════════════════════════════════════════════════

def _mhdr(title: str) -> str:
    return f'<div class="mhdr">{title}</div>'


def _odds_grid(options: List[Dict], value_keys: set = None) -> str:
    """
    Build a flex grid of Stake-style odds buttons.
    Each option: {label, odds, prob, edge (optional)}
    value_keys: set of label strings that are value bets
    """
    cells = ""
    for opt in options:
        label   = opt.get("label", "—")
        odds    = opt.get("odds", 1.0)
        prob    = opt.get("prob", 0.0)
        edge    = opt.get("edge")
        is_val  = value_keys and label in value_keys
        cls     = "ob vb" if is_val else "ob"
        edge_h  = f'<div class="edg">+{edge:.1f}% edge</div>' if is_val and edge else ""
        cells += f"""
        <div class="{cls}" style="min-width:90px;flex:1">
            <div class="lbl">{label}</div>
            <div class="val">{odds:.2f}</div>
            <div class="pct">{prob*100:.1f}%</div>
            {edge_h}
        </div>"""

    return f'<div style="display:flex;gap:8px;margin:8px 0 18px;flex-wrap:wrap">{cells}</div>'


def _htft_grid(htft: Dict, home: str, away: str) -> str:
    """3x3 HT/FT grid."""
    order = ["1", "X", "2"]
    names = {"1": home[:10], "X": "Draw", "2": away[:10]}
    rows_html = ""
    for ht_r in order:
        for ft_r in order:
            key = f"{ht_r}/{ft_r}"
            d = htft.get(key, {"prob": 0, "odds": 999})
            prob, odds = d["prob"], d["odds"]
            bg_alpha = min(0.15 + prob * 2.5, 0.85)
            rows_html += f"""
            <div class="htft" style="background:rgba(26,38,55,{bg_alpha:.2f})">
                <div class="hl">{names[ht_r]} / {names[ft_r]}</div>
                <div class="hv">{odds:.2f}</div>
                <div class="hp">{prob*100:.1f}%</div>
            </div>"""

    return f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0 18px">{rows_html}</div>'


def _scorer_grid(scorers: List[Dict], title: str, use_first: bool = False) -> str:
    """Grid of player scorer cards."""
    if not scorers:
        return f'<div style="color:#6b8099;font-size:.8rem;margin-bottom:12px">No lineup data available for {title}.</div>'

    prob_key  = "first_prob"  if use_first else "prob"
    odds_key  = "first_odds" if use_first else "odds"

    cards = ""
    for s in scorers[:12]:
        name  = s.get("name", "—")
        pos   = s.get("position", "—")
        odds  = s.get(odds_key, 99)
        prob  = s.get(prob_key, 0)
        goals = s.get("goals_in_comp", 0)
        is_key = s.get("is_key_scorer", False)

        cls        = "sc key" if is_key else "sc"
        goals_bdg  = f'<span style="background:#f9a825;color:#000;padding:1px 5px;border-radius:3px;font-size:.62rem;font-weight:700;margin-right:4px">{goals}g</span>' if goals else ""
        star       = " &#9733;" if is_key else ""

        cards += f"""
        <div class="{cls}">
            <div class="sn">{goals_bdg}{name}{star}</div>
            <div class="sp">{pos}</div>
            <div class="so">{odds:.2f}</div>
            <div class="spct">{prob*100:.1f}%</div>
        </div>"""

    return f"""
    <div style="margin-bottom:6px;font-size:.8rem;color:#7fa8d1;font-weight:600">{title}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:6px;margin-bottom:20px">
        {cards}
    </div>"""


def _correct_score_grid(cs_list: List[Dict]) -> str:
    """Compact grid for correct scores."""
    top = cs_list[:16]
    cells = ""
    for s in top:
        score = f"{s['home']}-{s['away']}"
        p     = s["probability"]
        o     = round(1.0 / p, 2) if p > 0 else 999
        cells += f"""
        <div style="background:#1a2637;border:1px solid #2d4a6e;border-radius:6px;
                    padding:7px 5px;text-align:center">
            <div style="color:#cdd9e5;font-size:.9rem;font-weight:700">{score}</div>
            <div style="color:#00e5ff;font-size:1rem;font-weight:700">{o:.2f}</div>
            <div style="color:#8899aa;font-size:.66rem">{p*100:.1f}%</div>
        </div>"""

    return f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(70px,1fr));gap:6px;margin:8px 0 18px">{cells}</div>'


# ════════════════════════════════════════════════════════════════════════
# Value bet index helpers
# ════════════════════════════════════════════════════════════════════════

def _vbet_labels(vbets: List[Dict], market: str) -> set:
    """Return set of selection labels that are value bets for a given market."""
    return {v["selection"] for v in vbets if v["market"] == market}


def _vbet_edges(vbets: List[Dict], market: str) -> Dict[str, float]:
    return {v["selection"]: v["edge_pct"] for v in vbets if v["market"] == market}


# ════════════════════════════════════════════════════════════════════════
# Market tab renderers
# ════════════════════════════════════════════════════════════════════════

def _tab_main(markets: Dict, home: str, away: str, vbets: List[Dict]):
    h = home[:14]
    a = away[:14]

    # Value-bet sets per market (for highlighting)
    vk_1x2 = _vbet_labels(vbets, "1X2")
    vk_dnb  = _vbet_labels(vbets, "DNB")
    vk_dc   = _vbet_labels(vbets, "DC")
    vk_btts = _vbet_labels(vbets, "BTTS")
    ve_1x2  = _vbet_edges(vbets, "1X2")
    ve_dnb  = _vbet_edges(vbets, "DNB")
    ve_dc   = _vbet_edges(vbets, "DC")
    ve_btts = _vbet_edges(vbets, "BTTS")

    # 1X2 match winner
    x = markets["1x2"]
    st.markdown(_mhdr("Match Winner — Full Time"), unsafe_allow_html=True)
    st.markdown(_odds_grid([
        {"label": h,      "odds": x["home"]["odds"], "prob": x["home"]["prob"],
         "edge": ve_1x2.get("Home")},
        {"label": "Draw", "odds": x["draw"]["odds"], "prob": x["draw"]["prob"],
         "edge": ve_1x2.get("Draw")},
        {"label": a,      "odds": x["away"]["odds"], "prob": x["away"]["prob"],
         "edge": ve_1x2.get("Away")},
    ], value_keys=vk_1x2), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # BTTS Full Time
        b = markets["btts"]
        st.markdown(_mhdr("Both Teams to Score — Full Time"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": "Yes", "odds": b["yes"]["odds"], "prob": b["yes"]["prob"],
             "edge": ve_btts.get("Yes")},
            {"label": "No",  "odds": b["no"]["odds"],  "prob": b["no"]["prob"],
             "edge": ve_btts.get("No")},
        ], value_keys=vk_btts), unsafe_allow_html=True)

        # Draw No Bet
        d = markets["draw_no_bet"]
        st.markdown(_mhdr("Draw No Bet"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": h, "odds": d["home"]["odds"], "prob": d["home"]["prob"],
             "edge": ve_dnb.get("Home")},
            {"label": a, "odds": d["away"]["odds"], "prob": d["away"]["prob"],
             "edge": ve_dnb.get("Away")},
        ], value_keys=vk_dnb), unsafe_allow_html=True)

    with col2:
        # Double Chance
        dc = markets["double_chance"]
        st.markdown(_mhdr("Double Chance"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": f"{h} or Draw", "odds": dc["home_draw"]["odds"], "prob": dc["home_draw"]["prob"],
             "edge": ve_dc.get("1X")},
            {"label": f"{a} or Draw", "odds": dc["away_draw"]["odds"], "prob": dc["away_draw"]["prob"],
             "edge": ve_dc.get("X2")},
            {"label": f"{h} or {a}", "odds": dc["home_away"]["odds"], "prob": dc["home_away"]["prob"],
             "edge": ve_dc.get("12")},
        ], value_keys=vk_dc), unsafe_allow_html=True)

        # Clean Sheet
        cs_mkt = markets["clean_sheet"]
        st.markdown(_mhdr("Clean Sheet"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": f"{h} yes", "odds": cs_mkt["away"]["odds"], "prob": cs_mkt["away"]["prob"]},
            {"label": f"{h} no",  "odds": round(1.0/(1-cs_mkt["away"]["prob"]),2) if cs_mkt["away"]["prob"] < 0.99 else 99,
             "prob": 1-cs_mkt["away"]["prob"]},
            {"label": f"{a} yes", "odds": cs_mkt["home"]["odds"], "prob": cs_mkt["home"]["prob"]},
            {"label": f"{a} no",  "odds": round(1.0/(1-cs_mkt["home"]["prob"]),2) if cs_mkt["home"]["prob"] < 0.99 else 99,
             "prob": 1-cs_mkt["home"]["prob"]},
        ]), unsafe_allow_html=True)

    # 1st Goal
    fg = markets["first_goal"]
    st.markdown(_mhdr("1st Goal of Match"), unsafe_allow_html=True)
    st.markdown(_odds_grid([
        {"label": h,           "odds": fg["home"]["odds"], "prob": fg["home"]["prob"]},
        {"label": "No Goal",   "odds": fg["none"]["odds"], "prob": fg["none"]["prob"]},
        {"label": a,           "odds": fg["away"]["odds"], "prob": fg["away"]["prob"]},
    ]), unsafe_allow_html=True)

    # Correct Score
    st.markdown(_mhdr("Correct Score (Top 16)"), unsafe_allow_html=True)
    st.markdown(_correct_score_grid(markets["correct_score"]), unsafe_allow_html=True)


def _tab_goals(markets: Dict, home: str, away: str):
    # Total Goals Exact
    tge = markets["total_goals_exact"]
    st.markdown(_mhdr("Total Goals — Exact Count"), unsafe_allow_html=True)
    st.markdown(_odds_grid([
        {"label": f"{r['goals']} Goals", "odds": r["odds"], "prob": r["prob"]}
        for r in tge
    ]), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # BTTS 1st Half
        b1 = markets["btts_ht"]
        st.markdown(_mhdr("Both Teams to Score — 1st Half"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": "Yes", "odds": b1["yes"]["odds"], "prob": b1["yes"]["prob"]},
            {"label": "No",  "odds": b1["no"]["odds"],  "prob": b1["no"]["prob"]},
        ]), unsafe_allow_html=True)

    with col2:
        # BTTS 2nd Half
        b2 = markets["btts_2h"]
        st.markdown(_mhdr("Both Teams to Score — 2nd Half"), unsafe_allow_html=True)
        st.markdown(_odds_grid([
            {"label": "Yes", "odds": b2["yes"]["odds"], "prob": b2["yes"]["prob"]},
            {"label": "No",  "odds": b2["no"]["odds"],  "prob": b2["no"]["prob"]},
        ]), unsafe_allow_html=True)

    # Asian Total table
    st.markdown(_mhdr("Asian Total (Over / Under)"), unsafe_allow_html=True)
    rows = []
    for line, d in markets["asian_total"].items():
        rows.append({
            "Line":       float(line),
            "Over %":     f"{d['over']['prob']*100:.1f}%",
            "Over Odds":  f"{d['over']['odds']:.2f}",
            "Under %":    f"{d['under']['prob']*100:.1f}%",
            "Under Odds": f"{d['under']['odds']:.2f}",
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Line"),
        use_container_width=True,
    )


def _tab_asian(markets: Dict, home: str, away: str):
    h = home[:14]
    a = away[:14]

    st.markdown(_mhdr(f"Asian Handicap ({h} perspective)"), unsafe_allow_html=True)
    rows = []
    for h_val, d in markets["asian_handicap"].items():
        rows.append({
            "Handicap":    f"{h_val:+.2f}",
            f"{h} %":      f"{d['home']['prob']*100:.1f}%",
            f"{h} Odds":   f"{d['home']['odds']:.2f}",
            "Push":        f"{d['push']*100:.1f}%",
            f"{a} %":      f"{d['away']['prob']*100:.1f}%",
            f"{a} Odds":   f"{d['away']['odds']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Handicap"), use_container_width=True)


def _tab_half(markets: Dict, home: str, away: str):
    h = home[:14]
    a = away[:14]

    # HT 1X2
    ht = markets["halftime"]
    vk_ht = set()  # (extend to check value bets if needed)
    st.markdown(_mhdr("Half-Time Result (1X2)"), unsafe_allow_html=True)
    st.markdown(_odds_grid([
        {"label": h,      "odds": ht["home"]["odds"], "prob": ht["home"]["prob"]},
        {"label": "Draw", "odds": ht["draw"]["odds"], "prob": ht["draw"]["prob"]},
        {"label": a,      "odds": ht["away"]["odds"], "prob": ht["away"]["prob"]},
    ]), unsafe_allow_html=True)

    # HT/FT combo
    htft_mkt = markets.get("htft_combo", {})
    if htft_mkt:
        st.markdown(_mhdr("Half-Time / Full-Time"), unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#6b8099;font-size:.75rem;margin-bottom:6px">'
            'Format: HT result / FT result &nbsp;|&nbsp; 1=Home, X=Draw, 2=Away</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_htft_grid(htft_mkt, h, a), unsafe_allow_html=True)


def _tab_goalscorers(data: Dict, home: str, away: str):
    home_sc = data.get("home_scorers", [])
    away_sc = data.get("away_scorers", [])
    home_1st = data.get("home_1st_scorers", [])
    away_1st = data.get("away_1st_scorers", [])

    if not (home_sc or away_sc):
        st.info("Goalscorer predictions require lineup data. Lineups are usually confirmed ~1 hour before kickoff.")
        return

    lineup_note = data.get("lineup_confirmed", False)
    tsc = data.get("top_scorers_count", 0)
    note_parts = []
    if lineup_note:
        note_parts.append("Confirmed lineup")
    if tsc > 0:
        note_parts.append(f"{tsc} WC scorers loaded")
    if note_parts:
        st.caption("Data: " + " · ".join(note_parts) + " · Poisson model per player")

    st.markdown(_mhdr("Anytime Goalscorer"), unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_scorer_grid(home_sc, home, use_first=False), unsafe_allow_html=True)
    with col2:
        st.markdown(_scorer_grid(away_sc, away, use_first=False), unsafe_allow_html=True)

    st.markdown(_mhdr("1st Goalscorer"), unsafe_allow_html=True)
    col1b, col2b = st.columns(2)
    with col1b:
        st.markdown(_scorer_grid(home_1st[:8], home, use_first=True), unsafe_allow_html=True)
    with col2b:
        st.markdown(_scorer_grid(away_1st[:8], away, use_first=True), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# Match header
# ════════════════════════════════════════════════════════════════════════

def _render_match_header(pred: Dict):
    home     = pred["home_team"]
    away     = pred["away_team"]
    conf     = pred["confidence"]
    markets  = pred["markets"]
    diag     = pred.get("diagnostics", {})
    data     = pred.get("data", {})
    odds_src = pred.get("bookmaker_odds_source")

    conf_pct  = conf["total"]
    predicted = conf.get("predicted_outcome", "?")
    pred_prob = conf.get("predicted_outcome_prob", 0)

    if conf_pct >= 93:
        conf_bg, conf_tag = "#1b5e20", "ABOVE 93% THRESHOLD"
    elif conf_pct >= 65:
        conf_bg, conf_tag = "#7c3a00", "Medium confidence"
    else:
        conf_bg, conf_tag = "#6b0f0f", "Below threshold"

    lam_h = markets["lam_home"]
    lam_a = markets["lam_away"]

    oddsrc_badge = ""
    if odds_src == "sofascore":
        oddsrc_badge = '<span style="background:#004d40;color:#00e676;padding:2px 8px;border-radius:4px;font-size:.72rem;margin-left:10px">LIVE ODDS</span>'
    elif odds_src == "manual":
        oddsrc_badge = '<span style="background:#1a2e4a;color:#7fa8d1;padding:2px 8px;border-radius:4px;font-size:.72rem;margin-left:10px">MANUAL ODDS</span>'

    # Confidence bar
    bar_w = min(conf_pct, 100)
    bar_col = "#00e676" if conf_pct >= 93 else ("#ff9800" if conf_pct >= 65 else "#ef5350")

    st.markdown(f"""
    <div class="mhead">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
            <div>
                <div style="font-size:1.5rem;font-weight:800;color:#ffffff;letter-spacing:1px">
                    {home}  <span style="color:#2d4a6e">vs</span>  {away}
                </div>
                <div style="color:#6b8099;font-size:.78rem;margin-top:4px">
                    FIFA World Cup 2026
                    {oddsrc_badge}
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-size:1.7rem;font-weight:800;color:#00e5ff">{conf_pct:.1f}%</div>
                <div style="font-size:.72rem;color:#cdd9e5">{conf_tag}</div>
                <div style="font-size:.75rem;color:#9ab;margin-top:2px">
                    Predicted: <b style="color:#cdd9e5">{predicted}</b> ({pred_prob:.1f}%)
                </div>
            </div>
        </div>
        <div style="margin-top:10px;background:#0e1826;border-radius:4px;height:6px;overflow:hidden">
            <div style="width:{bar_w}%;height:100%;background:{bar_col};border-radius:4px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    co = st.columns(6)
    co[0].metric(f"{home[:10]} xG",  f"{lam_h:.2f}")
    co[1].metric(f"{away[:10]} xG",  f"{lam_a:.2f}")
    co[2].metric(f"{home[:8]} ELO",  f"{diag.get('elo_home', 0):.0f}")
    co[3].metric(f"{away[:8]} ELO",  f"{diag.get('elo_away', 0):.0f}")
    co[4].metric(f"{home[:8]} Form", f"{diag.get('home_form_score', 0.5):.2f}")
    co[5].metric(f"{away[:8]} Form", f"{diag.get('away_form_score', 0.5):.2f}")

    # Weather caption
    weather = data.get("weather")
    if weather:
        wimp = weather.get("impact_factor", 1.0)
        icon = "🌧" if wimp < 0.95 else "☀"
        st.caption(
            f"{icon} {weather.get('description', '')} · "
            f"{weather.get('temperature_c', '?')}°C · "
            f"Wind {weather.get('wind_speed_kmh', '?')} km/h · "
            f"Weather impact {wimp:.2f}x"
        )

    # Missing players warning
    for mp, label in [
        (data.get("missing_home_players", []), home),
        (data.get("missing_away_players", []), away),
    ]:
        if mp:
            st.warning(f"{label} missing: {', '.join(mp)}")

    # Lineups expander
    home_lu = data.get("home_lineup", [])
    away_lu = data.get("away_lineup", [])
    confirmed = data.get("lineup_confirmed", False)
    h_form = data.get("home_formation", "")
    a_form = data.get("away_formation", "")
    if home_lu or away_lu:
        lu_label = "Confirmed" if confirmed else "Expected (unconfirmed)"
        form_str = f" · {h_form} vs {a_form}" if (h_form or a_form) else ""
        with st.expander(f"Starting Lineups — {lu_label}{form_str}", expanded=False):
            lc1, lc2 = st.columns(2)
            def _pname(p):
                return (p.get("name", "") if isinstance(p, dict) else str(p))
            with lc1:
                st.markdown(f"**{home}**" + (f"  `{h_form}`" if h_form else ""))
                for i, p in enumerate(home_lu[:11], 1):
                    pos = p.get("position", "") if isinstance(p, dict) else ""
                    st.write(f"{i}. {_pname(p)}" + (f" · *{pos}*" if pos else ""))
            with lc2:
                st.markdown(f"**{away}**" + (f"  `{a_form}`" if a_form else ""))
                for i, p in enumerate(away_lu[:11], 1):
                    pos = p.get("position", "") if isinstance(p, dict) else ""
                    st.write(f"{i}. {_pname(p)}" + (f" · *{pos}*" if pos else ""))


# ════════════════════════════════════════════════════════════════════════
# Value bets summary
# ════════════════════════════════════════════════════════════════════════

def _render_value_bets(vbets: List[Dict], odds_src: Optional[str]):
    if vbets:
        st.markdown(
            f'<div style="background:#0a2718;border:1px solid #00e676;border-radius:8px;'
            f'padding:12px 16px;margin:14px 0">'
            f'<div style="color:#00e676;font-weight:700;font-size:.9rem;margin-bottom:8px">'
            f'Value Bets — {len(vbets)} found</div>',
            unsafe_allow_html=True
        )
        for v in vbets:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid #0d3320">'
                f'<span style="color:#cdd9e5;font-size:.82rem">'
                f'<b>{v["market"]}</b> — {v["selection"]}</span>'
                f'<span style="color:#8899aa;font-size:.78rem">'
                f'Our {v["model_odds"]:.2f} vs Book {v["bookie_odds"]:.2f}</span>'
                f'<span style="color:#00e676;font-weight:700;font-size:.82rem">'
                f'+{v["edge_pct"]:.1f}% edge &nbsp; EV {v["expected_value"]:+.3f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    elif odds_src:
        st.caption("No value bets detected vs current bookmaker odds.")


# ════════════════════════════════════════════════════════════════════════
# Data analysis expander
# ════════════════════════════════════════════════════════════════════════

def _render_data_analysis(pred: Dict):
    data  = pred.get("data", {})
    conf  = pred["confidence"]
    diag  = pred.get("diagnostics", {})

    with st.expander("Data Analysis & Model Breakdown", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        comp = conf["components"]
        c1.metric("Prediction Clarity", f"{comp['prediction_clarity']:.1f}%")
        c2.metric("Data Quality",       f"{comp['data_quality']:.1f}%")
        c3.metric("Model Agreement",    f"{comp['model_agreement']:.1f}%")
        c4.metric("Lineup Certainty",   f"{comp['lineup_certainty']:.1f}%")

        st.progress(
            min(conf["total"] / 100, 1.0),
            text=f"Overall confidence: {conf['total']:.1f}%  (93% required threshold)"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Past matches analysed:**")
            st.write(f"- {pred['home_team']}: {data.get('home_form_count', 0)} form matches")
            st.write(f"- {pred['away_team']}: {data.get('away_form_count', 0)} form matches")
            st.write(f"- H2H history: {data.get('h2h_count', 0)} matches")
            st.write(f"- WC scorers loaded: {data.get('top_scorers_count', 0)}")

        with col_b:
            st.markdown("**Model components:**")
            home_c = diag.get("home_composite", {})
            away_c = diag.get("away_composite", {})
            if home_c:
                st.write(f"- {pred['home_team']} base xG: {home_c.get('base_lam', 0):.2f}")
                st.write(f"- {pred['home_team']} form adj: {home_c.get('form_adj', 1):.2f}x")
                st.write(f"- {pred['home_team']} ELO adj: {home_c.get('elo_adj', 1):.2f}x")
            if away_c:
                st.write(f"- {pred['away_team']} base xG: {away_c.get('base_lam', 0):.2f}")
                st.write(f"- {pred['away_team']} form adj: {away_c.get('form_adj', 1):.2f}x")
                st.write(f"- {pred['away_team']} ELO adj: {away_c.get('elo_adj', 1):.2f}x")

        # H2H results
        h2h = data.get("h2h", [])
        if h2h:
            st.markdown("**Head-to-Head (last 5):**")
            home_wins = sum(1 for m in h2h[:5] if m.get("result") == "home")
            draws     = sum(1 for m in h2h[:5] if m.get("result") == "draw")
            away_wins = sum(1 for m in h2h[:5] if m.get("result") == "away")
            st.write(
                f"{pred['home_team']} wins: **{home_wins}**  |  "
                f"Draws: **{draws}**  |  "
                f"{pred['away_team']} wins: **{away_wins}**"
            )
            for m in h2h[:5]:
                st.caption(
                    f"{m.get('date','?')[:10]}  {m.get('home_team','?')} "
                    f"{m.get('home_score','?')}-{m.get('away_score','?')} "
                    f"{m.get('away_team','?')}"
                )

        sources = data.get("data_sources", [])
        if sources:
            st.caption("Sources: " + "  ·  ".join(sources))


# ════════════════════════════════════════════════════════════════════════
# Main prediction renderer
# ════════════════════════════════════════════════════════════════════════

def render_prediction(pred: Dict):
    if "error" in pred:
        st.error(f"Prediction error: {pred['error']}")
        return

    home    = pred["home_team"]
    away    = pred["away_team"]
    markets = pred["markets"]
    vbets   = pred.get("value_bets", [])
    data    = pred.get("data", {})
    oddsrc  = pred.get("bookmaker_odds_source")

    # Match header (confidence, xG, lineups, weather)
    _render_match_header(pred)

    # Value bets summary strip
    _render_value_bets(vbets, oddsrc)

    st.divider()

    # Market tabs — matching Stake's navigation
    t_main, t_goals, t_asian, t_half, t_scorers = st.tabs([
        "Main", "Goals", "Asian Lines", "Half", "Goalscorers"
    ])

    with t_main:
        _tab_main(markets, home, away, vbets)

    with t_goals:
        _tab_goals(markets, home, away)

    with t_asian:
        _tab_asian(markets, home, away)

    with t_half:
        _tab_half(markets, home, away)

    with t_scorers:
        _tab_goalscorers(data, home, away)

    # Data analysis
    _render_data_analysis(pred)


# ════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#0a1524,#0d2040);
            padding:18px 28px;border-radius:12px;margin-bottom:18px;
            border:1px solid #1e3351">
    <h1 style="margin:0;color:#00e5ff;font-family:monospace;letter-spacing:3px;font-size:1.7rem">
        FIFA 2026 — MATCH PREDICTOR
    </h1>
    <p style="margin:5px 0 0;color:#6b8099;font-size:.83rem">
        Poisson · Dixon-Coles · ELO · Form · Sentiment · Sofascore · Goalscorer Model
    </p>
</div>
""", unsafe_allow_html=True)

engine = load_engine()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FIFA 2026 Predictor")
    st.caption(f"ELO database: {len(engine.elo_ratings)} teams")
    st.divider()

    mode = st.radio(
        "Prediction mode",
        ["Today's Matches", "Specific Match"],
        index=0,
    )

    if "Today" in mode:
        target_date = st.date_input("Match date", value=date.today())
        date_str    = target_date.isoformat()
    else:
        st.markdown("**Enter teams:**")
        home_input = st.text_input("Home Team", placeholder="e.g. Brazil")
        away_input = st.text_input("Away Team", placeholder="e.g. Morocco")
        date_str   = date.today().isoformat()

    st.divider()

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"5-min cache · Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    st.caption(
        "**Data sources:**\n"
        "football-data.org · Sofascore\n"
        "TheSportsDB · NewsAPI · GNews\n"
        "Open-Meteo (weather)"
    )

# ── Main ───────────────────────────────────────────────────────────────
if "Today" in mode:
    with st.spinner("Fetching today's fixtures..."):
        fixtures = fetch_fixtures(engine, date_str)

    if not fixtures:
        st.warning(
            "No FIFA 2026 matches found for this date. "
            "Try a different date or use Specific Match mode."
        )
        st.stop()

    d_label    = date.fromisoformat(date_str).strftime("%A %d %B %Y")
    auto_count = sum(1 for f in fixtures if f.get("sofascore_id"))
    st.markdown(
        f"**{len(fixtures)} match{'es' if len(fixtures) != 1 else ''} · {d_label}**"
        f"  —  {auto_count} with live odds"
    )

    tab_labels = []
    for f in fixtures:
        h = f.get("home_team", "?")
        a = f.get("away_team", "?")
        badge = "[LIVE] " if f.get("sofascore_id") else ""
        tab_labels.append(f"{badge}{h} vs {a}")

    tabs = st.tabs(tab_labels)

    for tab, fixture in zip(tabs, fixtures):
        with tab:
            h     = fixture.get("home_team", "")
            a     = fixture.get("away_team", "")
            fid   = str(fixture.get("id", ""))
            sfid  = fixture.get("sofascore_id")
            city  = fixture.get("city") or fixture.get("venue") or "Dallas"
            mdate = (fixture.get("date", "")[:10] if fixture.get("date") else date_str)
            status = (fixture.get("sofascore_status") or fixture.get("status") or "").strip()

            if status:
                s_lo = status.lower()
                if "ended" in s_lo or "finished" in s_lo:
                    st.info(f"Match finished — {status}")
                elif "progress" in s_lo or "live" in s_lo:
                    st.success(f"LIVE — {status}")
                else:
                    st.caption(f"Status: {status}")

            with st.spinner(f"Predicting {h} vs {a}..."):
                try:
                    pred = fetch_prediction(engine, h, a, fid, sfid, city, mdate)
                    render_prediction(pred)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

else:  # Specific Match
    if not home_input.strip() or not away_input.strip():
        st.info("Enter both team names in the sidebar to start.")
    else:
        h = home_input.strip()
        a = away_input.strip()
        st.markdown(f"## {h}  vs  {a}")
        with st.spinner(f"Predicting {h} vs {a}..."):
            try:
                pred = fetch_prediction(engine, h, a, "", None, "Dallas", date_str)
                render_prediction(pred)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
