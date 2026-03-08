"""Human-centric win probability query tool.

Usage examples
--------------
# Query a specific game by game_pk:
python scripts/query_game.py --game-pk 745444

# Find all 2024 games between two teams:
python scripts/query_game.py --home LAD --away SDP --season 2024

# Show the 10 biggest upsets in 2024 (highest win prob that lost):
python scripts/query_game.py --season 2024 --show-upsets

# Compare win probabilities for both teams across the season:
python scripts/query_game.py --home LAD --season 2024 --show-schedule

The output includes:
  - Win probability with a visual probability bar
  - SHAP-based factor attribution (what drove the prediction)
  - Elo ratings for each team
  - Key stats comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Team metadata
# ---------------------------------------------------------------------------

_PROCESSED_DIR = Path("data/processed")

# Retrosheet code → full team name (current era)
_RETRO_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHA": "Chicago White Sox",
    "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCA": "Kansas City Royals",
    "LAN": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYA": "New York Yankees",
    "NYN": "New York Mets",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDN": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFN": "San Francisco Giants",
    "SLN": "St. Louis Cardinals",
    "TBA": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals",
    "ANA": "Los Angeles Angels",
    "ATH": "Athletics",
    "FLO": "Florida Marlins",
    "MON": "Montreal Expos",
}

# Common abbreviation aliases (user input → Retrosheet code)
_ALIAS: dict[str, str] = {
    # FanGraphs / common → Retrosheet
    "LAD": "LAN",
    "SD": "SDN",
    "SDP": "SDN",
    "SF": "SFN",
    "SFG": "SFN",
    "STL": "SLN",
    "SLN": "SLN",
    "KC": "KCA",
    "KCR": "KCA",
    "NYY": "NYA",
    "NYM": "NYN",
    "CWS": "CHA",
    "CHW": "CHA",
    "CHC": "CHN",
    "TB": "TBA",
    "TBR": "TBA",
    "WAS": "WAS",
    "WSH": "WAS",
    "WSN": "WAS",
    "LAA": "ANA",
    "MIA": "MIA",
    "FLA": "FLO",
    "OAK": "OAK",
    "ATH": "ATH",
    "TEX": "TEX",
    "SEA": "SEA",
    "HOU": "HOU",
    "BOS": "BOS",
    "TOR": "TOR",
    "ATL": "ATL",
    "PHI": "PHI",
    "PIT": "PIT",
    "CIN": "CIN",
    "CLE": "CLE",
    "DET": "DET",
    "MIN": "MIN",
    "MIL": "MIL",
    "COL": "COL",
    "BAL": "BAL",
    "ARI": "ARI",
    # Retrosheet codes pass through unchanged
    "LAN": "LAN",
    "SDN": "SDN",
    "SFN": "SFN",
    "KCA": "KCA",
    "NYA": "NYA",
    "NYN": "NYN",
    "CHA": "CHA",
    "CHN": "CHN",
    "TBA": "TBA",
    "ANA": "ANA",
}


def _resolve_team(code: str) -> str:
    """Resolve user-supplied team abbreviation to Retrosheet code."""
    upper = code.upper()
    return _ALIAS.get(upper, upper)


def _team_name(retro: str) -> str:
    return _RETRO_NAMES.get(retro, retro)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_features(season: int | None = None) -> pd.DataFrame:
    """Load one or all seasons of feature data."""
    feat_dir = _PROCESSED_DIR / "features"
    if season:
        path = feat_dir / f"features_{season}.parquet"
        if not path.exists():
            sys.exit(f"No feature data for season {season}.  Run build_features.py first.")
        df = pd.read_parquet(path)
    else:
        frames = [pd.read_parquet(f) for f in sorted(feat_dir.glob("features_*.parquet"))]
        df = pd.concat(frames, ignore_index=True)
    if "is_spring" not in df.columns:
        df["is_spring"] = 0.0
    else:
        df["is_spring"] = df["is_spring"].fillna(0.0)
    if "game_type" not in df.columns:
        df["game_type"] = "R"
    else:
        df["game_type"] = df["game_type"].fillna("R")
    return df


def _load_model(model_dir: Path = Path("data/models"), model_type: str = "stacked"):
    """Load the production model."""
    from winprob.model.artifacts import latest_artifact, load_model

    art = latest_artifact(model_type, model_dir=model_dir, version="v3")
    if art is None:
        sys.exit(f"No '{model_type}' production model found.  Run train_model.py first.")
    return load_model(art)


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

_FEATURE_LABELS: dict[str, str] = {
    "home_elo": "Home team Elo rating",
    "away_elo": "Away team Elo rating",
    "elo_diff": "Elo advantage (home − away)",
    "home_win_pct_30": "Home team 30-game win%",
    "away_win_pct_30": "Away team 30-game win%",
    "home_pythag_30": "Home team 30-game Pythagorean",
    "away_pythag_30": "Away team 30-game Pythagorean",
    "home_win_pct_ewm": "Home team recent form (EWMA)",
    "away_win_pct_ewm": "Away team recent form (EWMA)",
    "home_pythag_ewm": "Home team recent Pythagorean (EWMA)",
    "away_pythag_ewm": "Away team recent Pythagorean (EWMA)",
    "home_win_pct_home_only": "Home team at-home win%",
    "away_win_pct_away_only": "Away team on-road win%",
    "home_sp_era": "Home starter ERA (prior season)",
    "away_sp_era": "Away starter ERA (prior season)",
    "home_sp_k9": "Home starter K/9 (prior season)",
    "away_sp_k9": "Away starter K/9 (prior season)",
    "home_bat_woba": "Home team wOBA (prior season)",
    "away_bat_woba": "Away team wOBA (prior season)",
    "home_pit_fip": "Home team FIP (prior season)",
    "away_pit_fip": "Away team FIP (prior season)",
    "pythag_diff_30": "Pythagorean edge (home − away, 30g)",
    "pythag_diff_ewm": "Recent Pythagorean edge (EWMA)",
    "home_away_split_diff": "Home/road performance edge",
    "sp_era_diff": "Pitcher ERA edge (lower = home advantage)",
    "woba_diff": "Batting quality edge (wOBA)",
    "fip_diff": "Pitching quality edge (FIP)",
    "park_run_factor": "Park run factor (1.0 = neutral)",
    "season_progress": "Point in season (0=opening, 1=end)",
    "home_streak": "Home team win streak (+) / loss streak (−)",
    "away_streak": "Away team win streak (+) / loss streak (−)",
    "home_rest_days": "Home team rest days",
    "away_rest_days": "Away team rest days",
}


def _compute_shap(model: object, X_row: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Compute SHAP-style feature attributions for a single game row.

    For logistic regression: uses coefficient × z-score (closed-form).
    For tree models: uses SHAP TreeExplainer.
    """
    base = getattr(model, "base", model)
    x = X_row.values[0].astype(float)

    try:
        if hasattr(base, "booster_"):  # LightGBM
            import shap

            explainer = shap.TreeExplainer(base)
            vals = explainer.shap_values(X_row)
            shap_arr = vals[1] if isinstance(vals, list) else vals
            return pd.Series(shap_arr[0], index=feature_cols)

        elif hasattr(base, "get_booster"):  # XGBoost
            import shap

            explainer = shap.TreeExplainer(base)
            vals = explainer.shap_values(X_row)
            return pd.Series(vals[0], index=feature_cols)

        elif hasattr(base, "named_steps"):  # sklearn Pipeline (logistic)
            # Closed-form: coef_i × (x_i - mean_i) / scale_i
            # This is the log-odds contribution of each feature relative to the
            # training mean.  Positive = pushes toward home win.
            scaler = base.named_steps["scaler"]
            lr = base.named_steps["lr"]
            coef = lr.coef_[0]  # shape: (n_features,)
            z = (x - scaler.mean_) / scaler.scale_  # standardized deviation from mean
            shap_arr = coef * z
            return pd.Series(shap_arr, index=feature_cols)

    except Exception:
        pass
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _prob_bar(p: float, width: int = 40) -> str:
    """ASCII probability bar."""
    filled = round(p * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {p * 100:.1f}%"


def _format_game(
    row: pd.Series, shap_vals: pd.Series | None = None, *, verbose: bool = True
) -> str:
    home = _team_name(str(row.get("home_retro", "")))
    away = _team_name(str(row.get("away_retro", "")))
    prob = float(row.get("prob", 0.5))
    home_win = row.get("home_win")
    date = str(row.get("date", ""))[:10]

    lines = [
        "",
        f"  {'─' * 60}",
        f"  {'MLB Win Probability':^60}",
        f"  {'─' * 60}",
        f"  Date   : {date}",
        f"  Matchup: {away} @ {home}",
    ]

    if pd.notna(home_win):
        winner = home if home_win == 1.0 else away
        result = "HOME WIN ✓" if home_win == 1.0 else "AWAY WIN ✗"
        lines.append(f"  Result : {result} ({winner})")

    lines += [
        "",
        f"  {home:<30}  {_prob_bar(prob)}",
        f"  {away:<30}  {_prob_bar(1 - prob)}",
    ]

    if verbose:
        # Key stats
        lines += [
            "",
            f"  {'KEY STATS':^60}",
            f"  {'─' * 60}",
            f"  {'Metric':<28} {'Home':>12}  {'Away':>12}",
            f"  {'Elo rating':<28} {row.get('home_elo', 1500):>12.0f}  {row.get('away_elo', 1500):>12.0f}",
            f"  {'30-game win%':<28} {row.get('home_win_pct_30', 0.5):>11.1%}  {row.get('away_win_pct_30', 0.5):>11.1%}",
            f"  {'30-game Pythagorean':<28} {row.get('home_pythag_30', 0.5):>11.1%}  {row.get('away_pythag_30', 0.5):>11.1%}",
            f"  {'At-home / On-road win%':<28} {row.get('home_win_pct_home_only', 0.5):>11.1%}  {row.get('away_win_pct_away_only', 0.5):>11.1%}",
            f"  {'SP ERA (prior season)':<28} {row.get('home_sp_era', 4.5):>12.2f}  {row.get('away_sp_era', 4.5):>12.2f}",
            f"  {'SP K/9 (prior season)':<28} {row.get('home_sp_k9', 8.5):>12.2f}  {row.get('away_sp_k9', 8.5):>12.2f}",
            f"  {'Team wOBA (prior season)':<28} {row.get('home_bat_woba', 0.32):.3f}        {row.get('away_bat_woba', 0.32):.3f}",
            f"  {'Team FIP (prior season)':<28} {row.get('home_pit_fip', 4.2):>12.2f}  {row.get('away_pit_fip', 4.2):>12.2f}",
            f"  {'Rest days':<28} {row.get('home_rest_days', 2):>12.0f}  {row.get('away_rest_days', 2):>12.0f}",
            f"  {'Streak':<28} {row.get('home_streak', 0):>+12.0f}  {row.get('away_streak', 0):>+12.0f}",
            f"  {'Park run factor':<28} {row.get('park_run_factor', 1.0):>24.3f}",
        ]

        # SHAP attribution
        if shap_vals is not None and len(shap_vals) > 0:
            top = shap_vals.abs().nlargest(8)
            lines += [
                "",
                f"  {'TOP FACTORS (SHAP attribution)':^60}",
                f"  {'─' * 60}",
                f"  {'Factor':<40} {'Effect':>10}",
            ]
            for feat, _ in top.items():
                val = shap_vals[feat]
                label = _FEATURE_LABELS.get(str(feat), str(feat))
                direction = "↑ favors home" if val > 0 else "↓ favors away"
                lines.append(f"  {label:<40} {val:>+.4f}  {direction}")

    lines.append(f"  {'─' * 60}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predict on a feature row
# ---------------------------------------------------------------------------


def _predict_row(model: object, row: pd.Series, feature_cols: list[str]) -> float:
    from winprob.model.train import _predict_proba

    X = row[feature_cols].values.astype(float).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=feature_cols)
    return float(_predict_proba(model, X_df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query MLB win probability predictions in human-readable form.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--game-pk", type=int, help="MLB game_pk to query")
    ap.add_argument("--home", type=str, help="Home team abbreviation (e.g. LAD, NYY)")
    ap.add_argument("--away", type=str, help="Away team abbreviation (e.g. SDP, BOS)")
    ap.add_argument("--season", type=int, help="Season year (e.g. 2024)")
    ap.add_argument("--date", type=str, help="Game date YYYY-MM-DD")
    ap.add_argument("--show-upsets", action="store_true", help="Show biggest upsets in the season")
    ap.add_argument("--show-schedule", action="store_true", help="Show all queried team games")
    ap.add_argument("--top-n", type=int, default=10, help="Number of results for --show-upsets")
    ap.add_argument(
        "--model-type", default="stacked", choices=["logistic", "lightgbm", "xgboost", "stacked"]
    )
    ap.add_argument("--model-dir", type=Path, default=Path("data/models"))
    ap.add_argument("--no-shap", action="store_true", help="Skip SHAP computation")
    ap.add_argument("--brief", action="store_true", help="Compact one-line output per game")
    args = ap.parse_args()

    # Load data
    df = _load_features(args.season)
    model, meta = _load_model(args.model_dir, args.model_type)
    feature_cols = meta.feature_cols

    # Compute predictions for all rows
    X = df[feature_cols].astype(float)
    from winprob.model.train import _predict_proba

    probs = _predict_proba(model, X)
    df = df.copy()
    df["prob"] = probs

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered = df.copy()

    if args.game_pk:
        filtered = filtered[filtered["game_pk"] == args.game_pk]

    if args.home:
        retro = _resolve_team(args.home)
        filtered = filtered[filtered["home_retro"] == retro]

    if args.away:
        retro = _resolve_team(args.away)
        filtered = filtered[filtered["away_retro"] == retro]

    if args.date:
        filtered = filtered[filtered["date"].astype(str) == args.date]

    if filtered.empty:
        print("No games matched your query.")
        return

    # ── Show upsets ──────────────────────────────────────────────────────────
    if args.show_upsets:
        # Biggest upsets = games where the heavy favourite lost
        has_result = filtered[filtered["home_win"].notna()].copy()
        has_result["fav_home"] = has_result["prob"] >= 0.5
        has_result["upset"] = (has_result["fav_home"] & (has_result["home_win"] == 0)) | (
            ~has_result["fav_home"] & (has_result["home_win"] == 1)
        )
        upsets = has_result[has_result["upset"]].copy()
        upsets["prob_edge"] = (upsets["prob"] - 0.5).abs()
        upsets = upsets.nlargest(args.top_n, "prob_edge")

        print(f"\n  Top {args.top_n} biggest upsets")
        print(f"  {'Date':<12} {'Away':>22} {'@':>2} {'Home':<22} {'Fav prob':>9} {'Winner':<6}")
        print(f"  {'─' * 80}")
        for _, r in upsets.iterrows():
            home_n = _team_name(str(r["home_retro"]))[:20]
            away_n = _team_name(str(r["away_retro"]))[:20]
            fav_prob = max(r["prob"], 1 - r["prob"])
            winner = "HOME" if r["home_win"] == 1 else "AWAY"
            fav = "home" if r["prob"] >= 0.5 else "away"
            print(
                f"  {str(r['date'])[:10]:<12} {away_n:>22}  @ {home_n:<22} {fav_prob:>8.1%} {winner} (fav: {fav})"
            )
        return

    # ── Show schedule ────────────────────────────────────────────────────────
    if args.show_schedule:
        print(f"\n  {'Date':<12} {'Away':>22} {'@':>2} {'Home':<22} {'P(home)':>8} {'Result':<10}")
        print(f"  {'─' * 80}")
        for _, r in filtered.sort_values("date").iterrows():
            home_n = _team_name(str(r["home_retro"]))[:20]
            away_n = _team_name(str(r["away_retro"]))[:20]
            result = ""
            if pd.notna(r.get("home_win")):
                result = "HOME WIN" if r["home_win"] == 1 else "AWAY WIN"
            print(
                f"  {str(r['date'])[:10]:<12} {away_n:>22}  @ {home_n:<22} {r['prob']:>7.1%}  {result}"
            )
        return

    # ── Detailed per-game output ──────────────────────────────────────────────
    for _, row in filtered.sort_values("date").iterrows():
        if args.brief:
            home_n = _team_name(str(row["home_retro"]))
            away_n = _team_name(str(row["away_retro"]))
            result = ""
            if pd.notna(row.get("home_win")):
                result = "✓ HOME WIN" if row["home_win"] == 1 else "✗ AWAY WIN"
            print(
                f"{str(row['date'])[:10]}  {away_n} @ {home_n}  P(home)={row['prob']:.1%}  {result}"
            )
        else:
            shap_vals = None
            if not args.no_shap:
                X_row = pd.DataFrame(
                    row[feature_cols].values.astype(float).reshape(1, -1),
                    columns=feature_cols,
                )
                shap_vals = _compute_shap(model, X_row, feature_cols)
            print(_format_game(row, shap_vals))


if __name__ == "__main__":
    main()
