"""Tool functions and flat JSON schemas for MCP and other consumers.

Twelve tools: 10 that wrap existing project code, plus find_ev_bets and
get_live_odds as stubs until odds/EV modules exist. Flat schemas (no nested
objects, no $defs) for compatibility with tool-calling clients.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from mlb_predict.app.data_cache import (
    TEAM_NAMES,
    available_model_types,
    get_active_model_type,
    get_features,
    get_model,
    is_ready,
)
from mlb_predict.standings import (
    DIVISION_DISPLAY_ORDER,
    DIVISIONS,
    compute_league_leaders,
    compute_predicted_standings,
)
from mlb_predict.tools.knowledge import (
    get_feature_description,
    get_glossary_term,
    get_model_docs,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Flat JSON schemas for each tool (required fields only where needed)
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_predictions",
            "description": "Find game predictions by team, season, or date",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Retrosheet team code e.g. LAD, NYA, BOS",
                    },
                    "season": {"type": "integer", "description": "Year e.g. 2026"},
                    "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_prediction",
            "description": "Get SHAP breakdown and key stats for a single game",
            "parameters": {
                "type": "object",
                "properties": {
                    "game_pk": {"type": "integer", "description": "MLB game ID"},
                },
                "required": ["game_pk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_models",
            "description": "Get cross-validation summary for all models",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_stats",
            "description": "Get batting and pitching stats for all teams in a season",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "Year e.g. 2026"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_standings",
            "description": "Get predicted and actual division standings",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "Year e.g. 2026"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_upsets",
            "description": "Find biggest upsets (favorites that lost)",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "Year e.g. 2026"},
                    "min_prob": {"type": "number", "description": "Min favorite prob e.g. 0.65"},
                    "limit": {"type": "integer", "description": "Max number to return e.g. 20"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_drift_metrics",
            "description": "Get prediction drift summary vs previous snapshot",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Get short description of a model type",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "e.g. stacked, lightgbm, logistic",
                    },
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_feature",
            "description": "Get description of a feature or sabermetric term",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Feature name e.g. home_elo or term e.g. woba",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_season_summary",
            "description": "List available seasons and active model",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_ev_bets",
            "description": "Find today's positive expected value bets vs model",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_odds",
            "description": "Get live odds for today's games",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _safe_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float, str)):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def _tool_query_predictions(team: str | None, season: int | None, date: str | None) -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet."})
    df = get_features()
    if team:
        team_upper = (team or "").strip().upper()
        mask = (df["home_retro"] == team_upper) | (df["away_retro"] == team_upper)
        df = df[mask]
    if season is not None:
        df = df[df["season"] == int(season)]
    if date:
        df = df[df["date"].astype(str).str.startswith(str(date)[:10])]
    df = df.head(50)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "game_pk": int(r.get("game_pk", 0)),
                "date": str(r.get("date", ""))[:10],
                "home": str(r.get("home_retro", "")),
                "away": str(r.get("away_retro", "")),
                "prob_home": round(float(r["prob"]), 4) if pd.notna(r.get("prob")) else None,
            }
        )
    return json.dumps({"total": len(rows), "games": rows})


def _tool_explain_prediction(game_pk: int) -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet."})
    df = get_features()
    matches = df[df["game_pk"] == game_pk]
    if matches.empty:
        return json.dumps({"error": f"Game {game_pk} not found."})
    row = matches.iloc[0]
    model, meta, feature_cols = get_model()
    shap_vals: dict[str, float] = {}
    try:
        base = getattr(model, "base", model)
        x = row[feature_cols].values.astype(float)
        if hasattr(base, "named_steps"):
            scaler = base.named_steps["scaler"]
            lr = base.named_steps["lr"]
            z = (x - scaler.mean_) / scaler.scale_
            shap_arr = lr.coef_[0] * z
            shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, shap_arr)}
        elif hasattr(base, "booster_") or hasattr(base, "get_booster"):
            import shap

            X_df = pd.DataFrame([x], columns=feature_cols)
            explainer = shap.TreeExplainer(base)
            sv = explainer.shap_values(X_df)
            arr = sv[1][0] if isinstance(sv, list) else sv[0]
            shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, arr)}
    except Exception as e:
        logger.warning("SHAP failed for game_pk=%s: %s", game_pk, e)
    top = sorted(
        [
            {"feature": k, "value": v, "label": get_feature_description(k)}
            for k, v in shap_vals.items()
        ],
        key=lambda x: abs(cast(float, x["value"])),
        reverse=True,
    )[:10]
    out = {
        "game_pk": game_pk,
        "date": str(row.get("date", ""))[:10],
        "home": str(row.get("home_retro", "")),
        "away": str(row.get("away_retro", "")),
        "prob_home": _safe_float(row.get("prob")),
        "top_factors": top,
    }
    return json.dumps(out)


def _tool_compare_models() -> str:
    paths = [
        _REPO_ROOT / "data" / "models" / "cv_summary_v3.json",
        _REPO_ROOT / "data" / "models" / "cv_summary_v2.json",
        _REPO_ROOT / "data" / "models" / "cv_summary.json",
    ]
    for p in paths:
        if p.exists():
            data = json.loads(p.read_text())
            return json.dumps(data if isinstance(data, list) else [data])
    return json.dumps([])


def _tool_get_team_stats(season: int | None) -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet."})
    return json.dumps(
        {
            "message": "Full team batting/pitching stats are available on the dashboard at /api/team-stats. Pass season (e.g. 2026) as query. This tool does not call the MLB API.",
            "season": season or 2026,
        }
    )


def _tool_get_standings(season: int | None) -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet."})
    season = season or 2026
    df = get_features()
    pred_df = compute_predicted_standings(df, season=season)
    if pred_df.empty:
        return json.dumps({"season": season, "divisions": [], "message": "No predicted standings."})
    league_leaders = compute_league_leaders(pred_df)
    divisions = []
    for div_id in DIVISION_DISPLAY_ORDER:
        div_info = DIVISIONS.get(div_id, {})
        div_df = pred_df[pred_df["division_id"] == div_id].sort_values(
            "pred_win_pct", ascending=False
        )
        if div_df.empty:
            continue
        teams = []
        for _, row in div_df.iterrows():
            code = row.get("retro_code", "")
            teams.append(
                {
                    "retro_code": code,
                    "team_name": TEAM_NAMES.get(code, ""),
                    "pred_w": int(row.get("pred_wins", 0)),
                    "pred_l": int(row.get("pred_losses", 0)),
                    "pred_win_pct": round(float(row.get("pred_win_pct", 0)), 3),
                }
            )
        divisions.append({"division": div_info.get("name", div_id), "teams": teams})
    return json.dumps({"season": season, "divisions": divisions, "league_leaders": league_leaders})


def _tool_find_upsets(
    season: int | None,
    min_prob: float | None,
    limit: int | None,
) -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet."})
    df = get_features()
    if season is not None:
        df = df[df["season"] == int(season)]
    played = df[df["home_win"].notna()].copy()
    if played.empty:
        return json.dumps({"upsets": [], "message": "No completed games."})
    min_p = min_prob if min_prob is not None else 0.65
    fav_home = played["prob"] >= 0.5
    fav_lost = (fav_home & (played["home_win"] == 0)) | (~fav_home & (played["home_win"] == 1))
    played["fav_prob"] = played["prob"].where(played["prob"] >= 0.5, 1 - played["prob"])
    upsets = played[fav_lost & (played["fav_prob"] >= min_p)].nlargest(limit or 20, "fav_prob")
    rows = []
    for _, r in upsets.iterrows():
        rows.append(
            {
                "game_pk": int(r.get("game_pk", 0)),
                "date": str(r.get("date", ""))[:10],
                "home": str(r.get("home_retro", "")),
                "away": str(r.get("away_retro", "")),
                "fav_prob": round(float(r.get("fav_prob", 0.5)), 4),
                "winner": "home" if r.get("home_win") == 1 else "away",
            }
        )
    return json.dumps({"upsets": rows})


def _tool_get_drift_metrics() -> str:
    drift_dir = _REPO_ROOT / "data" / "processed" / "drift"
    global_path = drift_dir / "global_run_metrics.parquet"
    if not global_path.exists():
        return json.dumps({"message": "No drift metrics yet. Run predictions to compute drift."})
    try:
        df = pd.read_parquet(global_path)
        if df.empty:
            return json.dumps({"message": "No drift metrics rows."})
        row = df.iloc[-1]
        return json.dumps(
            {
                "run_ts_utc": str(row.get("run_ts_utc", "")),
                "season": int(row.get("season", 0)),
                "mean_abs_delta": round(float(row.get("mean_abs_delta", 0)), 5),
                "p95_abs_delta": round(float(row.get("p95_abs_delta", 0)), 5),
                "pct_gt_0p01": round(float(row.get("pct_gt_0p01", 0)), 4),
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_get_model_info(model_type: str) -> str:
    doc = get_model_docs(model_type)
    return json.dumps({"model_type": (model_type or "").strip().lower(), "description": doc})


def _tool_describe_feature(name: str) -> str:
    key = (name or "").strip()
    desc = get_feature_description(key)
    glossary = get_glossary_term(key)
    out = {"feature_or_term": key, "description": desc}
    if glossary:
        out["glossary"] = glossary
    return json.dumps(out)


def _tool_get_season_summary() -> str:
    if not is_ready():
        return json.dumps({"error": "Data not loaded yet.", "seasons": [], "active_model": None})
    try:
        df = get_features()
        seasons = sorted(df["season"].dropna().unique().astype(int).tolist(), reverse=True)
    except Exception:
        seasons = []
    active = get_active_model_type() if is_ready() else None
    available = list(available_model_types()) if is_ready() else []
    return json.dumps({"seasons": seasons, "active_model": active, "available_models": available})


def _tool_find_ev_bets() -> str:
    return json.dumps(
        {
            "message": "Odds and EV features are not available yet. They will be added in a future update."
        }
    )


def _tool_get_live_odds() -> str:
    """Return live MLB game odds when The Odds API key is configured; else not-available message."""
    from mlb_predict.external.odds import OddsClient

    client = OddsClient()
    if not client.is_available():
        return json.dumps(
            {
                "message": "Live odds are not configured. Set ODDS_API_KEY or add the key in Dashboard → Live Odds API Key."
            }
        )
    events = client.get_game_odds_sync()
    client.events_to_retro(events)
    if not events:
        return json.dumps({"message": "No upcoming games with odds right now.", "events": []})
    out = []
    for ev in events[:20]:
        home = ev.get("home_team") or ""
        away = ev.get("away_team") or ""
        commence = ev.get("commence_time") or ""
        books = ev.get("bookmakers") or []
        h2h = []
        for b in books:
            for m in b.get("markets") or []:
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes") or []:
                    h2h.append(
                        {"book": b.get("key"), "team": o.get("name"), "price": o.get("price")}
                    )
        out.append(
            {"home_team": home, "away_team": away, "commence_time": commence, "moneyline": h2h[:4]}
        )
    return json.dumps({"events": out, "count": len(events)})


def run_tool(name: str, params: dict) -> str:
    """Execute a tool by name with the given params; returns JSON string."""
    p = params or {}
    if name == "query_predictions":
        return _tool_query_predictions(
            p.get("team"),
            p.get("season") if "season" in p else None,
            p.get("date"),
        )
    if name == "explain_prediction":
        return _tool_explain_prediction(int(p.get("game_pk", 0)))
    if name == "compare_models":
        return _tool_compare_models()
    if name == "get_team_stats":
        return _tool_get_team_stats(p.get("season"))
    if name == "get_standings":
        return _tool_get_standings(p.get("season"))
    if name == "find_upsets":
        return _tool_find_upsets(
            p.get("season"),
            p.get("min_prob"),
            p.get("limit"),
        )
    if name == "get_drift_metrics":
        return _tool_get_drift_metrics()
    if name == "get_model_info":
        return _tool_get_model_info(str(p.get("model_type", "")))
    if name == "describe_feature":
        return _tool_describe_feature(str(p.get("name", "")))
    if name == "get_season_summary":
        return _tool_get_season_summary()
    if name == "find_ev_bets":
        return _tool_find_ev_bets()
    if name == "get_live_odds":
        return _tool_get_live_odds()
    return json.dumps({"error": f"Unknown tool: {name}"})
