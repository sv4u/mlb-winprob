"""FastAPI application — MLB Win Probability dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from winprob.app.admin import (
    PipelineKind,
    conflicting_pipeline,
    gather_data_status,
    gather_model_status,
    get_state,
    run_pipeline,
)
from winprob.app.data_cache import (
    TEAM_NAMES,
    available_model_types,
    get_active_model_type,
    get_features,
    get_git_commit,
    get_model,
    startup,
    switch_model,
)
from winprob.app.timing import TimingMiddleware, timed_operation
from winprob.standings import (
    DIVISION_DISPLAY_ORDER,
    DIVISIONS,
    compute_league_leaders,
    compute_predicted_standings,
    merge_predicted_actual,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MLB Win Probability", version="3.0")
app.add_middleware(TimingMiddleware)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that logs the full traceback and returns a safe JSON error."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    logger.debug("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return JSONResponse(
        {"error": "Internal server error", "detail": str(exc)},
        status_code=500,
    )


_BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_BASE / "templates"))

_static = _BASE / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


_DEFAULT_MODEL_TYPE = "stacked"


def _reload_app() -> None:
    """Re-run startup() to pick up new data/models after a pipeline completes."""
    model_type = os.environ.get("WINPROB_MODEL_TYPE", _DEFAULT_MODEL_TYPE)
    startup(model_type)


@app.on_event("startup")
async def _startup() -> None:
    model_type = os.environ.get("WINPROB_MODEL_TYPE", _DEFAULT_MODEL_TYPE)
    logger.info("Application startup: model_type=%s", model_type)
    startup(model_type)
    logger.info("Startup complete — serving on model '%s'", model_type)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/version")
def api_version() -> dict:
    """Return the current application version and git commit hash."""
    return {
        "version": "3.0",
        "git_commit": get_git_commit(),
    }


@app.get("/api/seasons", response_model=None)
def api_seasons() -> list[int] | JSONResponse:
    """List all available seasons."""
    try:
        df = get_features()
        return sorted(df["season"].dropna().unique().astype(int).tolist())
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.get("/api/teams")
def api_teams() -> list[dict]:
    """List all known teams with their Retrosheet codes and names."""
    df = get_features()
    teams = set(df["home_retro"].dropna().tolist()) | set(df["away_retro"].dropna().tolist())
    return sorted(
        [{"code": t, "name": TEAM_NAMES.get(t, t)} for t in teams],
        key=lambda x: x["name"],
    )


@app.get("/api/games")
def api_games(
    season: Annotated[int | None, Query()] = None,
    home: Annotated[str | None, Query()] = None,
    away: Annotated[str | None, Query()] = None,
    date: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    sort: Annotated[str, Query()] = "date",
    order: Annotated[str, Query()] = "desc",
) -> dict:
    """Query games with optional filters."""
    logger.debug(
        "GET /api/games season=%s home=%s away=%s date=%s limit=%d offset=%d",
        season,
        home,
        away,
        date,
        limit,
        offset,
    )
    with timed_operation("games_query"):
        df = get_features()
        if season:
            df = df[df["season"] == season]
        if home:
            df = df[df["home_retro"] == home.upper()]
        if away:
            df = df[df["away_retro"] == away.upper()]
        if date:
            df = df[df["date"].astype(str) == date]

        valid_sorts = {"date", "prob", "season", "game_pk"}
        sort_col = sort if sort in valid_sorts else "date"
        ascending = order.lower() != "desc"
        df = df.sort_values(sort_col, ascending=ascending)

        total = len(df)
        page = df.iloc[offset : offset + limit]

        rows = []
        for _, r in page.iterrows():
            rows.append(
                {
                    "game_pk": int(r.get("game_pk", 0) or 0),
                    "date": str(r.get("date", ""))[:10],
                    "season": int(r.get("season", 0) or 0),
                    "home_retro": str(r.get("home_retro", "")),
                    "home_name": TEAM_NAMES.get(str(r.get("home_retro", "")), ""),
                    "away_retro": str(r.get("away_retro", "")),
                    "away_name": TEAM_NAMES.get(str(r.get("away_retro", "")), ""),
                    "prob_home": round(float(r["prob"]), 4) if pd.notna(r.get("prob")) else None,
                    "home_win": (int(r["home_win"]) if pd.notna(r.get("home_win")) else None),
                    "home_elo": round(float(r["home_elo"]), 1)
                    if pd.notna(r.get("home_elo"))
                    else None,
                    "away_elo": round(float(r["away_elo"]), 1)
                    if pd.notna(r.get("away_elo"))
                    else None,
                }
            )

    return {"total": total, "offset": offset, "limit": limit, "games": rows}


@app.get("/api/games/{game_pk}", response_model=None)
def api_game_detail(game_pk: int) -> dict | JSONResponse:
    """Full feature breakdown + SHAP attribution for a single game."""
    logger.debug("GET /api/games/%d", game_pk)
    df = get_features()
    matches = df[df["game_pk"] == game_pk]
    if matches.empty:
        return JSONResponse({"error": "game_pk not found"}, status_code=404)

    row = matches.iloc[0]
    model, meta, feature_cols = get_model()

    # SHAP attribution
    shap_vals: dict[str, float] = {}
    with timed_operation("shap_attribution"):
        try:
            base = getattr(model, "base", model)
            x = row[feature_cols].values.astype(float)
            if hasattr(base, "named_steps"):  # logistic
                scaler = base.named_steps["scaler"]
                lr = base.named_steps["lr"]
                coef = lr.coef_[0]
                z = (x - scaler.mean_) / scaler.scale_
                shap_arr = coef * z
                shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, shap_arr)}
            elif hasattr(base, "booster_") or hasattr(base, "get_booster"):
                import shap

                X_df = pd.DataFrame([x], columns=feature_cols)
                explainer = shap.TreeExplainer(base)
                sv = explainer.shap_values(X_df)
                arr = sv[1][0] if isinstance(sv, list) else sv[0]
                shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, arr)}
        except Exception as exc:
            logger.warning("SHAP attribution failed for game_pk=%d: %s", game_pk, exc)

    # Top SHAP factors (sorted by absolute value)
    top_factors = sorted(
        [{"feature": k, "value": v} for k, v in shap_vals.items()],
        key=lambda x: abs(x["value"]),  # type: ignore[arg-type]
        reverse=True,
    )[:12]

    stats = {
        k: (round(float(v), 4) if pd.notna(v) else None)
        for k, v in row.items()
        if k in feature_cols
    }

    return {
        "game_pk": game_pk,
        "date": str(row.get("date", ""))[:10],
        "season": int(row.get("season", 0) or 0),
        "home_retro": str(row.get("home_retro", "")),
        "home_name": TEAM_NAMES.get(str(row.get("home_retro", "")), ""),
        "away_retro": str(row.get("away_retro", "")),
        "away_name": TEAM_NAMES.get(str(row.get("away_retro", "")), ""),
        "prob_home": round(float(row["prob"]), 4) if pd.notna(row.get("prob")) else None,
        "home_win": int(row["home_win"]) if pd.notna(row.get("home_win")) else None,
        "stats": stats,
        "top_factors": top_factors,
    }


@app.get("/api/upsets")
def api_upsets(
    season: Annotated[int | None, Query()] = None,
    home: Annotated[str | None, Query()] = None,
    away: Annotated[str | None, Query()] = None,
    min_prob: Annotated[float, Query(ge=0.5, le=1.0)] = 0.65,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[dict]:
    """Return the biggest upsets (heavy favorites that lost)."""
    with timed_operation("upsets_query"):
        df = get_features()
        if season:
            df = df[df["season"] == season]
        if home:
            df = df[df["home_retro"] == home.upper()]
        if away:
            df = df[df["away_retro"] == away.upper()]
        has_result = df[df["home_win"].notna() & df["prob"].notna()].copy()
        has_result["fav_home"] = has_result["prob"] >= 0.5
        has_result["fav_prob"] = has_result["prob"].clip(lower=0.5)
        has_result.loc[~has_result["fav_home"], "fav_prob"] = (
            1 - has_result.loc[~has_result["fav_home"], "prob"]
        )
        has_result = has_result[has_result["fav_prob"] >= min_prob]
        has_result["upset"] = (has_result["fav_home"] & (has_result["home_win"] == 0)) | (
            ~has_result["fav_home"] & (has_result["home_win"] == 1)
        )
        upsets = has_result[has_result["upset"]].nlargest(limit, "fav_prob")
        result = []
        for _, r in upsets.iterrows():
            result.append(
                {
                    "game_pk": int(r.get("game_pk", 0) or 0),
                    "date": str(r.get("date", ""))[:10],
                    "season": int(r.get("season", 0) or 0),
                    "home_name": TEAM_NAMES.get(str(r.get("home_retro", "")), ""),
                    "away_name": TEAM_NAMES.get(str(r.get("away_retro", "")), ""),
                    "prob_home": round(float(r["prob"]), 4),
                    "fav_prob": round(float(r["fav_prob"]), 4),
                    "fav_team": "home" if r["fav_home"] else "away",
                    "winner": "home" if r["home_win"] == 1 else "away",
                }
            )
    return result


@app.get("/api/cv-summary")
def api_cv_summary() -> list[dict]:
    """Return cross-validation results from the latest training run."""
    import json

    paths = [
        Path("data/models/cv_summary_v3.json"),
        Path("data/models/cv_summary_v2.json"),
        Path("data/models/cv_summary.json"),
    ]
    for p in paths:
        if p.exists():
            return json.loads(p.read_text())
    return []


@app.get("/api/standings")
async def api_standings(
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
) -> dict:
    """Return predicted vs actual standings grouped by division.

    Predicted standings are computed from the per-game win probabilities
    in the features cache.  Actual standings are fetched live from the
    MLB Stats API when available.
    """
    logger.debug("GET /api/standings season=%d", season)
    from winprob.mlbapi.client import MLBAPIClient
    from winprob.mlbapi.standings import fetch_standings

    with timed_operation("predicted_standings"):
        df = get_features()
        pred_df = compute_predicted_standings(df, season=season)

    actual_df = pd.DataFrame()
    try:
        async with timed_operation("mlb_api_standings"):
            async with MLBAPIClient() as client:
                actual_df = await fetch_standings(client, season=season)
    except Exception as exc:
        logger.warning("Could not fetch live standings for season=%d: %s", season, exc)

    # Suppress actual data when season hasn't started (all teams 0-0)
    season_started = not actual_df.empty and actual_df["wins"].sum() + actual_df["losses"].sum() > 0

    if not pred_df.empty and season_started:
        standings = merge_predicted_actual(pred_df, actual_df)
    elif not pred_df.empty:
        standings = pred_df
    else:
        return {"season": season, "divisions": [], "league_leaders": {}}

    league_leaders = compute_league_leaders(standings)

    divisions: list[dict] = []
    for div_id in DIVISION_DISPLAY_ORDER:
        div_info = DIVISIONS.get(div_id, {})
        div_df = standings[standings["division_id"] == div_id].sort_values(
            "pred_win_pct",
            ascending=False,
        )
        if div_df.empty:
            continue

        teams: list[dict] = []
        for _, row in div_df.iterrows():
            team_entry: dict = {
                "retro_code": row.get("retro_code", ""),
                "mlb_id": int(row.get("mlb_id", 0)),
                "team_name": TEAM_NAMES.get(row.get("retro_code", ""), row.get("team_name", "")),
                "pred_wins": float(row.get("pred_wins", 0)),
                "pred_losses": float(row.get("pred_losses", 0)),
                "pred_win_pct": float(row.get("pred_win_pct", 0)),
                "pred_division_rank": int(row.get("pred_division_rank", 0)),
                "pred_gb": str(row.get("pred_gb_str", "-")),
                "pred_total_games": int(row.get("pred_total_games", 0)),
            }
            if "actual_wins" in row and pd.notna(row.get("actual_wins")):
                team_entry.update(
                    {
                        "actual_wins": int(row["actual_wins"]),
                        "actual_losses": int(row["actual_losses"]),
                        "actual_win_pct": round(float(row["actual_win_pct"]), 3),
                        "actual_gb": str(row.get("actual_gb", "-")),
                        "actual_division_rank": int(row.get("actual_division_rank", 0)),
                        "runs_scored": int(row.get("runs_scored", 0)),
                        "runs_allowed": int(row.get("runs_allowed", 0)),
                        "run_diff": int(row.get("run_diff", 0)),
                        "wins_delta": round(float(row.get("wins_delta", 0)), 1),
                        "pct_delta": round(float(row.get("pct_delta", 0)), 3),
                        "rank_delta": int(row.get("rank_delta", 0)),
                    }
                )
            teams.append(team_entry)

        divisions.append(
            {
                "division_id": div_id,
                "division_name": div_info.get("name", ""),
                "league": div_info.get("league", ""),
                "teams": teams,
            }
        )

    return {
        "season": season,
        "divisions": divisions,
        "league_leaders": league_leaders,
    }


@app.get("/api/team-stats")
async def api_team_stats(
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
) -> dict:
    """Return batting and pitching stats for all teams in a season."""
    from winprob.mlbapi.client import MLBAPIClient
    from winprob.mlbapi.standings import fetch_all_team_stats, fetch_standings

    try:
        async with timed_operation("mlb_api_team_stats"):
            async with MLBAPIClient() as client:
                standings = await fetch_standings(client, season=season)
                if standings.empty:
                    return {"season": season, "teams": []}
                total_games = standings["wins"].sum() + standings["losses"].sum()
                if total_games == 0:
                    return {"season": season, "teams": [], "message": "Season has not started yet."}
                full = await fetch_all_team_stats(
                    client,
                    standings_df=standings,
                    season=season,
                )
    except Exception as exc:
        logger.warning("Could not fetch team stats for season=%d: %s", season, exc)
        return {"season": season, "teams": [], "error": str(exc)}

    teams: list[dict] = []
    for _, row in full.iterrows():
        teams.append(
            {
                "team_id": int(row.get("team_id", 0)),
                "team_name": row.get("team_name", ""),
                "division_name": row.get("division_name", ""),
                "league_name": row.get("league_name", ""),
                "record": f"{row.get('wins', 0)}-{row.get('losses', 0)}",
                "pct": round(float(row.get("pct", 0)), 3),
                "run_diff": int(row.get("run_diff", 0)),
                "batting": {
                    "avg": round(float(row.get("bat_avg", 0)), 3),
                    "obp": round(float(row.get("bat_obp", 0)), 3),
                    "slg": round(float(row.get("bat_slg", 0)), 3),
                    "ops": round(float(row.get("bat_ops", 0)), 3),
                    "runs": int(row.get("bat_runs", 0)),
                    "hits": int(row.get("bat_hits", 0)),
                    "doubles": int(row.get("bat_doubles", 0)),
                    "triples": int(row.get("bat_triples", 0)),
                    "hr": int(row.get("bat_hr", 0)),
                    "rbi": int(row.get("bat_rbi", 0)),
                    "sb": int(row.get("bat_sb", 0)),
                    "bb": int(row.get("bat_bb", 0)),
                    "so": int(row.get("bat_so", 0)),
                },
                "pitching": {
                    "era": round(float(row.get("pit_era", 0)), 2),
                    "wins": int(row.get("pit_wins", 0)),
                    "losses": int(row.get("pit_losses", 0)),
                    "saves": int(row.get("pit_saves", 0)),
                    "ip": str(row.get("pit_ip", "")),
                    "hits": int(row.get("pit_hits", 0)),
                    "bb": int(row.get("pit_bb", 0)),
                    "so": int(row.get("pit_so", 0)),
                    "whip": round(float(row.get("pit_whip", 0)), 2),
                    "hr": int(row.get("pit_hr", 0)),
                },
            }
        )

    return {"season": season, "teams": teams}


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------


def _ctx(request: Request, **extra: object) -> dict:
    """Build a base template context with version info and active model."""
    return {
        "request": request,
        "git_commit": get_git_commit(),
        "active_model": get_active_model_type(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    return templates.TemplateResponse("index.html", _ctx(request))


@app.get("/game/{game_pk}", response_class=HTMLResponse)
async def page_game(request: Request, game_pk: int):
    return templates.TemplateResponse("game.html", _ctx(request, game_pk=game_pk))


@app.get("/season/2026", response_class=HTMLResponse)
async def page_season_2026(request: Request):
    """Dedicated 2026 season schedule + pre-season predictions page."""
    df = get_features()
    season_df = df[df["season"] == 2026]
    total = len(season_df)
    first_date = str(season_df["date"].min())[:10] if total else "—"
    return templates.TemplateResponse(
        "season_2026.html",
        _ctx(request, total_games=total, first_date=first_date),
    )


@app.get("/standings", response_class=HTMLResponse)
async def page_standings(request: Request):
    """Full standings page: predicted vs actual, all divisions + league leaders."""
    return templates.TemplateResponse("standings.html", _ctx(request))


@app.get("/sitemap", response_class=HTMLResponse)
async def page_sitemap(request: Request):
    """Sitemap page listing all routes in the application."""
    return templates.TemplateResponse("sitemap.html", _ctx(request))


@app.get("/sitemap.xml", response_class=Response)
async def xml_sitemap(request: Request) -> Response:
    """XML sitemap for search engine crawlers."""
    base = str(request.base_url).rstrip("/")
    paths = ["/", "/season/2026", "/standings", "/wiki", "/dashboard", "/sitemap"]
    urls = "\n".join(f"  <url><loc>{base}{p}</loc></url>" for p in paths)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


@app.get("/wiki", response_class=HTMLResponse)
async def page_wiki(request: Request):
    """Technical wiki describing models, data sources, and training pipeline."""
    return templates.TemplateResponse("wiki.html", _ctx(request))


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    """Admin dashboard with retrain/ingest controls and system status."""
    return templates.TemplateResponse("dashboard.html", _ctx(request))


# ---------------------------------------------------------------------------
# Model switching API
# ---------------------------------------------------------------------------

_VALID_MODEL_TYPES = ("logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked")


class _SwitchModelRequest(BaseModel):
    model_type: str


@app.get("/api/active-model")
def api_active_model() -> dict:
    """Return the currently active model type and available alternatives."""
    return {
        "model_type": get_active_model_type(),
        "available": available_model_types(),
    }


@app.post("/api/admin/switch-model")
def api_switch_model(body: _SwitchModelRequest) -> dict:
    """Hot-swap the active prediction model at runtime."""
    model_type = body.model_type.lower().strip()
    if model_type not in _VALID_MODEL_TYPES:
        return {"ok": False, "message": f"Unknown model type '{model_type}'."}
    try:
        logger.info("API request to switch model to '%s'", model_type)
        with timed_operation("model_switch"):
            switch_model(model_type)
        os.environ["WINPROB_MODEL_TYPE"] = model_type
        return {"ok": True, "model_type": model_type, "message": f"Switched to {model_type}."}
    except Exception as exc:
        logger.error("Model switch failed: %s", exc)
        return {"ok": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Admin API — pipeline control and status
# ---------------------------------------------------------------------------


@app.get("/api/admin/status")
def api_admin_status() -> dict:
    """Full system status: data coverage, model inventory, pipeline states."""
    return {
        "version": "3.0",
        "git_commit": get_git_commit(),
        "data": gather_data_status(),
        "models": gather_model_status(),
        "pipelines": {
            "ingest": get_state(PipelineKind.INGEST).to_dict(),
            "update": get_state(PipelineKind.UPDATE).to_dict(),
            "retrain": get_state(PipelineKind.RETRAIN).to_dict(),
        },
    }


@app.post("/api/admin/ingest")
async def api_admin_ingest() -> dict:
    """Full re-ingestion: clears all processed data and re-ingests every season."""
    blocker = conflicting_pipeline()
    if blocker is not None:
        return {
            "ok": False,
            "message": f"Cannot start ingest — {blocker.value} pipeline is running.",
        }
    asyncio.create_task(run_pipeline(PipelineKind.INGEST, on_success=_reload_app))
    return {"ok": True, "message": "Full re-ingestion started."}


@app.post("/api/admin/update")
async def api_admin_update() -> dict:
    """Update current season data only (non-destructive)."""
    blocker = conflicting_pipeline()
    if blocker is not None:
        return {
            "ok": False,
            "message": f"Cannot start update — {blocker.value} pipeline is running.",
        }
    asyncio.create_task(run_pipeline(PipelineKind.UPDATE, on_success=_reload_app))
    return {"ok": True, "message": "Season update started."}


@app.post("/api/admin/retrain")
async def api_admin_retrain() -> dict:
    """Clear all model artifacts and retrain from scratch."""
    blocker = conflicting_pipeline()
    if blocker is not None:
        return {
            "ok": False,
            "message": f"Cannot start retrain — {blocker.value} pipeline is running.",
        }
    asyncio.create_task(run_pipeline(PipelineKind.RETRAIN, on_success=_reload_app))
    return {"ok": True, "message": "Retrain pipeline started."}
