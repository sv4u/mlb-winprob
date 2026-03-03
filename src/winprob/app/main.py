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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from winprob.app.admin import (
    PipelineKind,
    PipelineStatus,
    gather_data_status,
    gather_model_status,
    get_state,
    run_pipeline,
)
from winprob.app.data_cache import (
    TEAM_NAMES,
    get_features,
    get_git_commit,
    get_model,
    startup,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MLB Win Probability", version="3.0")


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
    startup(model_type)


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
                "home_elo": round(float(r["home_elo"]), 1) if pd.notna(r.get("home_elo")) else None,
                "away_elo": round(float(r["away_elo"]), 1) if pd.notna(r.get("away_elo")) else None,
            }
        )

    return {"total": total, "offset": offset, "limit": limit, "games": rows}


@app.get("/api/games/{game_pk}", response_model=None)
def api_game_detail(game_pk: int) -> dict | JSONResponse:
    """Full feature breakdown + SHAP attribution for a single game."""
    df = get_features()
    matches = df[df["game_pk"] == game_pk]
    if matches.empty:
        return JSONResponse({"error": "game_pk not found"}, status_code=404)

    row = matches.iloc[0]
    model, meta, feature_cols = get_model()

    # SHAP attribution
    shap_vals: dict[str, float] = {}
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
        import logging

        logging.getLogger(__name__).warning(
            "SHAP attribution failed for game_pk=%d: %s", game_pk, exc
        )

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


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------


def _ctx(request: Request, **extra: object) -> dict:
    """Build a base template context with version info."""
    return {"request": request, "git_commit": get_git_commit(), **extra}


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


@app.get("/wiki", response_class=HTMLResponse)
async def page_wiki(request: Request):
    """Technical wiki describing models, data sources, and training pipeline."""
    return templates.TemplateResponse("wiki.html", _ctx(request))


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    """Admin dashboard with retrain/ingest controls and system status."""
    return templates.TemplateResponse("dashboard.html", _ctx(request))


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
            "retrain": get_state(PipelineKind.RETRAIN).to_dict(),
        },
    }


@app.post("/api/admin/ingest")
async def api_admin_ingest() -> dict:
    """Kick off the data-ingest pipeline in the background."""
    state = get_state(PipelineKind.INGEST)
    if state.status == PipelineStatus.RUNNING:
        return {"ok": False, "message": "Ingest pipeline is already running."}
    asyncio.create_task(run_pipeline(PipelineKind.INGEST, on_success=_reload_app))
    return {"ok": True, "message": "Ingest pipeline started."}


@app.post("/api/admin/retrain")
async def api_admin_retrain() -> dict:
    """Kick off the model-retrain pipeline in the background."""
    state = get_state(PipelineKind.RETRAIN)
    if state.status == PipelineStatus.RUNNING:
        return {"ok": False, "message": "Retrain pipeline is already running."}
    asyncio.create_task(run_pipeline(PipelineKind.RETRAIN, on_success=_reload_app))
    return {"ok": True, "message": "Retrain pipeline started."}
