"""FastAPI application — MLB Win Probability dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import grpc
import pandas as pd
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import inspect

from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

from mlb_predict.app.admin import (
    PipelineKind,
    PipelineOptions,
    conflicting_pipeline,
    gather_data_status,
    gather_model_status,
    get_state,
    run_pipeline,
    ws_repl_run,
    ws_shell_run,
)
from mlb_predict.app.data_cache import (
    TEAM_NAMES,
    available_model_types,
    get_active_model_type,
    get_features,
    get_git_commit,
    get_model,
    is_ready,
    startup,
    switch_model,
    try_startup,
)
from mlb_predict.app.game_detail_cache import get_game_detail_cached, set_game_detail_cached
from mlb_predict.app.response_cache import cache_get_response
from mlb_predict.app.timing import TimingMiddleware, timed_operation
from mlb_predict.mcp import create_mcp_app
from mlb_predict.standings import (
    DIVISION_DISPLAY_ORDER,
    DIVISIONS,
    compute_league_leaders,
    compute_predicted_standings,
    merge_predicted_actual,
)

logger = logging.getLogger(__name__)

_GRPC_ENABLED = os.environ.get("MLB_PREDICT_GRPC_ENABLED", "1").strip() == "1"
_GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))


def _live_api_enabled() -> bool:
    """True if live MLB/Odds API calls are allowed (MLB_PREDICT_LIVE_API != 0)."""
    return os.environ.get("MLB_PREDICT_LIVE_API", "1").strip() != "0"


_MTD_USE_NEW_KW = (
    "always_print_fields_with_no_presence" in inspect.signature(MessageToDict).parameters
)


def _grpc_dict(msg) -> dict:
    """Convert protobuf to JSON-serializable dict (snake_case, include defaults)."""
    defaults_kw = (
        {"always_print_fields_with_no_presence": True}
        if _MTD_USE_NEW_KW
        else {"including_default_value_fields": True}
    )
    return MessageToDict(msg, preserving_proto_field_name=True, **defaults_kw)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: load data, start gRPC server, create stubs. Shutdown: stop gRPC."""
    model_type = os.environ.get("MLB_PREDICT_MODEL_TYPE", _DEFAULT_MODEL_TYPE)
    logger.info("Application startup: model_type=%s", model_type)
    loaded = try_startup(model_type)
    if loaded:
        logger.info("Startup complete — serving on model '%s'", model_type)
    else:
        logger.info("Data not ready — server accepting requests; auto-bootstrap starting")
        asyncio.create_task(_auto_bootstrap())

    app.state._grpc_stubs = None
    app.state._grpc_server = None
    if _GRPC_ENABLED:
        try:
            from mlb_predict.grpc.server import start_grpc_server

            server = await start_grpc_server()
            if server:
                app.state._grpc_server = server
                from mlb_predict.grpc.generated.mlb_predict.v1 import (
                    admin_pb2_grpc,
                    games_pb2_grpc,
                    models_pb2_grpc,
                    standings_pb2_grpc,
                    system_pb2_grpc,
                )

                channel = grpc.aio.insecure_channel(f"localhost:{_GRPC_PORT}")
                app.state._grpc_stubs = {
                    "system": system_pb2_grpc.SystemServiceStub(channel),
                    "games": games_pb2_grpc.GameServiceStub(channel),
                    "models": models_pb2_grpc.ModelServiceStub(channel),
                    "standings": standings_pb2_grpc.StandingsServiceStub(channel),
                    "admin": admin_pb2_grpc.AdminServiceStub(channel),
                }
                logger.info("gRPC gateway enabled — stubs ready")
        except Exception as exc:
            logger.warning("gRPC server or stubs failed: %s — running without gateway", exc)

    yield

    if getattr(app.state, "_grpc_server", None) is not None:
        from mlb_predict.grpc.server import stop_grpc_server

        await stop_grpc_server()
        app.state._grpc_stubs = None
        app.state._grpc_server = None


# MCP server (Streamable HTTP) — same process as web UI; mount at /mcp for Cursor/home network
_mcp_app = create_mcp_app()


@asynccontextmanager
async def _combined_lifespan(app: FastAPI):
    """Run app lifespan then MCP server lifespan so /mcp tools have access to data."""
    async with _lifespan(app):
        async with _mcp_app.lifespan(app):
            yield


app = FastAPI(title="MLB Win Probability", version="3.0", lifespan=_combined_lifespan)
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
app.mount("/mcp", _mcp_app)


_DEFAULT_MODEL_TYPE = "stacked"


def _reload_app() -> None:
    """Re-run startup() to pick up new data/models after a pipeline completes."""
    model_type = os.environ.get("MLB_PREDICT_MODEL_TYPE", _DEFAULT_MODEL_TYPE)
    startup(model_type)


async def _auto_bootstrap() -> None:
    """Auto-trigger ingest + retrain when no data/model exists on first startup."""
    logger.info("No data/model found — auto-bootstrapping with ingest + retrain")

    def _reload_after_retrain() -> None:
        model_type = os.environ.get("MLB_PREDICT_MODEL_TYPE", _DEFAULT_MODEL_TYPE)
        try:
            startup(model_type)
            logger.info("Auto-bootstrap complete — app is ready")
        except RuntimeError as exc:
            logger.error("Auto-bootstrap reload failed: %s", exc)

    await run_pipeline(PipelineKind.INGEST)
    ingest_state = get_state(PipelineKind.INGEST)
    if ingest_state.status.value != "success":
        logger.error("Auto-bootstrap ingest failed — retrain skipped")
        return
    await run_pipeline(PipelineKind.RETRAIN, on_success=_reload_after_retrain)


# ---------------------------------------------------------------------------
# API endpoints (gateway to gRPC when enabled)
# ---------------------------------------------------------------------------


def _stubs(request: Request):
    return getattr(request.app.state, "_grpc_stubs", None)


def _grpc_error_to_response(exc: grpc.RpcError) -> JSONResponse:
    if exc.code() == grpc.StatusCode.UNAVAILABLE:
        return JSONResponse(
            {"error": "System initializing — data not loaded yet.", "status": "initializing"},
            status_code=503,
        )
    if exc.code() == grpc.StatusCode.NOT_FOUND:
        return JSONResponse({"error": exc.details() or "Not found"}, status_code=404)
    return JSONResponse(
        {"error": exc.details() or str(exc)},
        status_code=500,
    )


@app.get("/api/health", response_model=None)
async def api_health(request: Request) -> dict | JSONResponse:
    """Lightweight health/readiness probe."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["system"].Health(common_pb2.Empty())
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    ready = is_ready()
    payload = {"ready": ready, "version": "3.0"}
    if not ready:
        return JSONResponse(payload, status_code=503)
    return payload


@app.get("/api/version", response_model=None)
async def api_version(request: Request) -> dict | JSONResponse:
    """Return the current application version and git commit hash."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["system"].Version(common_pb2.Empty())
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    return {"version": "3.0", "git_commit": get_git_commit()}


def _not_ready_json() -> JSONResponse:
    """Return a 503 JSON response when the app isn't ready yet."""
    return JSONResponse(
        {"error": "System initializing — data not loaded yet.", "status": "initializing"},
        status_code=503,
    )


@app.get("/api/seasons", response_model=None)
@cache_get_response(ttl_seconds=60)
async def api_seasons(request: Request) -> list[int] | dict | JSONResponse:
    """List all available seasons."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["system"].Seasons(common_pb2.Empty())
            return list(r.seasons)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
    try:
        df = get_features()
        return sorted(df["season"].dropna().unique().astype(int).tolist())
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.get("/api/teams", response_model=None)
@cache_get_response(ttl_seconds=60)
async def api_teams(request: Request) -> list[dict] | JSONResponse:
    """List all known teams with their Retrosheet codes and names."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["system"].Teams(common_pb2.Empty())
            return _grpc_dict(r).get("teams", [])
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
    df = get_features()
    teams = set(df["home_retro"].dropna().tolist()) | set(df["away_retro"].dropna().tolist())
    return sorted(
        [{"code": t, "name": TEAM_NAMES.get(t, t)} for t in teams],
        key=lambda x: x["name"],
    )


@app.get("/api/games", response_model=None)
@cache_get_response(ttl_seconds=45)
async def api_games(
    request: Request,
    season: Annotated[int | None, Query()] = None,
    home: Annotated[str | None, Query()] = None,
    away: Annotated[str | None, Query()] = None,
    date: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query(description="Search matchup (team name or abbrev)")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    sort: Annotated[str, Query()] = "date",
    order: Annotated[str, Query()] = "desc",
) -> dict | JSONResponse:
    """Query games with optional filters."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import games_pb2

            req = games_pb2.GetGamesRequest(
                limit=limit,
                offset=offset,
                sort=sort,
                order=order,
            )
            if season is not None:
                req.season = season
            if home is not None:
                req.home = home
            if away is not None:
                req.away = away
            if date is not None:
                req.date = date
            r = await stubs["games"].GetGames(req)
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
    logger.debug(
        "GET /api/games season=%s home=%s away=%s date=%s q=%s limit=%d offset=%d",
        season,
        home,
        away,
        date,
        q,
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
        if q and q.strip():
            q_ = q.strip().lower()
            hretro = df["home_retro"].astype(str).str.lower()
            aretro = df["away_retro"].astype(str).str.lower()
            home_name = df["home_retro"].map(lambda r: (TEAM_NAMES.get(str(r), "") or "").lower())
            away_name = df["away_retro"].map(lambda r: (TEAM_NAMES.get(str(r), "") or "").lower())
            mask = (
                hretro.str.contains(q_, na=False, regex=False)
                | aretro.str.contains(q_, na=False, regex=False)
                | home_name.astype(str).str.contains(q_, na=False, regex=False)
                | away_name.astype(str).str.contains(q_, na=False, regex=False)
            )
            df = df[mask]

        valid_sorts = {"date", "prob", "season", "game_pk"}
        sort_col = sort if sort in valid_sorts else "date"
        ascending = order.lower() != "desc"
        df = df.sort_values(sort_col, ascending=ascending)

        total = len(df)
        page = df.iloc[offset : offset + limit]

        # Vectorized row construction (avoids slow iterrows())
        def _prob_home_series(s):
            return s.apply(lambda v: round(float(v), 4) if pd.notna(v) else None)

        game_pks = page["game_pk"].fillna(0).astype(int).tolist()
        dates = page["date"].astype(str).str[:10].tolist()
        seasons = page["season"].fillna(0).astype(int).tolist()
        game_types = page["game_type"].fillna("R").astype(str).tolist()
        home_retros = page["home_retro"].astype(str).tolist()
        away_retros = page["away_retro"].astype(str).tolist()
        prob_list = _prob_home_series(page["prob"]).tolist()
        home_win_list = page["home_win"].apply(lambda v: int(v) if pd.notna(v) else None).tolist()
        home_elo_list = (
            page["home_elo"].apply(lambda v: round(float(v), 1) if pd.notna(v) else None).tolist()
        )
        away_elo_list = (
            page["away_elo"].apply(lambda v: round(float(v), 1) if pd.notna(v) else None).tolist()
        )

        rows = [
            {
                "game_pk": pk,
                "date": d,
                "season": se,
                "game_type": gt,
                "home_retro": hr,
                "home_name": TEAM_NAMES.get(hr, ""),
                "away_retro": ar,
                "away_name": TEAM_NAMES.get(ar, ""),
                "prob_home": ph,
                "home_win": hw,
                "home_elo": he,
                "away_elo": ae,
            }
            for pk, d, se, gt, hr, ar, ph, hw, he, ae in zip(
                game_pks,
                dates,
                seasons,
                game_types,
                home_retros,
                away_retros,
                prob_list,
                home_win_list,
                home_elo_list,
                away_elo_list,
            )
        ]

    return {"total": total, "offset": offset, "limit": limit, "games": rows}


@app.get("/api/games/{game_pk}", response_model=None)
async def api_game_detail(request: Request, game_pk: int) -> dict | JSONResponse:
    """Full feature breakdown + SHAP attribution for a single game."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import games_pb2

            r = await stubs["games"].GetGameDetail(games_pb2.GetGameDetailRequest(game_pk=game_pk))
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
    logger.debug("GET /api/games/%d", game_pk)

    from mlb_predict.app.odds_cache import get_cached_odds, is_odds_configured, match_odds_for_game

    # Return cached payload (SHAP + stats) when available; only add fresh live_odds
    cached = get_game_detail_cached(game_pk)
    if cached is not None:
        events = await get_cached_odds()
        live_odds = match_odds_for_game(
            events, cached.get("home_retro", ""), cached.get("away_retro", "")
        )
        return {**cached, "live_odds": live_odds, "odds_configured": is_odds_configured()}

    df = get_features()
    matches = df[df["game_pk"] == game_pk]
    if matches.empty:
        return JSONResponse({"error": "game_pk not found"}, status_code=404)

    row = matches.iloc[0]
    model, meta, feature_cols = get_model()

    shap_vals: dict[str, float] = {}
    with timed_operation("shap_attribution"):
        try:
            from mlb_predict.grpc.services.games import _extract_tree_model

            x = row[feature_cols].values.astype(float)
            tree_model = _extract_tree_model(model)
            base = getattr(model, "base", model)
            if tree_model is not None:
                import shap

                X_df = pd.DataFrame([x], columns=feature_cols)
                explainer = shap.TreeExplainer(tree_model)
                sv = explainer.shap_values(X_df)
                arr = sv[1][0] if isinstance(sv, list) else sv[0]
                shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, arr)}
            elif hasattr(base, "named_steps"):
                scaler = base.named_steps["scaler"]
                lr = base.named_steps["lr"]
                coef = lr.coef_[0]
                z = (x - scaler.mean_) / scaler.scale_
                shap_arr = coef * z
                shap_vals = {f: round(float(v), 5) for f, v in zip(feature_cols, shap_arr)}
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

    home_retro = str(row.get("home_retro", ""))
    away_retro = str(row.get("away_retro", ""))
    events = await get_cached_odds()
    live_odds = match_odds_for_game(events, home_retro, away_retro)

    payload = {
        "game_pk": game_pk,
        "date": str(row.get("date", ""))[:10],
        "season": int(row.get("season", 0) or 0),
        "game_type": str(row.get("game_type", "R")),
        "home_retro": home_retro,
        "home_name": TEAM_NAMES.get(home_retro, ""),
        "away_retro": away_retro,
        "away_name": TEAM_NAMES.get(away_retro, ""),
        "prob_home": round(float(row["prob"]), 4) if pd.notna(row.get("prob")) else None,
        "home_win": int(row["home_win"]) if pd.notna(row.get("home_win")) else None,
        "stats": stats,
        "top_factors": top_factors,
        "live_odds": live_odds,
        "odds_configured": is_odds_configured(),
    }
    set_game_detail_cached(game_pk, payload)
    return payload


@app.get("/api/games/{game_pk}/play-by-play", response_model=None)
async def api_game_play_by_play(game_pk: int) -> dict | JSONResponse:
    """Return play-by-play data for a game from the MLB Stats API (feed/live)."""
    if not _live_api_enabled():
        return JSONResponse(
            {
                "error": "Live data disabled.",
                "detail": "Set MLB_PREDICT_LIVE_API=1 to enable play-by-play.",
            },
            status_code=503,
        )
    from mlb_predict.mlbapi.client import MLBAPIClient, MLBNotFoundError
    from mlb_predict.mlbapi.game_feed import fetch_game_feed

    try:
        async with timed_operation("mlb_api_game_feed"):
            async with MLBAPIClient() as client:
                data = await fetch_game_feed(client, game_pk=game_pk)
        return data
    except MLBNotFoundError:
        return JSONResponse(
            {"error": "Game not found or no play-by-play available."}, status_code=404
        )
    except Exception as exc:
        logger.warning("Play-by-play fetch failed for game_pk=%d: %s", game_pk, exc)
        return JSONResponse(
            {"error": "Failed to load play-by-play.", "detail": str(exc)},
            status_code=502,
        )


@app.get("/api/odds", response_model=None)
async def api_odds() -> dict:
    """Return all current MLB game odds from The Odds API (cached)."""
    from mlb_predict.app.odds_cache import get_cached_odds, is_odds_configured, match_odds_for_game

    from mlb_predict.external.odds import _to_retro

    events = await get_cached_odds()
    enriched: list[dict] = []
    for ev in events:
        hr = _to_retro(ev.get("home_team") or "")
        ar = _to_retro(ev.get("away_team") or "")
        matched = match_odds_for_game([ev], hr, ar)
        if matched:
            matched["home_retro"] = hr
            matched["away_retro"] = ar
            enriched.append(matched)
    return {"configured": is_odds_configured(), "count": len(enriched), "events": enriched}


@app.get("/api/ev-opportunities", response_model=None)
async def api_ev_opportunities(
    min_edge: Annotated[float, Query(ge=0.0, le=0.5)] = 0.0,
) -> dict:
    """Positive-EV moneyline bets: model edge over best market odds."""
    from mlb_predict.app.odds_cache import (
        compute_ev_opportunities,
        get_cached_odds,
        is_odds_configured,
    )

    if not is_odds_configured():
        return {"configured": False, "count": 0, "opportunities": []}
    if not is_ready():
        return {"configured": True, "count": 0, "opportunities": []}

    events = await get_cached_odds()
    df = get_features()
    opps = compute_ev_opportunities(events, df, min_edge=min_edge)
    return {"configured": True, "count": len(opps), "opportunities": opps}


@app.get("/api/upsets", response_model=None)
async def api_upsets(
    request: Request,
    season: Annotated[int | None, Query()] = None,
    home: Annotated[str | None, Query()] = None,
    away: Annotated[str | None, Query()] = None,
    min_prob: Annotated[float, Query(ge=0.5, le=1.0)] = 0.65,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[dict] | JSONResponse:
    """Return the biggest upsets (heavy favorites that lost)."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import games_pb2

            req = games_pb2.GetUpsetsRequest(min_prob=min_prob, limit=limit)
            if season is not None:
                req.season = season
            if home is not None:
                req.home = home
            if away is not None:
                req.away = away
            r = await stubs["games"].GetUpsets(req)
            return _grpc_dict(r).get("upsets", [])
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
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
@cache_get_response(ttl_seconds=300)
async def api_cv_summary(request: Request) -> list[dict]:
    """Return cross-validation results from the latest training run."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["models"].GetCVSummary(common_pb2.Empty())
            return json.loads(r.json) if r.json else []
        except grpc.RpcError:
            pass
    paths = [
        Path("data/models/cv_summary_v3.json"),
        Path("data/models/cv_summary_v2.json"),
        Path("data/models/cv_summary.json"),
    ]
    for p in paths:
        if p.exists():
            return json.loads(p.read_text())
    return []


def _build_standings_payload(
    standings: pd.DataFrame,
    season: int,
) -> tuple[list[dict], dict]:
    """Build divisions list and league_leaders from a standings DataFrame."""
    league_leaders = compute_league_leaders(standings)
    divisions = []
    for div_id in DIVISION_DISPLAY_ORDER:
        div_info = DIVISIONS.get(div_id, {})
        div_df = standings[standings["division_id"] == div_id].sort_values(
            "pred_win_pct",
            ascending=False,
        )
        if div_df.empty:
            continue
        teams = []
        for _, row in div_df.iterrows():
            team_entry = {
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
    return divisions, league_leaders


@app.get("/api/standings", response_model=None)
@cache_get_response(ttl_seconds=60)
async def api_standings(
    request: Request,
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
    include_spring: Annotated[
        bool, Query(description="Include spring training standings subsection")
    ] = False,
) -> dict | JSONResponse:
    """Return predicted vs actual standings grouped by division.

    Main standings are regular-season only. Actual standings are from the MLB Stats API.
    When include_spring=true, also returns spring training predicted standings (no actuals).
    """
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import standings_pb2

            r = await stubs["standings"].GetStandings(
                standings_pb2.GetStandingsRequest(season=season)
            )
            out = _grpc_dict(r)
            if include_spring:
                with timed_operation("predicted_standings_spring"):
                    df = get_features()
                    pred_spring = compute_predicted_standings(df, season=season, game_type="S")
                if not pred_spring.empty:
                    divs_spring, leaders_spring = _build_standings_payload(pred_spring, season)
                    out["spring"] = {"divisions": divs_spring, "league_leaders": leaders_spring}
            return out
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not is_ready():
        return _not_ready_json()
    logger.debug("GET /api/standings season=%d include_spring=%s", season, include_spring)
    from mlb_predict.mlbapi.client import MLBAPIClient
    from mlb_predict.mlbapi.standings import fetch_standings

    with timed_operation("predicted_standings"):
        df = get_features()
        pred_df = compute_predicted_standings(df, season=season, game_type="R")

    actual_df = pd.DataFrame()
    if _live_api_enabled():
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
        out = {"season": season, "divisions": [], "league_leaders": {}}
        if include_spring:
            with timed_operation("predicted_standings_spring"):
                pred_spring = compute_predicted_standings(df, season=season, game_type="S")
            if not pred_spring.empty:
                divs_spring, leaders_spring = _build_standings_payload(pred_spring, season)
                out["spring"] = {"divisions": divs_spring, "league_leaders": leaders_spring}
        return out

    divisions, league_leaders = _build_standings_payload(standings, season)
    out = {
        "season": season,
        "game_type": "regularSeason",
        "divisions": divisions,
        "league_leaders": league_leaders,
    }
    if include_spring:
        with timed_operation("predicted_standings_spring"):
            pred_spring = compute_predicted_standings(df, season=season, game_type="S")
        if not pred_spring.empty:
            divs_spring, leaders_spring = _build_standings_payload(pred_spring, season)
            out["spring"] = {"divisions": divs_spring, "league_leaders": leaders_spring}
    return out


@app.get("/api/team-stats", response_model=None)
async def api_team_stats(
    request: Request,
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
) -> dict | JSONResponse:
    """Return batting and pitching stats for all teams in a season."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import standings_pb2

            r = await stubs["standings"].GetTeamStats(
                standings_pb2.GetTeamStatsRequest(season=season)
            )
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    if not _live_api_enabled():
        return {
            "season": season,
            "teams": [],
            "message": "Live data disabled. Set MLB_PREDICT_LIVE_API=1 to enable.",
        }
    from mlb_predict.mlbapi.client import MLBAPIClient
    from mlb_predict.mlbapi.standings import fetch_all_team_stats, fetch_standings

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


@app.get("/api/leaders", response_model=None)
async def api_leaders(
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
    league_id: Annotated[int | None, Query(description="AL=103, NL=104; omit for both")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    stat_group: Annotated[str, Query(description="hitting or pitching")] = "hitting",
) -> dict:
    """Return league leaders (top N per category) for a season from the MLB Stats API."""
    if not _live_api_enabled():
        return {
            "season": season,
            "league_id": league_id,
            "stat_group": stat_group,
            "leader_tables": [],
            "message": "Live data disabled. Set MLB_PREDICT_LIVE_API=1 to enable.",
        }
    from mlb_predict.mlbapi.client import MLBAPIClient
    from mlb_predict.mlbapi.leaders import fetch_leaders

    try:
        async with timed_operation("mlb_api_leaders"):
            async with MLBAPIClient() as client:
                tables = await fetch_leaders(
                    client,
                    season=season,
                    league_id=league_id,
                    limit=limit,
                    stat_group=stat_group,
                )
        return {
            "season": season,
            "league_id": league_id,
            "stat_group": stat_group,
            "leader_tables": tables,
        }
    except Exception as exc:
        logger.warning("Leaders fetch failed season=%d: %s", season, exc)
        return {
            "season": season,
            "league_id": league_id,
            "stat_group": stat_group,
            "leader_tables": [],
            "error": str(exc),
        }


@app.get("/api/player-stats", response_model=None)
async def api_player_stats(
    season: Annotated[int, Query(ge=2000, le=2030)] = 2026,
    group: Annotated[str, Query(description="hitting or pitching")] = "hitting",
    league_id: Annotated[int | None, Query(description="AL=103, NL=104; omit for both")] = None,
    limit: Annotated[int, Query(ge=1, le=250)] = 250,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Return full player stats table (batting or pitching) for a season from the MLB Stats API."""
    if not _live_api_enabled():
        return {
            "season": season,
            "group": group,
            "league_id": league_id,
            "players": [],
            "message": "Live data disabled. Set MLB_PREDICT_LIVE_API=1 to enable.",
        }
    from mlb_predict.mlbapi.client import MLBAPIClient
    from mlb_predict.mlbapi.leaders import fetch_player_stats

    try:
        async with timed_operation("mlb_api_player_stats"):
            async with MLBAPIClient() as client:
                players = await fetch_player_stats(
                    client,
                    season=season,
                    group=group,
                    league_id=league_id,
                    limit=limit,
                    offset=offset,
                )
        return {
            "season": season,
            "group": group,
            "league_id": league_id,
            "offset": offset,
            "count": len(players),
            "players": players,
        }
    except Exception as exc:
        logger.warning("Player stats fetch failed season=%d: %s", season, exc)
        return {
            "season": season,
            "group": group,
            "league_id": league_id,
            "count": 0,
            "players": [],
            "error": str(exc),
        }


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


def _init_page(request: Request) -> HTMLResponse:
    """Return the initialization progress page."""
    return templates.TemplateResponse(
        "initializing.html",
        {"request": request, "git_commit": get_git_commit()},
    )


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    if not is_ready():
        return _init_page(request)
    return templates.TemplateResponse("index.html", _ctx(request))


@app.get("/game/{game_pk}", response_class=HTMLResponse)
async def page_game(request: Request, game_pk: int):
    if not is_ready():
        return _init_page(request)
    return templates.TemplateResponse("game.html", _ctx(request, game_pk=game_pk))


@app.get("/season/2026", response_class=HTMLResponse)
async def page_season_2026(request: Request):
    """Dedicated 2026 season schedule + pre-season predictions page."""
    if not is_ready():
        return _init_page(request)
    df = get_features()
    season_df = df[df["season"] == 2026]
    total = len(season_df)
    first_date = str(season_df["date"].min())[:10] if total else "—"
    return templates.TemplateResponse(
        "season_2026.html",
        _ctx(request, total_games=total, first_date=first_date),
    )


@app.get("/season/{season}", response_class=HTMLResponse)
async def page_season_generic(request: Request, season: int):
    """Dynamic season page — redirects to the home page filtered by season."""
    from fastapi.responses import RedirectResponse

    if not is_ready():
        return _init_page(request)
    df = get_features()
    available = sorted(df["season"].dropna().unique().astype(int).tolist())
    if season not in available:
        return HTMLResponse(
            content=templates.get_template("404_season.html").render(
                request=request,
                git_commit=get_git_commit(),
                season=season,
                available_seasons=available,
            ),
            status_code=404,
        )
    return RedirectResponse(url=f"/?season={season}", status_code=302)


@app.get("/standings", response_class=HTMLResponse)
async def page_standings(request: Request):
    """Full standings page: predicted vs actual, all divisions + league leaders."""
    if not is_ready():
        return _init_page(request)
    return templates.TemplateResponse("standings.html", _ctx(request))


@app.get("/leaders", response_class=HTMLResponse)
async def page_leaders(request: Request):
    """League leaders page: top players by hitting/pitching categories (MLB Stats API)."""
    return templates.TemplateResponse("leaders.html", _ctx(request))


@app.get("/players", response_class=HTMLResponse)
async def page_players(request: Request):
    """Player stats page: full batting/pitching tables by season (MLB Stats API)."""
    return templates.TemplateResponse("players.html", _ctx(request))


@app.get("/sitemap", response_class=HTMLResponse)
async def page_sitemap(request: Request):
    """Sitemap page listing all routes in the application."""
    return templates.TemplateResponse("sitemap.html", _ctx(request))


@app.get("/sitemap.xml", response_class=Response)
async def xml_sitemap(request: Request) -> Response:
    """XML sitemap for search engine crawlers."""
    base = str(request.base_url).rstrip("/")
    paths = [
        "/",
        "/season/2026",
        "/standings",
        "/leaders",
        "/players",
        "/odds",
        "/wiki",
        "/dashboard",
        "/sitemap",
    ]
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
    if not is_ready():
        return _init_page(request)
    return templates.TemplateResponse("wiki.html", _ctx(request))


@app.get("/odds", response_class=HTMLResponse)
async def page_odds_hub(request: Request):
    """Odds hub: EV+ opportunities, all odds board, and EV calculator."""
    return templates.TemplateResponse("odds_hub.html", _ctx(request))


@app.get("/tools/ev-calculator", response_class=HTMLResponse)
async def page_ev_calculator(request: Request):
    """Legacy EV calculator URL — redirects to the Odds Hub."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/odds", status_code=301)


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


class _OddsConfigRequest(BaseModel):
    """Body for POST /api/admin/odds-config. Key is never logged or re-exposed."""

    api_key: str


class _PipelineOptionsRequest(BaseModel):
    """Body for POST /api/admin/ingest and /api/admin/update with options."""

    include_preseason: bool = False
    seasons: list[int] | None = None
    refresh_mlbapi: bool = True
    refresh_retro: bool = True


@app.get("/api/active-model", response_model=None)
async def api_active_model(request: Request) -> dict | JSONResponse:
    """Return the currently active model type and available alternatives."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["models"].GetActiveModel(common_pb2.Empty())
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    return {
        "model_type": get_active_model_type(),
        "available": list(available_model_types()),
    }


@app.post("/api/admin/switch-model", response_model=None)
async def api_switch_model(request: Request, body: _SwitchModelRequest) -> dict | JSONResponse:
    """Hot-swap the active prediction model at runtime."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import models_pb2

            r = await stubs["models"].SwitchModel(
                models_pb2.SwitchModelRequest(model_type=body.model_type)
            )
            return _grpc_dict(r)
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    model_type = body.model_type.lower().strip()
    if model_type not in _VALID_MODEL_TYPES:
        return {"ok": False, "message": f"Unknown model type '{model_type}'."}
    try:
        logger.info("API request to switch model to '%s'", model_type)
        with timed_operation("model_switch"):
            switch_model(model_type)
        os.environ["MLB_PREDICT_MODEL_TYPE"] = model_type
        return {"ok": True, "model_type": model_type, "message": f"Switched to {model_type}."}
    except Exception as exc:
        logger.error("Model switch failed: %s", exc)
        return {"ok": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Admin API — pipeline control and status
# ---------------------------------------------------------------------------


@app.get("/api/admin/status", response_model=None)
async def api_admin_status(request: Request) -> dict | JSONResponse:
    """Full system status: data coverage, model inventory, pipeline states."""
    stubs = _stubs(request)
    if stubs:
        try:
            from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2

            r = await stubs["admin"].GetStatus(common_pb2.Empty())
            return json.loads(r.json) if r.json else {}
        except grpc.RpcError as e:
            return _grpc_error_to_response(e)
    from mlb_predict.external.odds_config import get_odds_config_status

    odds_status = get_odds_config_status()
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
        "odds_configured": odds_status.get("configured") is True,
    }


@app.get("/api/admin/odds-config", response_model=None)
async def api_admin_get_odds_config() -> dict:
    """Return whether the Odds API key is configured and from where. Never returns the key."""
    from mlb_predict.external.odds_config import get_odds_config_status

    return get_odds_config_status()


@app.post("/api/admin/odds-config", response_model=None)
async def api_admin_save_odds_config(body: _OddsConfigRequest) -> dict:
    """Save the Odds API key to data/processed/odds/config.json. Key is never logged."""
    from mlb_predict.external.odds_config import set_odds_api_key

    set_odds_api_key(body.api_key)
    return {"ok": True}


@app.post("/api/admin/ingest")
async def api_admin_ingest(body: _PipelineOptionsRequest | None = None) -> dict:
    """Full re-ingestion: clears all processed data and re-ingests every season."""
    blocker = conflicting_pipeline()
    if blocker is not None:
        return {
            "ok": False,
            "message": f"Cannot start ingest — {blocker.value} pipeline is running.",
        }
    opts: PipelineOptions | None = None
    if body:
        opts = PipelineOptions(
            include_preseason=body.include_preseason,
            seasons=body.seasons,
            refresh_mlbapi=body.refresh_mlbapi,
            refresh_retro=body.refresh_retro,
        )
    asyncio.create_task(run_pipeline(PipelineKind.INGEST, on_success=_reload_app, opts=opts))
    return {"ok": True, "message": "Full re-ingestion started."}


@app.post("/api/admin/update")
async def api_admin_update(body: _PipelineOptionsRequest | None = None) -> dict:
    """Update current season data only (non-destructive)."""
    blocker = conflicting_pipeline()
    if blocker is not None:
        return {
            "ok": False,
            "message": f"Cannot start update — {blocker.value} pipeline is running.",
        }
    opts: PipelineOptions | None = None
    if body:
        opts = PipelineOptions(
            include_preseason=body.include_preseason,
            seasons=body.seasons,
            refresh_mlbapi=body.refresh_mlbapi,
            refresh_retro=body.refresh_retro,
        )
    asyncio.create_task(run_pipeline(PipelineKind.UPDATE, on_success=_reload_app, opts=opts))
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


# ---------------------------------------------------------------------------
# WebSocket — Shell runner and Python REPL
# ---------------------------------------------------------------------------


@app.websocket("/ws/admin/shell")
async def ws_admin_shell(websocket: WebSocket) -> None:
    """WebSocket endpoint for executing shell commands with streaming output."""
    await ws_shell_run(websocket)


@app.websocket("/ws/admin/repl")
async def ws_admin_repl(websocket: WebSocket) -> None:
    """WebSocket endpoint for an interactive Python REPL session."""
    await ws_repl_run(websocket)
