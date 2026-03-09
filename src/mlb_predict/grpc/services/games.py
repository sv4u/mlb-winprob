"""GameService implementation: GetGames, GetGameDetail (with SHAP), GetUpsets."""

from __future__ import annotations

import logging
from typing import Any, cast

import grpc
import pandas as pd

from mlb_predict.app.data_cache import TEAM_NAMES, get_features, get_model, is_ready
from mlb_predict.app.timing import timed_operation
from mlb_predict.grpc.generated.mlb_predict.v1 import common_pb2, games_pb2, games_pb2_grpc

logger = logging.getLogger(__name__)

_TREE_MODEL_PREFERENCE = ["lightgbm", "xgboost", "catboost"]


def _extract_tree_model(model: Any) -> Any | None:
    """Extract a raw tree model suitable for TreeSHAP.

    Works for bare tree models, calibrated wrappers, and StackedEnsemble
    (picks the best available tree-based base model).
    """
    raw = getattr(model, "base", model)
    if hasattr(raw, "booster_") or hasattr(raw, "get_booster"):
        return raw
    if hasattr(model, "base_models") and hasattr(model, "base_keys"):
        for key in _TREE_MODEL_PREFERENCE:
            cal = model.base_models.get(key)
            if cal is None:
                continue
            inner = getattr(cal, "base", None)
            if inner is not None and (hasattr(inner, "booster_") or hasattr(inner, "get_booster")):
                return inner
    return None


def _row_to_game(r: Any) -> common_pb2.Game:
    """Build a Game protobuf from a feature row."""
    g = common_pb2.Game(
        game_pk=int(r.get("game_pk", 0) or 0),
        date=str(r.get("date", ""))[:10],
        season=int(r.get("season", 0) or 0),
        home_retro=str(r.get("home_retro", "")),
        home_name=TEAM_NAMES.get(str(r.get("home_retro", "")), ""),
        away_retro=str(r.get("away_retro", "")),
        away_name=TEAM_NAMES.get(str(r.get("away_retro", "")), ""),
    )
    if pd.notna(r.get("prob")):
        g.prob_home = round(float(r["prob"]), 4)
    if pd.notna(r.get("home_win")):
        g.home_win = int(r["home_win"])
    if pd.notna(r.get("home_elo")):
        g.home_elo = round(float(r["home_elo"]), 1)
    if pd.notna(r.get("away_elo")):
        g.away_elo = round(float(r["away_elo"]), 1)
    return g


class GameServicer(games_pb2_grpc.GameServiceServicer):
    """Implements GameService RPCs using the app data cache."""

    async def GetGames(
        self,
        request: games_pb2.GetGamesRequest,
        context: grpc.aio.ServicerContext,
    ) -> games_pb2.GetGamesResponse:
        """Query games with optional filters."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        df = get_features().copy()
        if request.HasField("season"):
            df = df[df["season"] == request.season]
        if request.HasField("home"):
            df = df[df["home_retro"] == request.home.upper()]
        if request.HasField("away"):
            df = df[df["away_retro"] == request.away.upper()]
        if request.HasField("date"):
            df = df[df["date"].astype(str) == request.date]

        valid_sorts = {"date", "prob", "season", "game_pk"}
        sort_col = request.sort if request.sort in valid_sorts else "date"
        ascending = (request.order or "desc").lower() != "desc"
        df = df.sort_values(sort_col, ascending=ascending)

        total = len(df)
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset
        page = df.iloc[offset : offset + limit]

        games = [_row_to_game(r) for _, r in page.iterrows()]
        return games_pb2.GetGamesResponse(
            total=total,
            offset=offset,
            limit=limit,
            games=games,
        )

    async def GetGameDetail(
        self,
        request: games_pb2.GetGameDetailRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.GameDetail:
        """Full feature breakdown + SHAP attribution for a single game."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        df = get_features()
        matches = df[df["game_pk"] == request.game_pk]
        if matches.empty:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("game_pk not found")
            raise Exception("Not found")

        row = matches.iloc[0]
        model, _meta, feature_cols = get_model()

        shap_vals: dict[str, float] = {}
        with timed_operation("shap_attribution"):
            try:
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
                logger.warning(
                    "SHAP attribution failed for game_pk=%d: %s",
                    request.game_pk,
                    exc,
                )

        top_factors = sorted(
            [{"feature": k, "value": v} for k, v in shap_vals.items()],
            key=lambda x: abs(cast(float, x["value"])),
            reverse=True,
        )[:12]

        stats = {
            k: (round(float(v), 4) if pd.notna(v) else 0.0)
            for k, v in row.items()
            if k in feature_cols and pd.notna(v)
        }

        detail = common_pb2.GameDetail(
            game_pk=request.game_pk,
            date=str(row.get("date", ""))[:10],
            season=int(row.get("season", 0) or 0),
            home_retro=str(row.get("home_retro", "")),
            home_name=TEAM_NAMES.get(str(row.get("home_retro", "")), ""),
            away_retro=str(row.get("away_retro", "")),
            away_name=TEAM_NAMES.get(str(row.get("away_retro", "")), ""),
            stats=stats,
            top_factors=[
                common_pb2.Factor(feature=f["feature"], value=f["value"]) for f in top_factors
            ],
        )
        if pd.notna(row.get("prob")):
            detail.prob_home = round(float(row["prob"]), 4)
        if pd.notna(row.get("home_win")):
            detail.home_win = int(row["home_win"])
        return detail

    async def GetUpsets(
        self,
        request: games_pb2.GetUpsetsRequest,
        context: grpc.aio.ServicerContext,
    ) -> games_pb2.GetUpsetsResponse:
        """Return the biggest upsets (heavy favorites that lost)."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        with timed_operation("upsets_query"):
            df = get_features().copy()
            if request.HasField("season"):
                df = df[df["season"] == request.season]
            if request.HasField("home"):
                df = df[df["home_retro"] == request.home.upper()]
            if request.HasField("away"):
                df = df[df["away_retro"] == request.away.upper()]
            has_result = df[df["home_win"].notna() & df["prob"].notna()].copy()
            has_result["fav_home"] = has_result["prob"] >= 0.5
            has_result["fav_prob"] = has_result["prob"].clip(lower=0.5)
            has_result.loc[~has_result["fav_home"], "fav_prob"] = (
                1 - has_result.loc[~has_result["fav_home"], "prob"]
            )
            has_result = has_result[has_result["fav_prob"] >= (request.min_prob or 0.65)]
            has_result["upset"] = (has_result["fav_home"] & (has_result["home_win"] == 0)) | (
                ~has_result["fav_home"] & (has_result["home_win"] == 1)
            )
            upsets = has_result[has_result["upset"]].nlargest(request.limit or 20, "fav_prob")
            entries = [
                games_pb2.UpsetEntry(
                    game_pk=int(r.get("game_pk", 0) or 0),
                    date=str(r.get("date", ""))[:10],
                    season=int(r.get("season", 0) or 0),
                    home_name=TEAM_NAMES.get(str(r.get("home_retro", "")), ""),
                    away_name=TEAM_NAMES.get(str(r.get("away_retro", "")), ""),
                    prob_home=round(float(r["prob"]), 4),
                    fav_prob=round(float(r["fav_prob"]), 4),
                    fav_team="home" if r["fav_home"] else "away",
                    winner="home" if r["home_win"] == 1 else "away",
                )
                for _, r in upsets.iterrows()
            ]
        return games_pb2.GetUpsetsResponse(upsets=entries)
