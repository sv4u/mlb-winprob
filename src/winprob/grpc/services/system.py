"""SystemService implementation: Health, Version, Seasons, Teams."""

from __future__ import annotations

import grpc

from winprob.app.data_cache import TEAM_NAMES, get_features, get_git_commit, is_ready
from winprob.grpc.generated.winprob.v1 import common_pb2, system_pb2, system_pb2_grpc


class SystemServicer(system_pb2_grpc.SystemServiceServicer):
    """Implements SystemService RPCs using the app data cache."""

    async def Health(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> system_pb2.HealthResponse:
        """Lightweight health/readiness probe."""
        return system_pb2.HealthResponse(ready=is_ready(), version="3.0")

    async def Version(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> system_pb2.VersionResponse:
        """Return the current application version and git commit hash."""
        return system_pb2.VersionResponse(
            version="3.0",
            git_commit=get_git_commit(),
        )

    async def Seasons(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> system_pb2.SeasonsResponse:
        """List all available seasons."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        df = get_features()
        seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
        return system_pb2.SeasonsResponse(seasons=seasons)

    async def Teams(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> system_pb2.TeamsResponse:
        """List all known teams with their Retrosheet codes and names."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        df = get_features()
        teams_set = set(df["home_retro"].dropna().tolist()) | set(
            df["away_retro"].dropna().tolist()
        )
        teams = sorted(
            [{"code": t, "name": TEAM_NAMES.get(t, t)} for t in teams_set],
            key=lambda x: x["name"],
        )
        return system_pb2.TeamsResponse(
            teams=[
                common_pb2.Team(code=t["code"], name=t["name"]) for t in teams
            ]
        )
