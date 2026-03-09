"""Async gRPC server; started within FastAPI lifespan, sharing uvicorn's event loop."""

from __future__ import annotations

import logging
import os

import grpc.aio

from mlb_predict.grpc.generated.mlb_predict.v1 import (
    admin_pb2_grpc,
    games_pb2_grpc,
    models_pb2_grpc,
    standings_pb2_grpc,
    system_pb2_grpc,
)
from mlb_predict.grpc.services.admin import AdminServicer
from mlb_predict.grpc.services.games import GameServicer
from mlb_predict.grpc.services.models import ModelServicer
from mlb_predict.grpc.services.standings import StandingsServicer
from mlb_predict.grpc.services.system import SystemServicer

logger = logging.getLogger(__name__)

_GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))
_server: grpc.aio.Server | None = None


def _add_servicers(server: grpc.aio.Server) -> None:
    """Register all gRPC servicers. Called before server.start()."""
    system_pb2_grpc.add_SystemServiceServicer_to_server(SystemServicer(), server)
    games_pb2_grpc.add_GameServiceServicer_to_server(GameServicer(), server)
    models_pb2_grpc.add_ModelServiceServicer_to_server(ModelServicer(), server)
    standings_pb2_grpc.add_StandingsServiceServicer_to_server(StandingsServicer(), server)
    admin_pb2_grpc.add_AdminServiceServicer_to_server(AdminServicer(), server)


async def start_grpc_server() -> grpc.aio.Server | None:
    """Create, configure, and start the gRPC server on GRPC_PORT.

    Uses the current asyncio event loop (uvicorn's when called from lifespan).
    Returns the server instance so the caller can await server.stop(grace) on shutdown.
    """
    if os.environ.get("MLB_PREDICT_GRPC_ENABLED", "1").strip() == "0":
        logger.info("gRPC disabled (MLB_PREDICT_GRPC_ENABLED=0)")
        return None
    server = grpc.aio.server()
    _add_servicers(server)
    server.add_insecure_port(f"[::]:{_GRPC_PORT}")
    await server.start()
    global _server
    _server = server
    logger.info("gRPC server listening on port %s", _GRPC_PORT)
    return server


async def stop_grpc_server(grace: float = 5.0) -> None:
    """Stop the gRPC server if it was started."""
    global _server
    if _server is not None:
        await _server.stop(grace)
        _server = None
        logger.info("gRPC server stopped")
