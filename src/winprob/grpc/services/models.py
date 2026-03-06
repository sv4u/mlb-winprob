"""ModelService implementation: GetActiveModel, SwitchModel, GetCVSummary."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import grpc

from winprob.app.data_cache import available_model_types, get_active_model_type, switch_model
from winprob.grpc.generated.winprob.v1 import common_pb2, models_pb2, models_pb2_grpc

logger = logging.getLogger(__name__)

_VALID_MODEL_TYPES = ("logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CV_PATHS = [
    _REPO_ROOT / "data" / "models" / "cv_summary_v3.json",
    _REPO_ROOT / "data" / "models" / "cv_summary_v2.json",
    _REPO_ROOT / "data" / "models" / "cv_summary.json",
]


class ModelServicer(models_pb2_grpc.ModelServiceServicer):
    """Implements ModelService RPCs using the app data cache."""

    async def GetActiveModel(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> models_pb2.ActiveModelResponse:
        """Return the currently active model type and available alternatives."""
        return models_pb2.ActiveModelResponse(
            model_type=get_active_model_type(),
            available=list(available_model_types()),
        )

    async def SwitchModel(
        self,
        request: models_pb2.SwitchModelRequest,
        context: grpc.aio.ServicerContext,
    ) -> models_pb2.SwitchModelResponse:
        """Hot-swap the active prediction model at runtime."""
        model_type = (request.model_type or "").lower().strip()
        if model_type not in _VALID_MODEL_TYPES:
            return models_pb2.SwitchModelResponse(
                ok=False,
                model_type=model_type or "",
                message=f"Unknown model type '{request.model_type}'.",
            )
        try:
            logger.info("gRPC request to switch model to '%s'", model_type)
            switch_model(model_type)
            os.environ["WINPROB_MODEL_TYPE"] = model_type
            return models_pb2.SwitchModelResponse(
                ok=True,
                model_type=model_type,
                message=f"Switched to {model_type}.",
            )
        except Exception as exc:
            logger.error("Model switch failed: %s", exc)
            return models_pb2.SwitchModelResponse(
                ok=False,
                model_type=model_type,
                message=str(exc),
            )

    async def GetCVSummary(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> models_pb2.CVSummaryResponse:
        """Return cross-validation results from the latest training run as JSON string."""
        for p in _CV_PATHS:
            if p.exists():
                data = json.loads(p.read_text())
                return models_pb2.CVSummaryResponse(json=json.dumps(data))
        return models_pb2.CVSummaryResponse(json="[]")
