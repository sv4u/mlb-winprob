"""AdminService implementation: GetStatus, RunPipeline (server-streaming)."""

from __future__ import annotations

import asyncio
import json
import logging

import grpc

from winprob.app.admin import (
    PipelineKind,
    conflicting_pipeline,
    gather_data_status,
    gather_model_status,
    get_state,
    run_pipeline,
)
from winprob.app.data_cache import get_git_commit
from winprob.grpc.generated.winprob.v1 import admin_pb2, common_pb2, admin_pb2_grpc

logger = logging.getLogger(__name__)

_VALID_PIPELINE_KINDS = {"ingest", "update", "retrain"}


def _state_to_proto(state) -> common_pb2.PipelineState:
    """Convert admin PipelineState dataclass to common_pb2.PipelineState."""
    d = state.to_dict()
    ps = common_pb2.PipelineState(
        kind=d.get("kind", ""),
        status=d.get("status", ""),
        log_tail=d.get("log_tail", [])[-80:],
        log_line_count=d.get("log_line_count", 0),
    )
    if d.get("started_at"):
        ps.started_at = d["started_at"]
    if d.get("finished_at"):
        ps.finished_at = d["finished_at"]
    if d.get("elapsed_seconds") is not None:
        ps.elapsed_seconds = d["elapsed_seconds"]
    if d.get("error"):
        ps.error = d["error"]
    return ps


class AdminServicer(admin_pb2_grpc.AdminServiceServicer):
    """Implements AdminService RPCs using the admin pipeline and status helpers."""

    async def GetStatus(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> admin_pb2.AdminStatusResponse:
        """Full system status: data coverage, model inventory, pipeline states."""
        payload = {
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
        return admin_pb2.AdminStatusResponse(json=json.dumps(payload))

    async def RunPipeline(
        self,
        request: admin_pb2.RunPipelineRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Run ingest/update/retrain pipeline; stream log lines and final state."""
        kind_str = (request.kind or "").lower().strip()
        if kind_str not in _VALID_PIPELINE_KINDS:
            yield admin_pb2.PipelineProgress(
                final_state=common_pb2.PipelineState(
                    kind=kind_str or "unknown",
                    status="failed",
                    error=f"Unknown pipeline kind '{request.kind}'.",
                )
            )
            return
        kind = PipelineKind(kind_str)

        blocker = conflicting_pipeline()
        if blocker is not None:
            err_state = common_pb2.PipelineState(
                kind=kind_str,
                status="failed",
                error=f"Cannot start {kind_str} — {blocker.value} pipeline is running.",
            )
            yield admin_pb2.PipelineProgress(final_state=err_state)
            return

        state = get_state(kind)
        last_sent = 0
        task = asyncio.create_task(run_pipeline(kind))

        while True:
            await asyncio.sleep(0.05)
            state = get_state(kind)
            for i in range(last_sent, len(state.log_lines)):
                yield admin_pb2.PipelineProgress(
                    log_line=state.log_lines[i]
                )
            last_sent = len(state.log_lines)
            if task.done():
                try:
                    await task
                except Exception:
                    pass
                state = get_state(kind)
                yield admin_pb2.PipelineProgress(
                    final_state=_state_to_proto(state)
                )
                return
