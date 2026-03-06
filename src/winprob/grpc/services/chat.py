"""ChatService implementation: SendMessage (streaming), GetStatus.

Uses ChatEngine for session store and tool-calling loop. Streams assistant
content to the client; when the model uses tools, the engine handles that
internally and continues streaming the next turn.
"""

from __future__ import annotations

import logging
import os

import grpc

from winprob.grpc.generated.winprob.v1 import chat_pb2, chat_pb2_grpc, common_pb2
from winprob.llm.engine import ChatEngine

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")

# Single shared engine (session store is in-process)
_engine: ChatEngine | None = None


def _get_engine() -> ChatEngine:
    """Lazy-init the chat engine."""
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    """Implements ChatService: SendMessage (server-streaming), GetStatus."""

    async def SendMessage(
        self,
        request: chat_pb2.ChatRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Stream assistant reply; engine handles tool loop internally."""
        session_id = request.session_id or "default"
        message = (request.message or "").strip()
        model = request.model or _DEFAULT_MODEL
        if not message:
            yield chat_pb2.ChatResponse(content="Send a non-empty message.", done=True)
            return

        engine = _get_engine()
        try:
            async for chunk in engine.stream_turn(session_id, message, model=model or None):
                if chunk:
                    yield chat_pb2.ChatResponse(content=chunk, done=False)
            yield chat_pb2.ChatResponse(content="", done=True)
        except Exception as e:
            logger.exception("SendMessage error")
            yield chat_pb2.ChatResponse(content=f"Error: {e!s}", done=True)

    async def GetStatus(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.ChatStatusResponse:
        """Return Ollama availability, default model, and session count."""
        engine = _get_engine()
        available = await engine.ollama_available()
        return chat_pb2.ChatStatusResponse(
            ollama_available=available,
            model=_DEFAULT_MODEL,
            session_count=engine.session_count(),
        )
