"""OllamaService implementation: translate gRPC to Ollama HTTP API, stream responses."""

from __future__ import annotations

import json
import logging
import os

import grpc
import aiohttp

from winprob.grpc.generated.winprob.v1 import chat_pb2, chat_pb2_grpc, common_pb2

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=10)


async def _ollama_available(session: aiohttp.ClientSession) -> bool:
    """Return True if Ollama is reachable (e.g. GET /api/tags succeeds)."""
    try:
        async with session.get(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


class OllamaAdapterServicer(chat_pb2_grpc.OllamaServiceServicer):
    """Translates gRPC OllamaService to Ollama HTTP API; streams responses as protobuf."""

    async def Chat(
        self,
        request: chat_pb2.OllamaChatRequest,
        context: grpc.aio.ServicerContext,
    ):
        """Stream chat completion from Ollama; yield OllamaChatResponse per chunk."""
        model = request.model or "qwen2.5:3b"
        stream = request.stream
        try:
            messages = json.loads(request.messages_json) if request.messages_json else []
        except json.JSONDecodeError:
            yield chat_pb2.OllamaChatResponse(
                content="Invalid messages_json.",
                done=True,
            )
            return

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if request.HasField("tools_json") and request.tools_json:
            try:
                payload["tools"] = json.loads(request.tools_json)
            except json.JSONDecodeError:
                pass

        try:
            async with aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT) as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        yield chat_pb2.OllamaChatResponse(
                            content=f"Ollama error {resp.status}: {text[:200]}",
                            done=True,
                        )
                        return

                    if not stream:
                        data = await resp.json()
                        msg = data.get("message", {})
                        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        yield chat_pb2.OllamaChatResponse(
                            content=content or "",
                            done=True,
                        )
                        return

                    buf = b""
                    async for chunk in resp.content.iter_chunked(8192):
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line.decode("utf-8"))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                            msg = obj.get("message", {}) if isinstance(obj.get("message"), dict) else {}
                            chunk_text = msg.get("content") or obj.get("response") or ""
                            done = obj.get("done", False)
                            tool_calls = None
                            if done and msg and "tool_calls" in msg and msg["tool_calls"]:
                                tool_calls = json.dumps(msg["tool_calls"])
                            out = chat_pb2.OllamaChatResponse(
                                content=chunk_text,
                                done=done,
                            )
                            if tool_calls:
                                out.tool_calls_json = tool_calls
                            yield out
                            if done:
                                return

        except aiohttp.ClientError as e:
            logger.warning("Ollama HTTP error: %s", e)
            yield chat_pb2.OllamaChatResponse(
                content="Ollama is unreachable. Start Ollama (e.g. docker run ollama/ollama) or set OLLAMA_HOST.",
                done=True,
            )
        except Exception as e:
            logger.exception("Ollama adapter error")
            yield chat_pb2.OllamaChatResponse(
                content=f"Error: {e!s}",
                done=True,
            )

    async def ListModels(
        self,
        request: common_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.ListModelsResponse:
        """List available Ollama models via GET /api/tags."""
        try:
            async with aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT) as session:
                async with session.get(f"{OLLAMA_HOST}/api/tags") as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return chat_pb2.ListModelsResponse(json=text)
        except Exception as e:
            logger.warning("Ollama list models failed: %s", e)
        return chat_pb2.ListModelsResponse(json='{"models":[]}')
