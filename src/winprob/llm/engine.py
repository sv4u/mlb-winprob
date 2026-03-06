"""Chat engine: session store, Ollama HTTP client, tool-calling loop, streaming.

Per-session history (20 message cap), 50-session LRU. Streams assistant
content; when the model returns tool_calls, runs tools and continues streaming
the next turn (up to 2 tool rounds). Graceful degradation when Ollama is down.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict

import aiohttp

from winprob.llm.context import build_system_prompt
from winprob.llm.tools import TOOL_SCHEMAS, run_tool

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
_MAX_SESSION_MESSAGES = 20
_MAX_SESSIONS = 50
_MAX_TOOL_ROUNDS = 2
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=10)

# Keyword fallback: if tool-calling fails, map user text to a tool name
_FALLBACK_PATTERNS = [
    (r"\b(odds|bet|ev|edge|kelly|wager)\b", "find_ev_bets"),
    (r"\b(why|explain|factor|shap|driver)\b", "explain_prediction"),
    (r"\b(standings|rank|division|playoff)\b", "get_standings"),
    (r"\b(drift|accuracy|calibration)\b", "get_drift_metrics"),
]


def _fallback_tool_for_message(text: str) -> str | None:
    """Return a suggested tool name from keyword matching, or None."""
    lower = (text or "").lower()
    for pattern, tool in _FALLBACK_PATTERNS:
        if re.search(pattern, lower):
            return tool
    return None


class ChatEngine:
    """Ollama chat with session history and tool-calling loop."""

    def __init__(
        self,
        *,
        ollama_host: str = OLLAMA_HOST,
        default_model: str = _DEFAULT_MODEL,
        max_session_messages: int = _MAX_SESSION_MESSAGES,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        self.ollama_host = ollama_host.rstrip("/")
        self.default_model = default_model
        self.max_session_messages = max_session_messages
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[dict]] = OrderedDict()

    def _touch_session(self, session_id: str) -> None:
        """Move session to end (LRU)."""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)

    def _get_or_create_messages(self, session_id: str) -> list[dict]:
        """Get message list for session; evict oldest if at capacity."""
        if session_id not in self._sessions:
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = []
        self._touch_session(session_id)
        return self._sessions[session_id]

    def append_user(self, session_id: str, content: str) -> None:
        """Append a user message and trim to max_session_messages."""
        messages = self._get_or_create_messages(session_id)
        messages.append({"role": "user", "content": content or ""})
        while len(messages) > self.max_session_messages:
            messages.pop(0)

    def append_assistant_and_tool(
        self,
        session_id: str,
        assistant_content: str,
        tool_calls: list[dict],
        tool_results: list[tuple[str, str]],
    ) -> None:
        """Append assistant message (with tool_calls) and tool result messages."""
        messages = self._get_or_create_messages(session_id)
        msg: dict = {"role": "assistant", "content": assistant_content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
        for tc_id, result in tool_results:
            messages.append({"role": "tool", "content": result, "tool_call_id": tc_id})
        while len(messages) > self.max_session_messages:
            messages.pop(0)

    def append_assistant_only(self, session_id: str, content: str) -> None:
        """Append assistant message only (final turn, no tools)."""
        messages = self._get_or_create_messages(session_id)
        messages.append({"role": "assistant", "content": content or ""})
        while len(messages) > self.max_session_messages:
            messages.pop(0)

    def session_count(self) -> int:
        """Return number of active sessions."""
        return len(self._sessions)

    async def ollama_available(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.ollama_host}/api/tags") as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def stream_turn(
        self,
        session_id: str,
        user_message: str,
        model: str | None = None,
    ):
        """Async generator: yield assistant content chunks. Handles tool loop internally."""
        model = model or self.default_model
        self.append_user(session_id, user_message)
        messages = self._get_or_create_messages(session_id)
        system = build_system_prompt()
        payload_messages = [{"role": "system", "content": system}] + messages
        round_count = 0

        while round_count <= _MAX_TOOL_ROUNDS:
            payload = {
                "model": model,
                "messages": payload_messages,
                "stream": True,
                "tools": TOOL_SCHEMAS,
            }
            accumulated = ""
            tool_calls_this_turn: list[dict] = []

            try:
                async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
                    async with session.post(
                        f"{self.ollama_host}/api/chat",
                        json=payload,
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            yield f"Ollama error {resp.status}: {text[:200]}"
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
                                msg = (
                                    obj.get("message", {})
                                    if isinstance(obj.get("message"), dict)
                                    else {}
                                )
                                content_delta = msg.get("content") or obj.get("response") or ""
                                if content_delta:
                                    accumulated += content_delta
                                    yield content_delta
                                done = obj.get("done", False)
                                if done and msg.get("tool_calls"):
                                    tool_calls_this_turn = msg["tool_calls"]

            except aiohttp.ClientError as e:
                logger.warning("Ollama HTTP error: %s", e)
                yield "Ollama is unreachable. Start Ollama or set OLLAMA_HOST."
                return
            except Exception as e:
                logger.exception("Chat engine error")
                yield f"Error: {e!s}"
                return

            if not tool_calls_this_turn:
                self.append_assistant_only(session_id, accumulated)
                return

            round_count += 1
            if round_count > _MAX_TOOL_ROUNDS:
                self.append_assistant_only(session_id, accumulated)
                return

            tool_results: list[tuple[str, str]] = []
            for tc in tool_calls_this_turn:
                tc_id = tc.get("id") or ""
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(name, args)
                tool_results.append((tc_id, result))

            self.append_assistant_and_tool(
                session_id,
                accumulated,
                tool_calls_this_turn,
                tool_results,
            )
            payload_messages = [{"role": "system", "content": system}] + self._get_or_create_messages(session_id)
