"""Interactive REPL chat client for the MLB Win Probability ChatService.

Connects to the gRPC server and streams assistant replies. Use /status to
check Ollama availability and session count; /quit or Ctrl+D to exit.

Usage
-----
    python scripts/chat.py                    # default localhost:50051
    python scripts/chat.py --grpc-port 50052
    python scripts/chat.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root so winprob is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


async def _run(
    grpc_host: str,
    grpc_port: int,
    model: str | None,
    session_id: str,
) -> None:
    import grpc
    from winprob.grpc.generated.winprob.v1 import chat_pb2, chat_pb2_grpc, common_pb2

    channel = grpc.aio.insecure_channel(f"{grpc_host}:{grpc_port}")
    stub = chat_pb2_grpc.ChatServiceStub(channel)

    print("Chat (gRPC). Commands: /status, /quit")
    print("─" * 50)

    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line.startswith("/quit"):
            break
        if line.startswith("/status"):
            try:
                r = await stub.GetStatus(common_pb2.Empty())
                print(
                    f"  Ollama: {'ok' if r.ollama_available else 'unavailable'}  "
                    f"Model: {r.model or 'default'}  Sessions: {r.session_count}"
                )
            except Exception as e:
                print(f"  Error: {e}")
            continue

        req = chat_pb2.ChatRequest(message=line, session_id=session_id)
        if model:
            req.model = model
        print("Assistant: ", end="", flush=True)
        try:
            stream = stub.SendMessage(req)
            async for chunk in stream:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
        except grpc.RpcError as e:
            print(f"\n  RPC error: {e.code()} — {e.details()}")

    await channel.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive chat with MLB Win Probability (gRPC).")
    ap.add_argument("--grpc-host", default="127.0.0.1", help="gRPC server host")
    ap.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port")
    ap.add_argument("--model", default=None, help="Ollama model (e.g. qwen2.5:3b)")
    ap.add_argument("--session", default="cli", help="Session ID for conversation history")
    args = ap.parse_args()
    asyncio.run(_run(args.grpc_host, args.grpc_port, args.model, args.session))


if __name__ == "__main__":
    main()
