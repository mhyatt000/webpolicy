# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`webpolicy` is a lightweight WebSocket-based server/client library for serving robot policies. It is a port of the openpi web server/client without the model weights — fewer dependencies, same protocol. Observations (numpy arrays) are sent from client to server; actions (numpy arrays) are returned.

## Commands

```bash
# Install (editable, with dev deps)
uv sync

# Run tests
uv run pytest src/webpolicy/

# Run a single test file
uv run pytest src/webpolicy/msgpack_numpy_test.py

# Start a dummy policy server (for testing)
uv run python scripts/serve_test.py --host 0.0.0.0 --port 8000

# Run a test client against the server
uv run python scripts/client_test.py --host 0.0.0.0 --port 8000 --env aloha_sim --num_steps 10
```

## Architecture

**Core communication layer** (`src/webpolicy/`):
- `base_policy.py` — Abstract `BasePolicy` with `step(obs: dict) -> dict` and `reset()`. All policies implement this.
- `server.py` — `Server` wraps a `BasePolicy`, runs an async WebSocket server. On connect, sends metadata; then loops: receives msgpack-encoded obs, calls `policy.step()`, sends back msgpack-encoded action. Errors are sent as a string frame before closing with `INTERNAL_ERROR`.
- `client.py` — `Client` also implements `BasePolicy`. Connects synchronously, retrying until the server is up. `step()` sends obs and receives action over the same websocket.
- `msgpack_numpy.py` — Custom msgpack extension that serializes `np.ndarray` and `np.generic` as typed byte buffers. Used for all wire communication. Does not fall back to pickle.
- `action_chunk_broker.py` — `ActionChunkBroker` wraps a policy to cache a chunk of actions and return them one-at-a-time, only re-calling the inner policy when the chunk is exhausted.
- `image_tools.py` — Utilities: `convert_to_uint8` and `resize_with_pad` (PIL-based, replicates `tf.image.resize_with_pad`).

**Runtime layer** (`src/webpolicy/runtime/`):
- `runtime.py` — `Runtime` orchestrates an episode loop: calls `environment.reset()`, then steps `environment → agent → action → environment` at a configurable Hz for N episodes.
- `agent.py` / `environment.py` / `subscriber.py` — Abstract base classes for the runtime components.
- `agents/policy_agent.py` — `PolicyAgent` bridges `BasePolicy` → `Agent` (calls `policy.infer()` instead of `policy.step()`).

**Key design note:** `Server`/`Client` use `policy.step()`, while the runtime's `PolicyAgent` calls `policy.infer()`. These are two different method names on `BasePolicy` subclasses — take care when implementing a policy that needs to work with both.

## Serialization

All observation and action dicts are serialized with the custom `msgpack_numpy` module. Values must be `np.ndarray` or plain Python types (no object arrays, void, or complex dtypes). Nested dicts are supported.
