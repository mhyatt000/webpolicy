import asyncio
import socket
import threading
import time

import numpy as np
import websockets.asyncio.server

from webpolicy.base_policy import BasePolicy
from webpolicy.client import Client
from webpolicy.server import Server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(server: Server):
    loop = asyncio.new_event_loop()
    started = threading.Event()
    stop_server = asyncio.Event()

    async def _run():
        async with websockets.asyncio.server.serve(
            server._handler,
            server._host,
            server._port,
            compression=None,
            max_size=None,
        ):
            started.set()
            await stop_server.wait()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
        loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    started.wait(timeout=2)
    if not started.is_set():
        raise RuntimeError("Server failed to start")

    def _shutdown():
        loop.call_soon_threadsafe(stop_server.set)
        thread.join(timeout=2)

    return _shutdown


class DummyPolicy(BasePolicy):
    def __init__(self):
        self._num_calls = 0

    def step(self, obs: dict) -> dict:
        self._num_calls += 1
        return {
            "action": obs["obs"] + 1.0,
            "call_count": np.array([self._num_calls], dtype=np.int32),
        }

    def reset(self) -> None:
        self._num_calls = 0


def test_server_client_roundtrip_with_dummy_policy():
    port = _free_port()
    policy = DummyPolicy()
    server = Server(policy=policy, host="127.0.0.1", port=port, metadata={"policy": "dummy"})
    shutdown = _start_server(server)

    ws_client = None
    try:
        ws_client = Client(host="127.0.0.1", port=port)
        assert ws_client.get_server_metadata() == {"policy": "dummy"}

        obs = {"obs": np.array([1.0, 2.0], dtype=np.float32)}
        action1 = ws_client.step(obs)
        np.testing.assert_array_equal(action1["action"], np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(action1["call_count"], np.array([1], dtype=np.int32))

        action2 = ws_client.step(obs)
        np.testing.assert_array_equal(action2["action"], np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(action2["call_count"], np.array([2], dtype=np.int32))
    finally:
        if ws_client is not None:
            ws_client._ws.close()
        shutdown()
        time.sleep(0.05)
