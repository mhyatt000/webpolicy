import asyncio
import rich
import logging
import traceback
import signal
from contextlib import contextmanager

from . import base_policy as _base_policy
from . import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


class TimeoutException(Exception):
    pass


@contextmanager
def _timeout_context(seconds: int) -> None:
    def _handler(signum, frame):
        raise TimeoutException(f"Policy step exceeded time limit of {seconds} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)

class Server:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `step` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        timeout: float | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self._timeout = timeout

    def start(self) -> None:
        self.serve()

    def serve(self) -> None:
        print(f"Starting server on ws://{self._host}:{self._port}")
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if self._timeout is not None:
                    with _timeout_context(self._timeout):
                        action = self._policy.step(obs)
                else:
                    action = self._policy.step(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
