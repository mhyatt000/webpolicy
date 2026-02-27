import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from . import base_policy as _base_policy
from . import msgpack_numpy

_OP_KEY = "__webpolicy_op__"
_OP_STEP = "step"
_OP_RESET = "reset"
_ACTION_KEY = "action"
_RESET_ACK_KEY = "__webpolicy_reset_ack__"


class Client(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def step(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack({_OP_KEY: _OP_STEP, "obs": obs})
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        unpacked = msgpack_numpy.unpackb(response)

        # Preferred format from updated servers.
        if isinstance(unpacked, dict) and _ACTION_KEY in unpacked:
            return unpacked[_ACTION_KEY]

        # Backward compatibility for old servers that return raw action dicts.
        return unpacked

    @override
    def reset(self, payload: Dict | None = None) -> None:
        payload = payload or {}

        reset_request = {_OP_KEY: _OP_RESET}
        if payload:
            reset_request["payload"] = payload

        self._ws.send(self._packer.pack(reset_request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server reset:\n{response}")

        unpacked = msgpack_numpy.unpackb(response)
        if not isinstance(unpacked, dict) or unpacked.get(_RESET_ACK_KEY) is not True:
            raise RuntimeError(f"Unexpected reset response from server: {unpacked!r}")
