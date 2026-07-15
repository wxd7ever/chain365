"""Bounded-time WebSocket client for the OpenPI remote policy protocol."""

from __future__ import annotations

import functools
from typing import Any, Mapping

import numpy as np


def _protocol_imports():
    try:
        import msgpack
        import websockets
        import websockets.exceptions
        import websockets.sync.client
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI client dependencies are missing. Install them with "
            "`pip install 'msgpack>=1.0.5' 'websockets>=11'`."
        ) from exc
    return msgpack, websockets, websockets.exceptions


def _pack_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"OpenPI cannot serialize dtype {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"OpenPI cannot serialize {type(value).__name__}")


def _unpack_array(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=tuple(value[b"shape"]),
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class OpenPIWebsocketClient:
    """Connect once and reuse the same OpenPI socket across tasks and episodes."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 8000,
        api_key: str | None = None,
        connect_timeout_s: float = 15.0,
        infer_timeout_s: float = 120.0,
        max_retries: int = 1,
    ):
        if not host:
            raise ValueError("host must be non-empty")
        if port <= 0:
            raise ValueError("port must be positive")
        if connect_timeout_s <= 0 or infer_timeout_s <= 0:
            raise ValueError("OpenPI timeouts must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.api_key = api_key
        self.connect_timeout_s = float(connect_timeout_s)
        self.infer_timeout_s = float(infer_timeout_s)
        self.max_retries = int(max_retries)
        self._msgpack, self._websockets, self._ws_exceptions = _protocol_imports()
        self._packer = functools.partial(self._msgpack.packb, default=_pack_array)
        self._unpacker = functools.partial(
            self._msgpack.unpackb,
            object_hook=_unpack_array,
        )
        self._ws = None
        self.server_metadata: Mapping[str, Any] = {}
        self._connect()

    def _connect(self) -> None:
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        kwargs = {
            "compression": None,
            "max_size": None,
            "open_timeout": self.connect_timeout_s,
            "ping_interval": 120.0,
            "ping_timeout": 600.0,
        }
        try:
            self._ws = self._websockets.sync.client.connect(
                self.uri,
                additional_headers=headers,
                **kwargs,
            )
        except TypeError:
            self._ws = self._websockets.sync.client.connect(
                self.uri,
                extra_headers=headers,
                **kwargs,
            )
        metadata = self._ws.recv(timeout=self.connect_timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"OpenPI server returned text during handshake: {metadata}")
        unpacked = self._unpacker(metadata)
        if not isinstance(unpacked, Mapping):
            raise RuntimeError("OpenPI server metadata must be a mapping")
        self.server_metadata = dict(unpacked)

    def reset(self) -> None:
        """OpenPI's current protocol has no remote reset message."""

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("OpenPI observation must be a mapping")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self._ws is None:
                    self._connect()
                self._ws.send(self._packer(dict(observation)))
                response = self._ws.recv(timeout=self.infer_timeout_s)
                if isinstance(response, str):
                    raise RuntimeError(f"OpenPI inference server error: {response}")
                unpacked = self._unpacker(response)
                if not isinstance(unpacked, Mapping):
                    raise RuntimeError("OpenPI inference response must be a mapping")
                return dict(unpacked)
            except (OSError, TimeoutError, self._ws_exceptions.WebSocketException) as exc:
                last_error = exc
                self.close()
                if attempt >= self.max_retries:
                    break
        raise RuntimeError(
            f"OpenPI inference failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
