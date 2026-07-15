from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from openpi_client import _pack_array, _protocol_imports, _unpack_array  # noqa: E402


def test_openpi_msgpack_round_trip_preserves_numpy_arrays():
    msgpack, _, ws_exceptions = _protocol_imports()
    value = {
        "image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "state": np.arange(16, dtype=np.float32),
    }
    encoded = msgpack.packb(value, default=_pack_array)
    decoded = msgpack.unpackb(encoded, object_hook=_unpack_array)
    assert np.array_equal(decoded["image"], value["image"])
    assert np.array_equal(decoded["state"], value["state"])
    assert issubclass(ws_exceptions.WebSocketException, Exception)
