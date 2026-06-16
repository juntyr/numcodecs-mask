import json
from typing import ClassVar

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
import pytest
from numcodecs.abc import Codec


def test_from_config():
    config = dict(
        id="mask.meta",
        mask=np.nan,
        codec=dict(id="zlib", level=1),
        bitmap_codec=dict(id="packbits"),
    )

    codec = numcodecs.registry.get_codec(config)
    assert codec.__class__.__name__ == "MaskMetaCodec"
    assert codec.__class__.__module__ == "numcodecs_mask"

    assert (
        repr(codec)
        == "MaskMetaCodec(mask=nan, codec=Zlib(level=1), bitmap_codec=PackBits())"
    )

    assert json.dumps(codec.get_config(), sort_keys=True) == json.dumps(
        config, sort_keys=True
    )


def check_roundtrip(data: np.ndarray, mask: int | float):
    config = dict(
        id="mask.meta",
        mask=mask,
        codec=dict(id="noise", seed=42),
        bitmap_codec=dict(id="packbits"),
    )

    codec = numcodecs.registry.get_codec(config)

    assert (
        repr(codec)
        == f"MaskMetaCodec(mask={mask!r}, codec=NoiseCodec(seed=42), bitmap_codec=PackBits())"
    )

    assert json.dumps(codec.get_config(), sort_keys=True) == json.dumps(
        config, sort_keys=True
    )

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    if isinstance(mask, int) or not np.isnan(mask):
        assert np.all((decoded == mask) | (data != mask))
    else:
        assert np.all(np.isnan(decoded) | ~np.isnan(data))


@np.errstate(invalid="ignore")
@pytest.mark.parametrize(
    "mask", [0.0, -0.0, 1.0, -1.0, np.nan, -np.nan, np.inf, -np.inf, 4.2, -2.4]
)
def test_roundtrip(mask):
    check_roundtrip(np.zeros(tuple()), mask=mask)
    check_roundtrip(np.zeros((0,)), mask=mask)
    if np.isfinite(mask):
        check_roundtrip(np.arange(1000).reshape(10, 10, 10), mask=mask)
    check_roundtrip(np.array([4.2, -2.4, np.nan, -np.nan, 0.0, -0.0]), mask=mask)
    check_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]), mask=mask)
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder("<"),
        ),
        mask=mask,
    )
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder(">"),
        ),
        mask=mask,
    )


class NoiseCodec(Codec):
    __slots__: tuple[str, ...] = ("_seed",)
    _seed: int

    codec_id: ClassVar[str] = "noise"  # type: ignore

    def __init__(self, *, seed: int):
        self._seed = seed

    def encode(self, buf):
        rng = np.random.Generator(np.random.PCG64(seed=self._seed))
        return buf + rng.normal(scale=0.1, size=buf.shape).astype(buf.dtype)

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(buf, out)

    def get_config(self) -> dict:
        return dict(id=type(self).codec_id, seed=self._seed)

    def __repr__(self) -> str:
        return f"NoiseCodec(seed={self._seed!r})"


numcodecs.registry.register_codec(NoiseCodec)
