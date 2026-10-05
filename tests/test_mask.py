import json
from typing import ClassVar

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
import pytest
from numcodecs.abc import Codec

from numcodecs_mask.abc import MaskAwareCodecMixin


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


class SparseCodec(Codec, MaskAwareCodecMixin):
    """Test codec that only stores the unmasked values and records the masks
    it was given."""

    __slots__: tuple[str, ...] = ("masks",)

    codec_id: ClassVar[str] = "sparse-test"  # type: ignore

    def __init__(self):
        self.masks: list[np.ndarray] = []

    def encode(self, buf):
        return self.encode_masked(buf, np.zeros(np.shape(buf), dtype=np.bool))

    def decode(self, buf, out=None):
        return self.decode_masked(buf, np.zeros(np.shape(out), dtype=np.bool), out)

    def encode_masked(self, buf, mask):
        a = numcodecs.compat.ensure_ndarray(buf)
        assert mask.shape == a.shape and mask.dtype == np.bool
        self.masks.append(np.copy(mask))
        return a[~mask].tobytes()

    def decode_masked(self, buf, mask, out=None):
        assert out is not None and mask.shape == out.shape
        decoded = np.zeros(out.shape, dtype=out.dtype)
        decoded[~mask] = np.frombuffer(
            numcodecs.compat.ensure_bytes(buf), dtype=out.dtype
        )
        return numcodecs.compat.ndarray_copy(decoded, out)

    def get_config(self) -> dict:
        return dict(id=type(self).codec_id)


def test_mask_aware_codec():
    from numcodecs_mask import MaskMetaCodec

    data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 0.0]])

    inner = SparseCodec()
    codec = MaskMetaCodec(mask=np.nan, codec=inner, bitmap_codec=dict(id="packbits"))

    encoded = codec.encode(data)
    # only the unmasked values were passed on to the inner codec
    assert len(inner.masks) == 1
    np.testing.assert_array_equal(inner.masks[0], np.isnan(data))

    decoded = codec.decode(encoded)
    np.testing.assert_array_equal(decoded, data)


def test_nested_mask_aware_codecs():
    from numcodecs_mask import MaskMetaCodec

    data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 0.0], [0.0, 7.0, np.nan]])

    inner = SparseCodec()
    codec = MaskMetaCodec(
        mask=np.nan,
        codec=MaskMetaCodec(mask=0.0, codec=inner, bitmap_codec=dict(id="packbits")),
        bitmap_codec=dict(id="packbits"),
    )

    encoded = codec.encode(data)
    # the innermost codec receives the union of both masks
    assert len(inner.masks) == 1
    np.testing.assert_array_equal(inner.masks[0], np.isnan(data) | (data == 0.0))

    decoded = codec.decode(encoded)
    np.testing.assert_array_equal(decoded, data)


def test_mask_unaware_codec_unchanged():
    from numcodecs_mask import MaskMetaCodec

    # codecs without mask support are used exactly as before
    data = np.array([1.0, np.nan, 3.0])
    codec = MaskMetaCodec(
        mask=np.nan, codec=dict(id="zlib", level=1), bitmap_codec=dict(id="packbits")
    )
    np.testing.assert_array_equal(codec.decode(codec.encode(data)), data)


def test_structural_subclass():
    from numcodecs_mask import MaskMetaCodec
    from numcodecs_mask.abc import MaskAwareCodecMixin

    class StructuralSparseCodec(Codec):
        """Implements the protocol without inheriting from the mixin."""

        codec_id: ClassVar[str] = "structural-sparse-test"  # type: ignore

        def encode(self, buf):
            return self.encode_masked(buf, np.zeros(np.shape(buf), dtype=np.bool))

        def decode(self, buf, out=None):
            return self.decode_masked(buf, np.zeros(np.shape(out), dtype=np.bool), out)

        def encode_masked(self, buf, mask):
            a = numcodecs.compat.ensure_ndarray(buf)
            return a[~mask].tobytes()

        def decode_masked(self, buf, mask, out=None):
            assert out is not None and mask.shape == out.shape
            decoded = np.zeros(out.shape, dtype=out.dtype)
            decoded[~mask] = np.frombuffer(
                numcodecs.compat.ensure_bytes(buf), dtype=out.dtype
            )
            return numcodecs.compat.ndarray_copy(decoded, out)

        def get_config(self) -> dict:
            return dict(id=type(self).codec_id)

    assert issubclass(StructuralSparseCodec, MaskAwareCodecMixin)
    assert isinstance(StructuralSparseCodec(), MaskAwareCodecMixin)
    assert not issubclass(NoiseCodec, MaskAwareCodecMixin)

    data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 0.0]])
    codec = MaskMetaCodec(
        mask=np.nan,
        codec=StructuralSparseCodec(),
        bitmap_codec=dict(id="packbits"),
    )
    encoded = codec.encode(data)
    # 4 unmasked float64 values -> the sparse codec stored exactly those
    decoded = codec.decode(encoded)
    np.testing.assert_array_equal(decoded, data)
