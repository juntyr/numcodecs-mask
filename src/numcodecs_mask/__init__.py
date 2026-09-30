"""
Masking codecs for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["MaskMetaCodec", "MaskAwareCodecMixin"]

from collections.abc import Callable
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12

from .abc import MaskAwareCodecMixin


class MaskMetaCodec(Codec, CodecCombinatorMixin, MaskAwareCodecMixin):
    """
    Meta-codec that masks a value during encoding and restores it during decoding.

    The data is encoded with the `codec`, the [boolean][numpy.bool] mask of
    where values need to be restored is encoded with the `bitmap_codec`.

    Multiple [`MaskMetaCodec`][.]s can be nested to mask several values.
    The masked value can be replaced with a fill value before further encoding
    by including a
    [`numcodecs_replace.ReplaceFilterCodec`][numcodecs_replace.ReplaceFilterCodec]
    in the `codec`, e.g. by stacking using the
    [`numcodecs-combinators`](https://numcodecs-combinators.readthedocs.io)
    package.

    If the `codec` implements the
    [`MaskAwareCodecMixin`][numcodecs_mask.abc.MaskAwareCodecMixin], it is
    given the mask so that it can skip the masked values instead of encoding
    a fill value for them. The [`MaskMetaCodec`][.] implements the mixin
    itself, such that nested mask meta-codecs forward the union of their masks
    to the innermost codec.

    Parameters
    ----------
    mask : int | float
        The value to be masked.
    codec : dict | Codec
        The configuration or instantiated codec that encodes the data.
    bitmap_codec : dict | Codec
        The configuration or instantiated codec that encodes the
        [boolean][numpy.bool] mask of where values need to be restored.

        For instance, the [`numcodecs.PackBits`][numcodecs.packbits.PackBits]
        codec can be used to pack the mask into a byte array.
    """

    __slots__: tuple[str, ...] = ("_mask", "_codec", "_bitmap_codec")
    _mask: int | float
    _codec: Codec
    _bitmap_codec: Codec

    codec_id: str = "mask.meta"  # type: ignore

    def __init__(
        self,
        *,
        mask: int | float,
        codec: dict | Codec,
        bitmap_codec: dict | Codec,
    ) -> None:
        self._mask = mask
        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._bitmap_codec = (
            bitmap_codec
            if isinstance(bitmap_codec, Codec)
            else numcodecs.registry.get_codec(bitmap_codec)
        )

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return self._encode(buf, None)

    def encode_masked(
        self, buf: Buffer, mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]]
    ) -> Buffer:
        """Encode the data in `buf`, ignoring the values where `mask` is
        [`True`][True] (they are neither encoded nor restored by this codec).

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol. The values at masked positions are unspecified.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the data, of
            the values that do not need to be preserved.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return self._encode(buf, mask)

    def _encode(
        self,
        buf: Buffer,
        outer_mask: None | np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    ) -> bytes:
        a = np.copy(numcodecs.compat.ensure_ndarray(buf))
        dtype, shape = a.dtype, a.shape

        is_masked = self._is_masked(a)

        # message: dtype shape encoded-dtype encoded-shape [padding] encoded
        #          bitmap-dtype bitmap-shape [padding] bitmap
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        encoded_buf: Buffer
        if isinstance(self._codec, MaskAwareCodecMixin):
            inner_mask = is_masked if outer_mask is None else (is_masked | outer_mask)
            encoded_buf = self._codec.encode_masked(a, inner_mask)  # type: ignore
        else:
            encoded_buf = self._codec.encode(a)
        encoded = numcodecs.compat.ensure_ndarray(encoded_buf)

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        bitmap = self._bitmap_codec.encode(is_masked)
        bitmap = numcodecs.compat.ensure_ndarray(bitmap)

        message.append(leb128.u.encode(len(bitmap.dtype.str)))
        message.append(bitmap.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(bitmap.ndim))
        for s in bitmap.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with bitmap itemsize
        message.append(
            b"\0"
            * (bitmap.dtype.itemsize - (sum(len(m) for m in message) % bitmap.itemsize))
        )

        # ensure that the bitmap values are encoded in little endian binary
        message.append(bitmap.astype(bitmap.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return self._decode(buf, None, out)

    def decode_masked(
        self,
        buf: Buffer,
        mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        out: None | Buffer = None,
    ) -> Buffer:
        """
        Decode the data in `buf`, which was encoded with the same `mask`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the decoded
            data, that was passed to
            [`encode_masked`][numcodecs_mask.MaskMetaCodec.encode_masked].
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The values at masked positions are unspecified.
        """

        return self._decode(buf, mask, out)

    def _decode(
        self,
        buf: Buffer,
        outer_mask: None | np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        out: None | Buffer,
    ) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)
        b_io = BytesIO(b)

        # message: dtype shape encoded-dtype encoded-shape [padding] encoded
        #          bitmap-dtype bitmap-shape [padding] bitmap
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        bitmap_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        bitmap_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        bitmap_size = reduce(lambda a, b: a * b, bitmap_shape, 1)

        # remove padding to align with bitmap itemsize
        b_io.read(bitmap_dtype.itemsize - (b_io.tell() % bitmap_dtype.itemsize))

        bitmap = (
            np.frombuffer(
                b_io.read(bitmap_size * bitmap_dtype.itemsize),
                dtype=bitmap_dtype.newbyteorder("<"),
                count=bitmap_size,
            )
            .astype(bitmap_dtype)
            .reshape(bitmap_shape)
        )

        is_masked = np.empty(shape, dtype=np.bool)
        self._bitmap_codec.decode(bitmap, out=is_masked)

        decoded = np.empty(shape, dtype=dtype)
        if isinstance(self._codec, MaskAwareCodecMixin):
            inner_mask = is_masked if outer_mask is None else (is_masked | outer_mask)
            self._codec.decode_masked(encoded, inner_mask, out=decoded)  # type: ignore
        else:
            self._codec.decode(encoded, out=decoded)

        decoded[is_masked] = self._mask

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def _is_masked(
        self, a: np.ndarray
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        if isinstance(self._mask, int) or not np.isnan(self._mask):
            return a == self._mask  # type: ignore
        return np.isnan(a)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this mask meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this mask meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            mask=self._mask,
            codec=self._codec.get_config(),
            bitmap_codec=self._bitmap_codec.get_config(),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mask={self._mask!r}, codec={self._codec!r}, bitmap_codec={self._bitmap_codec!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "MaskMetaCodec":
        """
        Apply the `mapper` to this mask meta-codec.

        In the returned [`MaskMetaCodec`][..], the `codec` and
        `bitmap_codec` are replaced by their mapped codecs.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the
        [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(codec, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to the wrapped `codec` and
            `bitmap_codec` to map over this mask meta-codec.

        Returns
        -------
        mapped : MaskMetaCodec
            The mapped mask meta-codec.
        """

        return MaskMetaCodec(
            mask=self._mask,
            codec=mapper(self._codec),
            bitmap_codec=mapper(self._bitmap_codec),
        )


numcodecs.registry.register_codec(MaskMetaCodec)
