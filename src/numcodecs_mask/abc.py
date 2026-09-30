"""
This module defines the [`MaskAwareCodecMixin`][numcodecs_mask.abc.MaskAwareCodecMixin] mixin, an interface for codecs that can make use of a mask of values that do not need to be preserved.
"""

__all__ = ["MaskAwareCodecMixin"]

from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Buffer  # MSPV 3.12


class MaskAwareCodecMixin(ABC):
    """
    Mixin class for [`Codec`][numcodecs.abc.Codec]s that can make use of a
    [boolean][numpy.bool] mask of values that do not need to be preserved.

    A meta-codec that restores masked values itself after decoding, such as
    the [`MaskMetaCodec`][numcodecs_mask.MaskMetaCodec], calls
    [`encode_masked`][numcodecs_mask.abc.MaskAwareCodecMixin.encode_masked]
    and [`decode_masked`][numcodecs_mask.abc.MaskAwareCodecMixin.decode_masked]
    instead of [`encode`][numcodecs.abc.Codec.encode] and
    [`decode`][numcodecs.abc.Codec.decode] on codecs that implement this
    mixin. Such codecs can then skip the masked values entirely, e.g. an
    entropy coder can avoid spending any bits on them and can treat them as
    missing when it conditions on neighbouring values.

    The values at masked positions are unspecified both in the data passed to
    [`encode_masked`][numcodecs_mask.abc.MaskAwareCodecMixin.encode_masked]
    and in the data produced by
    [`decode_masked`][numcodecs_mask.abc.MaskAwareCodecMixin.decode_masked].
    Meta-codecs that implement this mixin should forward the mask, combined
    with any values they mask themselves, to their inner codecs.
    """

    __slots__ = ()

    @abstractmethod
    def encode_masked(
        self, buf: Buffer, mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]]
    ) -> Buffer:
        """
        Encode the data in `buf`, ignoring the values where `mask` is
        [`True`][True].

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

    @abstractmethod
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
            [`encode_masked`][numcodecs_mask.abc.MaskAwareCodecMixin.encode_masked].
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The values at masked positions are unspecified.
        """
