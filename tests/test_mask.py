import json

import numcodecs
import numcodecs.registry
import numpy as np


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
