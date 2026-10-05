[![image](https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-mask/ci.yml?branch=main)](https://github.com/juntyr/numcodecs-mask/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-mask.svg)](https://pypi.python.org/pypi/numcodecs-mask)
[![image](https://img.shields.io/pypi/l/numcodecs-mask.svg)](https://github.com/juntyr/numcodecs-mask/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjuntyr%2Fnumcodecs-mask%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-mask)
[![image](https://readthedocs.org/projects/numcodecs-mask/badge/?version=latest)](https://numcodecs-mask.readthedocs.io/en/latest/?badge=latest)

# numcodecs-mask

Masking codecs for the [`numcodecs`] buffer compression API.

The `MaskMetaCodec` masks a value (e.g. NaN) during encoding and restores it during decoding. Inner codecs that implement the `MaskAwareCodecMixin` are given the mask and can skip the masked values entirely (e.g. entropy coders can avoid spending bits on them and treat them as missing in their context modelling); nested `MaskMetaCodec`s forward the union of their masks.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-mask` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
