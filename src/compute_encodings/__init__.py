"Class to compute encodings and add to the data"

from .base_class import EncodingsSaverBase
from .encoding_saver_hgraphs import EncodingsSaver
from .encoding_saver_lrgb import EncodingsSaverLRGB
from .encoding_saver_cc_ca import EncodingsSaverForCCCA

__all__ = [
    "EncodingsSaverBase",
    "EncodingsSaver",
    "EncodingsSaverLRGB",
    "EncodingsSaverForCCCA",
]
