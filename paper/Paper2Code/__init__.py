"""DeepSeek-V4.1-Flash reference implementation.

Faithful PyTorch reference for the architectural contributions of
"DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression".
"""

from .config import DeepSeekV41Config
from .model import DeepSeekV41Flash
from .ced import CED, TransformerBlock, SWAAttention
from .csa2 import CSA2Layer, CSA2State
from .moe import DeepSeekMoE
from .mhc import mHCBlock
from .engram import Engram
from .dspark import DSpark
from .vision import DeepSeekViT, PixelUnshuffle, MLPProjector
from .fp4 import quantize_kv, dequantize_e2m1
from .kv_cache import GlobalKVCache, SWACache

__all__ = [
    "DeepSeekV41Config", "DeepSeekV41Flash", "CED", "TransformerBlock",
    "SWAAttention", "CSA2Layer", "CSA2State", "DeepSeekMoE", "mHCBlock",
    "Engram", "DSpark", "DeepSeekViT", "PixelUnshuffle", "MLPProjector",
    "quantize_kv", "dequantize_e2m1", "GlobalKVCache", "SWACache",
]
