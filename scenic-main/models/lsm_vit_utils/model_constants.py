"""Vit Model Configuration Constants."""

from typing import Dict


HIDDEN_SIZES: Dict[str, int] = {
    'Deb': 16,
    'Ti': 192,
    'TiShallow': 192,
    'XS': 256,
    'S': 384,
    'SShallow': 384,
    'M': 512,
    'B': 768,
    'L': 1024,
    'H': 1280,
    'g': 1408,
    'G': 1664,
    'e': 1792,
}
MLP_DIMS: Dict[str, int] = {
    'Deb': 32,
    'Ti': 768,
    'TiShallow': 768,
    'XS': 1024,
    'S': 1536,
    'SShallow': 1536,
    'M': 2048,
    'B': 3072,
    'L': 4096,
    'H': 5120,
    'g': 6144,
    'G': 8192,
    'e': 15360,
}
NUM_HEADS: Dict[str, int] = {
    'Deb': 2,
    'Ti': 3,
    'TiShallow': 3,
    'XS': 4,
    'S': 6,
    'SShallow': 6,
    'M': 8,
    'B': 12,
    'L': 16,
    'H': 16,
    'g': 16,
    'G': 16,
    'e': 16,
}
NUM_LAYERS: Dict[str, int] = {
    'Deb': 2,
    'Ti': 12,
    'TiShallow': 4,
    'XS': 8,
    'S': 12,
    'SShallow': 4,
    'M': 12,
    'B': 12,
    'L': 24,
    'H': 32,
    'g': 40,
    'G': 48,
    'e': 56,
}


DECODER_HIDDEN_SIZES: Dict[str, int] = {
    'Deb': 16,
    'Ti': 128,
    'TiShallow': 128,
    'XS': 192,
    'S': 256,
    'M': 384,
    'B': 512,
    'L': 512,
    'H': 512,
}
DECODER_MLP_DIMS: Dict[str, int] = {
    'Deb': 32,
    'Ti': 512,
    'TiShallow': 512,
    'XS': 768,
    'S': 1024,
    'M': 1536,
    'B': 2048,
    'L': 2048,
    'H': 2048,
}
DECODER_NUM_LAYERS: Dict[str, int] = {
    'Deb': 2,
    'Ti': 2,
    'TiShallow': 2,
    'XS': 2,
    'S': 4,
    'M': 4,
    'B': 8,
    'L': 8,
    'H': 8,
}
DECODER_NUM_HEADS: Dict[str, int] = {
    'Deb': 2,
    'Ti': 4,
    'TiShallow': 4,
    'XS': 4,
    'S': 8,
    'M': 8,
    'B': 16,
    'L': 16,
    'H': 16,
}
