import importlib.metadata

from .tokenizer import train_bpe, Tokenizer
from .model import nn_basic

# __version__ = importlib.metadata.version("cs336_basics")
__version__ = "0.1.0"

__all__ = [
    # Tokenizer
    'train_bpe',
    'Tokenizer',
    # Model
    'nn_basic',
    '__version__'
]
