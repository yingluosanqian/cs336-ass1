import importlib.metadata

from .tokenizer import train_bpe, Tokenizer

# __version__ = importlib.metadata.version("cs336_basics")
__version__ = "0.1.0"

__all__ = [
    'train_bpe',
    'Tokenizer',
    '__version__'
]
