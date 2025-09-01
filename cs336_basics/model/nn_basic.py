import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from einops import rearrange, einsum
from jaxtyping import Float, Int

from .initialize import init_linear_weights, init_embedding_weights


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weights: Float[Tensor, "d_out d_in"] = nn.Parameter(
            init_linear_weights((out_features, in_features),
                                in_features, out_features, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weights: Float[LongTensor, "vocab_size d_model"] = nn.Parameter(
            init_embedding_weights((num_embeddings, embedding_dim),
                                   device=device, dtype=dtype)
        )

    def forward(self, token_ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq d_model"]:
        return self.weights[token_ids]
