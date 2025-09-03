import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from einops import rearrange, einsum
from jaxtyping import Float, Int

from .initialize import init_linear_weights, init_embedding_weights, init_rmsnorm_weights


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
        self.weights: Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            init_embedding_weights((num_embeddings, embedding_dim),
                                   device=device, dtype=dtype)
        )

    def forward(self, token_ids: Int[LongTensor, "... seq"]) -> Float[Tensor, "... seq d_model"]:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights: Float[Tensor, "d_model"] = nn.Parameter(
            init_rmsnorm_weights((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Cast to FP32 first
        in_type = x.dtype
        x = x.to(torch.float32)

        rs = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(rs / self.d_model + self.eps)

        result = x / rms * self.weights
        return result.to(in_type)


class SiLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.silu = SiLu()
        self.w1_weight: Float[Tensor, "d_ff d_model"] = nn.Parameter(
            init_linear_weights((d_ff, d_model), d_model,
                                d_ff, device=device, dtype=dtype)
        )
        self.w2_weight: Float[Tensor, "d_model d_ff"] = nn.Parameter(
            init_linear_weights((d_model, d_ff), d_ff,
                                d_model, device=device, dtype=dtype)
        )
        self.w3_weight: Float[Tensor, "d_ff d_model"] = nn.Parameter(
            init_linear_weights((d_ff, d_model), d_model,
                                d_ff, device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        x_w1 = einsum(x, self.w1_weight, "... d_model, dff d_model -> ... dff")
        x_w3 = einsum(x, self.w3_weight, "... d_model, dff d_model -> ... dff")
        return einsum(self.silu(x_w1) * x_w3, self.w2_weight, "... dff, d_model dff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.cos_value: Float[Tensor, "max_seq_len half_d_k"]
        self.sin_value: Float[Tensor, "max_seq_len half_d_k"]
        self.d_k = d_k
        if d_k % 2 > 0:
            raise ValueError("Rotary positional embedding requires even d_k")
        pos_id = torch.arange(max_seq_len, device=device).float()
        dim_id = torch.arange(0, d_k, 2, device=device).float()
        theta_exp = torch.pow(theta, dim_id / d_k)

        pos_id = rearrange(pos_id, "max_seq_len -> max_seq_len 1")
        dim_id = rearrange(dim_id, "half_d_k -> 1 half_d_k")
        angles = pos_id / theta_exp
        cos_value: Float[Tensor, "max_seq_len half_d_k"] = torch.cos(angles)
        sin_value: Float[Tensor, "max_seq_len half_d_k"] = torch.sin(angles)
        # Save cos_value and sin_value
        self.register_buffer("cos_value", cos_value, persistent=False)
        self.register_buffer("sin_value", sin_value, persistent=False)

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d_k"]:
        d_k = x.shape[-1]
        if d_k != self.d_k:
            raise ValueError(
                f"Input d_k ({d_k}) does not match layer d_k ({self.d_k})")
        cos_value: Float[Tensor,
                         "seq half_d_k"] = self.cos_value[token_positions]
        sin_value: Float[Tensor,
                         "seq half_d_k"] = self.sin_value[token_positions]

        x = rearrange(
            x, "... seq (half_d_k pair) -> ... seq half_d_k pair", pair=2)
        x0: Float[Tensor, "... seq half_d_k"] = x[..., 0]
        x1: Float[Tensor, "... seq half_d_k"] = x[..., 1]

        rotated_x0: Float[Tensor,
                          "... seq half_d_k"] = cos_value * x0 - sin_value * x1
        rotated_x1: Float[Tensor,
                          "... seq half_d_k"] = sin_value * x0 + cos_value * x1

        rope_x = rearrange([rotated_x0, rotated_x1],
                           "pair ... seq half_d_k -> ... seq (half_d_k pair)", pair=2)
        return rope_x
