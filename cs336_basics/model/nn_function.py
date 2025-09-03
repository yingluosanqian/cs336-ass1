import torch
import torch.nn as nn
import math
from torch import Tensor, LongTensor, BoolTensor
from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    query: Float[Tensor, "... seq d_qk"],
    key: Float[Tensor, "... seq d_qk"],
    value: Float[Tensor, "... seq d_v"],
    *,
    attn_mask: Bool[Tensor, "... seq_q seq_kv"] | None = None,
) -> Float[Tensor, "... seq d_v"]:
    scale = math.sqrt(query.shape[-1])
    P = einsum(
        query, key, "... seq_q d_qk, ... seq_kv d_qk -> ... seq_q seq_kv") / scale
    P = P.masked_fill(~attn_mask, float(
        '-inf')) if attn_mask is not None else P
    S = softmax(P, dim=-1)
    return einsum(S, value, "... seq_q seq_kv, ... seq_kv d_v -> ... seq_q d_v")
