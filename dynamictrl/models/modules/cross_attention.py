import math
from typing import Tuple, Union, Optional, Any

import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import einops
from einops import repeat

from diffusers import AutoencoderKL
from timm.models.vision_transformer import Mlp
from timm.models.layers import to_2tuple
from transformers import (
    AutoTokenizer,
    MT5EncoderModel,
    BertModel,
)

memory_efficient_attention = None
try:
    import xformers
except:
    pass

try:
    from xformers.ops import memory_efficient_attention
except:
    memory_efficient_attention = None

def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

MEMORY_LAYOUTS = {
    "torch": (
        lambda x, head_dim: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
        lambda x: (1, x, 1, 1),
    ),
    "xformers": (
        lambda x, head_dim: x,
        lambda x: x,
        lambda x: (1, 1, x, 1),
    ),
    "math": (
        lambda x, head_dim: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
        lambda x: (1, x, 1, 1),
    ),
}

def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: Optional[torch.Tensor],
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Precomputed frequency tensor for complex exponentials.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        if xk is not None:
            xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        if xk is not None:
            xk_ = torch.view_as_complex(
                xk.float().reshape(*xk.shape[:-1], -1, 2)
            )  # [B, S, H, D//2]
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out

def vanilla_attention(q, k, v, mask, dropout_p, scale=None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = einops.rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = einops.repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    if dropout_p != 0:
        scores = F.dropout(p_attn, p=dropout_p, training=True)
    return torch.bmm(p_attn, v)

def attention(q, k, v, head_dim, dropout_p=0, mask=None, scale=None, mode="xformers"):
    """
    q, k, v: [B, L, H, D]
    """
    pre_attn_layout = MEMORY_LAYOUTS[mode][0]
    post_attn_layout = MEMORY_LAYOUTS[mode][1]
    q = pre_attn_layout(q, head_dim)
    k = pre_attn_layout(k, head_dim)
    v = pre_attn_layout(v, head_dim)

    # scores = ATTN_FUNCTION[mode](q, k.to(q), v.to(q), mask, scale=scale)
    if mode == "torch":
        assert scale is None
        scores = F.scaled_dot_product_attention(
            q, k.to(q), v.to(q), mask, dropout_p
        )  # , scale=scale)
    elif mode == "xformers":
        scores = memory_efficient_attention(
            q, k.to(q), v.to(q), mask, dropout_p, scale=scale
        )
    else:
        scores = vanilla_attention(q, k.to(q), v.to(q), mask, dropout_p, scale=scale)

    scores = post_attn_layout(scores)
    return scores


class CrossAttention(nn.Module):
    """
    Use QK Normalization.
    """

    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
        attn_mode="xformers",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 192
        ), "Only support head_dim <= 128 and divisible by 8"

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )

        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.attn_mode = attn_mode

    def set_attn_mode(self, mode):
        self.attn_mode = mode

    def forward(self, x, y, freqs_cis_img=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num_heads * head_dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // num_heads), RoPE for image
        """
        b, s1, _ = x.shape
        _, s2, _ = y.shape

        q = self.q_proj(x).view(b, s1, self.num_heads, self.head_dim)
        kv = self.kv_proj(y).view(
            b, s2, 2, self.num_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        q = self.q_norm(q).to(q)
        k = self.k_norm(k).to(k)

        # Apply RoPE if needed
        if freqs_cis_img is not None:
            qq, _ = apply_rotary_emb(q, None, freqs_cis_img)
            assert qq.shape == q.shape, f"qq: {qq.shape}, q: {q.shape}"
            q = qq
        context = attention(q, k, v, self.head_dim, self.attn_drop, mode=self.attn_mode)
        context = context.reshape(b, s1, -1)
        
        out = self.out_proj(context)
        out = self.proj_drop(out)

        out_tuple = (out,)

        return out_tuple
