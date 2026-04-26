"""Transformer Decoder 训练和推理时使用的注意力 mask 工具。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> Tensor:
    """返回 [seq_len, seq_len] 的布尔 mask，True 表示屏蔽未来 token。"""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def padding_mask(tokens: Tensor, padding_idx: int = 0) -> Tensor:
    """返回 [batch, seq_len] 的布尔 mask，True 表示屏蔽 PAD token。"""
    return tokens.eq(padding_idx)


def make_decoder_masks(
    tokens: Tensor,
    padding_idx: Optional[int] = 0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """构造标准 Decoder self-attention 所需的 mask。

    PyTorch MultiheadAttention 的约定：
    - attn_mask: [target_len, target_len]，True 表示不能关注该位置。
    - key_padding_mask: [batch, target_len]，True 表示忽略该 key。
    """
    attn_mask = causal_mask(tokens.size(1), device=tokens.device)
    key_padding_mask = None
    if padding_idx is not None:
        key_padding_mask = padding_mask(tokens, padding_idx=padding_idx)
    return attn_mask, key_padding_mask
