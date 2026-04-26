"""Transformer Decoder 的输入嵌入模块。"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Token 嵌入层，并按 Transformer 习惯乘以 sqrt(d_model)。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        """将 [batch, seq_len] 的 token id 转成向量表示。"""
        return self.embedding(tokens) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """原始 Transformer 使用的固定正弦位置编码。"""

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # pe 的形状固定为 [1, max_seq_len, d_model]，前面的 1 方便和 batch 维广播相加。
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """给 [batch, seq_len, d_model] 的嵌入向量加上位置编码。"""
        seq_len = x.size(1)
        return x + self.pe[:, start_pos : start_pos + seq_len].to(dtype=x.dtype)


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码，适合位置含义和任务强相关的场景。"""

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        seq_len = x.size(1)
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=x.device,
            dtype=torch.long,
        )
        return x + self.embedding(positions).unsqueeze(0)


class DecoderInputEmbedding(nn.Module):
    """Decoder 层之前的输入模块：Token 嵌入 + 位置编码 + Dropout。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        padding_idx: Optional[int] = 0,
        learned_position: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        position_cls = LearnedPositionalEncoding if learned_position else SinusoidalPositionalEncoding
        self.position_encoding = position_cls(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        x = self.token_embedding(tokens)
        x = self.position_encoding(x, start_pos=start_pos)
        return self.dropout(x)
