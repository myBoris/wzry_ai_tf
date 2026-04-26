"""文字 token 输入的编码器。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import TextEncoderConfig
from .input_embedding import DecoderInputEmbedding


class TextTokenEncoder(nn.Module):
    """把文字 token 序列编码成 memory tokens。

    输入形状：
        input_ids: [batch, text_len]

    输出形状：
        text_tokens: [batch, text_len, d_model]
        text_padding_mask: [batch, text_len]，True 表示该文字 token 无效。
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = DecoderInputEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_text_len,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
            learned_position=config.learned_position,
        )
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, config.d_model))

        if config.num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                norm_first=True,
            )
            self.encoder: Optional[nn.TransformerEncoder] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_layers,
            )
        else:
            self.encoder = None

        self.norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        text_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """编码文字 token，并返回文字 token 与 padding mask。"""
        if input_ids.dim() != 2:
            raise ValueError("input_ids 必须是 [batch, text_len] 形状")

        batch_size, text_len = input_ids.shape
        if text_len > self.config.max_text_len:
            raise ValueError(f"文字 token 长度不能超过 {self.config.max_text_len}")

        if text_padding_mask is None:
            if self.config.padding_idx is None:
                text_padding_mask = torch.zeros(
                    batch_size,
                    text_len,
                    dtype=torch.bool,
                    device=input_ids.device,
                )
            else:
                text_padding_mask = input_ids.eq(self.config.padding_idx)
        elif text_padding_mask.shape != (batch_size, text_len):
            raise ValueError("text_padding_mask 必须是 [batch, text_len] 形状")
        else:
            text_padding_mask = text_padding_mask.to(device=input_ids.device, dtype=torch.bool)

        text_tokens = self.embedding(input_ids)
        text_tokens = text_tokens + self.modality_embedding

        if self.encoder is not None:
            text_tokens = self.encoder(
                text_tokens,
                src_key_padding_mask=text_padding_mask,
            )

        text_tokens = self.norm(text_tokens)
        text_tokens = text_tokens.masked_fill(text_padding_mask.unsqueeze(-1), 0.0)
        return text_tokens, text_padding_mask

    @classmethod
    def from_config(cls, config: TextEncoderConfig) -> "TextTokenEncoder":
        """使用 TextEncoderConfig 创建文字编码器。"""
        return cls(config)
