"""用于序列预测或动作 token 预测的多层 Transformer Decoder。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from torch import Tensor, nn

from .attention_mask import make_decoder_masks
from .config import TransformerDecoderConfig
from .decoder_layer import TransformerDecoderLayer
from .input_embedding import DecoderInputEmbedding
from .output_decode import OutputProjection


@dataclass
class DecoderOutput:
    """可选的结构化输出，便于调试和分析。"""

    logits: Tensor
    hidden_states: Optional[Tensor] = None
    attentions: Optional[List[Dict[str, Optional[Tensor]]]] = None


class TransformerDecoder(nn.Module):
    """标准 Transformer Decoder 堆叠。

    游戏 AI 中的典型用法：
    - input_ids: 历史动作 token，形状为 [batch, target_len]
    - memory: 可选的游戏状态编码，形状为 [batch, source_len, d_model]
    - logits: 下一步动作分数，形状为 [batch, target_len, output_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: Optional[int] = None,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        padding_idx: Optional[int] = 0,
        activation: str = "gelu",
        cross_attention: bool = True,
        norm_first: bool = True,
        learned_position: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")

        self.padding_idx = padding_idx
        self.d_model = d_model
        self.output_dim = output_dim or vocab_size

        self.embedding = DecoderInputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
            learned_position=learned_position,
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    cross_attention=cross_attention,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = OutputProjection(d_model, self.output_dim)

    def forward(
        self,
        input_ids: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        return_hidden: bool = False,
        return_attentions: bool = False,
        return_dict: bool = False,
    ) -> Union[Tensor, DecoderOutput]:
        """解码一个目标 token 序列。

        参数：
            input_ids: 目标 token id，形状为 [batch, target_len]。
            memory: 可选的 Encoder 状态，形状为 [batch, source_len, d_model]。
            tgt_mask: 可选的因果 mask；不传时会自动创建。
            tgt_key_padding_mask: 可选的目标序列 padding mask；默认由 padding_idx 创建。
            memory_key_padding_mask: 可选的 memory padding mask。
            start_pos: 自回归生成时的位置偏移。
            return_hidden: 是否在 DecoderOutput 中返回最终隐状态。
            return_attentions: 是否在 DecoderOutput 中返回每层注意力权重。
            return_dict: 是否返回 DecoderOutput；否则只返回 logits。
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids 必须是 [batch, target_len] 形状")

        # 调用方不传 mask 时，自动构造 Decoder 最常用的因果 mask 和 padding mask。
        if tgt_mask is None or (tgt_key_padding_mask is None and self.padding_idx is not None):
            auto_tgt_mask, auto_padding_mask = make_decoder_masks(
                input_ids,
                padding_idx=self.padding_idx,
            )
            if tgt_mask is None:
                tgt_mask = auto_tgt_mask
            if tgt_key_padding_mask is None:
                tgt_key_padding_mask = auto_padding_mask

        hidden_states = self.embedding(input_ids, start_pos=start_pos)
        attentions: List[Dict[str, Optional[Tensor]]] = []

        # 逐层堆叠 Decoder Layer；需要分析注意力时才保存权重，避免训练时多占显存。
        for layer in self.layers:
            if return_attentions:
                hidden_states, layer_attn = layer(
                    hidden_states,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_weights=True,
                )
                attentions.append(layer_attn)
            else:
                hidden_states = layer(
                    hidden_states,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_weights=False,
                )

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_projection(hidden_states)

        if return_dict or return_hidden or return_attentions:
            return DecoderOutput(
                logits=logits,
                hidden_states=hidden_states if return_hidden else None,
                attentions=attentions if return_attentions else None,
            )
        return logits

    @classmethod
    def from_config(cls, config: TransformerDecoderConfig) -> "TransformerDecoder":
        """使用 TransformerDecoderConfig 创建 Decoder。"""
        return cls(**config.to_dict())
