"""使用 PyTorch 模块实现的标准 Transformer Decoder Layer。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

from torch import Tensor, nn


class FeedForward(nn.Module):
    """Transformer 层内部的位置前馈网络。"""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation == "relu":
            activation_layer: nn.Module = nn.ReLU()
        elif activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError("activation 只支持 'relu' 或 'gelu'")

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerDecoderLayer(nn.Module):
    """单个 Decoder 块：masked self-attention、可选 cross-attention、FFN。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        cross_attention: bool = True,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.cross_attention = cross_attention
        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = (
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            if cross_attention
            else None
        )
        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _self_attention(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        need_weights: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        output, weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.dropout1(output), weights if need_weights else None

    def _cross_attention(
        self,
        x: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor],
        need_weights: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.cross_attn is None:
            raise RuntimeError("当前层未启用 cross_attention")

        output, weights = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.dropout2(output), weights if need_weights else None

    def _feed_forward(self, x: Tensor) -> Tensor:
        return self.dropout3(self.feed_forward(x))

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Optional[Tensor]]]]:
        """执行一层 Decoder。

        参数：
            tgt: Decoder 输入，形状为 [batch, target_len, d_model]。
            memory: 可选的 Encoder 输出，形状为 [batch, source_len, d_model]。
            tgt_mask: self-attention 的因果 mask，形状为 [target_len, target_len]。
            tgt_key_padding_mask: tgt 的 padding mask，形状为 [batch, target_len]。
            memory_key_padding_mask: memory 的 padding mask，形状为 [batch, source_len]。
            need_weights: 为 True 时返回每个 head 的注意力权重。
        """
        attentions: Dict[str, Optional[Tensor]] = {
            "self_attn": None,
            "cross_attn": None,
        }

        use_cross_attention = self.cross_attention and memory is not None

        if self.norm_first:
            # Pre-LN：先归一化再进入子层，深层模型训练通常更稳定。
            self_output, self_weights = self._self_attention(
                self.norm1(tgt),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=need_weights,
            )
            tgt = tgt + self_output
            attentions["self_attn"] = self_weights

            if use_cross_attention:
                cross_output, cross_weights = self._cross_attention(
                    self.norm2(tgt),
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_weights=need_weights,
                )
                tgt = tgt + cross_output
                attentions["cross_attn"] = cross_weights
                tgt = tgt + self._feed_forward(self.norm3(tgt))
            else:
                tgt = tgt + self._feed_forward(self.norm2(tgt))
        else:
            # Post-LN：先残差相加，再做 LayerNorm，更接近原始 Transformer 论文结构。
            self_output, self_weights = self._self_attention(
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=need_weights,
            )
            tgt = self.norm1(tgt + self_output)
            attentions["self_attn"] = self_weights

            if use_cross_attention:
                cross_output, cross_weights = self._cross_attention(
                    tgt,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_weights=need_weights,
                )
                tgt = self.norm2(tgt + cross_output)
                attentions["cross_attn"] = cross_weights
                tgt = self.norm3(tgt + self._feed_forward(tgt))
            else:
                tgt = self.norm2(tgt + self._feed_forward(tgt))

        if need_weights:
            return tgt, attentions
        return tgt
