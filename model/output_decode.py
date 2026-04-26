"""将 Decoder 隐状态解码为动作或 token logits 的输出头。"""

from __future__ import annotations

from torch import Tensor, nn


class OutputProjection(nn.Module):
    """把 Decoder 隐状态线性投影到词表或动作空间。"""

    def __init__(self, d_model: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, output_dim, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """将 [batch, seq_len, d_model] 投影成 [batch, seq_len, output_dim]。"""
        return self.projection(hidden_states)
