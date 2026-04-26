"""图片和文字输入的多模态融合编码器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .config import MultiModalEncoderConfig
from .image_encoder import ImageTokenEncoder
from .text_encoder import TextTokenEncoder


@dataclass
class MultiModalEncoderOutput:
    """多模态编码器的结构化输出。"""

    memory: Tensor
    memory_key_padding_mask: Tensor
    image_tokens: Optional[Tensor] = None
    text_tokens: Optional[Tensor] = None


class MultiModalEncoder(nn.Module):
    """将最多 10 张图片和一段文字融合成 Decoder 可用的 memory。"""

    def __init__(self, config: MultiModalEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = ImageTokenEncoder(config.image)
        self.text_encoder = TextTokenEncoder(config.text)
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        images: Optional[Tensor] = None,
        text_input_ids: Optional[Tensor] = None,
        image_padding_mask: Optional[Tensor] = None,
        text_padding_mask: Optional[Tensor] = None,
        return_dict: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], MultiModalEncoderOutput]:
        """编码图片和文字。

        参数：
            images: 图片输入，形状为 [batch, num_images, channels, height, width]。
            text_input_ids: 文字 token id，形状为 [batch, text_len]。
            image_padding_mask: 图片 padding mask，形状为 [batch, num_images]。
            text_padding_mask: 文字 padding mask，形状为 [batch, text_len]。
            return_dict: 为 True 时返回 MultiModalEncoderOutput。
        """
        if images is None and text_input_ids is None:
            raise ValueError("images 和 text_input_ids 至少需要提供一个")

        token_parts = []
        mask_parts = []
        image_tokens = None
        text_tokens = None

        if images is not None:
            image_tokens, image_mask = self.image_encoder(
                images,
                image_padding_mask=image_padding_mask,
            )
            token_parts.append(image_tokens)
            mask_parts.append(image_mask)

        if text_input_ids is not None:
            text_tokens, text_mask = self.text_encoder(
                text_input_ids,
                text_padding_mask=text_padding_mask,
            )
            token_parts.append(text_tokens)
            mask_parts.append(text_mask)

        batch_sizes = {tokens.size(0) for tokens in token_parts}
        if len(batch_sizes) != 1:
            raise ValueError("图片 batch 和文字 batch 必须一致")

        memory = torch.cat(token_parts, dim=1)
        memory_key_padding_mask = torch.cat(mask_parts, dim=1)
        memory = self.fusion_norm(self.dropout(memory))
        memory = memory.masked_fill(memory_key_padding_mask.unsqueeze(-1), 0.0)

        if return_dict:
            return MultiModalEncoderOutput(
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                image_tokens=image_tokens,
                text_tokens=text_tokens,
            )
        return memory, memory_key_padding_mask

    @classmethod
    def from_config(cls, config: MultiModalEncoderConfig) -> "MultiModalEncoder":
        """使用 MultiModalEncoderConfig 创建多模态编码器。"""
        return cls(config)
