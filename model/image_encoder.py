"""最多 10 张图片输入的编码器。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import ImageEncoderConfig


class ImageTokenEncoder(nn.Module):
    """把每张图片编码成一个 memory token。

    输入形状：
        images: [batch, num_images, channels, height, width]

    输出形状：
        image_tokens: [batch, num_images, d_model]
        image_padding_mask: [batch, num_images]，True 表示该图片位置无效。
    """

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.image_channels
        for out_channels in config.cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Linear(config.cnn_channels[-1], config.d_model)
        self.image_position = nn.Embedding(config.max_images, config.d_model)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        images: Tensor,
        image_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """编码图片，并返回图片 token 与图片 padding mask。"""
        if images.dim() != 5:
            raise ValueError("images 必须是 [batch, num_images, channels, height, width] 形状")

        batch_size, num_images, channels, height, width = images.shape
        if num_images > self.config.max_images:
            raise ValueError(f"最多只能输入 {self.config.max_images} 张图片")
        if channels != self.config.image_channels:
            raise ValueError(f"图片通道数必须是 {self.config.image_channels}")

        if image_padding_mask is None:
            image_padding_mask = torch.zeros(
                batch_size,
                num_images,
                dtype=torch.bool,
                device=images.device,
            )
        elif image_padding_mask.shape != (batch_size, num_images):
            raise ValueError("image_padding_mask 必须是 [batch, num_images] 形状")
        else:
            image_padding_mask = image_padding_mask.to(device=images.device, dtype=torch.bool)

        # 如果输入是 uint8 图片，自动转成 [0, 1] 浮点；如果已经是 float，则保持原值。
        if not torch.is_floating_point(images):
            images = images.float() / 255.0

        flat_images = images.reshape(batch_size * num_images, channels, height, width)
        features = self.cnn(flat_images).flatten(1)
        image_tokens = self.projection(features).view(batch_size, num_images, self.config.d_model)

        positions = torch.arange(num_images, device=images.device, dtype=torch.long)
        image_tokens = image_tokens + self.image_position(positions).unsqueeze(0)
        image_tokens = image_tokens + self.modality_embedding
        image_tokens = self.norm(self.dropout(image_tokens))

        # 无效图片位置清零，真正的屏蔽由 memory_key_padding_mask 完成。
        image_tokens = image_tokens.masked_fill(image_padding_mask.unsqueeze(-1), 0.0)
        return image_tokens, image_padding_mask

    @classmethod
    def from_config(cls, config: ImageEncoderConfig) -> "ImageTokenEncoder":
        """使用 ImageEncoderConfig 创建图片编码器。"""
        return cls(config)
