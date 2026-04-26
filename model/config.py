"""Transformer Decoder 的配置定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class TransformerDecoderConfig:
    """集中管理 Transformer Decoder 的超参数。

    说明：
        vocab_size: 输入 token 词表大小，游戏场景中通常是动作 token 数量。
        output_dim: 输出维度，默认等于 vocab_size；也可以设为动作类别数量。
        d_model: 模型隐藏维度。
        num_layers: Decoder 层数。
        num_heads: 多头注意力头数。
        dim_feedforward: 前馈网络中间层维度。
        max_seq_len: 支持的最大序列长度。
        dropout: Dropout 概率。
        padding_idx: PAD token id；如果为 None，则不自动生成 padding mask。
        activation: 前馈网络激活函数，支持 "gelu" 或 "relu"。
        cross_attention: 是否启用 encoder-decoder cross-attention。
        norm_first: 是否使用 Pre-LN 结构。
        learned_position: 是否使用可学习位置编码。
    """

    vocab_size: int
    output_dim: Optional[int] = None
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    padding_idx: Optional[int] = 0
    activation: str = "gelu"
    cross_attention: bool = True
    norm_first: bool = True
    learned_position: bool = False

    def __post_init__(self) -> None:
        """在创建配置时做基础合法性检查，尽早暴露参数错误。"""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0")
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim 必须大于 0")
        if self.d_model <= 0:
            raise ValueError("d_model 必须大于 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads 必须大于 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward 必须大于 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len 必须大于 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("activation 只支持 'gelu' 或 'relu'")

    @property
    def resolved_output_dim(self) -> int:
        """返回最终输出维度；未显式设置时使用 vocab_size。"""
        return self.output_dim or self.vocab_size

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典，方便保存到 JSON/YAML。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TransformerDecoderConfig":
        """从字典创建配置对象。"""
        return cls(**values)


@dataclass
class ImageEncoderConfig:
    """图片编码器配置。

    说明：
        d_model: 输出 token 的隐藏维度，必须和 Decoder 的 d_model 一致。
        image_channels: 输入图片通道数，RGB 图片通常为 3。
        max_images: 单次最多输入图片数量，本项目默认最多 10 张。
        cnn_channels: 简单 CNN 每一层的输出通道数。
        dropout: 图片 token 的 Dropout 概率。
    """

    d_model: int = 512
    image_channels: int = 3
    max_images: int = 10
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """检查图片编码器参数是否合法。"""
        if self.d_model <= 0:
            raise ValueError("图片编码器 d_model 必须大于 0")
        if self.image_channels <= 0:
            raise ValueError("image_channels 必须大于 0")
        if self.max_images <= 0:
            raise ValueError("max_images 必须大于 0")
        if self.max_images > 10:
            raise ValueError("max_images 不能超过 10")
        if len(self.cnn_channels) == 0:
            raise ValueError("cnn_channels 至少需要一层")
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("cnn_channels 中的通道数都必须大于 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("图片编码器 dropout 必须在 [0, 1) 范围内")

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典，方便保存到 JSON/YAML。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ImageEncoderConfig":
        """从字典创建图片编码器配置。"""
        return cls(**values)


@dataclass
class TextEncoderConfig:
    """文字编码器配置。

    说明：
        vocab_size: 文字 token 词表大小。
        d_model: 输出 token 的隐藏维度，必须和 Decoder 的 d_model 一致。
        max_text_len: 最大文字 token 长度。
        num_layers: TransformerEncoder 层数；为 0 时只使用 embedding。
        num_heads: 多头注意力头数。
        dim_feedforward: 前馈网络中间层维度。
        dropout: Dropout 概率。
        padding_idx: PAD token id；如果为 None，则不自动生成文字 padding mask。
        activation: 前馈网络激活函数，支持 "gelu" 或 "relu"。
        learned_position: 是否使用可学习位置编码。
    """

    vocab_size: int
    d_model: int = 512
    max_text_len: int = 128
    num_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    padding_idx: Optional[int] = 0
    activation: str = "gelu"
    learned_position: bool = False

    def __post_init__(self) -> None:
        """检查文字编码器参数是否合法。"""
        if self.vocab_size <= 0:
            raise ValueError("文字 vocab_size 必须大于 0")
        if self.d_model <= 0:
            raise ValueError("文字编码器 d_model 必须大于 0")
        if self.max_text_len <= 0:
            raise ValueError("max_text_len 必须大于 0")
        if self.num_layers < 0:
            raise ValueError("num_layers 不能小于 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads 必须大于 0")
        if self.d_model % self.num_heads != 0:
            raise ValueError("文字编码器 d_model 必须能被 num_heads 整除")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward 必须大于 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("文字编码器 dropout 必须在 [0, 1) 范围内")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("activation 只支持 'gelu' 或 'relu'")

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典，方便保存到 JSON/YAML。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TextEncoderConfig":
        """从字典创建文字编码器配置。"""
        return cls(**values)


@dataclass
class MultiModalEncoderConfig:
    """图片 + 文字融合编码器配置。"""

    text: TextEncoderConfig
    image: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """检查图片和文字编码器是否能拼接到同一个 memory。"""
        if self.image.d_model != self.text.d_model:
            raise ValueError("图片编码器和文字编码器的 d_model 必须一致")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("多模态编码器 dropout 必须在 [0, 1) 范围内")

    @property
    def d_model(self) -> int:
        """返回融合后 memory 的隐藏维度。"""
        return self.text.d_model

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典，方便保存到 JSON/YAML。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "MultiModalEncoderConfig":
        """从字典创建多模态编码器配置。"""
        copied = dict(values)
        if isinstance(copied.get("text"), dict):
            copied["text"] = TextEncoderConfig.from_dict(copied["text"])
        if isinstance(copied.get("image"), dict):
            copied["image"] = ImageEncoderConfig.from_dict(copied["image"])
        return cls(**copied)


@dataclass
class MultiModalTransformerConfig:
    """多模态输入 + Transformer Decoder 的完整模型配置。

    字段说明：
        encoder: 图片编码器、文字编码器和融合层的配置。
        decoder: Transformer Decoder 的配置，决定动作/token logits 的输出维度。
        json_output_template: 可选的 JSON 输出模板。传入后模型会额外创建
            `JsonOutputHead`，用 Decoder 最后一步 hidden state 预测模板中的字段。

    JSON 模板示例：

    ```python
    {
        "action": {"$type": "enum", "choices": ["move", "attack", "skill"]},
        "target": {
            "x": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
            "y": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
        },
        "use_skill": {"$type": "boolean", "default": False},
    }
    ```

    模板中带 `$type` 的字段会被模型预测；普通常量字段会在渲染 JSON 时原样保留。
    """

    encoder: MultiModalEncoderConfig
    decoder: TransformerDecoderConfig
    json_output_template: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """检查 Encoder 和 Decoder 的隐藏维度是否一致。"""
        if self.encoder.d_model != self.decoder.d_model:
            raise ValueError("多模态 Encoder 和 Decoder 的 d_model 必须一致")

    def to_dict(self) -> Dict[str, Any]:
        """转成普通字典，方便保存到 JSON/YAML。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "MultiModalTransformerConfig":
        """从字典创建完整多模态模型配置。"""
        copied = dict(values)
        if isinstance(copied.get("encoder"), dict):
            copied["encoder"] = MultiModalEncoderConfig.from_dict(copied["encoder"])
        if isinstance(copied.get("decoder"), dict):
            copied["decoder"] = TransformerDecoderConfig.from_dict(copied["decoder"])
        return cls(**copied)
