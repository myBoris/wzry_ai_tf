# PyTorch Transformer Decoder 实现说明

这份实现放在 `model/` 目录下，目标是实现一个支持“最多 10 张图片 + 文字输入”的多模态 Transformer Decoder，可用于序列生成，也可用于“游戏画面/文字指令 -> 动作序列”的策略模型。

## 目录结构

```text
model/
+-- __init__.py              # 对外导出模块
+-- attention_mask.py        # causal mask 与 padding mask
+-- config.py                # Decoder 与多模态输入配置
+-- image_encoder.py         # 最多 10 张图片编码为 image tokens
+-- input_embedding.py       # token embedding 与 position encoding
+-- decoder_layer.py         # 单层 Transformer Decoder Layer
+-- multimodal_encoder.py    # 图片 token 与文字 token 融合为 memory
+-- multimodal_transformer.py # 完整多模态模型
+-- text_encoder.py          # 文字 token 编码为 text tokens
+-- transformer_decoder.py   # 多层 Decoder 堆叠
`-- output_decode.py         # 输出投影到动作/词表 logits
```

## 多模态输入结构

模型现在支持两类输入：

- `images`：最多 10 张图片，形状为 `[batch, num_images, channels, height, width]`。
- `text_input_ids`：文字 token，形状为 `[batch, text_len]`。

整体流程如下：

```text
最多 10 张图片
    |
    v
ImageTokenEncoder -> image tokens

文字 token ids
    |
    v
TextTokenEncoder -> text tokens

image tokens + text tokens
    |
    v
MultiModalEncoder -> memory
    |
    v
TransformerDecoder(历史动作 tokens, memory)
    |
    v
动作/token logits
```

当前 `ImageTokenEncoder` 会把每张图片压缩成一个 `d_model` 维 token；也就是说最多 10 张图片会产生最多 10 个 image tokens。后续如果想保留更多视觉细节，可以把 `image_encoder.py` 替换成 ResNet、ViT 或 patch-level encoder。

## 标准 Decoder 结构

一个标准 Transformer Decoder Layer 由三块组成：

```text
target tokens
    |
    v
Token Embedding + Position Encoding
    |
    v
Masked Multi-Head Self-Attention
    |
    v
Encoder-Decoder Cross-Attention，可选
    |
    v
Feed Forward Network
    |
    v
Output Projection -> logits
```

在本项目里可以这样理解：

- `input_ids`：历史动作 token，例如移动、普攻、释放技能、撤退等。
- `memory`：可选的游戏状态编码结果，例如视觉编码器或状态编码器输出。
- `logits`：每个时间步对下一步动作 token 的预测分数。

## 关键张量形状

默认使用 PyTorch 的 `batch_first=True`，所以主要张量形状如下：

```text
input_ids:               [batch, target_len]
images:                  [batch, num_images, channels, height, width]
text_input_ids:          [batch, text_len]
embedding output:        [batch, target_len, d_model]
image tokens:            [batch, num_images, d_model]
text tokens:             [batch, text_len, d_model]
memory:                  [batch, num_images + text_len, d_model]
logits:                  [batch, target_len, output_dim]
tgt_mask:                [target_len, target_len]
tgt_key_padding_mask:    [batch, target_len]
memory_key_padding_mask: [batch, source_len]
```

`tgt_mask` 是 causal mask，用来防止当前位置看到未来动作。  
`padding_mask` 用来忽略 padding token。

## 使用示例

```python
import torch

from model import TransformerDecoder


batch_size = 2
target_len = 8
source_len = 16

action_vocab_size = 32
d_model = 128

decoder = TransformerDecoder(
    vocab_size=action_vocab_size,
    output_dim=action_vocab_size,
    d_model=d_model,
    num_layers=4,
    num_heads=8,
    dim_feedforward=512,
    max_seq_len=128,
    padding_idx=0,
    cross_attention=True,
)

# 历史动作序列，0 通常作为 PAD。
input_ids = torch.randint(1, action_vocab_size, (batch_size, target_len))

# 这里假设 memory 来自游戏画面/小地图/数值状态编码器。
memory = torch.randn(batch_size, source_len, d_model)

logits = decoder(input_ids, memory=memory)
print(logits.shape)  # [2, 8, 32]

# 取最后一个时间步的下一动作预测。
next_action_logits = logits[:, -1, :]
next_action = next_action_logits.argmax(dim=-1)
print(next_action)
```

完整多模态模型示例：

```python
import torch

from model import (
    ImageEncoderConfig,
    MultiModalEncoderConfig,
    MultiModalTransformerConfig,
    MultiModalTransformerDecoder,
    TextEncoderConfig,
    TransformerDecoderConfig,
)


batch_size = 2
num_images = 10
text_len = 32
target_len = 8

text_vocab_size = 5000
action_vocab_size = 32
d_model = 128

config = MultiModalTransformerConfig(
    encoder=MultiModalEncoderConfig(
        image=ImageEncoderConfig(
            d_model=d_model,
            image_channels=3,
            max_images=10,
        ),
        text=TextEncoderConfig(
            vocab_size=text_vocab_size,
            d_model=d_model,
            max_text_len=128,
            num_layers=2,
            num_heads=8,
        ),
    ),
    decoder=TransformerDecoderConfig(
        vocab_size=action_vocab_size,
        output_dim=action_vocab_size,
        d_model=d_model,
        num_layers=4,
        num_heads=8,
        dim_feedforward=512,
        cross_attention=True,
    ),
)

model = MultiModalTransformerDecoder.from_config(config)

# 图片可以是 float，也可以是 uint8；uint8 会在模型里自动转成 [0, 1]。
images = torch.randn(batch_size, num_images, 3, 224, 224)
text_input_ids = torch.randint(1, text_vocab_size, (batch_size, text_len))
decoder_input_ids = torch.randint(1, action_vocab_size, (batch_size, target_len))

logits = model(
    decoder_input_ids=decoder_input_ids,
    images=images,
    text_input_ids=text_input_ids,
)
print(logits.shape)  # [2, 8, 32]
```

如果一个 batch 中有的样本图片不足 10 张，可以用 `image_padding_mask` 标记无效图片位置：

```python
image_padding_mask = torch.zeros(batch_size, num_images, dtype=torch.bool)
image_padding_mask[0, 7:] = True  # 第 0 个样本只有 7 张有效图片

logits = model(
    decoder_input_ids=decoder_input_ids,
    images=images,
    text_input_ids=text_input_ids,
    image_padding_mask=image_padding_mask,
)
```

也可以使用配置对象统一管理超参数：

```python
from model import TransformerDecoder, TransformerDecoderConfig


config = TransformerDecoderConfig(
    vocab_size=32,
    output_dim=32,
    d_model=128,
    num_layers=4,
    num_heads=8,
    dim_feedforward=512,
    max_seq_len=128,
    padding_idx=0,
    cross_attention=True,
)

decoder = TransformerDecoder.from_config(config)
```

## JSON 输出模板

如果希望模型推理时直接返回 JSON，可以在 `MultiModalTransformerConfig` 中传入 `json_output_template`。模板里用 `$type` 定义字段格式，目前支持：

- `enum`：枚举分类，需要 `choices`。
- `string`：字符串；如果要用固定输出头训练，也需要 `choices`。
- `integer`：整数，固定输出头需要 `min` 和 `max`。
- `number`：浮点数，可选 `min`、`max`、`precision`。
- `boolean`：布尔值。

示例：

```python
json_output_template = {
    "action": {"$type": "enum", "choices": ["move", "attack", "skill"]},
    "target": {
        "x": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
        "y": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
    },
    "skill_id": {"$type": "integer", "min": 1, "max": 3, "default": 1},
    "use_skill": {"$type": "boolean", "default": False},
}
```

配置后调用：

```python
output = model(
    decoder_input_ids=decoder_input_ids,
    images=images,
    text_input_ids=text_input_ids,
    return_json=True,
    return_dict=True,
)

print(output.json_values)
print(output.json_field_outputs["action"].shape)  # 可用于分类 loss
```

`json_values` 是已经渲染好的 JSON 兼容对象；`json_field_outputs` 保留每个字段的 raw tensor，训练时可以分别计算分类或回归损失。

## 如何一步步实现

### 1. 先定义动作词表

把动作离散化成 token，例如：

```text
0: PAD
1: BOS
2: EOS
3: MOVE_UP
4: MOVE_DOWN
5: MOVE_LEFT
6: MOVE_RIGHT
7: ATTACK
8: SKILL_1
9: SKILL_2
10: SKILL_3
```

如果动作包含方向、技能目标、连续坐标，可以先做粗粒度离散化。后续也可以把输出头拆成多个 head，例如动作类型、方向、技能目标分别预测。

### 2. 输入嵌入

`DecoderInputEmbedding` 做两件事：

- `TokenEmbedding`：把动作 token id 转成向量。
- `PositionEncoding`：加入位置信息，让模型知道第几个时间步。

### 3. 构造 Mask

`causal_mask` 保证第 `t` 个动作只能看见 `0..t` 的历史动作，不能偷看未来。

`padding_mask` 保证 batch 中补齐用的 PAD token 不参与注意力。

### 4. 实现单层 Decoder

`TransformerDecoderLayer` 的核心流程是：

```text
x = x + MaskedSelfAttention(LayerNorm(x))
x = x + CrossAttention(LayerNorm(x), memory)
x = x + FeedForward(LayerNorm(x))
```

这里默认使用 Pre-LN，也就是先 LayerNorm 再进入子模块，训练更稳定。若想更接近原始论文的 Post-LN，可以把 `norm_first=False`。

### 5. 堆叠多层 Decoder

`TransformerDecoder` 用 `nn.ModuleList` 堆叠多个 `TransformerDecoderLayer`，最后经过 `LayerNorm` 和 `OutputProjection` 得到 logits。

### 6. 接入游戏 AI

一个常见设计是：

```text
最多 10 张游戏截图 + 文字指令/状态描述
    |
    v
MultiModalEncoder -> memory
    |
    v
TransformerDecoder(历史动作 tokens, memory)
    |
    v
下一步动作 logits
```

训练时可以用专家数据做行为克隆：

```text
输入:  [BOS, a1, a2, ..., aN-1]
标签:  [a1,  a2, ..., aN]
损失:  CrossEntropyLoss(ignore_index=PAD)
```

推理时则每次把历史动作喂给 Decoder，取最后一个时间步的 logits，选择下一步动作。
