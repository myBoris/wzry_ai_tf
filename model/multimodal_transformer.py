"""图片 + 文字输入，Transformer Decoder 输出动作/token logits 的完整模型。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

from torch import Tensor, nn

from .config import MultiModalTransformerConfig
from .json_output import JsonOutputHead
from .multimodal_encoder import MultiModalEncoder
from .transformer_decoder import DecoderOutput, TransformerDecoder


@dataclass
class MultiModalTransformerOutput:
    """完整多模态模型的结构化输出。

    字段说明：
        logits: Decoder 对每个时间步输出 token/action 的预测分数，
            形状为 `[batch, target_len, output_dim]`。
        memory: 多模态 Encoder 输出的融合特征。只有 `return_memory=True` 时返回。
        memory_key_padding_mask: 与 memory 对齐的 padding mask。True 表示该 memory
            位置无效，例如被补齐的图片或文字 token。
        hidden_states: Decoder 最后一层隐状态。只有 `return_hidden=True` 时对外返回；
            若只是为了生成 JSON，内部会临时取 hidden states，但不会默认暴露。
        attentions: Decoder 每层注意力权重。只在调试或可视化时建议打开。
        json_field_outputs: JSON 输出头的原始 tensor。训练时可以用它给每个字段算 loss。
        json_values: 已经按模板解码并渲染好的 JSON 兼容对象，推理或接口调用时使用。
    """

    logits: Tensor
    memory: Optional[Tensor] = None
    memory_key_padding_mask: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    attentions: Optional[List[Dict[str, Optional[Tensor]]]] = None
    json_field_outputs: Optional[Dict[str, Tensor]] = None
    json_values: Optional[List[Any]] = None


class MultiModalTransformerDecoder(nn.Module):
    """最多 10 张图片 + 文字输入，再由 Transformer Decoder 预测输出。

    基础输出仍然是普通的 `logits`，用于动作 token 或序列训练。
    如果配置了 `json_output_template`，模型会额外创建一个 `JsonOutputHead`：

    ```text
    images/text -> MultiModalEncoder -> memory
    decoder_input_ids + memory -> TransformerDecoder -> hidden_states/logits
    hidden_states[:, -1, :] -> JsonOutputHead -> json_field_outputs -> json_values
    ```

    这样可以同时保留 token 级训练能力，以及面向业务接口的结构化 JSON 输出。
    """

    def __init__(self, config: MultiModalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MultiModalEncoder(config.encoder)
        self.decoder = TransformerDecoder.from_config(config.decoder)

        # JSON 输出头是可选的。没有模板时，模型行为和普通多模态 Decoder 完全一致；
        # 有模板时，return_json=True 会用 Decoder 最后一步 hidden state 预测模板字段。
        self.json_output_head: Optional[JsonOutputHead] = None
        if config.json_output_template is not None:
            self.json_output_head = JsonOutputHead(
                d_model=config.decoder.d_model,
                output_template=config.json_output_template,
            )

    def forward(
        self,
        decoder_input_ids: Tensor,
        images: Optional[Tensor] = None,
        text_input_ids: Optional[Tensor] = None,
        image_padding_mask: Optional[Tensor] = None,
        text_padding_mask: Optional[Tensor] = None,
        decoder_tgt_mask: Optional[Tensor] = None,
        decoder_tgt_key_padding_mask: Optional[Tensor] = None,
        start_pos: int = 0,
        return_memory: bool = False,
        return_hidden: bool = False,
        return_attentions: bool = False,
        return_json: bool = False,
        return_dict: bool = False,
    ) -> Union[Tensor, MultiModalTransformerOutput, List[Any]]:
        """执行完整多模态前向传播。

        参数：
            decoder_input_ids: Decoder 输入 token，形状为 [batch, target_len]。
            images: 图片输入，形状为 [batch, num_images, channels, height, width]。
            text_input_ids: 文字 token id，形状为 [batch, text_len]。
            image_padding_mask: 图片 padding mask，True 表示该图片无效。
            text_padding_mask: 文字 padding mask，True 表示该 token 无效。
            decoder_tgt_mask: Decoder 的因果 mask；默认自动创建。
            decoder_tgt_key_padding_mask: Decoder 输入 padding mask；默认自动创建。
            start_pos: 自回归生成时的位置偏移。
            return_memory: 是否返回多模态 memory。
            return_hidden: 是否返回 Decoder 最后一层隐状态。
            return_attentions: 是否返回 Decoder 注意力权重。
            return_json: 是否按 json_output_template 解码 JSON 输出。
            return_dict: 是否返回结构化输出；否则只返回 logits。

        返回：
            - 默认：返回 logits tensor。
            - `return_dict=True`：返回 `MultiModalTransformerOutput`。
            - `return_json=True` 且 `return_dict=False`：只返回 JSON 对象列表。
            - `return_json=True` 且 `return_dict=True`：同时返回 logits、字段 raw tensor
              和已经渲染好的 JSON 对象，方便训练/调试一起看。
        """
        if return_json and self.json_output_head is None:
            raise ValueError("return_json=True 需要先配置 json_output_template")

        # 先把图片和文字统一编码为 memory。memory 是 Decoder cross-attention 的来源，
        # 形状通常是 [batch, num_images + text_len, d_model]。
        encoder_output = self.encoder(
            images=images,
            text_input_ids=text_input_ids,
            image_padding_mask=image_padding_mask,
            text_padding_mask=text_padding_mask,
            return_dict=True,
        )

        # Decoder 始终返回 DecoderOutput，便于后面按需取 logits、hidden states、attention。
        # 生成 JSON 需要 hidden states，所以 return_json=True 时会强制内部打开 return_hidden。
        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            memory=encoder_output.memory,
            tgt_mask=decoder_tgt_mask,
            tgt_key_padding_mask=decoder_tgt_key_padding_mask,
            memory_key_padding_mask=encoder_output.memory_key_padding_mask,
            start_pos=start_pos,
            return_hidden=return_hidden or return_json,
            return_attentions=return_attentions,
            return_dict=True,
        )
        if not isinstance(decoder_output, DecoderOutput):
            raise RuntimeError("Decoder 应返回 DecoderOutput")

        json_field_outputs = None
        json_values = None
        if return_json:
            # JSON 输出头使用最后一个时间步的 hidden state 预测字段。这里不直接用 logits，
            # 是因为 logits 通常对应动作/token 词表，而 JSON 字段可能是多个不同类型的 head。
            if decoder_output.hidden_states is None:
                raise RuntimeError("生成 JSON 输出需要 Decoder hidden_states")
            if self.json_output_head is None:
                raise RuntimeError("json_output_head 未初始化")
            json_field_outputs = self.json_output_head(decoder_output.hidden_states)
            json_values = self.json_output_head.decode(json_field_outputs)

        # 纯推理场景下，调用方可能只关心 JSON 对象，不需要 logits 和中间状态。
        if return_json and not (return_dict or return_memory or return_hidden or return_attentions):
            return json_values or []

        # 只要调用方请求任一结构化信息，就统一返回 MultiModalTransformerOutput。
        # 注意：return_json 为了内部 decode 会计算 hidden_states，但只有显式
        # return_hidden=True 时才把 hidden_states 暴露给调用方，避免默认多占内存引用。
        if return_dict or return_memory or return_hidden or return_attentions or return_json:
            return MultiModalTransformerOutput(
                logits=decoder_output.logits,
                memory=encoder_output.memory if return_memory else None,
                memory_key_padding_mask=(
                    encoder_output.memory_key_padding_mask if return_memory else None
                ),
                hidden_states=decoder_output.hidden_states if return_hidden else None,
                attentions=decoder_output.attentions,
                json_field_outputs=json_field_outputs,
                json_values=json_values,
            )
        return decoder_output.logits

    def set_json_output_template(self, output_template: Mapping[str, Any]) -> None:
        """在训练前设置或替换 JSON 输出模板。

        适用场景：
            - 先用普通 logits 模型初始化，再根据任务动态挂载 JSON 输出头。
            - 实验不同 JSON schema，而不重建整个 Encoder/Decoder。

        注意：
            这会重新初始化 `JsonOutputHead` 的参数；如果已经训练过旧的 JSON 输出头，
            调用本方法后旧 head 的权重不会保留。
        """
        self.config.json_output_template = copy.deepcopy(dict(output_template))
        self.json_output_head = JsonOutputHead(
            d_model=self.config.decoder.d_model,
            output_template=output_template,
        )

    @classmethod
    def from_config(
        cls,
        config: MultiModalTransformerConfig,
    ) -> "MultiModalTransformerDecoder":
        """使用 MultiModalTransformerConfig 创建完整模型。"""
        return cls(config)
