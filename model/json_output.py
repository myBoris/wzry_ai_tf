"""JSON 输出模板与字段输出头。

这个模块把“模型最后输出一个 JSON”拆成两层：

1. `JsonOutputTemplate`
   负责解析用户给定的 JSON 模板。模板中的叶子字段可以用 `$type`
   标记输出格式，例如 enum、number、integer、boolean。解析后会得到
   一组 `JsonFieldSpec`，每个 spec 对应一个需要模型预测的字段。

2. `JsonOutputHead`
   负责把 Decoder 最后一个时间步的 hidden state 映射到这些字段。
   对枚举/布尔/整数范围字段，它输出分类 logits；对浮点数字段，
   它输出一个回归值。推理时再把这些 raw tensor 解码并填回模板。

这样做的好处是：模型训练时仍然可以用标准 tensor loss，推理时又能得到
结构化、可校验的 JSON，而不是让 Transformer 直接生成一长串 JSON 文本。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# JSON 字段路径中的一段。对象字段用 str，数组下标用 int。
# 例如 {"target": {"x": ...}} 中 target.x 的路径是 ("target", "x")。
JsonPathPart = Union[str, int]

# 一个完整 JSON 字段路径。使用 tuple 可以作为 dict key，方便缓存 spec。
JsonPath = Tuple[JsonPathPart, ...]

# 模板中用于声明“这个位置是一个需要模型预测的字段”的特殊 key。
FIELD_TYPE_KEY = "$type"

# 当前固定支持的字段格式。新增格式时需要同步扩展 JsonFieldSpec.normalize、
# JsonOutputHead._field_output_dim 和 JsonOutputHead._decode_field。
SUPPORTED_FIELD_TYPES = {"enum", "integer", "number", "boolean", "string"}


def path_to_key(path: JsonPath) -> str:
    """把 JSON 路径转成日志、loss 字典中常用的可读 key。

    例如路径 `("target", "x")` 会变成 `"target.x"`。训练代码里可以用
    这个字符串作为 loss 字典 key，例如 `losses["target.x"]`。
    """
    return ".".join(str(part) for part in path)


def _copy_json_value(value: Any) -> Any:
    """深拷贝 JSON 模板值，避免调用方后续修改原始模板影响模型内部状态。"""
    return copy.deepcopy(value)


def _lookup_value(values: Mapping[Union[str, JsonPath], Any], path: JsonPath) -> Any:
    """从用户传入的字段值里按路径查找。

    支持两种写法：
    - tuple 路径：`("target", "x")`
    - 点号字符串：`"target.x"`

    测试、日志和训练代码通常用点号字符串更直观；内部递归渲染时使用 tuple
    路径更稳，二者都支持可以减少调用方的心智负担。
    """
    if path in values:
        return values[path]

    text_key = path_to_key(path)
    if text_key in values:
        return values[text_key]

    return None


@dataclass(frozen=True)
class JsonFieldSpec:
    """单个 JSON 字段的输出格式定义。

    每一个被 `$type` 标记的模板叶子节点都会变成一个 `JsonFieldSpec`。

    字段含义：
        path: 字段在 JSON 中的位置，例如 ("target", "x")。
        field_type: 输出格式，目前支持 enum、integer、number、boolean、string。
        choices: enum/string 的可选值；enum 必填，string 可选。
        min_value/max_value: 数值范围；integer 用它们确定分类空间大小。
        precision: number 渲染成 JSON 时保留的小数位数。
        default: 调用 render 时如果没有给这个字段值，就使用默认值。
    """

    path: JsonPath
    field_type: str
    choices: Optional[Tuple[Any, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    precision: Optional[int] = None
    default: Any = None

    @classmethod
    def from_template_value(cls, path: JsonPath, value: Mapping[str, Any]) -> "JsonFieldSpec":
        """从模板中的一个 `$type` 字段解析出字段规格。

        这里会尽早校验模板是否可训练、可解码：
        - enum 必须提供 choices，因为输出 head 需要知道分类数。
        - integer 如果设置 min/max，必须是真整数，避免出现半个类别。
        - max 不能小于 min，否则后续归一化和分类空间都会不明确。
        """
        field_type = str(value.get(FIELD_TYPE_KEY, ""))
        if field_type not in SUPPORTED_FIELD_TYPES:
            raise ValueError(f"不支持的 JSON 字段类型: {field_type}")

        # choices 会被固定成 tuple，保证解析后的模板不可变，也方便测试比较。
        choices = value.get("choices")
        choice_tuple = tuple(choices) if choices is not None else None
        if field_type == "enum" and not choice_tuple:
            raise ValueError(f"{path_to_key(path)} enum 字段必须提供 choices")
        if field_type == "string" and choices is not None and not choice_tuple:
            raise ValueError(f"{path_to_key(path)} string 字段的 choices 不能为空")

        min_value = value.get("min")
        max_value = value.get("max")
        if min_value is not None and max_value is not None and float(max_value) < float(min_value):
            raise ValueError(f"{path_to_key(path)} max 不能小于 min")

        # integer 字段在输出头里被当作“有限类别”预测，因此边界必须能落到整数格点上。
        if field_type == "integer":
            for name, bound in (("min", min_value), ("max", max_value)):
                if bound is not None and float(bound) != int(bound):
                    raise ValueError(f"{path_to_key(path)} integer 字段的 {name} 必须是整数")

        precision = value.get("precision")
        if precision is not None and int(precision) < 0:
            raise ValueError(f"{path_to_key(path)} precision 不能小于 0")

        return cls(
            path=path,
            field_type=field_type,
            choices=choice_tuple,
            min_value=min_value,
            max_value=max_value,
            precision=int(precision) if precision is not None else None,
            default=value.get("default"),
        )

    @property
    def key(self) -> str:
        """返回点号形式的字段名，例如 `target.x`。"""
        return path_to_key(self.path)

    def default_value(self) -> Any:
        """返回字段缺省值。

        优先使用模板里的 `default`；没有 default 时，为每种字段类型选择一个
        保守默认值。这个默认值主要用于 render 时调用方没有提供某个字段的场景，
        也用于推理 decode 后做非严格渲染。
        """
        if self.default is not None:
            return _copy_json_value(self.default)
        if self.field_type in {"enum", "string"} and self.choices:
            return self.choices[0]
        if self.field_type == "integer":
            return int(self.min_value) if self.min_value is not None else 0
        if self.field_type == "number":
            return float(self.min_value) if self.min_value is not None else 0.0
        if self.field_type == "boolean":
            return False
        return None

    def normalize(self, value: Any, *, strict: bool = True) -> Any:
        """按字段格式规范化输出值。

        参数：
            value: 待写入 JSON 的字段值。可以来自用户传入，也可以来自模型 decode。
            strict: 是否严格校验。严格模式下越界或非法枚举会抛错；非严格模式下
                会自动回退到默认值或把数值裁剪到范围内，适合模型推理阶段。

        返回：
            JSON 兼容的 Python 值，例如 str、int、float、bool。
        """
        if value is None:
            value = self.default_value()

        if self.field_type == "enum":
            # enum 是有限分类，值必须在 choices 中。模型推理时如果出现非法值，
            # 非严格模式会回到第一个 choice，保证 JSON 始终可渲染。
            if self.choices is None:
                raise ValueError(f"{self.key} enum 字段缺少 choices")
            if value not in self.choices:
                if strict:
                    raise ValueError(f"{self.key} 必须是 choices 中的一个值")
                value = self.choices[0]
            return value

        if self.field_type == "string":
            # string 默认允许任意字符串；如果模板给了 choices，就按有限集合约束。
            text = str(value)
            if self.choices is not None and text not in self.choices:
                if strict:
                    raise ValueError(f"{self.key} 必须是 choices 中的一个字符串")
                text = str(self.choices[0])
            return text

        if self.field_type == "boolean":
            return bool(value)

        if self.field_type == "integer":
            # integer 先四舍五入，再按 min/max 校验或裁剪。
            integer = int(round(float(value)))
            if self.min_value is not None and integer < int(self.min_value):
                if strict:
                    raise ValueError(f"{self.key} 不能小于 {self.min_value}")
                integer = int(self.min_value)
            if self.max_value is not None and integer > int(self.max_value):
                if strict:
                    raise ValueError(f"{self.key} 不能大于 {self.max_value}")
                integer = int(self.max_value)
            return integer

        # number 是连续值。推理阶段通常来自回归头，可能略微越界，
        # 非严格模式会裁剪到模板声明的范围内。
        number = float(value)
        if self.min_value is not None and number < float(self.min_value):
            if strict:
                raise ValueError(f"{self.key} 不能小于 {self.min_value}")
            number = float(self.min_value)
        if self.max_value is not None and number > float(self.max_value):
            if strict:
                raise ValueError(f"{self.key} 不能大于 {self.max_value}")
            number = float(self.max_value)
        if self.precision is not None:
            number = round(number, self.precision)
        return number


@dataclass(frozen=True)
class JsonOutputTemplate:
    """可从 JSON 模板解析出的输出规格。

    `template` 保存原始 JSON 结构，`fields` 保存所有需要模型预测的叶子字段。
    常量字段会保留在 template 里，不会生成输出 head。例如：

    ```python
    {
        "action": {"$type": "enum", "choices": ["move", "attack"]},
        "schema_version": 1,
    }
    ```

    其中 `action` 会成为模型字段，`schema_version` 会在渲染 JSON 时原样保留。
    """

    template: Any
    fields: Tuple[JsonFieldSpec, ...]

    @classmethod
    def from_template(cls, template: Mapping[str, Any]) -> "JsonOutputTemplate":
        """解析用户传入的 JSON 模板。

        模板里至少要有一个带 `$type` 的字段，否则模型没有任何可预测目标。
        解析过程不会改动调用方传入的原始 dict。
        """
        fields: List[JsonFieldSpec] = []
        cls._collect_fields(template, (), fields)
        if not fields:
            raise ValueError("JSON 输出模板至少需要一个字段格式定义")
        return cls(template=_copy_json_value(template), fields=tuple(fields))

    @classmethod
    def _collect_fields(
        cls,
        node: Any,
        path: JsonPath,
        fields: List[JsonFieldSpec],
    ) -> None:
        """深度优先遍历模板，收集所有 `$type` 字段。

        遍历规则：
        - dict 且包含 `$type`：认为这是一个字段规格，不再继续往下遍历。
        - dict 且不包含 `$type`：继续遍历每个子字段。
        - list：把数组下标加入路径后继续遍历。
        - 普通常量：视为固定 JSON 内容，不需要模型预测。
        """
        if isinstance(node, Mapping):
            if FIELD_TYPE_KEY in node:
                fields.append(JsonFieldSpec.from_template_value(path, node))
                return
            for key, value in node.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object 的 key 必须是字符串")
                cls._collect_fields(value, (*path, key), fields)
            return

        if isinstance(node, list):
            for index, value in enumerate(node):
                cls._collect_fields(value, (*path, index), fields)

    @property
    def field_keys(self) -> Tuple[str, ...]:
        """返回所有模型预测字段的点号 key，顺序与模板遍历顺序一致。"""
        return tuple(field.key for field in self.fields)

    def render(
        self,
        values: Mapping[Union[str, JsonPath], Any],
        *,
        strict: bool = True,
    ) -> Any:
        """把字段值填入模板，返回 JSON 兼容的 Python 对象。

        `values` 只需要提供被 `$type` 标记的字段；普通常量字段会自动从模板复制。
        如果某个字段没提供，则使用 `JsonFieldSpec.default_value()`。
        """
        specs = {field.path: field for field in self.fields}

        def render_node(node: Any, path: JsonPath) -> Any:
            # 如果当前位置是模型字段，就取调用方给的值并做类型/范围规范化。
            spec = specs.get(path)
            if spec is not None:
                return spec.normalize(_lookup_value(values, path), strict=strict)

            # 普通 object 递归渲染。若这里还能看到 `$type`，说明字段解析状态不一致，
            # 直接报错比静默输出错误 JSON 更好排查。
            if isinstance(node, Mapping):
                if FIELD_TYPE_KEY in node:
                    raise ValueError(f"{path_to_key(path)} 缺少字段格式定义")
                return {key: render_node(value, (*path, key)) for key, value in node.items()}

            # 数组里也允许放字段规格或常量，路径用数组下标区分。
            if isinstance(node, list):
                return [render_node(value, (*path, index)) for index, value in enumerate(node)]

            # 常量字段直接深拷贝，避免返回对象和模板共享引用。
            return _copy_json_value(node)

        return render_node(self.template, ())

    def dumps(
        self,
        values: Mapping[Union[str, JsonPath], Any],
        *,
        strict: bool = True,
        ensure_ascii: bool = False,
    ) -> str:
        """把填充后的模板序列化成 JSON 字符串。

        默认 `ensure_ascii=False`，这样中文字段值会直接显示为中文，便于日志查看。
        `separators` 使用紧凑格式，方便把输出作为动作消息或接口 payload。
        """
        return json.dumps(
            self.render(values, strict=strict),
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
        )


class JsonOutputHead(nn.Module):
    """把 Decoder hidden state 解码成模板字段，再渲染为 JSON。

    输入：
        hidden_states: `[batch, d_model]` 或 `[batch, seq_len, d_model]`。
            如果传入序列，会默认取最后一个时间步，表示“基于当前历史生成下一步 JSON”。

    输出：
        一个 dict，key 是模板字段路径，例如 `"action"`、`"target.x"`；
        value 是对应字段的 raw tensor：
        - enum/string/boolean/integer: `[batch, num_classes]` 分类 logits。
        - number: `[batch, 1]` 回归值。

    训练时建议直接使用 forward 返回的 raw tensor 计算每个字段的 loss；
    推理时再调用 `decode()` 得到 JSON 兼容对象。
    """

    def __init__(
        self,
        d_model: int,
        output_template: Union[JsonOutputTemplate, Mapping[str, Any]],
    ) -> None:
        super().__init__()
        # 允许调用方传已经解析好的 JsonOutputTemplate，也允许直接传普通 dict 模板。
        if isinstance(output_template, JsonOutputTemplate):
            self.output_template = output_template
        else:
            self.output_template = JsonOutputTemplate.from_template(output_template)

        self.field_specs = self.output_template.fields
        self.heads = nn.ModuleDict()
        self._module_keys: Dict[str, str] = {}
        for index, field in enumerate(self.field_specs):
            # ModuleDict 的 key 不使用字段名本身，因为字段名可能包含点号、数组下标等
            # 不适合模块命名的字符。这里用稳定的 field_0、field_1 保存模块，
            # 再用 _module_keys 记录“字段 key -> 模块 key”的映射。
            module_key = f"field_{index}"
            self._module_keys[field.key] = module_key
            self.heads[module_key] = nn.Linear(d_model, self._field_output_dim(field))

    @staticmethod
    def _field_output_dim(field: JsonFieldSpec) -> int:
        """根据字段类型决定输出维度。

        enum/string/boolean/integer 都按分类问题处理；number 按单值回归处理。
        integer 必须声明 min/max，这样才能把整数范围映射成固定数量的类别。
        """
        if field.field_type in {"enum", "string"}:
            if not field.choices:
                raise ValueError(f"{field.key} 需要 choices 才能使用固定输出头")
            return len(field.choices)
        if field.field_type == "boolean":
            return 2
        if field.field_type == "integer":
            if field.min_value is None or field.max_value is None:
                raise ValueError(f"{field.key} integer 字段需要 min 和 max")
            return int(field.max_value) - int(field.min_value) + 1
        return 1

    def forward(self, hidden_states: Tensor) -> Dict[str, Tensor]:
        """返回每个 JSON 字段的 raw tensor 输出。

        如果输入是 `[batch, seq_len, d_model]`，会取 `hidden_states[:, -1, :]`。
        这和自回归推理一致：最后一个时间步代表下一步动作/结构化输出的上下文。
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        elif hidden_states.dim() != 2:
            raise ValueError("hidden_states 必须是 [batch, d_model] 或 [batch, seq_len, d_model]")

        return {
            field.key: self.heads[self._module_keys[field.key]](hidden_states)
            for field in self.field_specs
        }

    def decode(self, field_outputs: Mapping[str, Tensor]) -> List[Any]:
        """把字段输出 tensor 解码成 JSON 兼容对象列表。

        `field_outputs` 通常来自 `forward()`。返回值长度等于 batch size，每一项都是
        已经填好字段、范围已裁剪、可直接 `json.dumps` 的 Python 对象。
        """
        if not field_outputs:
            return []

        first_tensor = next(iter(field_outputs.values()))
        batch_size = first_tensor.size(0)
        results: List[Any] = []
        for batch_index in range(batch_size):
            values: Dict[str, Any] = {}
            for field in self.field_specs:
                # 每个字段只取当前 batch 的一行输出，然后按字段类型解码成 Python 值。
                output = field_outputs[field.key][batch_index]
                values[field.key] = self._decode_field(field, output)
            results.append(self.output_template.render(values, strict=False))
        return results

    @staticmethod
    def _decode_field(field: JsonFieldSpec, output: Tensor) -> Any:
        """把单个字段的一行 raw tensor 解码成 Python 值。"""
        if field.field_type in {"enum", "string"}:
            if not field.choices:
                raise ValueError(f"{field.key} 缺少 choices")
            # 分类字段直接取 argmax。训练时可以对同一个 tensor 使用 CrossEntropyLoss。
            index = int(output.argmax(dim=-1).item())
            return field.choices[index]

        if field.field_type == "boolean":
            # boolean 使用两个类别：0 -> False，1 -> True。
            return bool(int(output.argmax(dim=-1).item()))

        if field.field_type == "integer":
            # integer 也按分类处理。类别 0 对应 min_value，类别 i 对应 min_value + i。
            index = int(output.argmax(dim=-1).item())
            value = int(field.min_value or 0) + index
            return field.normalize(value, strict=False)

        raw_value = output.reshape(-1)[0]
        if field.min_value is not None and field.max_value is not None:
            # 有范围的 number 先过 sigmoid 映射到 [0, 1]，再线性缩放到 [min, max]。
            # 这样模型原始输出无论多大，都能得到合法范围内的 JSON 数值。
            value = torch.sigmoid(raw_value).item()
            value = float(field.min_value) + value * (float(field.max_value) - float(field.min_value))
        else:
            # 没有范围约束时直接使用回归头的原始标量。
            value = raw_value.item()
        return field.normalize(value, strict=False)
