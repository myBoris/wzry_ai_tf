"""模型测试共享用例。

本文件不直接作为 pytest 入口收集，而是给 CPU/GPU 两套测试入口复用。
每个用例都接收一个 `device` 参数：

- `test_model_cpu.py` 传入 `torch.device("cpu")`
- `test_model_gpu.py` 传入 `torch.device("cuda")`

这样 CPU 和 GPU 可以运行完全相同的行为断言，避免两份测试逻辑慢慢漂移。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
LOGGER = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    import torch

    from model import (  # noqa: E402
        ImageEncoderConfig,
        ImageTokenEncoder,
        JsonOutputHead,
        JsonOutputTemplate,
        MultiModalEncoderConfig,
        MultiModalTransformerConfig,
        MultiModalTransformerDecoder,
        TextEncoderConfig,
        TransformerDecoder,
        TransformerDecoderConfig,
        causal_mask,
        make_decoder_masks,
        padding_mask,
    )


def _json_action_template() -> dict:
    return {
        "action": {"$type": "enum", "choices": ["move", "attack", "skill"]},
        "target": {
            "x": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
            "y": {"$type": "number", "min": 0.0, "max": 1.0, "precision": 2},
        },
        "skill_id": {"$type": "integer", "min": 1, "max": 3, "default": 1},
        "use_skill": {"$type": "boolean", "default": False},
        "schema_version": 1,
    }


def _assert_shape(name: str, tensor: "torch.Tensor", expected: tuple[int, ...]) -> None:
    """断言张量形状，并在日志里留下可排查的上下文。"""
    actual = tuple(tensor.shape)
    LOGGER.info(
        "%s shape=%s expected=%s dtype=%s device=%s",
        name,
        actual,
        expected,
        tensor.dtype,
        tensor.device,
    )
    assert actual == expected, f"{name} shape mismatch: expected {expected}, got {actual}"


def _log_tensor_stats(name: str, tensor: "torch.Tensor") -> None:
    """记录张量的紧凑统计信息，避免日志里刷满大矩阵。"""
    if tensor.numel() == 0:
        LOGGER.info("%s shape=%s dtype=%s empty tensor", name, tuple(tensor.shape), tensor.dtype)
        return

    values = tensor.detach().float()
    LOGGER.info(
        "%s shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        values.min().item(),
        values.max().item(),
        values.mean().item(),
    )


def test_attention_masks_shape_and_values(device: "torch.device") -> None:
    """测试因果 mask 和 padding mask 的形状与关键取值。"""
    tokens = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ],
        device=device,
    )

    attn_mask = causal_mask(seq_len=4, device=tokens.device)
    pad_mask = padding_mask(tokens, padding_idx=0)
    auto_attn_mask, auto_pad_mask = make_decoder_masks(tokens, padding_idx=0)

    LOGGER.info("tokens=%s padding_idx=0 device=%s", tokens.cpu().tolist(), device)
    LOGGER.info("attn_mask=%s", attn_mask.int().cpu().tolist())
    LOGGER.info("pad_mask=%s", pad_mask.int().cpu().tolist())

    _assert_shape("attn_mask", attn_mask, (4, 4))
    _assert_shape("pad_mask", pad_mask, (2, 4))
    _assert_shape("auto_attn_mask", auto_attn_mask, (4, 4))
    assert auto_pad_mask is not None, "auto_pad_mask should be created when padding_idx is set"
    _assert_shape("auto_pad_mask", auto_pad_mask, (2, 4))

    assert attn_mask[0, 1].item() is True, "causal mask should block attention to future tokens"
    assert attn_mask[1, 0].item() is False, "causal mask should allow attention to previous tokens"
    expected_pad_mask = [
        [False, False, True, True],
        [False, False, False, True],
    ]
    actual_pad_mask = pad_mask.cpu().tolist()
    assert actual_pad_mask == expected_pad_mask, (
        f"padding mask mismatch: expected {expected_pad_mask}, got {actual_pad_mask}"
    )


def test_json_output_template_renders_field_formats(device: "torch.device") -> None:
    """测试 JSON 模板能定义每个字段的输出格式。"""
    LOGGER.info("json template render device=%s", device)
    output_template = JsonOutputTemplate.from_template(_json_action_template())

    rendered = output_template.render(
        {
            "action": "skill",
            "target.x": 0.123,
            "target.y": 0.987,
            "skill_id": 2,
            "use_skill": True,
        }
    )

    LOGGER.info("json template fields=%s rendered=%s", output_template.field_keys, rendered)
    assert output_template.field_keys == (
        "action",
        "target.x",
        "target.y",
        "skill_id",
        "use_skill",
    )
    assert rendered == {
        "action": "skill",
        "target": {"x": 0.12, "y": 0.99},
        "skill_id": 2,
        "use_skill": True,
        "schema_version": 1,
    }
    assert output_template.dumps({"action": "move"}) == (
        '{"action":"move","target":{"x":0.0,"y":0.0},'
        '"skill_id":1,"use_skill":false,"schema_version":1}'
    )


def test_json_output_head_decodes_hidden_states(device: "torch.device") -> None:
    """测试 JSON 输出头能把 hidden states 解码为 JSON 对象。"""
    torch.manual_seed(0)

    output_head = JsonOutputHead(d_model=16, output_template=_json_action_template()).to(device)
    hidden_states = torch.randn(2, 4, 16, device=device)

    field_outputs = output_head(hidden_states)
    json_values = output_head.decode(field_outputs)

    LOGGER.info("json field keys=%s values=%s", list(field_outputs), json_values)
    assert set(field_outputs) == {"action", "target.x", "target.y", "skill_id", "use_skill"}
    assert field_outputs["action"].shape == (2, 3)
    assert field_outputs["target.x"].shape == (2, 1)
    assert len(json_values) == 2
    assert json_values[0]["action"] in {"move", "attack", "skill"}
    assert 0.0 <= json_values[0]["target"]["x"] <= 1.0
    assert 1 <= json_values[0]["skill_id"] <= 3
    assert isinstance(json_values[0]["use_skill"], bool)


def test_decoder_config_rejects_invalid_heads(device: "torch.device") -> None:
    """d_model 不能被 num_heads 整除时，配置应直接报错。"""
    LOGGER.info("reject invalid decoder config d_model=30 num_heads=8 device=%s", device)
    with pytest.raises(ValueError, match="d_model"):
        TransformerDecoderConfig(
            vocab_size=16,
            d_model=30,
            num_heads=8,
        )
    LOGGER.info("invalid decoder config rejected")


def test_transformer_decoder_forward_shape(device: "torch.device") -> None:
    """测试纯 Transformer Decoder 的输出形状。"""
    torch.manual_seed(0)

    config = TransformerDecoderConfig(
        vocab_size=32,
        output_dim=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        max_seq_len=16,
        dropout=0.0,
        cross_attention=True,
    )
    LOGGER.info("decoder forward config=%s", config)
    decoder = TransformerDecoder.from_config(config).to(device)
    decoder.eval()

    input_ids = torch.randint(1, config.vocab_size, (2, 6), device=device)
    memory = torch.randn(2, 5, config.d_model, device=device)
    LOGGER.info(
        "decoder input_ids shape=%s sample=%s device=%s",
        tuple(input_ids.shape),
        input_ids[0].cpu().tolist(),
        device,
    )
    _log_tensor_stats("decoder memory", memory)

    with torch.no_grad():
        logits = decoder(input_ids=input_ids, memory=memory)

    _log_tensor_stats("decoder logits", logits)
    _assert_shape("decoder logits", logits, (2, 6, config.resolved_output_dim))


def test_transformer_decoder_return_dict(device: "torch.device") -> None:
    """测试 Decoder 能返回 hidden states 和注意力权重。"""
    torch.manual_seed(0)

    config = TransformerDecoderConfig(
        vocab_size=24,
        d_model=48,
        num_layers=2,
        num_heads=4,
        dim_feedforward=96,
        max_seq_len=16,
        dropout=0.0,
        cross_attention=True,
    )
    LOGGER.info("decoder return_dict config=%s", config)
    decoder = TransformerDecoder.from_config(config).to(device)
    decoder.eval()

    input_ids = torch.randint(1, config.vocab_size, (2, 5), device=device)
    memory = torch.randn(2, 7, config.d_model, device=device)
    LOGGER.info(
        "return_dict input_ids shape=%s sample=%s device=%s",
        tuple(input_ids.shape),
        input_ids[0].cpu().tolist(),
        device,
    )
    _log_tensor_stats("return_dict memory", memory)

    with torch.no_grad():
        output = decoder(
            input_ids=input_ids,
            memory=memory,
            return_hidden=True,
            return_attentions=True,
            return_dict=True,
        )

    _log_tensor_stats("return_dict logits", output.logits)
    _assert_shape("return_dict logits", output.logits, (2, 5, config.resolved_output_dim))
    assert output.hidden_states is not None, "hidden_states should be returned when return_hidden=True"
    _assert_shape("hidden_states", output.hidden_states, (2, 5, config.d_model))
    assert output.attentions is not None, "attentions should be returned when return_attentions=True"
    LOGGER.info("attention layers=%d expected=%d", len(output.attentions), config.num_layers)
    assert len(output.attentions) == config.num_layers
    _assert_shape("self_attn", output.attentions[0]["self_attn"], (2, config.num_heads, 5, 5))
    _assert_shape("cross_attn", output.attentions[0]["cross_attn"], (2, config.num_heads, 5, 7))


def test_image_encoder_rejects_more_than_ten_images(device: "torch.device") -> None:
    """图片数量超过 10 张时，应抛出错误。"""
    config = ImageEncoderConfig(
        d_model=32,
        image_channels=3,
        max_images=10,
        cnn_channels=(8, 16),
        dropout=0.0,
    )
    encoder = ImageTokenEncoder.from_config(config).to(device)
    encoder.eval()

    images = torch.randn(2, 11, 3, 32, 32, device=device)
    LOGGER.info(
        "too many images shape=%s max_images=%d device=%s",
        tuple(images.shape),
        config.max_images,
        device,
    )

    with pytest.raises(ValueError, match="最多"):
        encoder(images)
    LOGGER.info("too many images rejected")


def test_multimodal_model_forward_with_images_and_text(device: "torch.device") -> None:
    """测试完整多模态模型：最多 10 张图片 + 文字 + Decoder。"""
    torch.manual_seed(0)

    batch_size = 2
    num_images = 4
    text_len = 6
    target_len = 5
    text_vocab_size = 100
    action_vocab_size = 32
    d_model = 64

    config = MultiModalTransformerConfig(
        encoder=MultiModalEncoderConfig(
            image=ImageEncoderConfig(
                d_model=d_model,
                image_channels=3,
                max_images=10,
                cnn_channels=(8, 16),
                dropout=0.0,
            ),
            text=TextEncoderConfig(
                vocab_size=text_vocab_size,
                d_model=d_model,
                max_text_len=16,
                num_layers=1,
                num_heads=4,
                dim_feedforward=128,
                dropout=0.0,
            ),
            dropout=0.0,
        ),
        decoder=TransformerDecoderConfig(
            vocab_size=action_vocab_size,
            output_dim=action_vocab_size,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            dim_feedforward=128,
            max_seq_len=16,
            dropout=0.0,
            cross_attention=True,
        ),
    )
    LOGGER.info("multimodal forward config=%s", config)
    model = MultiModalTransformerDecoder.from_config(config).to(device)
    model.eval()

    images = torch.randn(batch_size, num_images, 3, 32, 32, device=device)
    text_input_ids = torch.randint(1, text_vocab_size, (batch_size, text_len), device=device)
    decoder_input_ids = torch.randint(
        1,
        action_vocab_size,
        (batch_size, target_len),
        device=device,
    )
    _log_tensor_stats("multimodal images", images)
    LOGGER.info(
        "text_input_ids shape=%s sample=%s device=%s",
        tuple(text_input_ids.shape),
        text_input_ids[0].cpu().tolist(),
        device,
    )
    LOGGER.info(
        "decoder_input_ids shape=%s sample=%s device=%s",
        tuple(decoder_input_ids.shape),
        decoder_input_ids[0].cpu().tolist(),
        device,
    )

    with torch.no_grad():
        logits = model(
            decoder_input_ids=decoder_input_ids,
            images=images,
            text_input_ids=text_input_ids,
        )

    _log_tensor_stats("multimodal logits", logits)
    _assert_shape("multimodal logits", logits, (batch_size, target_len, action_vocab_size))


def test_multimodal_model_return_json_output(device: "torch.device") -> None:
    """测试完整模型能按 JSON 模板输出结构化结果。"""
    torch.manual_seed(0)

    batch_size = 2
    num_images = 2
    text_len = 4
    target_len = 3
    text_vocab_size = 50
    action_vocab_size = 16
    d_model = 32

    config = MultiModalTransformerConfig(
        encoder=MultiModalEncoderConfig(
            image=ImageEncoderConfig(
                d_model=d_model,
                image_channels=3,
                max_images=10,
                cnn_channels=(8, 16),
                dropout=0.0,
            ),
            text=TextEncoderConfig(
                vocab_size=text_vocab_size,
                d_model=d_model,
                max_text_len=16,
                num_layers=0,
                num_heads=4,
                dim_feedforward=64,
                dropout=0.0,
            ),
            dropout=0.0,
        ),
        decoder=TransformerDecoderConfig(
            vocab_size=action_vocab_size,
            output_dim=action_vocab_size,
            d_model=d_model,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            max_seq_len=16,
            dropout=0.0,
            cross_attention=True,
        ),
        json_output_template=_json_action_template(),
    )
    model = MultiModalTransformerDecoder.from_config(config).to(device)
    model.eval()

    images = torch.randn(batch_size, num_images, 3, 32, 32, device=device)
    text_input_ids = torch.randint(1, text_vocab_size, (batch_size, text_len), device=device)
    decoder_input_ids = torch.randint(
        1,
        action_vocab_size,
        (batch_size, target_len),
        device=device,
    )

    with torch.no_grad():
        output = model(
            decoder_input_ids=decoder_input_ids,
            images=images,
            text_input_ids=text_input_ids,
            return_json=True,
            return_dict=True,
        )

    LOGGER.info("model json values=%s", output.json_values)
    _assert_shape("json logits", output.logits, (batch_size, target_len, action_vocab_size))
    assert output.hidden_states is None
    assert output.json_field_outputs is not None
    assert output.json_field_outputs["action"].shape == (batch_size, 3)
    assert output.json_values is not None
    assert len(output.json_values) == batch_size
    assert output.json_values[0]["action"] in {"move", "attack", "skill"}
    assert output.json_values[0]["schema_version"] == 1

    with torch.no_grad():
        json_only = model(
            decoder_input_ids=decoder_input_ids,
            images=images,
            text_input_ids=text_input_ids,
            return_json=True,
        )

    assert isinstance(json_only, list)
    assert len(json_only) == batch_size


def test_multimodal_model_return_memory_and_masks(device: "torch.device") -> None:
    """测试多模态 memory 和 padding mask 是否按图片、文字顺序拼接。"""
    torch.manual_seed(0)

    batch_size = 2
    num_images = 5
    text_len = 6
    target_len = 4
    text_vocab_size = 80
    action_vocab_size = 20
    d_model = 32

    config = MultiModalTransformerConfig(
        encoder=MultiModalEncoderConfig(
            image=ImageEncoderConfig(
                d_model=d_model,
                image_channels=3,
                max_images=10,
                cnn_channels=(8, 16),
                dropout=0.0,
            ),
            text=TextEncoderConfig(
                vocab_size=text_vocab_size,
                d_model=d_model,
                max_text_len=16,
                num_layers=0,
                num_heads=4,
                dim_feedforward=64,
                dropout=0.0,
                padding_idx=0,
            ),
            dropout=0.0,
        ),
        decoder=TransformerDecoderConfig(
            vocab_size=action_vocab_size,
            d_model=d_model,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            max_seq_len=16,
            dropout=0.0,
            cross_attention=True,
        ),
    )
    LOGGER.info("multimodal return_memory config=%s", config)
    model = MultiModalTransformerDecoder.from_config(config).to(device)
    model.eval()

    images = torch.randn(batch_size, num_images, 3, 32, 32, device=device)
    image_padding_mask = torch.zeros(batch_size, num_images, dtype=torch.bool, device=device)
    image_padding_mask[0, 3:] = True

    text_input_ids = torch.randint(1, text_vocab_size, (batch_size, text_len), device=device)
    text_input_ids[1, -2:] = 0

    decoder_input_ids = torch.randint(
        1,
        action_vocab_size,
        (batch_size, target_len),
        device=device,
    )
    LOGGER.info("image_padding_mask=%s", image_padding_mask.int().cpu().tolist())
    LOGGER.info("text_input_ids=%s", text_input_ids.cpu().tolist())
    LOGGER.info("decoder_input_ids shape=%s device=%s", tuple(decoder_input_ids.shape), device)

    with torch.no_grad():
        output = model(
            decoder_input_ids=decoder_input_ids,
            images=images,
            text_input_ids=text_input_ids,
            image_padding_mask=image_padding_mask,
            return_memory=True,
            return_dict=True,
        )

    _log_tensor_stats("return_memory logits", output.logits)
    _assert_shape("return_memory logits", output.logits, (batch_size, target_len, action_vocab_size))
    assert output.memory is not None, "memory should be returned when return_memory=True"
    _assert_shape("memory", output.memory, (batch_size, num_images + text_len, d_model))
    assert output.memory_key_padding_mask is not None, "memory_key_padding_mask should be returned with memory"
    _assert_shape(
        "memory_key_padding_mask",
        output.memory_key_padding_mask,
        (batch_size, num_images + text_len),
    )

    expected_mask = torch.cat([image_padding_mask, text_input_ids.eq(0)], dim=1)
    LOGGER.info("expected_memory_key_padding_mask=%s", expected_mask.int().cpu().tolist())
    LOGGER.info(
        "actual_memory_key_padding_mask=%s",
        output.memory_key_padding_mask.int().cpu().tolist(),
    )
    assert torch.equal(output.memory_key_padding_mask, expected_mask), (
        "memory_key_padding_mask mismatch: "
        f"expected {expected_mask.cpu().tolist()}, "
        f"got {output.memory_key_padding_mask.cpu().tolist()}"
    )


MODEL_TEST_CASES: tuple[tuple[str, Callable[["torch.device"], None]], ...] = (
    ("attention_masks_shape_and_values", test_attention_masks_shape_and_values),
    ("json_output_template_renders_field_formats", test_json_output_template_renders_field_formats),
    ("json_output_head_decodes_hidden_states", test_json_output_head_decodes_hidden_states),
    ("decoder_config_rejects_invalid_heads", test_decoder_config_rejects_invalid_heads),
    ("transformer_decoder_forward_shape", test_transformer_decoder_forward_shape),
    ("transformer_decoder_return_dict", test_transformer_decoder_return_dict),
    ("image_encoder_rejects_more_than_ten_images", test_image_encoder_rejects_more_than_ten_images),
    ("multimodal_model_forward_with_images_and_text", test_multimodal_model_forward_with_images_and_text),
    ("multimodal_model_return_json_output", test_multimodal_model_return_json_output),
    ("multimodal_model_return_memory_and_masks", test_multimodal_model_return_memory_and_masks),
)
