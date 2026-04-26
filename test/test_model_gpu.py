"""GPU 版本模型测试入口。"""

from __future__ import annotations

import pytest

from model_test_cases import MODEL_TEST_CASES, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="需要安装 PyTorch 才能测试模型"),
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="需要 CUDA GPU 才能运行 GPU 测试"),
]


@pytest.mark.parametrize(
    ("case_name", "case_func"),
    [pytest.param(name, func, id=name) for name, func in MODEL_TEST_CASES],
)
def test_model_gpu(case_name: str, case_func) -> None:
    """在 CUDA GPU 上运行共享模型测试用例。"""
    del case_name
    case_func(torch.device("cuda"))
