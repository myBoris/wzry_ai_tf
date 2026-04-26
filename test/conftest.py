"""pytest 本地配置。"""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_utf8_stdio() -> None:
    """避免 Windows 非 UTF-8 控制台把中文测试日志转成 \\u 转义。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_prefer_utf8_stdio()

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
