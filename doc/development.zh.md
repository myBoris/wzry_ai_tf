# 开发环境说明

本项目使用 `uv` 管理 Python 环境、依赖和测试命令。

## 安装 uv

如果本机还没有安装 `uv`，Windows PowerShell 可以使用：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后重新打开终端，确认版本：

```powershell
uv --version
```

## 创建环境并安装依赖

项目根目录已经包含：

```text
pyproject.toml
.python-version
```

进入项目目录后执行：

```powershell
uv sync --dev
```

这会自动创建 `.venv`，并安装运行依赖和测试依赖。

## 运行测试

运行全部 CPU/GPU 测试：

```powershell
uv run pytest
```

或者显式指定测试目录：

```powershell
uv run pytest test
```

只运行 CPU 版本：

```powershell
uv run pytest -m cpu
```

只运行 GPU 版本：

```powershell
uv run pytest -m gpu
```

如果机器没有 CUDA GPU，GPU 测试会自动跳过。

## 运行 Python 代码

```powershell
uv run python
```

也可以运行某个脚本：

```powershell
uv run python your_script.py
```

## 当前依赖

运行依赖：

- `torch`

开发依赖：

- `pytest`

如果后续要加入图像处理、数据集读取或训练可视化，可以继续在 `pyproject.toml` 中追加依赖，然后执行：

```powershell
uv sync --dev
```
