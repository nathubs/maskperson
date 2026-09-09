"""运行时设备选择：CUDA 优先，CPU 兜底。

不直接 ``import torch``，而是延迟到函数调用时 —
ultralytics 已强依赖 torch，理论上一定可用，但延迟 import
让模块文件本身可被静态分析工具在不安装 torch 的环境下读取。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


def resolve_device(mode: str = "auto") -> str:
    """将 ``mode`` 解析为最终设备字符串（``cpu`` / ``cuda`` / ``cuda:N``）。

    Args:
        mode: ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N``。

    Raises:
        RuntimeError: 显式指定 ``cuda`` 但 CUDA 不可用，或索引越界。
        ValueError: 未知模式或 ``cuda:N`` 索引非整数。
    """
    import torch

    if mode == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "已显式指定 --device cuda，但 torch.cuda.is_available() 返回 False。"
                "请确认：1) 机器有 NVIDIA GPU；2) 已安装匹配 CUDA 版本的 NVIDIA 驱动；"
                "3) `uv sync` 安装的是 CUDA 版 torch 而非 CPU 版。"
            )
        return "cuda"
    if mode.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"已显式指定 --device {mode}，但 CUDA 不可用。")
        try:
            idx = int(mode.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"非法 device 格式: {mode}（索引必须为整数）") from e
        total = torch.cuda.device_count()
        if idx >= total:
            raise RuntimeError(
                f"指定 {mode} 超出可用设备数（共 {total} 块）"
            )
        return mode
    raise ValueError(
        f"未知 device 模式: {mode}（应为 auto / cpu / cuda / cuda:N）"
    )


def apply_device(model: "YOLO", device: str) -> str:
    """将 ultralytics 模型搬到 ``device`` 并返回人类可读的设备描述。

    示例返回值：``"cuda (NVIDIA GeForce RTX 4090)"`` 或 ``"cpu"``。
    """
    import torch

    model.to(device)
    if device.startswith("cuda"):
        idx = 0 if device == "cuda" else int(device.split(":", 1)[1])
        name = torch.cuda.get_device_name(idx)
        return f"{device} ({name})"
    return device