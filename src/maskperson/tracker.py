"""跟踪器模块 — 目标跟踪 + 时序平滑 + 内存管理"""

from __future__ import annotations

from typing import Any

import numpy as np


class TrackCache:
    """滑动窗口缓存，支持时序平滑和过期清理。

    同时支持 CPU（numpy）和 GPU（torch.Tensor）两种 mask 后端，由构造时的
    ``device`` 参数决定；后续 ``update`` 返回相同后端的 mask。

    Attributes:
        mask_history:  track_id -> 近期 mask 列表（numpy 或 torch.Tensor）
        last_seen:     track_id -> 最后出现帧号
    """

    def __init__(
        self,
        smooth_window_size: int,
        max_age: int,
        device: str = "cpu",
    ) -> None:
        self.smooth_window_size = smooth_window_size
        self.max_age = max_age
        self.device = device
        self.mask_history: dict[int, list[Any]] = {}
        self.last_seen: dict[int, int] = {}

    def update(
        self, track_id: int, mask: Any, frame_idx: int
    ) -> Any:
        """更新缓存并返回平滑后的 mask。

        Args:
            track_id: 目标跟踪 ID
            mask:     当前帧的原始掩码 [H, W]（numpy 或 torch.Tensor）
            frame_idx: 当前帧编号

        Returns:
            平滑后的布尔掩码，与输入同后端。
        """
        self.last_seen[track_id] = frame_idx

        if track_id not in self.mask_history:
            self.mask_history[track_id] = []

        hist = self.mask_history[track_id]
        hist.append(mask)
        if len(hist) > self.smooth_window_size:
            hist.pop(0)

        if self.device.startswith("cuda"):
            # GPU 路径
            import torch
            stacked = torch.stack([m.to(torch.float32) for m in hist], dim=0)
            return stacked.mean(dim=0) > 0.5

        # CPU 路径（保持原 numpy 语义）
        np_hist = [np.asarray(m, dtype=np.float32) for m in hist]
        avg_mask = np.mean(np_hist, axis=0)
        return avg_mask > 0.5

    def clean_old(self, frame_idx: int) -> None:
        """清理长时间消失的 track，释放内存。"""
        stale = [
            tid
            for tid, last_frame in self.last_seen.items()
            if frame_idx - last_frame > self.max_age
        ]
        for tid in stale:
            del self.mask_history[tid]
            del self.last_seen[tid]

    @property
    def active_count(self) -> int:
        return len(self.mask_history)
