"""马赛克算法模块 — 向量化实现，避免 Python 双循环拖慢 GPU 流水线。

原实现用 ``for y, x in zip(...)`` 逐像素赋值，1920x1080 单帧轻松吃掉 200-500 ms，
导致 H20/A100 等 GPU 长期空载等待 CPU。新实现把"按 block 取均值再回填"
拆成两步：先 ``view + sum`` 求块内像素均值得到一张 ``[H/block, W/block, C]`` 小图，
再用 ``kron`` / 复制把块均值广播回原图尺寸，最后用 mask 一次性赋值。
"""

from __future__ import annotations

import cv2
import numpy as np


def expand_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    """膨胀 mask，覆盖快速移动导致的边缘漏出区域。

    Args:
        mask:    布尔掩码 [H, W], True/1 = 人体区域
        expand_pixels: 膨胀半径（像素），越大覆盖越多边缘

    Returns:
        膨胀后的布尔掩码
    """
    if expand_pixels <= 0:
        return mask.astype(bool)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
    )
    mask_binary = (mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    return dilated > 0


def _block_means(image: np.ndarray, block_size: int) -> np.ndarray:
    """计算每个 ``block_size x block_size`` 块的像素均值。

    关键技巧：``reshape`` + ``sum`` 把每个块看成 ``(N, block*block, C)``，
    然后对中间轴 ``sum`` 得到 ``(N, C)``，最后 ``reshape`` 回块网格。
    比 ``cv2.resize(INTER_AREA)`` 在小 block（如 16/20）上更准。

    Args:
        image: [H, W, C] uint8。
        block_size: 块边长，需能整除 H/W（否则右侧/底部用 INTER_AREA 兜底）。

    Returns:
        [H/block, W/block, C] float32 均值。
    """
    h, w = image.shape[:2]
    bh = h // block_size
    bw = w // block_size

    if bh == 0 or bw == 0:
        # 整张图都不够一个 block：用 INTER_AREA 兜底
        return cv2.resize(image, (max(bw, 1), max(bh, 1)),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    # 裁掉右侧/底部不足一格的像素（少数几列/几行，视觉无感）
    cropped = image[: bh * block_size, : bw * block_size]
    # reshape 为 (bh, block, bw, block, C) -> 转置成 (bh, bw, block, block, C)
    blocks = cropped.reshape(bh, block_size, bw, block_size, image.shape[2])
    blocks = blocks.transpose(0, 2, 1, 3, 4)
    return blocks.mean(axis=(2, 3))


def create_pixel_mosaic(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    expand_pixels: int = 0,
) -> np.ndarray:
    """像素马赛克 — 先膨胀 mask，再对覆盖区域做方块化（向量化）。

    Args:
        image:         原始帧 [H, W, C], BGR 格式 (cv2)。
        mask:          布尔掩码 [H, W], True/1 = 人体区域。
        block_size:    马赛克方块大小，越大越模糊。
        expand_pixels: mask 膨胀半径。

    Returns:
        处理后的帧。
    """
    if block_size <= 0:
        raise ValueError(f"block_size 必须 > 0，实际 {block_size}")

    out_img = image
    mask_expanded = expand_mask(mask, expand_pixels)
    if not mask_expanded.any():
        return out_img

    h, w = image.shape[:2]
    bh = h // block_size
    bw = w // block_size

    if bh == 0 or bw == 0:
        # 极端情况：图像极小，按 INTER_AREA 整图缩放再放大
        small = cv2.resize(image, (bw, bh),
                           interpolation=cv2.INTER_AREA)
        mos = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out_img = np.where(mask_expanded[..., None], mos, out_img)
        return out_img

    # 1) 求每个块的均值 → [bh, bw, C]（float32）
    means = _block_means(image, block_size)
    # 2) 广播回原图尺寸 [H, W, C]：每个块重复 block_size 次；round+cast 与原 for 循环行为一致
    mos_full = np.repeat(np.repeat(means, block_size, axis=0),
                         block_size, axis=1).round().astype(np.uint8)
    # 3) 拼成原图尺寸的 mosaic：mos_full 覆盖左上完整块区，剩余行/列从原图补齐
    mos_padded = image.copy()
    mos_padded[: bh * block_size, : bw * block_size] = mos_full
    # 4) 对 mask 区域一次性赋值；非 mask 区域保留原图
    return np.where(mask_expanded[..., None], mos_padded, image)


def create_solid_black_mosaic(
    image: np.ndarray,
    mask: np.ndarray,
    expand_pixels: int = 0,
) -> np.ndarray:
    """实心黑色覆盖 — mask 区域全部置 0。最强脱敏，完全不可识别。

    忽略 ``block_size``（对纯黑无意义）。

    Args:
        image:         原始帧 [H, W, C]。
        mask:          布尔掩码 [H, W]。
        expand_pixels: mask 膨胀半径。
    """
    mask_expanded = expand_mask(mask, expand_pixels)
    if not mask_expanded.any():
        return image
    black = np.zeros_like(image)
    return np.where(mask_expanded[..., None], black, image)


def create_checkerboard_mosaic(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    expand_pixels: int = 0,
) -> np.ndarray:
    """黑白棋盘格覆盖 — mask 区域用 ``block_size`` 大小的黑白方块交错填充。

    高对比度棋盘格消除所有轮廓信息；视觉上比纯黑更明显"已被脱敏"。
    ``block_size`` 控制方格大小，越大越粗糙。

    Args:
        image:         原始帧 [H, W, C]。
        mask:          布尔掩码 [H, W]。
        block_size:    棋盘方块边长（像素）。0 或负数当作 1。
        expand_pixels: mask 膨胀半径。
    """
    if block_size <= 0:
        block_size = 1

    mask_expanded = expand_mask(mask, expand_pixels)
    if not mask_expanded.any():
        return image

    h, w = image.shape[:2]
    bh = h // block_size
    bw = w // block_size

    # 构造 [bh, bw] 的 0/1 棋盘，然后广播到原图尺寸 [H, W]
    pattern = np.indices((bh, bw)).sum(axis=0) % 2  # 0 = 黑, 1 = 白
    # 全图尺寸的棋盘（pad 剩余行/列从原图补齐，pattern 自然在左上完整区）
    full_pattern = np.zeros((h, w), dtype=np.uint8)
    full_pattern[: bh * block_size, : bw * block_size] = np.repeat(
        np.repeat(pattern, block_size, axis=0), block_size, axis=1
    )

    # mask 区域内：偶数格黑，奇数格白
    cb = np.where(full_pattern[..., None] == 0, 0, 255).astype(np.uint8)
    cb = np.broadcast_to(cb, image.shape).copy()
    return np.where(mask_expanded[..., None], cb, image)


def apply_mosaic(
    image: np.ndarray,
    mask: np.ndarray,
    style: str,
    block_size: int,
    expand_pixels: int = 0,
) -> np.ndarray:
    """按 ``style`` 分派到对应脱敏函数。

    Args:
        image, mask, block_size, expand_pixels: 同各风格函数。
        style: ``pixel`` / ``solid_black`` / ``checkerboard``。

    Raises:
        ValueError: 未知风格。
    """
    if style == "pixel":
        return create_pixel_mosaic(
            image, mask,
            block_size=block_size,
            expand_pixels=expand_pixels,
        )
    if style == "solid_black":
        return create_solid_black_mosaic(image, mask, expand_pixels=expand_pixels)
    if style == "checkerboard":
        return create_checkerboard_mosaic(
            image, mask,
            block_size=block_size,
            expand_pixels=expand_pixels,
        )
    raise ValueError(
        f"未知 mosaic_style: {style!r}（应为 pixel / solid_black / checkerboard）"
    )
