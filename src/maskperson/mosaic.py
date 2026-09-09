"""马赛克算法模块"""

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
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
    )
    mask_binary = (mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    return dilated > 0


def create_pixel_mosaic(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    expand_pixels: int = 0,
) -> np.ndarray:
    """像素马赛克 — 先膨胀 mask，再对覆盖区域做方块化。

    Args:
        image:         原始帧 [H, W, C], BGR 格式 (cv2)
        mask:          布尔掩码 [H, W], True/1 = 人体区域
        block_size:    马赛克方块大小，越大越模糊
        expand_pixels: mask 膨胀半径

    Returns:
        处理后的帧
    """
    out_img = image.copy()
    mask_expanded = expand_mask(mask, expand_pixels)

    y_coords, x_coords = np.where(mask_expanded > 0.5)

    for y, x in zip(y_coords, x_coords):
        bx = (x // block_size) * block_size
        by = (y // block_size) * block_size
        bx_end = min(bx + block_size, image.shape[1])
        by_end = min(by + block_size, image.shape[0])
        block_mean = np.mean(
            out_img[by:by_end, bx:bx_end], axis=(0, 1)
        ).astype(np.uint8)
        out_img[y, x] = block_mean
    return out_img
