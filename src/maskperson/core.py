"""核心处理逻辑 — 视频帧循环 + YOLO 推理。

性能关键路径：
  - YOLO 推理在 GPU（fp16）。
  - mask resize / 时序平滑 / mask 膨胀 / 像素马赛克全部留在 GPU 上完成，
    避免 1920x1080 的大 mask 在 CPU 上做 4M 像素操作拖慢整条流水线。

profile 显示（1080p，14 track）：
  - CPU mask resize 单帧 ~80 ms × 14 = 1.1 s（最大瓶颈）
  - GPU mosaic 仅 ~5 ms
  - GPU infer ~34 ms
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from maskperson.config import Config
from maskperson.device import apply_device, resolve_device
from maskperson.tracker import TrackCache
from maskperson.video import VideoCapture, VideoWriter


def _mosaic_with_mask_gpu(
    image_gpu: torch.Tensor,
    mask_gpu: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """GPU 像素马赛克：块均值 + 广播 + mask 选择，全程 CUDA。

    Args:
        image_gpu: [H, W, 3] uint8，CUDA。
        mask_gpu:  [H, W] bool，CUDA。
        block_size: 块边长。
    """
    h, w = image_gpu.shape[:2]
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return image_gpu

    img_crop = image_gpu[: bh * block_size, : bw * block_size].float()
    blocks = img_crop.reshape(bh, block_size, bw, block_size, 3)
    blocks = blocks.permute(0, 2, 1, 3, 4)            # [bh, bw, bs, bs, C]
    means = blocks.mean(dim=(2, 3))                    # [bh, bw, C]

    mos = means.repeat_interleave(block_size, dim=0) \
              .repeat_interleave(block_size, dim=1)    # [bh*bs, bw*bs, C]
    mos = mos.to(torch.uint8)

    mos_full = image_gpu.clone()
    mos_full[: bh * block_size, : bw * block_size] = mos

    m = mask_gpu[: bh * block_size, : bw * block_size, None]
    return torch.where(
        m,
        mos_full[: bh * block_size, : bw * block_size],
        image_gpu[: bh * block_size, : bw * block_size],
    )


def process_video(cfg: Config) -> None:
    """主处理流程：读取视频 → YOLO 推理（GPU）→ 马赛克（GPU）→ 输出。"""
    # 抑制 ultralytics 的 'half' deprecation 等噪声告警；
    # WARNING 级别保留真正的错误，INFO 级一次性提示（如 half）会被屏蔽。
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    model = YOLO(cfg.model_weight)
    device = resolve_device(cfg.device)
    device_desc = apply_device(model, device)
    print(f"推理设备: {device_desc}")

    # H20/A100 等支持 fp16 的 GPU 启用半精度，吞吐约 2×
    # ultralytics >= 8.3 已移除 track() 的 half= 参数，改用 model.half() 全局切换
    use_gpu = device.startswith("cuda")
    if use_gpu:
        try:
            model.half()
            print("半精度推理: 开启（fp16）")
        except Exception as e:  # pragma: no cover — 极少数 GPU/CUDA 版本不支持
            print(f"半精度推理: 关闭（{e}）")
            use_gpu = False
    else:
        print("半精度推理: 关闭（CPU）")

    cache = TrackCache(
        smooth_window_size=cfg.smooth_window_size,
        max_age=cfg.track_max_age,
        device=device,
    )

    Path(cfg.temp_no_audio).parent.mkdir(parents=True, exist_ok=True)

    # 预计算 mask 膨胀 kernel（GPU 上一次性建好）
    if use_gpu and cfg.expand_pixels > 0:
        k = cfg.expand_pixels * 2 + 1
        expand_kernel = torch.ones(
            (1, 1, k, k), dtype=torch.float32, device=device
        )

    with VideoCapture(cfg.input_video) as cap:
        info = cap.info
        print(f"开始处理视频: {cfg.input_video}")
        print(f"  分辨率: {info.width}x{info.height}, FPS: {info.fps}")

        with VideoWriter(
            cfg.temp_no_audio,
            fps=info.fps,
            width=info.width,
            height=info.height,
        ) as writer:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # === YOLO 推理（GPU）===
                results = model.track(
                    frame,
                    conf=cfg.conf_thresh,
                    iou=cfg.iou_thresh,
                    classes=[0],
                    persist=True,
                    verbose=False,
                    retina_masks=False,
                )
                res = results[0]

                # === 上传 frame 到 GPU ===
                frame_gpu = torch.as_tensor(frame, device=device)

                if res.boxes is not None and res.boxes.id is not None:
                    track_ids = res.boxes.id.cpu().numpy().astype(int)  # type: ignore[union-attr]
                    # masks_data [N, h_mask, w_mask] 在 GPU（ultralytics 内部已上传）
                    masks_data = res.masks.data  # type: ignore[union-attr]

                    # 把所有 mask 一次性上采样到原图尺寸（替代 CPU cv2.resize，节省 ~80ms/帧）
                    masks_resized = F.interpolate(
                        masks_data[:, None],  # [N, 1, h, w]
                        size=(info.height, info.width),
                        mode="nearest",
                    ).squeeze(1)  # [N, H, W]

                    out_gpu = frame_gpu
                    for n, tid in enumerate(track_ids):
                        m = masks_resized[n]  # [H, W] GPU
                        # mask 膨胀（GPU conv2d，替代 cv2.dilate）
                        if cfg.expand_pixels > 0:
                            m = F.conv2d(
                                m[None, None].float(),
                                expand_kernel,
                                padding=k // 2,
                            )[0, 0] > 0.5
                        else:
                            m = m > 0.5

                        # 时序平滑（GPU）：直接调用 TrackCache.update（同后端）
                        smooth = cache.update(int(tid), m, frame_idx)
                        out_gpu = _mosaic_with_mask_gpu(
                            out_gpu, smooth, cfg.mosaic_block_size,
                        )

                    anonymized = out_gpu.cpu().numpy()
                else:
                    anonymized = frame

                # 清理过期 track
                cache.clean_old(frame_idx)
                writer.write(anonymized)
                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"  已处理 {frame_idx} 帧，活跃 track: {cache.active_count}")

    print("  帧处理完成")
