"""核心处理逻辑 — 视频帧循环 + YOLO 推理"""

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from maskperson.config import Config
from maskperson.device import apply_device, resolve_device
from maskperson.mosaic import create_pixel_mosaic
from maskperson.tracker import TrackCache
from maskperson.video import VideoCapture, VideoWriter


def process_video(cfg: Config) -> None:
    """主处理流程：读取视频 → YOLO 推理 → 马赛克 → 输出。"""
    model = YOLO(cfg.model_weight)
    device = resolve_device(cfg.device)
    device_desc = apply_device(model, device)
    print(f"推理设备: {device_desc}")

    # H20/A100 等支持 fp16 的 GPU 启用半精度，吞吐约 2×
    # ultralytics >= 8.3 已移除 track() 的 half= 参数，改用 model.half() 全局切换
    if device.startswith("cuda"):
        try:
            model.half()
            print("半精度推理: 开启（fp16）")
        except Exception as e:  # pragma: no cover — 极少数 GPU/CUDA 版本不支持
            print(f"半精度推理: 关闭（{e}）")
    else:
        print("半精度推理: 关闭（CPU）")

    cache = TrackCache(
        smooth_window_size=cfg.smooth_window_size,
        max_age=cfg.track_max_age,
    )

    Path(cfg.temp_no_audio).parent.mkdir(parents=True, exist_ok=True)

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

                # YOLO 推理 — 只检测人体 (class 0)
                # retina_masks=False: 在推理网络分辨率(640)出 mask，后续一次性 resize 到原图，
                # 避免 1920x1080 大 mask 在 CPU/GPU 间搬运。
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
                anonymized = frame.copy()

                if res.boxes is not None and res.boxes.id is not None:
                    track_ids = res.boxes.id.cpu().numpy().astype(int)  # type: ignore[union-attr]
                    masks_data = res.masks.data.cpu().numpy()  # type: ignore[union-attr]  # [N, H_mask, W_mask]

                    for tid, mask_raw in zip(track_ids, masks_data):
                        # resize mask 到原图尺寸
                        mask_resized = cv2.resize(
                            mask_raw,
                            (info.width, info.height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        smooth_mask = cache.update(tid, mask_resized, frame_idx)
                        anonymized = create_pixel_mosaic(
                            anonymized,
                            smooth_mask,
                            cfg.mosaic_block_size,
                            expand_pixels=cfg.expand_pixels,
                        )

                # 清理过期 track
                cache.clean_old(frame_idx)
                writer.write(anonymized)
                frame_idx += 1

                if frame_idx % 50 == 0:
                    print(f"  已处理 {frame_idx} 帧，活跃 track: {cache.active_count}")

    print("  帧处理完成")
