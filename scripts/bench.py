"""独立 benchmark — 直接读所有帧到内存，绕开 IO 测核心流水线瓶颈。

用法：
    .venv/bin/python scripts/bench.py -i input/input.mp4 -n 50

输出：每个阶段（推理 / 后处理 / 写盘）的平均耗时和占比。
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-n", "--frames", type=int, default=50,
                   help="测试用的前 N 帧（默认 50）")
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--expand", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) 一次性读所有帧到内存（绕开 IO 瓶颈）
    cap = cv2.VideoCapture(args.input)
    frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if len(frames) >= args.frames:
            break
    cap.release()
    print(f"加载 {len(frames)} 帧，shape={frames[0].shape}, "
          f"内存占用 ~{len(frames) * frames[0].nbytes / 1024**2:.0f} MB")

    # 2) 初始化模型（延迟 import 让 IO 阶段数据先打印）
    from ultralytics import YOLO
    from maskperson.device import apply_device, resolve_device

    model = YOLO("models/yolov8s-seg.pt")
    device = resolve_device("auto")
    desc = apply_device(model, device)
    print(f"推理设备: {desc}")
    if device.startswith("cuda"):
        model.half()
        print("fp16: 开启")

    # 3) 分阶段计时（warmup 2 帧后开始统计）
    t_total: list[float] = []
    t_infer: list[float] = []
    t_post: list[float] = []

    from maskperson.mosaic import create_pixel_mosaic

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        results = model.track(
            frame, classes=[0], persist=True, verbose=False,
            retina_masks=False,
        )
        t1 = time.perf_counter()
        out = frame.copy()
        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            masks = res.masks.data.cpu().numpy()
            for m in masks:
                mr = cv2.resize(m, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                out = create_pixel_mosaic(out, mr, args.block_size, args.expand)
        t2 = time.perf_counter()

        if i >= 2:  # 跳过 warmup
            t_total.append((t2 - t0) * 1000)
            t_infer.append((t1 - t0) * 1000)
            t_post.append((t2 - t1) * 1000)

    n = len(t_total)
    print(f"\n=== {n} 帧统计（跳过前 2 帧 warmup）===")
    print(f"总耗时/帧 : mean={statistics.mean(t_total):.1f} ms, "
          f"median={statistics.median(t_total):.1f} ms")
    print(f"YOLO 推理  : mean={statistics.mean(t_infer):.1f} ms")
    print(f"后处理    : mean={statistics.mean(t_post):.1f} ms")
    post_pct = 100 * statistics.mean(t_post) / statistics.mean(t_total)
    infer_pct = 100 - post_pct
    print(f"占比 推理={infer_pct:.0f}% / 后处理={post_pct:.0f}%")
    fps = 1000 / statistics.mean(t_total)
    print(f"理论 FPS   : {fps:.1f}")


if __name__ == "__main__":
    main()
