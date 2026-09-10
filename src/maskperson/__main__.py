"""CLI 入口 — python -m maskperson 或 maskperson 命令"""

from __future__ import annotations

import argparse
import sys

from maskperson.config import load_config
from maskperson.core import process_video
from maskperson.video import extract_audio, has_audio_stream, merge_av


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskperson",
        description="视频人体脱敏工具 — YOLO 跟踪 + 马赛克",
    )
    parser.add_argument(
        "-c", "--config",
        help="配置文件路径 (TOML 格式)",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_video",
        help="输入视频路径",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_video",
        help="输出视频路径",
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_weight",
        help="YOLO 模型权重路径",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help="马赛克方块大小 (默认 20)",
    )
    parser.add_argument(
        "--expand",
        type=int,
        dest="expand_pixels",
        help="mask 膨胀半径，覆盖移动边缘 (默认 15)",
    )
    parser.add_argument(
        "--mosaic-style",
        choices=["pixel", "solid_black", "checkerboard"],
        dest="mosaic_style",
        help=(
            "脱敏风格：pixel=模糊马赛克（默认，向后兼容）；"
            "solid_black=实心黑色（最强脱敏，完全不可识别）；"
            "checkerboard=黑白棋盘格（高对比度，最强 + 视觉提示）。"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        dest="imgsz",
        help=(
            "YOLO 推理分辨率（默认 640）。"
            "调大（如 960/1280）可显著改善小目标检测，但 GPU 显存占用和耗时也会增加。"
            "需为 32 的倍数。"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help=(
            "推理设备。auto=自动检测（CUDA 优先，CPU 兜底，默认），"
            "cuda=强制 GPU（不可用则报错），cpu=强制 CPU。"
            " 指定 cuda:N 选择具体卡可通过 Config/TOML/MASKPERSON_DEVICE 配置。"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # 合并命令行参数到配置
    cfg = load_config(args.config)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            setattr(cfg, key, value)

    # 1. 处理帧
    process_video(cfg)

    # 2. 探测源视频是否含音频轨；没有则整段跳过音频处理
    has_audio = has_audio_stream(cfg.input_video)

    # 3. 提取音频（仅当有音频轨时）
    if has_audio:
        print("  提取音频...")
        extract_audio(cfg.input_video, cfg.temp_audio)
    else:
        print("  源视频无音频轨，跳过音频提取")

    # 4. 合并 / 封装
    print("  封装输出...")
    import cv2

    cap = cv2.VideoCapture(cfg.temp_no_audio)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    merge_av(
        video_path=cfg.temp_no_audio,
        audio_path=cfg.temp_audio if has_audio else "",
        output_path=cfg.output_video,
        fps=fps,
        width=width,
        height=height,
    )

    print(f"  脱敏完成: {cfg.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
