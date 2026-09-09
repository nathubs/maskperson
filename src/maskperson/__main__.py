"""CLI 入口 — python -m maskperson 或 maskperson 命令"""

from __future__ import annotations

import argparse
import sys

from maskperson.config import load_config
from maskperson.core import process_video
from maskperson.video import extract_audio, merge_av


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
        help="mask 膨胀半径，覆盖移动边缘 (默认 5)",
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

    # 2. 提取音频
    print("  提取音频...")
    extract_audio(cfg.input_video, cfg.temp_audio)

    # 3. 合并音视频
    print("  合并音视频...")
    import cv2

    cap = cv2.VideoCapture(cfg.temp_no_audio)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    merge_av(
        video_path=cfg.temp_no_audio,
        audio_path=cfg.temp_audio,
        output_path=cfg.output_video,
        fps=fps,
        width=width,
        height=height,
    )

    print(f"  脱敏完成: {cfg.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
