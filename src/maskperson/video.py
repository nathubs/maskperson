"""视频处理模块 — 读写 + ffmpeg 封装"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoCapture:
    """cv2.VideoCapture 上下文管理器。"""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {path}")

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read(self) -> tuple[bool, np.ndarray]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoCapture:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class VideoWriter:
    """cv2.VideoWriter 上下文管理器。"""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise OSError(f"无法创建输出视频: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def extract_audio(input_video: str, output_audio: str) -> None:
    """从视频提取音频轨道。"""
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn", "-acodec", "aac",
            output_audio,
        ],
        capture_output=True,
        check=True,
    )


def merge_av(
    video_path: str,
    audio_path: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    """合并音视频流，使用 libx264 重新编码。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-crf", "23",
            "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "aac",
            output_path,
        ],
        capture_output=True,
        check=True,
    )
