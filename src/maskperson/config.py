"""配置管理"""

import os
from dataclasses import dataclass
from pathlib import Path

import toml


@dataclass
class Config:
    input_video: str = "input/input.mp4"
    output_video: str = "output/anonymized.mp4"
    temp_no_audio: str = "temp/no_audio.mp4"
    temp_audio: str = "temp/audio.aac"

    model_weight: str = "models/yolov8s-seg.pt"
    conf_thresh: float = 0.35
    iou_thresh: float = 0.5

    mosaic_block_size: int = 20
    expand_pixels: int = 5
    smooth_window_size: int = 5
    track_max_age: int = 30

    # 推理设备：auto（CUDA 优先，CPU 兜底） / cpu / cuda / cuda:N
    device: str = "auto"


def load_config(config_path: str | None = None) -> Config:
    """加载配置，优先级：命令行 > 环境变量 > 配置文件 > 默认值"""
    cfg = Config()

    # 1. 从配置文件加载
    if config_path and Path(config_path).exists():
        data = toml.load(config_path)
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # 2. 环境变量覆盖（支持 MASKPERSON_* 前缀）
    for field in cfg.__dataclass_fields__:
        env_key = f"MASKPERSON_{field.upper()}"
        if env_val := os.environ.get(env_key):
            value = cfg.__dataclass_fields__[field].type
            if value is int:
                setattr(cfg, field, int(env_val))
            elif value is float:
                setattr(cfg, field, float(env_val))
            else:
                setattr(cfg, field, env_val)

    return cfg
