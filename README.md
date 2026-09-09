# maskperson

视频人体脱敏工具 — 基于 YOLO 跟踪 + 像素级马赛克。

## 功能

- 使用 YOLOv8-Seg 分割模型检测人体
- ByteTrack 多目标跟踪，保持身份一致性
- 时序平滑，消除马赛克闪烁
- mask 膨胀，覆盖快速移动边缘漏出
- 自动提取并合并音频轨道
- 支持配置文件 / 环境变量 / 命令行参数覆盖

## 项目结构

```
maskperson/
├── src/maskperson/          # 源码
│   ├── __init__.py
│   ├── __main__.py          # CLI 入口
│   ├── config.py            # 配置管理
│   ├── core.py              # 主处理流程
│   ├── mosaic.py            # 马赛克算法 + mask 膨胀
│   ├── tracker.py           # 跟踪器 + 时序平滑
│   └── video.py             # 视频读写 + ffmpeg 封装
├── tests/                   # 单元测试
├── configs/
│   └── default.toml         # 默认配置
├── scripts/
│   └── run.sh              # 快速运行脚本
├── models/                  # YOLO 模型权重
├── input/                   # 输入视频
├── output/                  # 输出视频
└── temp/                    # 临时文件
```

## 安装

```bash
uv sync
```

## 下载模型

```bash
.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')"
mv yolov8s-seg.pt models/
```

## 使用

```bash
# 命令行参数
.venv/bin/python -m maskperson -i input.mp4 -o output.mp4

# 配置文件
.venv/bin/python -m maskperson -c configs/default.toml

# 环境变量
MASKPERSON_INPUT_VIDEO=input.mp4 MASKPERSON_OUTPUT_VIDEO=out.mp4 .venv/bin/python -m maskperson

# 快速脚本
chmod +x scripts/run.sh
./scripts/run.sh input.mp4 output.mp4
```

## 配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_video` | `input/input.mp4` | 输入视频路径 |
| `output_video` | `output/anonymized.mp4` | 输出视频路径 |
| `model_weight` | `models/yolov8s-seg.pt` | YOLO 模型路径 |
| `conf_thresh` | `0.35` | 置信度阈值 |
| `mosaic_block_size` | `20` | 马赛克块大小，越大越模糊 |
| `expand_pixels` | `5` | mask 膨胀半径，覆盖移动边缘 |
| `smooth_window_size` | `5` | 时序平滑窗口 |
| `track_max_age` | `30` | 目标消失后保留帧数 |
| `device` | `auto` | 推理设备：`auto` / `cpu` / `cuda` / `cuda:N` |

## 设备选择

默认自动检测 — NVIDIA 机器用 GPU，无 GPU 机器自动降级 CPU。

```bash
# 自动（默认）：检测到 CUDA 则用，否则 CPU
.venv/bin/python -m maskperson -i in.mp4 -o out.mp4

# 强制 CPU
.venv/bin/python -m maskperson --device cpu -i in.mp4 -o out.mp4

# 强制 GPU（不可用时报错并提示排查）
.venv/bin/python -m maskperson --device cuda -i in.mp4 -o out.mp4
```

也可通过 `configs/default.toml` 或环境变量 `MASKPERSON_DEVICE` 设置，例如指定第二块显卡：

```bash
MASKPERSON_DEVICE=cuda:1 .venv/bin/python -m maskperson -i in.mp4 -o out.mp4
```

启动时日志会打印实际使用的设备：

```
推理设备: cuda (NVIDIA GeForce RTX 4090)
# 或
推理设备: cpu
```

**安装阶段**：`pyproject.toml` 默认从 PyPI 拉取 torch wheel（兼容 GPU/CPU）。如需更小体积的纯 CPU wheel（避开 NVIDIA 驱动版本要求）：

```bash
uv sync --no-extra default --extra cpu
```

## 开发

```bash
uv pip install pytest ruff mypy types-toml
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
.venv/bin/python -m mypy src/
```
