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

## 开发

```bash
uv pip install pytest ruff mypy types-toml
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
.venv/bin/python -m mypy src/
```
