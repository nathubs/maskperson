#!/bin/bash
# 快速运行脚本

set -e

# 确保模型存在
if [ ! -f "models/yolov8s-seg.pt" ]; then
    echo "⚠️  模型不存在，正在下载..."
    .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')"
    mkdir -p models
    mv yolov8s-seg.pt models/ 2>/dev/null || true
fi

# 默认参数
INPUT="${1:-input/input.mp4}"
OUTPUT="${2:-output/anonymized.mp4}"

echo "输入: $INPUT"
echo "输出: $OUTPUT"

.venv/bin/python -m maskperson -i "$INPUT" -o "$OUTPUT"
