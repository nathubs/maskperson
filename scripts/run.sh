#!/bin/bash
# 快速运行脚本

set -e

# 确保模型存在（默认 medium，精度更好；用户可通过 -m 切换）
MODEL="${MASKPERSON_MODEL:-models/yolov8m-seg.pt}"
if [ ! -f "$MODEL" ]; then
    echo "⚠️  模型 $MODEL 不存在，正在下载..."
    .venv/bin/python -c "from ultralytics import YOLO; YOLO('${MODEL##*/}')"
    mkdir -p models
    mv "${MODEL##*/}" "$MODEL" 2>/dev/null || true
fi

# 默认参数
INPUT="${1:-input/input.mp4}"
OUTPUT="${2:-output/anonymized.mp4}"

echo "输入: $INPUT"
echo "输出: $OUTPUT"
echo "模型: $MODEL"

.venv/bin/python -m maskperson -m "$MODEL" -i "$INPUT" -o "$OUTPUT"
