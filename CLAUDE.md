# maskperson

视频人体脱敏工具 — YOLOv8-Seg 跟踪 + 像素级马赛克。

## 项目结构

```
src/maskperson/       # 源码
tests/                # 单元测试
configs/             # 配置文件
scripts/run.sh       # 快速运行
models/              # YOLO 权重
input/                # 输入视频
output/               # 输出视频
temp/                 # 临时文件
```

## 运行

使用 `.venv/bin/python` 而非 `uv run`（uv run 在多平台解析 torch 依赖时有冲突）。

```bash
.venv/bin/python -m maskperson -i input/input.mp4 -o output/anonymized.mp4
./scripts/run.sh input.mp4 output.mp4
```

## 依赖问题

当前 uv lockfile 中 torch==2.5.1+cpu 的平台标记与 ultralytics 的 torch 依赖声明在 Windows 解析路径下冲突。实际运行不受影响（Linux x86_64 下正常）。如需重新 `uv sync`，先手动装 CPU torch：
```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m mypy src/
```
