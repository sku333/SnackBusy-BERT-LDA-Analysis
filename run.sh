#!/bin/bash
# -*- coding: utf-8 -*-
#=============================================================================
# 硬折扣零食店评论分析 — 深度学习升级版 一键运行脚本
#=============================================================================

set -e

echo "============================================"
echo " 硬折扣零食店评论分析 — 深度学习升级版"
echo "============================================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 1. 检查 Python 环境 ----
echo ""
echo "[1/4] 检查 Python 环境..."
if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "错误: 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  Python: $PYTHON_VERSION"

# ---- 2. 安装依赖 ----
echo ""
echo "[2/4] 安装 Python 依赖..."
$PYTHON -m pip install -r requirements.txt --quiet

# ---- 3. 检查中文字体 ----
echo ""
echo "[3/4] 检查中文字体..."
FONT_PATHS=(
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
)
FONT_FOUND=false
for fp in "${FONT_PATHS[@]}"; do
    if [ -f "$fp" ]; then
        echo "  找到字体: $fp"
        FONT_FOUND=true
        break
    fi
done
if [ "$FONT_FOUND" = false ]; then
    echo "  警告: 未找到中文字体，词云可能显示为方块。"
    echo "  Ubuntu/Debian: sudo apt-get install fonts-wqy-zenhei"
    echo "  CentOS/RHEL:   sudo yum install wqy-zenhei-fonts"
fi

# ---- 4. 运行分析 ----
echo ""
echo "[4/4] 开始运行分析..."
echo ""

# 如果指定了 --local 参数，使用本地模型
if [ "$1" = "--local" ]; then
    echo "  模式: 本地模型加载"
    $PYTHON main.py --local
else
    echo "  模式: HF 镜像站下载"
    $PYTHON main.py "$@"
fi

echo ""
echo "============================================"
echo " 分析完成！请查看 output/ 目录下的结果。"
echo "============================================"
