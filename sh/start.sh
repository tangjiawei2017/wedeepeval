#!/bin/bash

# 数据集生成平台启动脚本

# 默认环境
ENV=${1:-development}

echo "🚀 启动数据集生成平台..."
echo "🌍 环境: $ENV"

# 设置环境变量
export ENV=$ENV

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_DIR"

# 检查环境配置文件是否存在
if [ ! -f "env.$ENV" ]; then
    echo "❌ 环境配置文件 env.$ENV 不存在"
    echo "可用的环境: development, production, testing"
    exit 1
fi

echo "✅ 使用配置文件: env.$ENV"

# 启动应用
python3 main.py 