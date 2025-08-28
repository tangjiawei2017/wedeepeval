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

# 确保日志目录存在
mkdir -p logs

# 启动应用（后台运行）
nohup python3 main.py > logs/app.log 2>&1 &

# 获取后台进程ID
PID=$!

# 等待一秒检查进程是否启动成功
sleep 1

# 检查进程是否还在运行
if kill -0 $PID 2>/dev/null; then
    echo "✅ 服务已成功启动，进程ID: $PID"
    echo "📝 日志文件: logs/app.log"
    echo "🌐 服务地址: http://localhost:8093"
    echo "📚 API文档: http://localhost:8093/docs"
    echo ""
    echo "💡 停止服务请运行: ./sh/stop.sh"
    echo "💡 查看日志请运行: tail -f logs/app.log"
else
    echo "❌ 服务启动失败，请检查日志: logs/app.log"
    exit 1
fi