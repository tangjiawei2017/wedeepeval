#!/bin/bash

# 停止数据集生成平台服务脚本

echo "🛑 正在停止数据集生成平台..."

# 查找并停止所有相关进程
echo "🔍 查找相关进程..."

# 查找Python主进程（支持多种Python命令）
PYTHON_PIDS=$(ps aux | grep -E "(/data/deepeval/deepeval-main/deepeval/bin/python main.py|Python main.py)" | grep -v grep | awk '{print $2}')

# 查找Uvicorn进程
UVICORN_PIDS=$(ps aux | grep "uvicorn" | grep -v grep | awk '{print $2}')

# 合并所有进程ID
ALL_PIDS=""
if [ ! -z "$PYTHON_PIDS" ]; then
    ALL_PIDS="$PYTHON_PIDS"
fi
if [ ! -z "$UVICORN_PIDS" ]; then
    if [ ! -z "$ALL_PIDS" ]; then
        ALL_PIDS="$ALL_PIDS $UVICORN_PIDS"
    else
        ALL_PIDS="$UVICORN_PIDS"
    fi
fi

# 停止所有找到的进程
if [ -z "$ALL_PIDS" ]; then
    echo "✅ 没有找到运行中的相关进程"
else
    echo "📋 找到以下进程: $ALL_PIDS"
    
    # 先尝试优雅停止
    echo "🔄 尝试优雅停止进程..."
    for pid in $ALL_PIDS; do
        echo "📤 发送TERM信号到进程 $pid"
        kill -TERM $pid 2>/dev/null
    done
    
    # 等待3秒让进程优雅退出
    sleep 3
    
    # 检查是否还有进程在运行，如果有则强制杀死
    REMAINING_PIDS=""
    for pid in $ALL_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            REMAINING_PIDS="$REMAINING_PIDS $pid"
        fi
    done
    
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo "⚠️  以下进程仍在运行，强制停止: $REMAINING_PIDS"
        for pid in $REMAINING_PIDS; do
            echo "🔪 强制杀死进程 $pid"
            kill -9 $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ 进程 $pid 已强制停止"
            else
                echo "❌ 无法停止进程 $pid"
            fi
        done
    else
        echo "✅ 所有进程已优雅停止"
    fi
fi

echo "🎉 服务停止完成！" 