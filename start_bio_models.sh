#!/bin/bash

# 生物序列模型部署脚本
# 此脚本用于设置环境变量并启动ToolUniverse生物序列模型服务

# 设置默认值
DEFAULT_MODEL_PATH="/data/models"
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"

# 从环境变量或参数中获取配置
MODEL_PATH=${MODEL_BASE_PATH:-$DEFAULT_MODEL_PATH}
PORT=${PORT:-$DEFAULT_PORT}
HOST=${HOST:-$DEFAULT_HOST}

echo "========================================="
echo "ToolUniverse 生物序列模型服务启动脚本"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "服务端口: $PORT"
echo "监听地址: $HOST"
echo "========================================="

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请确保模型文件已正确部署或设置正确的MODEL_BASE_PATH环境变量"
    echo "继续启动可能会导致模型加载失败..."
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消启动"
        exit 1
    fi
fi

# 设置环境变量
export MODEL_BASE_PATH=$MODEL_PATH

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查必要的Python包
echo "检查Python依赖..."
python -c "import torch; print('PyTorch版本:', torch.__version__)" || {
    echo "错误: 未安装PyTorch"
    exit 1
}

python -c "import transformers; print('Transformers版本:', transformers.__version__)" || {
    echo "错误: 未安装Transformers"
    exit 1
}

# 启动服务
echo "启动ToolUniverse生物序列模型服务..."
echo "访问地址: http://$HOST:$PORT"
echo "按Ctrl+C停止服务"
echo

# 启动FastAPI应用
python -m uvicorn src.tooluniverse.main:app --host $HOST --port $PORT --reload