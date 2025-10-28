#!/bin/bash

# Agent API服务构建和运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数定义
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "ToolUniverse Agent API服务管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build    构建Docker镜像"
    echo "  run      运行容器"
    echo "  stop     停止容器"
    echo "  logs     查看日志"
    echo "  restart  重启容器"
    echo "  clean    清理容器和镜像"
    echo "  help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build    # 构建镜像"
    echo "  $0 run      # 运行容器"
    echo "  $0 logs     # 查看日志"
}

# 构建Docker镜像
build_image() {
    log_info "构建ToolUniverse Agent API Docker镜像..."
    docker build -t tooluniverse/agent-api:latest -f Dockerfile ../..
    log_info "镜像构建完成"
}

# 运行容器
run_container() {
    log_info "启动ToolUniverse Agent API容器..."
    
    # 检查容器是否已存在
    if docker ps -a --format 'table {{.Names}}' | grep -q "agent-api-server"; then
        log_warn "容器已存在，正在删除..."
        docker stop agent-api-server || true
        docker rm agent-api-server || true
    fi
    
    # 启动新容器
    docker run -d \
        --name agent-api-server \
        --network tooluniverse_model-network \
        -p 5000:5000 \
        -e PYTHONPATH=/app/src \
        -e PYTHONUNBUFFERED=1 \
        tooluniverse/agent-api:latest
    
    log_info "容器已启动，API地址: http://localhost:5000"
    log_info "健康检查: curl http://localhost:5000/health"
}

# 停止容器
stop_container() {
    log_info "停止ToolUniverse Agent API容器..."
    docker stop agent-api-server || log_warn "容器未运行或已停止"
    log_info "容器已停止"
}

# 查看日志
show_logs() {
    log_info "显示ToolUniverse Agent API容器日志..."
    docker logs -f agent-api-server
}

# 重启容器
restart_container() {
    log_info "重启ToolUniverse Agent API容器..."
    stop_container
    run_container
}

# 清理容器和镜像
clean() {
    log_info "清理ToolUniverse Agent API容器和镜像..."
    
    # 停止并删除容器
    docker stop agent-api-server || true
    docker rm agent-api-server || true
    
    # 删除镜像
    docker rmi tooluniverse/agent-api:latest || log_warn "镜像不存在或已被删除"
    
    log_info "清理完成"
}

# 主逻辑
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    logs)
        show_logs
        ;;
    restart)
        restart_container
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "未知命令: $1"
        show_help
        exit 1
        ;;
esac