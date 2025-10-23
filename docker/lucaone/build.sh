#!/bin/bash

# LucaOne Docker镜像构建脚本

set -e

# 配置变量
IMAGE_NAME="tooluniverse/lucaone"
IMAGE_TAG="latest"
CONTAINER_NAME="lucaone-server"
HOST_PORT=8002
CONTAINER_PORT=8000
METRICS_PORT=8003

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    print_info "Docker已安装"
}

# 构建Docker镜像
build_image() {
    print_info "开始构建LucaOne Docker镜像..."
    
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        print_info "Docker镜像构建成功: ${IMAGE_NAME}:${IMAGE_TAG}"
    else
        print_error "Docker镜像构建失败"
        exit 1
    fi
}

# 运行容器
run_container() {
    print_info "启动LucaOne容器..."
    
    # 停止并删除已存在的容器
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warn "容器 ${CONTAINER_NAME} 已存在，正在停止并删除..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi
    
    # 启动新容器
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${HOST_PORT}:${CONTAINER_PORT} \
        -p ${METRICS_PORT}:${METRICS_PORT} \
        --gpus all \
        --memory=16g \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        print_info "容器启动成功: ${CONTAINER_NAME}"
        print_info "API服务地址: http://localhost:${HOST_PORT}"
        print_info "监控指标地址: http://localhost:${METRICS_PORT}"
    else
        print_error "容器启动失败"
        exit 1
    fi
}

# 查看容器日志
view_logs() {
    print_info "查看容器日志..."
    docker logs -f ${CONTAINER_NAME}
}

# 停止容器
stop_container() {
    print_info "停止LucaOne容器..."
    
    if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker stop ${CONTAINER_NAME}
        print_info "容器已停止: ${CONTAINER_NAME}"
    else
        print_warn "容器未运行: ${CONTAINER_NAME}"
    fi
}

# 删除容器
remove_container() {
    print_info "删除LucaOne容器..."
    
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME}
        print_info "容器已删除: ${CONTAINER_NAME}"
    else
        print_warn "容器不存在: ${CONTAINER_NAME}"
    fi
}

# 删除镜像
remove_image() {
    print_info "删除LucaOne镜像..."
    
    if docker images --format 'table {{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}:${IMAGE_TAG}$"; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
        print_info "镜像已删除: ${IMAGE_NAME}:${IMAGE_TAG}"
    else
        print_warn "镜像不存在: ${IMAGE_NAME}:${IMAGE_TAG}"
    fi
}

# 显示帮助信息
show_help() {
    echo "LucaOne Docker管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build      构建Docker镜像"
    echo "  run        运行容器"
    echo "  logs       查看容器日志"
    echo "  stop       停止容器"
    echo "  rm         删除容器"
    echo "  rmi        删除镜像"
    echo "  restart    重启容器"
    echo "  help       显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build   # 构建镜像"
    echo "  $0 run     # 运行容器"
    echo "  $0 logs    # 查看日志"
}

# 重启容器
restart_container() {
    print_info "重启LucaOne容器..."
    stop_container
    sleep 2
    run_container
}

# 主函数
main() {
    # 检查Docker是否安装
    check_docker
    
    # 根据参数执行相应操作
    case "${1:-help}" in
        build)
            build_image
            ;;
        run)
            run_container
            ;;
        logs)
            view_logs
            ;;
        stop)
            stop_container
            ;;
        rm)
            remove_container
            ;;
        rmi)
            remove_image
            ;;
        restart)
            restart_container
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"