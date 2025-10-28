#!/bin/bash
# 启动ToolUniverse生物模型服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker未运行，请先启动Docker"
        exit 1
    fi
    log_info "Docker运行正常"
}

# 检查Docker Compose是否可用
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose未安装或不可用"
        exit 1
    fi
    
    # 确定使用哪个命令
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    log_info "使用命令: $DOCKER_COMPOSE_CMD"
}

# 启动所有服务
start_services() {
    log_info "启动ToolUniverse生物模型服务..."
    
    # 使用docker-compose启动所有服务
    $DOCKER_COMPOSE_CMD up -d
    
    log_info "所有服务已启动"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待DNABERT2容器
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            log_info "DNABERT2服务已就绪"
            break
        fi
        attempt=$((attempt+1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "DNABERT2服务启动超时"
    fi
    
    # 等待LucaOne容器
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8002/health > /dev/null 2>&1; then
            log_info "LucaOne服务已就绪"
            break
        fi
        attempt=$((attempt+1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "LucaOne服务启动超时"
    fi
    
    # 等待Agent API
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            log_info "Agent API服务已就绪"
            break
        fi
        attempt=$((attempt+1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Agent API服务启动超时"
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    
    # 显示容器状态
    $DOCKER_COMPOSE_CMD ps
    
    # 显示服务访问地址
    echo ""
    echo "服务访问地址:"
    echo "- Agent API: http://localhost:5000"
    echo "- DNABERT2: http://localhost:8001"
    echo "- LucaOne: http://localhost:8002"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
}

# 主函数
main() {
    log_info "启动ToolUniverse生物模型服务..."
    
    # 检查Docker
    check_docker
    
    # 检查Docker Compose
    check_docker_compose
    
    # 启动服务
    start_services
    
    # 等待服务就绪
    wait_for_services
    
    # 显示状态
    show_status
    
    log_info "所有服务启动完成！"
}

# 处理命令行参数
case "${1:-}" in
    stop)
        log_info "停止所有服务..."
        
        # 停止并删除所有服务
        $DOCKER_COMPOSE_CMD down
        
        log_info "所有服务已停止"
        ;;
    status)
        show_status
        ;;
    restart)
        $0 stop
        sleep 3
        $0 start
        ;;
    logs)
        $DOCKER_COMPOSE_CMD logs -f
        ;;
    start|"")
        main
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac