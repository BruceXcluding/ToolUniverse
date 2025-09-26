#!/bin/bash

# ToolUniverse快速文档生成脚本
# 这个脚本会自动读取所有函数的docstring并生成完整的API文档
# 功能包括：
# 1. 自动生成工具配置索引
# 2. 生成Remote Tools文档
# 3. 安装Sphinx文档依赖
# 4. 生成增强API文档
# 5. 构建HTML文档
# 6. 提供本地服务器预览

# 设置错误时退出（遇到任何错误立即停止脚本）
set -e

# ===========================================
# 颜色定义 - 用于美化终端输出
# ===========================================
RED='\033[0;31m'      # 红色 - 错误信息
GREEN='\033[0;32m'    # 绿色 - 成功信息
YELLOW='\033[1;33m'   # 黄色 - 警告信息
BLUE='\033[0;34m'     # 蓝色 - 标题信息
PURPLE='\033[0;35m'   # 紫色 - 统计信息
NC='\033[0m'          # 重置颜色

# 显示脚本标题
echo -e "${BLUE}🧬 ToolUniverse 文档生成系统${NC}"
echo "========================================"

# ===========================================
# 目录路径设置
# ===========================================
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录（docs的上级目录）
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# 源码目录
SRC_DIR="$PROJECT_ROOT/src"

echo -e "${YELLOW}项目根目录: ${PROJECT_ROOT}${NC}"
echo -e "${YELLOW}源码目录: ${SRC_DIR}${NC}"

# ===========================================
# 步骤0: 生成工具配置索引（自动）
# ===========================================
# 这个步骤会自动扫描所有工具配置文件，生成索引文档
echo -e "\n${BLUE}🧩 生成工具配置索引（自动）${NC}"
cd "$SCRIPT_DIR"

# 直接调用 generate_config_index.py 生成工具配置索引
echo -e "${YELLOW}📑 正在生成工具配置索引...${NC}"
python generate_config_index.py || { echo -e "${RED}❌ 生成配置索引失败${NC}"; exit 1; }
echo -e "${GREEN}✅ 工具配置索引生成完成${NC}"

# ===========================================
# 步骤0.1: 生成Remote Tools文档（自动）
# ===========================================
# 这个步骤会生成远程工具的文档，包括MCP服务器等
echo -e "\n${BLUE}🌐 生成Remote Tools文档（自动）${NC}"
cd "$SCRIPT_DIR"

# 检查是否存在远程工具文档生成脚本
if [ -f "generate_remote_tools_docs.py" ]; then
    python generate_remote_tools_docs.py || { echo -e "${RED}❌ 生成Remote Tools文档失败${NC}"; exit 1; }
    echo -e "${GREEN}✅ Remote Tools文档生成完成${NC}"
else
    echo -e "${YELLOW}⚠️ 未找到 generate_remote_tools_docs.py${NC}"
fi

# ===========================================
# 步骤1: 安装Sphinx文档依赖
# ===========================================
# 安装构建文档所需的Python包
echo -e "\n${BLUE}📦 安装增强文档依赖${NC}"
cd "$PROJECT_ROOT"

# 静默安装Sphinx相关依赖包
# sphinx: 文档生成框架
# sphinx-rtd-theme: Read the Docs主题
# myst-parser: Markdown解析器
# sphinx-copybutton: 代码复制按钮
# sphinx-autodoc-typehints: 类型提示支持
pip install -q sphinx sphinx-rtd-theme myst-parser sphinx-copybutton sphinx-autodoc-typehints 2>/dev/null || true
echo -e "${GREEN}✅ 依赖安装完成${NC}"

# ===========================================
# 步骤2: 生成增强API文档
# ===========================================
# 使用sphinx-apidoc自动扫描Python源码，生成API文档
echo -e "\n${BLUE}📋 生成增强API文档${NC}"
cd "$SCRIPT_DIR"

# ===========================================
# 清理旧的构建文件
# ===========================================
# 清理构建目录
if [ -d "_build" ]; then rm -rf _build; fi

# 注释：保留api目录，因为可能包含手动生成的文档
# 如果需要完全重新生成，可以取消下面的注释
# if [ -d "api" ]; then rm -rf api; fi

# 清理模板目录
if [ -d "_templates" ]; then rm -rf _templates; fi

# ===========================================
# 创建自定义模板目录和文件
# ===========================================
# 创建模板目录
mkdir -p _templates

# 创建函数文档模板
cat > _templates/function.rst << 'EOF'
{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
EOF

# 创建类文档模板
cat > _templates/class.rst << 'EOF'
{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__,__call__,__str__,__repr__
EOF

# 创建模块文档模板
cat > _templates/module.rst << 'EOF'
{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. automodule:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__,__call__,__str__,__repr__
   :imported-members:
EOF

# 确保api目录存在
mkdir -p api

# ===========================================
# 使用sphinx-apidoc生成API文档
# ===========================================
# 检查sphinx-apidoc命令是否可用
if command -v sphinx-apidoc >/dev/null 2>&1; then
    # 使用sphinx-apidoc扫描源码并生成API文档
    # -f: 强制覆盖现有文件
    # -o api: 输出到api目录
    # ../src/tooluniverse: 源码路径
    # --separate: 为每个模块创建单独的文档文件
    # --module-first: 模块名在前
    # --maxdepth 6: 最大递归深度
    # --private: 包含私有成员
    # --force: 强制重新生成
    # --templatedir=_templates: 使用自定义模板
    sphinx-apidoc -f -o api ../src/tooluniverse \
        --separate \
        --module-first \
        --maxdepth 6 \
        --private \
        --force \
        --templatedir=_templates \
        2>/dev/null || true

    # 统计生成的API文档文件数量
    API_FILES=$(find api -name "*.rst" | wc -l | tr -d ' ')
    echo -e "${GREEN}✅ 生成了 ${API_FILES} 个API文档文件${NC}"

    # ===========================================
    # 显示发现的模块信息
    # ===========================================
    echo -e "\n${PURPLE}📋 发现的模块:${NC}"
    find api -name "*.rst" | sed 's|api/||' | sed 's|\.rst$||' | sort | head -20 | while read -r module; do
        echo -e "   📄 ${module}"
    done
    
    # 如果模块数量超过20个，显示剩余数量
    REMAINING=$(find api -name "*.rst" | wc -l | tr -d ' ')
    if [ "$REMAINING" -gt 20 ]; then
        echo -e "   ... 还有 $((REMAINING - 20)) 个模块"
    fi

    # ===========================================
    # 生成模块发现报告
    # ===========================================
    echo -e "\n${PURPLE}📋 模块发现报告:${NC}"
    echo -e "   🔍 核心模块: $(find "$SRC_DIR/tooluniverse" -maxdepth 1 -name "*.py" | wc -l | tr -d ' ')"
    echo -e "   🔍 组合脚本: $(find "$SRC_DIR/tooluniverse/compose_scripts" -name "*.py" 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
    echo -e "   🔍 外部工具: $(find "$SRC_DIR/tooluniverse/external" -name "*.py" 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
    echo -e "   🔍 工具脚本: $(find "$SRC_DIR/tooluniverse/scripts" -name "*.py" 2>/dev/null | wc -l | tr -d ' ' || echo "0")"
else
    echo -e "${RED}❌ sphinx-apidoc未找到${NC}"
    exit 1
fi

# ===========================================
# 步骤3: 构建增强HTML文档
# ===========================================
# 使用sphinx-build将RST文档转换为HTML格式
echo -e "\n${BLUE}🔧 构建增强HTML文档${NC}"

# 使用sphinx-build构建HTML文档
# -b html: 构建HTML格式
# . : 当前目录（docs目录）
# _build/html: 输出目录
# --keep-going: 遇到错误时继续构建
# -q: 静默模式
sphinx-build -b html . _build/html --keep-going -q || true

# ===========================================
# 检查构建结果
# ===========================================
# 检查是否成功生成了主页文件
if [ -f "_build/html/index.html" ]; then
    echo -e "${GREEN}✅ 文档构建成功${NC}"
else
    echo -e "${RED}❌ 文档构建失败${NC}"
    exit 1
fi

# ===========================================
# 步骤4: 生成详细统计信息
# ===========================================
# 统计生成的文档文件数量和代码统计信息
echo -e "\n${BLUE}📊 生成详细统计信息${NC}"

# 统计HTML文件数量
HTML_FILES=$(find _build/html -name "*.html" | wc -l | tr -d ' ')
# 统计API文档数量
API_DOCS=$(find _build/html/api -name "*.html" 2>/dev/null | wc -l | tr -d ' ' || echo "0")
# 计算文档总大小
DOC_SIZE=$(du -sh _build/html 2>/dev/null | cut -f1 || echo "未知")

# ===========================================
# 统计源码信息
# ===========================================
# 统计函数数量
FUNCTION_COUNT=$(grep -r "def " "$SRC_DIR" --include="*.py" | wc -l | tr -d ' ')
# 统计类数量
CLASS_COUNT=$(grep -r "class " "$SRC_DIR" --include="*.py" | wc -l | tr -d ' ')
# 统计模块数量
MODULE_COUNT=$(find "$SRC_DIR" -name "*.py" | wc -l | tr -d ' ')

# ===========================================
# 显示统计结果
# ===========================================
echo -e "\n${GREEN}🎉 增强文档生成完成！${NC}"
echo -e "${BLUE}📊 详细统计信息:${NC}"
echo -e "   📄 HTML文件: ${HTML_FILES}"
echo -e "   🔧 API文档: ${API_DOCS}"
echo -e "   📁 总大小: ${DOC_SIZE}"
echo -e "   🐍 Python模块: ${MODULE_COUNT}"
echo -e "   🔧 函数总数: ${FUNCTION_COUNT}"
echo -e "   🏗️ 类总数: ${CLASS_COUNT}"

# ===========================================
# 步骤5: 显示文档访问方式
# ===========================================
# 提供各种访问文档的方式和链接
BUILD_PATH="$SCRIPT_DIR/_build/html/index.html"
echo -e "\n${BLUE}📂 访问文档:${NC}"
echo -e "   🏠 主页: file://${BUILD_PATH}"
echo -e "   🔧 完整API: file://${SCRIPT_DIR}/_build/html/api/modules.html"

# ===========================================
# 步骤6: 可选的本地服务器启动
# ===========================================
# 检查是否在CI环境中运行
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ]; then
    echo -e "${BLUE}🤖 CI环境检测到，跳过服务器启动${NC}"
else
    # 询问用户是否要启动本地HTTP服务器来预览文档
    echo -e "\n${YELLOW}启动本地服务器查看文档? (y/n)${NC}"
    read -r response

    # 如果用户选择启动服务器
    if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🚀 启动服务器...${NC}"

    # ===========================================
    # 端口检测和分配
    # ===========================================
    # 尝试不同的端口，找到可用的端口
    PORTS=(8080 8081 8082 8083 8084)
    PORT=""

    # 遍历端口列表，找到第一个可用的端口
    for p in "${PORTS[@]}"; do
        if ! lsof -Pi :$p -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORT=$p
            break
        fi
    done

    # ===========================================
    # 启动HTTP服务器
    # ===========================================
    if [ -z "$PORT" ]; then
        # 如果没有找到可用端口，提供手动启动命令
        echo -e "${RED}❌ 无法找到可用端口，请手动启动服务器${NC}"
        echo -e "${YELLOW}💡 手动启动命令: cd _build/html && python -m http.server 8080${NC}"
    else
        # 显示访问地址
        echo -e "${GREEN}📡 访问地址: http://localhost:${PORT}${NC}"
        echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"

        # 切换到构建目录并启动服务器
        cd _build/html
        python -m http.server $PORT
    fi
fi
fi

# ===========================================
# 脚本完成总结
# ===========================================
# 显示脚本执行完成和新特性介绍
echo -e "\n${GREEN}✅ 增强文档系统完成！${NC}"
echo -e "${BLUE}💡 新特性:${NC}"
echo -e "   ✨ 自动发现所有模块和函数"
echo -e "   📊 详细的统计信息"
echo -e "   🔍 增强的模块发现"
echo -e "   📚 综合API索引"
echo -e "   🎨 自定义模板支持"
echo -e "   📱 响应式设计"
echo -e "   🔍 内置搜索功能"
