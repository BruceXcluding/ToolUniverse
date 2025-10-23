#!/usr/bin/env python3
"""
ToolUniverse监控仪表板启动脚本
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tooluniverse.bio_models.monitoring.dashboard import run_dashboard


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动ToolUniverse监控仪表板")
    parser.add_argument("--port", type=int, default=8501, help="仪表板端口 (默认: 8501)")
    parser.add_argument("--host", type=str, default="localhost", help="仪表板主机 (默认: localhost)")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    if args.debug:
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
    
    print(f"正在启动ToolUniverse监控仪表板...")
    print(f"访问地址: http://{args.host}:{args.port}")
    print("按Ctrl+C停止服务器")
    
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n仪表板已停止")


if __name__ == "__main__":
    main()