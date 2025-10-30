"""
Bio Models监控仪表板
提供可视化界面展示模型性能和系统资源使用情况
"""
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import these inside functions to avoid circular imports
# from . import get_logger, performance_monitor


class MetricsCollector:
    """指标收集器，用于收集和存储性能指标"""
    
    def __init__(self):
        self.metrics_history = []
        self.model_metrics = {}
        # 延迟导入以避免循环导入
        from . import get_logger
        self.logger = get_logger(__name__)
    
    def add_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """添加指标"""
        timestamp = datetime.now().isoformat()
        metric_entry = {
            "timestamp": timestamp,
            "name": metric_name,
            "value": value,
            "tags": tags or {}
        }
        self.metrics_history.append(metric_entry)
        
        # 限制历史记录数量
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
    
    def add_model_metric(self, model_name: str, metric_type: str, value: float) -> None:
        """添加模型特定指标"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {}
        
        if metric_type not in self.model_metrics[model_name]:
            self.model_metrics[model_name][metric_type] = []
        
        self.model_metrics[model_name][metric_type].append({
            "timestamp": datetime.now().isoformat(),
            "value": value
        })
        
        # 限制历史记录数量
        if len(self.model_metrics[model_name][metric_type]) > 1000:
            self.model_metrics[model_name][metric_type] = self.model_metrics[model_name][metric_type][-500:]
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """记录指标，与add_metric方法功能相同"""
        self.add_metric(metric_name, value, tags)
    
    def add_container_metric(self, container_name: str, metric_type: str, value: float) -> None:
        """添加容器特定指标"""
        metric_name = f"container_{container_name}_{metric_type}"
        self.add_metric(metric_name, value, {"container": container_name, "type": metric_type})
    
    def get_recent_metrics(self, metric_name: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近的指标"""
        filtered = [m for m in self.metrics_history if m["name"] == metric_name]
        return filtered[-count:] if filtered else []
    
    def get_model_metrics(self, model_name: str, metric_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取模型特定指标"""
        if model_name not in self.model_metrics or metric_type not in self.model_metrics[model_name]:
            return []
        
        return self.model_metrics[model_name][metric_type][-count:] if self.model_metrics[model_name][metric_type] else []
    
    def record_model_success(self, model_name: str, device: str) -> None:
        """记录模型加载成功"""
        self.add_metric("model_load_success", 1, {"model_name": model_name, "device": device})
        self.logger.info(f"模型 {model_name} 加载成功，设备: {device}")
    
    def record_model_failure(self, model_name: str, error_message: str) -> None:
        """记录模型加载失败"""
        self.add_metric("model_load_failure", 1, {"model_name": model_name, "error": error_message})
        self.logger.error(f"模型 {model_name} 加载失败: {error_message}")


# 全局指标收集器 - 延迟初始化以避免循环导入
metrics_collector = None

def get_metrics_collector():
    """获取全局指标收集器实例"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector


def create_dashboard() -> None:
    """创建Streamlit监控仪表板"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Please install it with: pip install streamlit")
        return
    
    st.set_page_config(
        page_title="Bio Models监控仪表板",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 Bio Models监控仪表板")
    
    # 侧边栏
    st.sidebar.header("控制面板")
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 1, 60, 5)
    
    # 主内容区域
    tab1, tab2, tab3 = st.tabs(["系统资源", "模型性能", "日志"])
    
    with tab1:
        display_system_resources()
    
    with tab2:
        display_model_performance()
    
    with tab3:
        display_logs()
    
    # 自动刷新
    time.sleep(refresh_interval)
    st.experimental_rerun()


def display_system_resources() -> None:
    """
    显示系统资源使用情况
    """
    st.header("系统资源使用情况")
    
    # 延迟导入以避免循环导入
    from . import performance_monitor
    from ..model_manager import ModelManager
    
    # 获取系统资源信息
    cpu_memory = performance_monitor.get_cpu_memory_info()
    cpu_usage = performance_monitor.get_cpu_usage()
    
    # 创建模型管理器实例以获取模型信息
    try:
        model_manager = ModelManager()
        loaded_models = model_manager.list_loaded_models()
        model_names = [model['name'] for model in loaded_models if model['type'] != 'container']
    except Exception:
        model_names = []
    
    # CPU和内存信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CPU使用率")
        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cpu_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("CPU使用率", f"{cpu_usage:.2f}%")
    
    with col2:
        st.subheader("内存使用情况")
        memory_percent = cpu_memory["percent"]
        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=memory_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("内存使用率", f"{memory_percent:.2f}%")
            st.metric("已用内存", f"{cpu_memory['used'] / (1024**3):.2f} GB")
            st.metric("总内存", f"{cpu_memory['total'] / (1024**3):.2f} GB")
    
    # GPU信息
    if performance_monitor.enable_gpu_monitoring:
        st.subheader("GPU使用情况")
        
        # 添加模型筛选选项
        model_filter = st.selectbox(
            "选择模型筛选（空表示显示所有模型使用的GPU）", 
            options=[""] + model_names,
            format_func=lambda x: "全部模型" if x == "" else x
        )
        
        # 获取需要显示的GPU ID列表
        gpu_ids_to_display = set()
        
        if model_filter:
            # 如果选择了特定模型，尝试获取该模型使用的GPU
            try:
                # 获取模型使用的GPU信息
                model_gpu_info = model_manager.gpu_scheduler.get_model_gpu_info(model_filter)
                if model_gpu_info:
                    # 从cuda设备字符串中提取GPU ID
                    gpu_id = int(model_gpu_info.split(':')[-1])
                    gpu_ids_to_display.add(gpu_id)
                else:
                    st.info(f"模型 {model_filter} 未使用GPU或未加载")
            except Exception as e:
                st.warning(f"获取模型GPU信息失败: {e}")
                # 如果获取失败，显示所有活跃GPU作为备选
                for gpu_id in range(performance_monitor.gpu_count):
                    gpu_memory = performance_monitor.get_gpu_memory_info(gpu_id)
                    if gpu_memory["allocated"] > 0:
                        gpu_ids_to_display.add(gpu_id)
        else:
            # 默认显示所有有显存使用的GPU
            for gpu_id in range(performance_monitor.gpu_count):
                gpu_memory = performance_monitor.get_gpu_memory_info(gpu_id)
                if gpu_memory["allocated"] > 0:
                    gpu_ids_to_display.add(gpu_id)
        
        # 显示选中的GPU信息
        active_gpus = []
        for gpu_id in sorted(gpu_ids_to_display):
            gpu_memory = performance_monitor.get_gpu_memory_info(gpu_id)
            active_gpus.append((gpu_id, gpu_memory))
        
        if not active_gpus:
            st.info("当前没有GPU正在使用")
        else:
            # 显示GPU信息，并标注使用该GPU的模型
            gpu_cols = st.columns(min(len(active_gpus), 4))
            for idx, (gpu_id, gpu_memory) in enumerate(active_gpus):
                with gpu_cols[idx % 4]:
                    memory_percent = (gpu_memory["allocated"] / gpu_memory["total"]) * 100 if gpu_memory["total"] > 0 else 0
                    
                    # 尝试获取使用该GPU的模型
                    using_models = []
                    try:
                        models_on_gpu = model_manager.gpu_scheduler.get_gpu_models_info(f"cuda:{gpu_id}")
                        if models_on_gpu:
                            using_models = models_on_gpu
                    except Exception:
                        pass
                    
                    # 显示GPU信息
                    st.metric(f"GPU {gpu_id}", f"{memory_percent:.2f}%")
                    st.metric("已用显存", f"{gpu_memory['allocated'] / (1024**3):.2f} GB")
                    st.metric("总显存", f"{gpu_memory['total'] / (1024**3):.2f} GB")
                    
                    # 显示使用该GPU的模型
                    if using_models:
                        st.text("使用中的模型:")
                        for model in using_models:
                            st.text(f"  - {model}")
        
        # PyTorch GPU内存
        torch_memory = performance_monitor.get_torch_gpu_memory_info()
        if torch_memory["allocated"] > 0 or torch_memory["reserved"] > 0:
            st.subheader("PyTorch GPU内存")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("已分配", f"{torch_memory['allocated']:.2f} GB")
            with col2:
                st.metric("已保留", f"{torch_memory['reserved']:.2f} GB")


def display_model_performance() -> None:
    """显示模型性能指标"""
    st.header("模型性能指标")
    
    # 获取指标收集器实例
    collector = get_metrics_collector()
    
    if not collector.model_metrics:
        st.info("暂无模型性能数据")
        return
    
    # 模型选择
    model_names = list(collector.model_metrics.keys())
    selected_model = st.selectbox("选择模型", model_names)
    
    if selected_model:
        model_data = collector.model_metrics[selected_model]
        
        # 显示不同类型的指标
        metric_types = list(model_data.keys())
        selected_metric = st.selectbox("选择指标类型", metric_types)
        
        if selected_metric and model_data[selected_metric]:
            # 准备数据
            timestamps = [m["timestamp"] for m in model_data[selected_metric]]
            values = [m["value"] for m in model_data[selected_metric]]
            
            # 创建图表
            if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "value": values
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                fig = px.line(
                    df, 
                    x="timestamp", 
                    y="value",
                    title=f"{selected_model} - {selected_metric}",
                    labels={"value": selected_metric, "timestamp": "时间"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 简单的文本显示
                latest_value = values[-1] if values else 0
                st.metric(f"最新{selected_metric}", f"{latest_value:.4f}")
                
                # 显示历史值
                st.subheader("历史值")
                for i, (ts, val) in enumerate(zip(timestamps[-10:], values[-10:])):
                    st.write(f"{ts}: {val:.4f}")


def display_logs() -> None:
    """显示日志信息"""
    st.header("日志信息")
    
    # 日志级别选择
    log_level = st.selectbox("选择日志级别", ["INFO", "WARNING", "ERROR", "DEBUG"])
    
    # 日志数量限制
    log_limit = st.slider("显示日志数量", 10, 1000, 100)
    
    # 这里应该从实际的日志存储中获取日志
    # 由于我们没有实现日志存储，这里只是示例
    st.info("日志显示功能需要与实际的日志存储系统集成")
    
    # 示例日志数据
    example_logs = [
        {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "模型加载完成", "model": "DNABERT-2"},
        {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "推理开始", "model": "LucaOne"},
        {"timestamp": datetime.now().isoformat(), "level": "WARNING", "message": "GPU内存使用率高", "gpu_id": 0, "usage": "85%"},
        {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "推理完成", "model": "AlphaFold", "time": "2.3s"},
    ]
    
    # 显示日志
    for log in example_logs[:log_limit]:
        if log["level"] == log_level or log_level == "INFO":
            with st.expander(f"{log['timestamp']} - {log['level']} - {log['message']}"):
                st.json(log)


def run_dashboard(host: str = "localhost", port: int = 8501) -> None:
    """运行监控仪表板"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Please install it with: pip install streamlit")
        return
    
    try:
        import subprocess
        import os
        
        # 获取当前脚本路径
        script_path = os.path.abspath(__file__)
        
        # 运行streamlit
        cmd = [
            "streamlit", "run", script_path,
            "--server.address", host,
            "--server.port", str(port)
        ]
        
        print(f"Starting dashboard at http://{host}:{port}")
        subprocess.run(cmd)
    except Exception as e:
        print(f"Failed to start dashboard: {e}")


if __name__ == "__main__":
    # 如果直接运行此脚本，启动仪表板
    run_dashboard()