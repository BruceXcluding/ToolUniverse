"""
Bio Models监控模块
提供统一的日志记录和性能监控功能
"""
import time
import json
import logging
import sys
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
from contextlib import contextmanager
import structlog
import psutil
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU monitoring will be disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available, GPU memory monitoring will be limited")

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from .dashboard import get_metrics_collector, MetricsCollector

# 创建全局指标收集器实例 - 延迟初始化
def get_metrics_collector_instance():
    """获取全局指标收集器实例"""
    return get_metrics_collector()

# 为了向后兼容，创建一个属性访问器
class _MetricsCollectorProxy:
    """指标收集器代理，用于延迟初始化"""
    def __getattr__(self, name):
        collector = get_metrics_collector()
        return getattr(collector, name)

metrics_collector = _MetricsCollectorProxy()


# Prometheus指标定义 - 避免重复注册
try:
    MODEL_LOAD_TIME = Histogram('bio_model_load_time_seconds', 'Time spent loading models', ['model_name'])
    INFERENCE_TIME = Histogram('bio_inference_time_seconds', 'Time spent on inference', ['model_name', 'task_type'])
    GPU_MEMORY_USAGE = Gauge('bio_gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id'])
    CPU_USAGE = Gauge('bio_cpu_usage_percent', 'CPU usage percentage')
    MEMORY_USAGE = Gauge('bio_memory_usage_bytes', 'Memory usage in bytes')
    MODEL_LOAD_COUNTER = Counter('bio_model_loads_total', 'Total model loads', ['model_name', 'status'])
    INFERENCE_COUNTER = Counter('bio_inferences_total', 'Total inferences', ['model_name', 'task_type', 'status'])
except ValueError as e:
    # 如果指标已经存在，跳过注册
    if "Duplicated timeseries" in str(e):
        pass
    else:
        raise


class PerformanceMonitor:
    """性能监控类，提供各种性能指标的收集和记录功能"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and PYNVML_AVAILABLE
        self.gpu_count = 0
        
        # 初始化GPU监控
        if self.enable_gpu_monitoring:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                print(f"Failed to initialize NVIDIA ML: {e}")
                self.enable_gpu_monitoring = False
    
    def get_gpu_memory_info(self, gpu_id: int = 0) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        if not self.enable_gpu_monitoring or gpu_id >= self.gpu_count:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 更新Prometheus指标
            GPU_MEMORY_USAGE.labels(gpu_id=str(gpu_id)).set(mem_info.used)
            
            return {
                "allocated": mem_info.used,
                "reserved": mem_info.used,
                "total": mem_info.total,
                "free": mem_info.free
            }
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return {"allocated": 0, "reserved": 0, "total": 0}
    
    def get_torch_gpu_memory_info(self) -> Dict[str, float]:
        """获取PyTorch GPU内存使用情况"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}
        
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)   # GB
            return {"allocated": allocated, "reserved": reserved}
        except Exception as e:
            print(f"Error getting PyTorch GPU memory info: {e}")
            return {"allocated": 0, "reserved": 0}
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存使用情况"""
        try:
            memory = psutil.virtual_memory()
            
            # 更新Prometheus指标
            MEMORY_USAGE.set(memory.used)
            
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except Exception as e:
            print(f"Error getting CPU memory info: {e}")
            return {"total": 0, "available": 0, "used": 0, "percent": 0}
    
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 更新Prometheus指标
            CPU_USAGE.set(cpu_percent)
            
            return cpu_percent
        except Exception as e:
            print(f"Error getting CPU usage: {e}")
            return 0.0
    
    def log_system_resources(self, logger: structlog.BoundLogger) -> None:
        """记录系统资源使用情况"""
        # CPU和内存
        cpu_memory = self.get_cpu_memory_info()
        cpu_usage = self.get_cpu_usage()
        
        logger.info(
            "system_resources",
            cpu_percent=cpu_usage,
            memory_total=cpu_memory["total"],
            memory_used=cpu_memory["used"],
            memory_percent=cpu_memory["percent"]
        )
        
        # GPU内存
        if self.enable_gpu_monitoring:
            for gpu_id in range(self.gpu_count):
                gpu_memory = self.get_gpu_memory_info(gpu_id)
                logger.info(
                    "gpu_resources",
                    gpu_id=gpu_id,
                    memory_allocated=gpu_memory["allocated"],
                    memory_total=gpu_memory["total"],
                    memory_percent=(gpu_memory["allocated"] / gpu_memory["total"]) * 100 if gpu_memory["total"] > 0 else 0
                )
        
        # PyTorch GPU内存
        torch_memory = self.get_torch_gpu_memory_info()
        if torch_memory["allocated"] > 0 or torch_memory["reserved"] > 0:
            logger.info(
                "torch_gpu_resources",
                memory_allocated=torch_memory["allocated"],
                memory_reserved=torch_memory["reserved"]
            )


# 初始化性能监控器
performance_monitor = PerformanceMonitor()


def configure_structlog() -> None:
    """配置structlog日志系统"""
    # 处理器链
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # 根据环境选择不同的输出格式
    if sys.stderr.isatty():  # 终端环境，使用彩色输出
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:  # 非终端环境，使用JSON格式
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 配置标准logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """获取结构化日志记录器"""
    return structlog.get_logger(name)


def log_io_shapes(logger: structlog.BoundLogger, input_data: Any, output_data: Any) -> None:
    """记录输入输出的形状信息"""
    input_info = _get_tensor_info(input_data)
    output_info = _get_tensor_info(output_data)
    
    logger.info(
        "io_shapes",
        input_shapes=input_info["shapes"],
        input_types=input_info["types"],
        output_shapes=output_info["shapes"],
        output_types=output_info["types"]
    )


def _get_tensor_info(data: Any) -> Dict[str, Any]:
    """获取张量/数组的形状和类型信息"""
    result = {"shapes": [], "types": []}
    
    def _extract_info(obj, path=""):
        if hasattr(obj, "shape"):  # numpy数组, torch张量等
            result["shapes"].append(f"{path}: {obj.shape}")
            result["types"].append(f"{path}: {type(obj).__name__}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _extract_info(item, f"{path}[{i}]" if path else f"[{i}]")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                _extract_info(value, f"{path}.{key}" if path else key)
    
    _extract_info(data)
    return result


def monitor_performance(model_name: str = None, task_type: str = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            # 记录开始时的资源使用情况
            performance_monitor.log_system_resources(logger.bind(event="start"))
            
            # 记录输入形状
            if args:
                log_io_shapes(logger, args[0] if len(args) > 0 else None, None)
            
            # 执行函数
            try:
                result = func(*args, **kwargs)
                status = "success"
                
                # 记录输出形状
                log_io_shapes(logger, args[0] if len(args) > 0 else None, result)
                
                return result
            except Exception as e:
                status = "error"
                logger.error(
                    "function_error",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                # 计算执行时间
                execution_time = time.time() - start_time
                
                # 记录结束时的资源使用情况
                performance_monitor.log_system_resources(logger.bind(event="end"))
                
                # 更新Prometheus指标
                if model_name:
                    if task_type:
                        INFERENCE_TIME.labels(model_name=model_name, task_type=task_type).observe(execution_time)
                        INFERENCE_COUNTER.labels(model_name=model_name, task_type=task_type, status=status).inc()
                    else:
                        MODEL_LOAD_TIME.labels(model_name=model_name).observe(execution_time)
                        MODEL_LOAD_COUNTER.labels(model_name=model_name, status=status).inc()
                
                # 记录执行时间
                logger.info(
                    "performance_metrics",
                    function=func.__name__,
                    execution_time=execution_time,
                    model_name=model_name,
                    task_type=task_type,
                    status=status
                )
        
        return wrapper
    return decorator


@contextmanager
def monitor_context(model_name: str = None, task_type: str = None):
    """性能监控上下文管理器"""
    logger = get_logger("performance_monitor")
    start_time = time.time()
    
    # 记录开始时的资源使用情况
    performance_monitor.log_system_resources(logger.bind(event="start"))
    
    try:
        yield logger
        status = "success"
    except Exception as e:
        status = "error"
        logger.error(
            "context_error",
            model_name=model_name,
            task_type=task_type,
            error=str(e),
            exc_info=True
        )
        raise
    finally:
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录结束时的资源使用情况
        performance_monitor.log_system_resources(logger.bind(event="end"))
        
        # 更新Prometheus指标
        if model_name:
            if task_type:
                INFERENCE_TIME.labels(model_name=model_name, task_type=task_type).observe(execution_time)
                INFERENCE_COUNTER.labels(model_name=model_name, task_type=task_type, status=status).inc()
            else:
                MODEL_LOAD_TIME.labels(model_name=model_name).observe(execution_time)
                MODEL_LOAD_COUNTER.labels(model_name=model_name, status=status).inc()
        
        # 记录执行时间
        logger.info(
            "performance_metrics",
            execution_time=execution_time,
            model_name=model_name,
            task_type=task_type,
            status=status
        )


def start_metrics_server(port: int = 8000) -> None:
    """启动Prometheus指标服务器"""
    try:
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        print(f"Failed to start metrics server: {e}")


# 初始化日志系统
configure_structlog()