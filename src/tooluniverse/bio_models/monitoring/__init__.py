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
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.gpu_count = 0
        self.pynvml_initialized = False
        
        # 优先使用PyTorch的GPU检测
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            print(f"PyTorch detected {self.gpu_count} GPUs")
        # 如果PyTorch不可用或没有检测到GPU，尝试使用pynvml
        elif self.enable_gpu_monitoring and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.pynvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"pynvml detected {self.gpu_count} GPUs")
            except Exception as e:
                print(f"Failed to initialize NVIDIA ML: {e}")
                self.enable_gpu_monitoring = False
                self.pynvml_initialized = False
    
    def get_gpu_memory_info(self, gpu_id: int = 0) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        # 基本检查
        if not self.enable_gpu_monitoring or gpu_id < 0:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        # 首先尝试使用PyTorch获取GPU信息
        if TORCH_AVAILABLE and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            try:
                # 设置当前设备以确保正确初始化
                torch.cuda.set_device(gpu_id)
                device_properties = torch.cuda.get_device_properties(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                
                # 安全地更新Prometheus指标
                try:
                    from prometheus_client import Gauge
                    from prometheus_client import REGISTRY
                    metric_name = 'bio_gpu_memory_usage_bytes'
                    if metric_name in REGISTRY._names_to_collectors:
                        gpu_metric = REGISTRY._names_to_collectors[metric_name]
                        gpu_metric.labels(gpu_id=str(gpu_id)).set(allocated)
                except Exception as metric_error:
                    print(f"Warning: Failed to update GPU metric: {metric_error}")
                
                return {
                    "allocated": allocated,
                    "reserved": torch.cuda.memory_reserved(gpu_id),
                    "total": device_properties.total_memory,
                    "free": device_properties.total_memory - allocated
                }
            except Exception as torch_error:
                print(f"Error getting PyTorch GPU memory info: {torch_error}")
        
        # 如果PyTorch方法失败且pynvml已初始化，尝试使用pynvml
        if self.pynvml_initialized and PYNVML_AVAILABLE:
            try:
                # 确保GPU ID有效
                if gpu_id < pynvml.nvmlDeviceGetCount():
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    return {
                        "allocated": mem_info.used,
                        "reserved": mem_info.used,
                        "total": mem_info.total,
                        "free": mem_info.free
                    }
            except Exception as e:
                print(f"Error getting GPU memory info: {e}")
        
        # 默认返回
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
        
        # GPU内存 - 添加模型关联功能
        if self.enable_gpu_monitoring:
            # 尝试导入GPUScheduler以获取模型-GPU关联信息
            try:
                from ..model_manager import GPUScheduler
                # 检查是否有全局的GPUScheduler实例
                import __main__
                if hasattr(__main__, 'gpu_scheduler'):
                    gpu_scheduler = __main__.gpu_scheduler
                else:
                    # 创建临时实例以获取所有GPU信息
                    gpu_scheduler = None
            except Exception:
                gpu_scheduler = None
                
            for gpu_id in range(self.gpu_count):
                gpu_memory = self.get_gpu_memory_info(gpu_id)
                device_name = f"cuda:{gpu_id}"
                
                # 尝试获取在该GPU上运行的模型
                models_on_gpu = []
                if gpu_scheduler and hasattr(gpu_scheduler, 'get_gpu_models_info'):
                    models_on_gpu = gpu_scheduler.get_gpu_models_info(device_name)
                
                logger.info(
                    "gpu_resources",
                    gpu_id=gpu_id,
                    memory_allocated=gpu_memory["allocated"],
                    memory_total=gpu_memory["total"],
                    memory_percent=(gpu_memory["allocated"] / gpu_memory["total"]) * 100 if gpu_memory["total"] > 0 else 0,
                    models=models_on_gpu  # 添加在该GPU上运行的模型列表
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
        # 处理None值
        if obj is None:
            result["shapes"].append(f"{path}: None")
            result["types"].append(f"{path}: NoneType")
        # 处理有shape属性的对象（numpy数组, torch张量等）
        elif hasattr(obj, "shape"):
            result["shapes"].append(f"{path}: {obj.shape}")
            result["types"].append(f"{path}: {type(obj).__name__}")
        # 处理列表和元组，记录长度信息
        elif isinstance(obj, (list, tuple)):
            result["shapes"].append(f"{path}: [{len(obj)} items]")
            result["types"].append(f"{path}: {type(obj).__name__}")
            # 递归处理前3个元素以避免日志过大
            for i, item in enumerate(obj[:3]):
                _extract_info(item, f"{path}[{i}]" if path else f"[{i}]")
            # 如果列表/元组长度大于3，标记还有更多元素
            if len(obj) > 3:
                result["shapes"].append(f"{path}: ... (and {len(obj) - 3} more items)")
        # 处理字典，记录键值对数量
        elif isinstance(obj, dict):
            result["shapes"].append(f"{path}: {len(obj)} key-value pairs")
            result["types"].append(f"{path}: {type(obj).__name__}")
            # 递归处理前3个键值对
            for i, (key, value) in enumerate(list(obj.items())[:3]):
                _extract_info(value, f"{path}.{key}" if path else key)
            # 如果字典长度大于3，标记还有更多元素
            if len(obj) > 3:
                result["shapes"].append(f"{path}: ... (and {len(obj) - 3} more keys)")
        # 处理字符串
        elif isinstance(obj, str):
            result["shapes"].append(f"{path}: str[{len(obj)}]")
            result["types"].append(f"{path}: str")
        # 处理数字
        elif isinstance(obj, (int, float)):
            result["shapes"].append(f"{path}: scalar")
            result["types"].append(f"{path}: {type(obj).__name__}")
        # 处理其他类型
        else:
            try:
                # 尝试获取长度（对于支持__len__的对象）
                length = len(obj)
                result["shapes"].append(f"{path}: length={length}")
            except:
                result["shapes"].append(f"{path}: unknown")
            result["types"].append(f"{path}: {type(obj).__name__}")
    
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
                
                # 使用MetricsCollector记录指标，避免Prometheus指标问题
                try:
                    # 确保metrics_collector已正确初始化
                    collector_instance = get_metrics_collector_instance()
                    if model_name and collector_instance:
                        if task_type:
                            # 记录推理指标
                            collector_instance.add_metric('inference_time', execution_time, {'model_name': model_name, 'task_type': task_type})
                            collector_instance.add_metric('inference_count', 1, {'model_name': model_name, 'task_type': task_type, 'status': status})
                        else:
                            # 记录模型加载指标 - 这解决了无法记录模型加载时间的问题
                            collector_instance.add_model_metric(model_name, 'load_time', execution_time)
                            collector_instance.add_metric('model_load_time', execution_time, {'model_name': model_name, 'status': status})
                            # 同时记录到日志中以便排查
                            logger.info(f"Model load time recorded: {execution_time:.4f}s for {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to update metrics with MetricsCollector: {e}")
                    # 即使指标收集失败，也要确保模型加载时间被记录到日志中
                    if not task_type and model_name:
                        logger.info(f"Fallback: Model load time was {execution_time:.4f}s for {model_name}")
                
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
        
        # 使用MetricsCollector记录指标，避免Prometheus指标问题
        try:
            # 确保metrics_collector已正确初始化
            collector_instance = get_metrics_collector_instance()
            if model_name and collector_instance:
                if task_type:
                    # 记录推理指标
                    collector_instance.add_metric('inference_time', execution_time, {'model_name': model_name, 'task_type': task_type})
                    collector_instance.add_metric('inference_count', 1, {'model_name': model_name, 'task_type': task_type, 'status': status})
                else:
                    # 记录模型加载指标 - 这解决了无法记录模型加载时间的问题
                    collector_instance.add_model_metric(model_name, 'load_time', execution_time)
                    collector_instance.add_metric('model_load_time', execution_time, {'model_name': model_name, 'status': status})
                    # 同时记录到日志中以便排查
                    logger.info(f"Model load time recorded: {execution_time:.4f}s for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to update metrics with MetricsCollector: {e}")
            # 即使指标收集失败，也要确保模型加载时间被记录到日志中
            if not task_type and model_name:
                logger.info(f"Fallback: Model load time was {execution_time:.4f}s for {model_name}")
        
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