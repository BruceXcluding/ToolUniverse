"""
生物序列模型管理器
"""
import os
import json
import logging
import re
import csv
import threading
from datetime import datetime
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import torch
import psutil
from .models.base_model import BaseModel
from .task_types import TaskType, SequenceType, ModelStatus, DeviceType
from .monitoring import get_logger, monitor_performance, monitor_context, metrics_collector
from .container_runtime import ContainerRuntime, ContainerConfig, ContainerStatus
from .container_client import ModelContainerClient, DNABERT2Client, LucaOneClient
from .container_monitor import ContainerMonitor
from .performance_collector import PerformanceCollector, PerformanceMetric
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    DNABERT2 = "dnabert2"
    LUCAONE = "lucaone"
    RNABERT = "rnabert"
    UTRBERT = "utrbert"
    CODONBERT = "codonbert"
    RNFM = "rnfm"


class DeploymentType(Enum):
    """部署类型枚举"""
    LOCAL = "local"
    CONTAINER = "container"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: ModelType
    model_path: str
    config_path: Optional[str] = None
    device: str = "auto"
    max_memory: Optional[int] = None
    deployment_type: DeploymentType = DeploymentType.LOCAL
    container_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.container_config is None:
            self.container_config = {}


@dataclass
class ContainerModelConfig:
    """容器模型配置"""
    name: str
    model_type: ModelType
    image: str
    ports: Dict[str, int]
    supported_tasks: List[TaskType] = None
    supported_sequences: List[SequenceType] = None
    environment: Dict[str, str] = None
    volumes: Dict[str, Dict[str, str]] = None
    gpu_enabled: bool = False
    memory_limit: str = "8g"
    restart_policy: str = "unless-stopped"
    healthcheck: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.supported_tasks is None:
            self.supported_tasks = []
        if self.supported_sequences is None:
            self.supported_sequences = []
        if self.environment is None:
            self.environment = {}
        if self.volumes is None:
            self.volumes = {}
        if self.healthcheck is None:
            self.healthcheck = {}


class ModelManager:
    """生物序列模型管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger(__name__)
        self.models: Dict[str, BaseModel] = {}
        self.loaded_models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.config = {}
        self.gpu_scheduler = GPUScheduler()
        self.lock = threading.Lock()  # 添加线程锁
        
        # 初始化容器运行时
        self.container_runtime = ContainerRuntime()
        self.container_clients: Dict[str, ModelContainerClient] = {}
        self.container_configs: Dict[str, ContainerModelConfig] = {}
        
        # 初始化容器监控
        self.container_monitor = ContainerMonitor(
            container_runtime=self.container_runtime,
            metrics_collector=metrics_collector,
            check_interval=30,
            metrics_history_size=1000
        )
        
        # 添加容器监控告警回调
        self.container_monitor.add_alert_callback(self._handle_container_alert)  # 存储容器配置
        
        # 初始化性能收集器
        self.performance_collector = PerformanceCollector()
        
        # 注册性能回调，将性能数据保存到文件
        self.performance_collector.register_performance_callback(self._save_performance_metric)
        
        # 加载配置
        if config_path:
            self._load_config(config_path)
        else:
            # 使用默认配置路径
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "bio_models", "model_config.json"
            )
            if os.path.exists(default_config_path):
                self._load_config(default_config_path)
        
        # 自动发现已运行的容器
        self._discover_containers()
    
    def _discover_containers(self):
        """自动发现并注册已运行的容器"""
        try:
            # 获取所有运行的容器
            containers = self.container_runtime.list_containers()
            self.logger.info(f"发现 {len(containers)} 个容器")
            
            for container_info in containers:
                # 获取容器名称
                container_name = container_info.name
                self.logger.info(f"处理容器: {container_name}, 状态: {container_info.status}")
                
                # 检查是否已经注册
                if container_name not in self.container_configs:
                    # 检查容器状态
                    if container_info.status == ContainerStatus.RUNNING:
                        # 将ContainerInfo对象转换为字典格式以便处理
                        container_dict = {
                            'name': container_info.name,
                            'status': container_info.status.value,
                            'image': container_info.image,
                            'ports': container_info.ports,
                            'api_endpoint': container_info.api_endpoint,
                            'metrics_endpoint': container_info.metrics_endpoint
                        }
                        
                        # 尝试识别模型类型
                        model_type = self._identify_model_type(container_name, container_dict)
                        self.logger.info(f"容器 {container_name} 识别的模型类型: {model_type}")
                        
                        if model_type:
                            # 获取容器端口信息
                            ports = self._extract_ports(container_dict)
                            self.logger.info(f"容器 {container_name} 提取的端口: {ports}")
                            
                            # 自动注册容器
                            # 根据模型类型设置支持的任务和序列类型
                            supported_tasks = []
                            supported_sequences = []
                            
                            if model_type == ModelType.DNABERT2:
                                supported_tasks = [TaskType.EMBEDDING, TaskType.CLASSIFICATION, TaskType.FINE_TUNING]
                                supported_sequences = [SequenceType.DNA]
                            elif model_type == ModelType.LUCAONE:
                                supported_tasks = [TaskType.EMBEDDING, TaskType.PREDICTION, TaskType.CLASSIFICATION]
                                supported_sequences = [SequenceType.DNA, SequenceType.RNA, SequenceType.PROTEIN]
                            
                            config = ContainerModelConfig(
                                name=container_name,
                                model_type=model_type,
                                image=container_dict.get('image', ''),
                                ports=ports,
                                supported_tasks=supported_tasks,
                                supported_sequences=supported_sequences,
                                environment={},  # 使用environment而不是env_vars
                                volumes={},
                                gpu_enabled=False
                            )
                            self.register_container_model(config)
                            
                            # 为已运行的容器创建客户端
                            if container_info.api_endpoint:
                                if model_type == ModelType.DNABERT2:
                                    client = DNABERT2Client(container_info.api_endpoint)
                                elif model_type == ModelType.LUCAONE:
                                    client = LucaOneClient(container_info.api_endpoint)
                                else:
                                    client = ModelContainerClient(container_info.api_endpoint)
                                
                                self.container_clients[container_name] = client
                                self.logger.info(f"已为容器 {container_name} 创建客户端: {container_info.api_endpoint}")
                            
                            self.logger.info(f"自动发现并注册容器模型: {container_name} ({model_type.value})")
                            
                            # 创建并注册模型实例到models字典中
                            class ContainerModel(BaseModel):
                                def __init__(self, name, model_type, api_endpoint):
                                    # 创建一个具有get方法的配置类
                                    class SimpleConfig:
                                        def __init__(self):
                                            self._dict = {}
                                        
                                        def __setattr__(self, name, value):
                                            if name == '_dict':
                                                super().__setattr__(name, value)
                                            else:
                                                self._dict[name] = value
                                                super().__setattr__(name, value)
                                        
                                        def get(self, key, default=None):
                                            return self._dict.get(key, default)
                                    
                                    config = SimpleConfig()
                                    config.model_type = model_type
                                    config.api_endpoint = api_endpoint
                                    config.supported_sequences = []  # 添加默认支持的序列类型
                                    config.supported_tasks = []  # 添加默认支持的任务类型
                                    
                                    super().__init__(model_name=name, config=config)
                                    self.model_name = name
                                    self.model_type = model_type
                                    self.api_endpoint = api_endpoint
                                    self.use_docker = True
                                    self.status = ModelStatus.LOADED
                                    # 修复：移除对不存在的container_clients属性的依赖
                                    # 直接设置客户端为None，让predict方法处理
                                    self.client = None
                                
                                def load_model(self, device=None):
                                    return True
                                
                                def unload_model(self):
                                    return True
                                
                                def is_loaded(self):
                                    return True
                                
                                def predict(self, sequences, task_type, **kwargs):
                                    if not self.client:
                                        return {"error": "容器客户端未初始化"}
                                    return self.client.predict(sequences, task_type, **kwargs)
                                
                                def get_memory_usage(self):
                                    return 0
                                
                                def get_model_info(self):
                                    return {"model_name": self.model_name, "model_type": self.model_type.value, "use_docker": True}
                            
                            # 创建模型实例并注册到models字典
                            model_instance = ContainerModel(container_name, model_type, container_info.api_endpoint)
                            self.models[container_name] = model_instance
                            self.logger.info(f"已将容器模型 {container_name} 注册到models字典")
                        else:
                            self.logger.warning(f"无法识别容器 {container_name} 的模型类型")
                    else:
                        self.logger.info(f"容器 {container_name} 未运行，状态: {container_info.status}")
                else:
                    self.logger.info(f"容器 {container_name} 已经注册")
        
        except Exception as e:
            self.logger.error(f"自动发现容器失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _identify_model_type(self, container_name: str, container_info: dict) -> Optional[ModelType]:
        """根据容器信息识别模型类型"""
        # 根据容器名称识别
        if 'dnabert' in container_name.lower():
            return ModelType.DNABERT2
        elif 'luca' in container_name.lower():  # 修改为更宽泛的匹配
            return ModelType.LUCAONE
        
        # 根据镜像名称识别
        image = container_info.get('image', '').lower()
        if 'dnabert' in image:
            return ModelType.DNABERT2
        elif 'luca' in image:
            return ModelType.LUCAONE
        
        # 根据环境变量识别
        env_vars = container_info.get('env_vars', {})
        if 'MODEL_TYPE' in env_vars:
            model_type_str = env_vars['MODEL_TYPE'].lower()
            if 'dnabert' in model_type_str:
                return ModelType.DNABERT2
            elif 'luca' in model_type_str:
                return ModelType.LUCAONE
        
        return None
    
    def _extract_ports(self, container_info: dict) -> Dict[str, int]:
        """从容器信息中提取端口映射"""
        ports = {}
        
        # 获取端口映射信息
        port_bindings = container_info.get('ports', {})
        
        # 尝试识别API端口和监控端口
        for container_port, host_bindings in port_bindings.items():
            if isinstance(host_bindings, list) and host_bindings:
                host_port = host_bindings[0].get('HostPort')
                if host_port:
                    # 根据端口号识别用途
                    if int(container_port) in [8000, 8001, 8002, 8003]:
                        ports['api'] = int(host_port)
                    elif int(container_port) in [8011, 8012, 8013]:
                        ports['metrics'] = int(host_port)
        
        # 如果无法识别，使用默认值
        if 'api' not in ports and 'metrics' not in ports:
            # 尝试从容器信息中获取API端点
            api_endpoint = container_info.get('api_endpoint', '')
            if api_endpoint:
                # 从URL中提取端口
                import re
                match = re.search(r':(\d+)', api_endpoint)
                if match:
                    ports['api'] = int(match.group(1))
        
        return ports
    
    def _load_config(self, config_path: str):
        """加载模型配置"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            # 解析配置中的环境变量
            self.config = self._resolve_env_vars(self.config)
            self.logger.info(f"已加载模型配置: {config_path}")
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            self.config = {}
    
    def _resolve_env_vars(self, obj):
        """递归解析对象中的环境变量
        
        支持格式: ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # 查找并替换环境变量
            pattern = r'\$\{([^:}]+):?([^}]*)\}'
            matches = re.findall(pattern, obj)
            
            for var_name, default_value in matches:
                env_value = os.getenv(var_name, default_value)
                obj = obj.replace(f'${{{var_name}:{default_value}}}', env_value)
            
            return obj
        else:
            return obj
    
    @monitor_performance(model_name="model_manager")
    def register_model(self, model: BaseModel):
        """
        注册模型
        
        Args:
            model: 模型实例
        """
        self.models[model.model_name] = model
        self.logger.info("模型已注册", model_name=model.model_name)
    
    @monitor_performance(model_name="model_manager")
    def load_model(self, model_name: str, device: Union[str, DeviceType] = DeviceType.AUTO) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            device: 设备类型
            
        Returns:
            bool: 是否加载成功
        """
        if model_name not in self.models:
            self.logger.error("模型不存在", model_name=model_name)
            return False
            
        model = self.models[model_name]
        
        # 检查模型是否已加载
        if model.is_loaded():
            self.logger.info("模型已加载", model_name=model_name)
            return True
            
        # 检查内存是否足够
        if not self._check_memory(model):
            self.logger.warning("内存不足，尝试卸载其他模型", model_name=model_name)
            self._unload_least_used_model()
            
        # 优化设备选择，优先使用GPUScheduler分配的GPU
        try:
            if device == DeviceType.AUTO and hasattr(self, 'gpu_scheduler'):
                available_device = self.gpu_scheduler.get_available_device()
                if available_device:
                    device = available_device
                    self.logger.info(f"通过GPUScheduler为模型 {model_name} 分配设备: {device}")
        except Exception as e:
            self.logger.warning(f"无法使用GPUScheduler分配设备: {str(e)}")
        
        # 加载模型
        try:
            with monitor_context(model_name=model_name):
                success = model.load_model(device)
                if success:
                    self.loaded_models[model_name] = model
                    self.logger.info("模型加载成功", model_name=model_name)
                    
                    # 记录模型加载指标
                    metrics_collector.add_model_metric(model_name, "load", 1)
                    
                    # 记录加载后的资源使用情况
                    self._log_resource_usage(model_name)
                    
                    # 使用GPUScheduler跟踪模型与GPU的关联
                    try:
                        # 尝试获取模型的实际设备
                        if hasattr(model, 'device'):
                            model_device = model.device
                            if model_device and isinstance(model_device, str) and model_device.startswith("cuda"):
                                self.gpu_scheduler.track_model_gpu_usage(model_name, model_device)
                                self.logger.info(f"已跟踪模型 {model_name} 与设备 {model_device} 的关联")
                    except Exception as e:
                        self.logger.warning(f"无法跟踪模型-GPU关联: {str(e)}")
                    
                    # 跟踪模型使用的GPU
                    if hasattr(model, 'device') and model.device and str(model.device).startswith('cuda:'):
                        self.gpu_scheduler.track_model_gpu_usage(model_name, str(model.device))
                return success
        except Exception as e:
            self.logger.error("模型加载失败", model_name=model_name, error=str(e))
            # 记录模型加载失败指标
            metrics_collector.add_model_metric(model_name, "load_failure", 1)
            return False
    
    @monitor_performance(model_name="model_manager")
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否卸载成功
        """
        if model_name not in self.loaded_models:
            self.logger.warning("模型未加载", model_name=model_name)
            return False
            
        model = self.loaded_models[model_name]
        try:
            with monitor_context(model_name=model_name):
                success = model.unload_model()
                if success:
                    # 取消跟踪模型使用的GPU
                    self.gpu_scheduler.untrack_model_gpu_usage(model_name)
                    
                    del self.loaded_models[model_name]
                    self.logger.info("模型卸载成功", model_name=model_name)
                    
                    # 记录模型卸载指标
                    metrics_collector.add_model_metric(model_name, "unload", 1)
                return success
        except Exception as e:
            self.logger.error("模型卸载失败", model_name=model_name, error=str(e))
            # 记录模型卸载失败指标
            metrics_collector.add_model_metric(model_name, "unload_failure", 1)
            return False
    
    def get_best_model(self, task_type: TaskType, sequence_type: Optional[SequenceType] = None) -> Optional[str]:
        """
        根据任务类型和序列类型获取最佳模型
        
        Args:
            task_type: 任务类型
            sequence_type: 序列类型
            
        Returns:
            Optional[str]: 最佳模型名称
        """
        # 首先检查容器模型
        for name, config in self.container_configs.items():
            container_info = self.container_runtime.get_container_info(name)
            if container_info and container_info.status == ContainerStatus.RUNNING:
                # 检查模型是否支持指定的任务类型和序列类型
                if task_type and task_type not in config.supported_tasks:
                    continue
                if sequence_type and sequence_type not in config.supported_sequences:
                    continue
                
                return name
        
        # 然后检查本地模型
        for name, config in self.model_configs.items():
            # 检查模型是否支持指定的任务类型和序列类型
            if task_type and task_type not in config.supported_tasks:
                continue
            if sequence_type and sequence_type not in config.supported_sequences:
                continue
            
            # 如果模型已加载，优先返回
            if name in self.loaded_models:
                return name
        
        # 如果没有已加载的模型，返回第一个支持的模型
        for name, config in self.model_configs.items():
            if task_type and task_type not in config.supported_tasks:
                continue
            if sequence_type and sequence_type not in config.supported_sequences:
                continue
            return name
        
        # 从配置中获取任务映射
        task_mapping = self.config.get("task_mapping", {})
        task_key = task_type.value
        
        if task_key not in task_mapping:
            self.logger.warning(f"未找到任务映射: {task_key}")
            return None
            
        task_config = task_mapping[task_key]
        default_model = task_config.get("default_model")
        
        # 如果没有指定序列类型，返回默认模型
        if not sequence_type:
            return default_model
            
        # 检查是否有针对特定序列类型的模型
        seq_key = sequence_type.value
        if seq_key in task_config and task_config[seq_key]:
            return task_config[seq_key]
            
        # 返回默认模型
        return default_model
    
    @monitor_performance(model_name="model_manager", task_type="prediction")
    def predict(
        self, 
        sequences: List[str], 
        task_type: TaskType, 
        model_name: Optional[str] = None,
        sequence_type: Optional[SequenceType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            sequences: 输入序列列表
            task_type: 任务类型
            model_name: 指定模型名称
            sequence_type: 序列类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        # 确定使用的模型
        if not model_name:
            model_name = self.get_best_model(task_type, sequence_type)
            
        if not model_name:
            return {"error": "没有找到合适的模型"}
        
        # 检查是否是容器模型
        if model_name in self.container_clients:
            return self._predict_with_container_model(model_name, sequences, task_type, sequence_type, **kwargs)
            
        # 加载本地模型
        if not self.load_model(model_name):
            return {"error": f"模型加载失败: {model_name}"}
            
        # 执行预测
        model = self.loaded_models[model_name]
        try:
            with monitor_context(model_name=model_name, task_type=task_type.value):
                # 记录输入信息
                self.logger.info(
                    "开始预测", 
                    model_name=model_name,
                    task_type=task_type.value,
                    sequence_count=len(sequences),
                    sequence_type=sequence_type.value if sequence_type else None
                )
                
                # 执行预测
                result = model.predict(sequences, task_type, **kwargs)
                
                # 记录预测成功指标
                metrics_collector.add_model_metric(model_name, "prediction_success", 1)
                
                # 记录输出信息
                if "predictions" in result:
                    self.logger.info(
                        "预测完成", 
                        model_name=model_name,
                        output_count=len(result["predictions"]) if isinstance(result["predictions"], list) else 1
                    )
                
                return result
        except Exception as e:
            self.logger.error("预测失败", model_name=model_name, error=str(e))
            # 记录预测失败指标
            metrics_collector.add_model_metric(model_name, "prediction_failure", 1)
            return {"error": f"预测失败: {str(e)}"}
    
    def list_models(self) -> Dict[str, Any]:
        """列出所有已注册的模型
        
        Returns:
            Dict[str, Any]: 模型列表
        """
        # 获取本地模型
        local_models = []
        for name, model in self.models.items():
            local_models.append({
                "name": name,
                "type": getattr(model, 'model_type', 'unknown'),
                "is_loaded": name in self.loaded_models,
                "deployment_type": "local"
            })
        
        # 获取容器模型
        container_models = []
        for name, config in self.container_configs.items():
            container_info = self.container_runtime.get_container_info(name)
            status = container_info.status.value if container_info else "unknown"
            container_models.append({
                "name": name,
                "type": config.model_type.value,
                "status": status,
                "deployment_type": "container",
                "image": config.image
            })
        
        return {
            "local_models": local_models,
            "container_models": container_models,
            "total_local": len(local_models),
            "total_container": len(container_models),
            "total": len(local_models) + len(container_models)
        }
    
    def list_loaded_models(self) -> Dict[str, Any]:
        """
        列出所有已加载的模型
        
        Returns:
            Dict[str, Any]: 已加载的模型列表
        """
        # 获取本地已加载的模型
        local_loaded = []
        for name in self.loaded_models:
            model = self.models[name]
            local_loaded.append({
                "name": name,
                "type": model.model_type.value,
                "device": model.device,
                "deployment_type": "local"
            })
        
        # 获取运行中的容器模型
        container_running = []
        for name in self.container_clients:
            container_info = self.container_runtime.get_container_info(name)
            if container_info and container_info.status == ContainerStatus.RUNNING:
                config = self.container_configs[name]
                container_running.append({
                    "name": name,
                    "type": config.model_type.value,
                    "image": config.image,
                    "deployment_type": "container",
                    "ports": config.ports
                })
        
        return {
            "local_loaded": local_loaded,
            "container_running": container_running,
            "total_local": len(local_loaded),
            "total_container": len(container_running),
            "total": len(local_loaded) + len(container_running)
        }
    
    def register_container_model(self, config: ContainerModelConfig) -> bool:
        """
        注册容器模型
        
        Args:
            config: 容器模型配置
            
        Returns:
            bool: 是否注册成功
        """
        try:
            with self.lock:
                self.container_configs[config.name] = config
                self.logger.info(f"容器模型注册成功: {config.name}")
                return True
        except Exception as e:
            self.logger.error(f"容器模型注册失败: {config.name}, 错误: {e}")
            return False
    
    def start_container_model(self, model_name: str) -> bool:
        """
        启动容器模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否启动成功
        """
        if model_name not in self.container_configs:
            self.logger.error(f"容器模型配置不存在: {model_name}")
            return False
        
        try:
            config = self.container_configs[model_name]
            
            # 创建容器配置
            container_config = ContainerConfig(
                name=config.name,
                image=config.image,
                ports=config.ports,
                environment=config.environment,
                volumes=config.volumes,
                gpu_enabled=config.gpu_enabled,
                memory_limit=config.memory_limit,
                restart_policy=config.restart_policy,
                healthcheck=config.healthcheck
            )
            
            # 启动容器
            container_info = self.container_runtime.start_container(container_config)
            
            if not container_info:
                self.logger.error(f"容器启动失败: {model_name}")
                return False
            
            # 创建容器客户端
            if container_info.api_endpoint:
                if config.model_type == ModelType.DNABERT2:
                    client = DNABERT2Client(container_info.api_endpoint)
                elif config.model_type == ModelType.LUCAONE:
                    client = LucaOneClient(container_info.api_endpoint)
                else:
                    client = ModelContainerClient(container_info.api_endpoint)
                
                self.container_clients[model_name] = client
                
                # 等待容器完全启动
                time.sleep(5)
                
                # 检查健康状态
                if client.health_check():
                    self.logger.info(f"容器模型启动成功: {model_name}")
                    return True
                else:
                    self.logger.error(f"容器模型健康检查失败: {model_name}")
                    return False
            else:
                self.logger.error(f"容器API端点未设置: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"容器模型启动失败: {model_name}, 错误: {e}")
            return False
    
    def stop_container_model(self, model_name: str) -> bool:
        """
        停止容器模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否停止成功
        """
        try:
            # 停止容器
            success = self.container_runtime.stop_container(model_name)
            
            if success:
                # 移除客户端
                if model_name in self.container_clients:
                    del self.container_clients[model_name]
                
                self.logger.info(f"容器模型停止成功: {model_name}")
            else:
                self.logger.error(f"容器模型停止失败: {model_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"容器模型停止失败: {model_name}, 错误: {e}")
            return False
    
    def remove_container_model(self, model_name: str) -> bool:
        """
        删除容器模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 删除容器
            success = self.container_runtime.remove_container(model_name)
            
            if success:
                # 移除客户端
                if model_name in self.container_clients:
                    del self.container_clients[model_name]
                
                self.logger.info(f"容器模型删除成功: {model_name}")
            else:
                self.logger.error(f"容器模型删除失败: {model_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"容器模型删除失败: {model_name}, 错误: {e}")
            return False
    
    def restart_container_model(self, model_name: str) -> bool:
        """
        重启容器模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否重启成功
        """
        try:
            # 重启容器
            success = self.container_runtime.restart_container(model_name)
            
            if success:
                # 等待容器完全启动
                time.sleep(5)
                
                # 更新客户端
                if model_name in self.container_configs and model_name in self.container_clients:
                    config = self.container_configs[model_name]
                    container_info = self.container_runtime.get_container_info(model_name)
                    
                    if container_info and container_info.api_endpoint:
                        if config.model_type == ModelType.DNABERT2:
                            client = DNABERT2Client(container_info.api_endpoint)
                        elif config.model_type == ModelType.LUCAONE:
                            client = LucaOneClient(container_info.api_endpoint)
                        else:
                            client = ModelContainerClient(container_info.api_endpoint)
                        
                        self.container_clients[model_name] = client
                
                self.logger.info(f"容器模型重启成功: {model_name}")
            else:
                self.logger.error(f"容器模型重启失败: {model_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"容器模型重启失败: {model_name}, 错误: {e}")
            return False
    
    def get_container_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取容器模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Optional[Dict[str, Any]]: 容器模型信息
        """
        try:
            # 获取容器信息
            container_info = self.container_runtime.get_container_info(model_name)
            
            if not container_info:
                return None
            
            # 转换为字典
            info = {
                "name": container_info.name,
                "status": container_info.status.value,
                "image": container_info.image,
                "ports": container_info.ports,
                "created_at": container_info.created_at,
                "started_at": container_info.started_at,
                "health_status": container_info.health_status,
                "api_endpoint": container_info.api_endpoint,
                "metrics_endpoint": container_info.metrics_endpoint
            }
            
            # 获取模型信息
            if model_name in self.container_clients:
                client = self.container_clients[model_name]
                try:
                    model_info = client.get_model_info()
                    if model_info:
                        info["model_info"] = model_info
                except Exception as e:
                    self.logger.error(f"获取容器模型信息失败: {model_name}, 错误: {e}")
            
            # 获取容器指标
            metrics = self.container_runtime.get_container_metrics(model_name)
            if metrics:
                info["metrics"] = metrics
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取容器模型信息失败: {model_name}, 错误: {e}")
            return None
    
    def list_container_models(self) -> List[Dict[str, Any]]:
        """
        列出所有容器模型
        
        Returns:
            List[Dict[str, Any]]: 容器模型列表
        """
        try:
            containers = self.container_runtime.list_containers(all_containers=True)
            
            result = []
            for container in containers:
                # 转换为字典
                info = {
                    "name": container.name,
                    "status": container.status.value,
                    "image": container.image,
                    "ports": container.ports,
                    "created_at": container.created_at,
                    "started_at": container.started_at,
                    "health_status": container.health_status,
                    "api_endpoint": container.api_endpoint,
                    "metrics_endpoint": container.metrics_endpoint
                }
                
                # 添加模型配置信息
                if container.name in self.container_configs:
                    config = self.container_configs[container.name]
                    info["model_type"] = config.model_type.value
                    info["registered"] = True
                else:
                    info["model_type"] = None
                    info["registered"] = False
                
                result.append(info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"列出容器模型失败: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        # 检查是否是容器模型
        if model_name in self.container_clients:
            return self.get_container_model_info(model_name) or {"error": "获取容器模型信息失败"}
        
        if model_name not in self.models:
            return {"error": "模型不存在"}
            
        model = self.models[model_name]
        return {
            "name": model.model_name,
            "supported_tasks": model.supported_tasks,
            "supported_sequences": model.supported_sequences,
            "memory_requirement": model.memory_requirement,
            "status": model.status.value,
            "device": model.device,
            "is_loaded": model.is_loaded(),
            "model_type": "local"
        }
    
    @monitor_performance(model_name="model_manager")
    def predict_batch(
        self,
        model_name: Optional[str] = None,
        sequences: List[str] = None,
        task_type: Optional[str] = None,
        sequence_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量预测
        
        Args:
            model_name: 模型名称
            sequences: 输入序列列表
            task_type: 任务类型
            sequence_type: 序列类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        if not sequences:
            return {"error": "没有提供输入序列"}
        
        # 如果没有指定模型，尝试选择合适的模型
        if not model_name:
            model_name = self.get_best_model(TaskType(task_type) if task_type else None, 
                                           SequenceType(sequence_type) if sequence_type else None)
            if not model_name:
                return {"error": "没有找到合适的模型"}
        
        # 检查是否是容器模型
        if model_name in self.container_clients:
            return self._predict_batch_with_container_model(model_name, sequences, task_type, sequence_type, **kwargs)
        
        # 加载本地模型
        if not self.load_model(model_name):
            return {"error": f"模型加载失败: {model_name}"}
        
        start_time = time.time()
        results = []
        
        # 获取模型
        model = self.loaded_models[model_name]
        
        # 批量处理
        for sequence in sequences:
            try:
                # 执行预测
                result = model.predict([sequence], TaskType(task_type) if task_type else None, **kwargs)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"批量预测失败: 模型={model_name}, 序列={sequence[:20]}..., 错误={str(e)}")
                results.append({"error": str(e)})
        
        # 记录指标
        processing_time = time.time() - start_time
        metrics_collector.add_model_metric(model_name, "batch_prediction_count", 1)
        metrics_collector.add_model_metric(model_name, "batch_prediction_time", processing_time)
        metrics_collector.add_model_metric(model_name, "batch_size", len(sequences))
        
        return {
            "results": results,
            "model_name": model_name,
            "model_type": "local",
            "batch_size": len(sequences),
            "processing_time": processing_time
        }
    
    def _predict_with_container_model(
        self,
        model_name: str,
        sequences: List[str],
        task_type: Union[str, TaskType],
        sequence_type: Optional[SequenceType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用容器模型进行预测
        
        Args:
            model_name: 模型名称
            sequences: 输入序列列表
            task_type: 任务类型
            sequence_type: 序列类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        # 转换任务类型
        if isinstance(task_type, TaskType):
            task_type_str = task_type.value
        else:
            task_type_str = task_type
            
        # 调用批量预测方法
        return self._predict_batch_with_container_model(
            model_name, sequences, task_type_str, 
            sequence_type.value if sequence_type else None, **kwargs
        )
    
    def _predict_batch_with_container_model(
        self,
        model_name: str,
        sequences: List[str],
        task_type: Optional[str] = None,
        sequence_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用容器模型进行批量预测
        
        Args:
            model_name: 模型名称
            sequences: 输入序列列表
            task_type: 任务类型
            sequence_type: 序列类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        start_time = time.time()
        
        # 获取容器客户端
        client = self.container_clients[model_name]
        
        try:
            # 根据模型类型执行预测
            if model_name in self.container_configs:
                config = self.container_configs[model_name]
                model_type = config.model_type
            else:
                # 尝试从容器信息中获取模型类型
                container_info = self.container_runtime.get_container_info(model_name)
                if container_info and "dnabert2" in container_info.image.lower():
                    model_type = ModelType.DNABERT2
                elif "lucaone" in container_info.image.lower():
                    model_type = ModelType.LUCAONE
                else:
                    model_type = ModelType.DNABERT2  # 默认类型
            
            # 执行批量预测
            if model_type == ModelType.DNABERT2:
                if isinstance(client, DNABERT2Client):
                    if task_type == "embedding":
                        # 处理嵌入任务
                        results = []
                        for sequence in sequences:
                            result = client.get_embedding(sequence, kwargs.get("embedding_dimension", 768))
                            results.append(result)
                        
                        # 转换结果格式为API期望的格式
                        formatted_results = []
                        for i, result in enumerate(results):
                            seq = sequences[i]  # 使用对应的序列
                            formatted_results.append({
                                "sequence": seq,
                                "sequence_length": len(seq),
                                "embedding": result.prediction  # 使用prediction字段中的嵌入向量
                            })
                        
                        return {
                            "success": True,
                            "results": formatted_results,
                            "model_name": model_name,
                            "model_type": model_type.value,
                            "batch_size": len(sequences),
                            "processing_time": time.time() - start_time,
                            "container_model": True
                        }
                    elif task_type == "classification":
                        result = client.classify_batch(sequences, kwargs)
                    elif task_type == "extraction":
                        result = client.extract_batch_features(sequences, kwargs)
                    elif task_type == "annotation":
                        result = client.annotate_batch_sequences(sequences, kwargs)
                    else:
                        # 默认分类
                        result = client.classify_batch(sequences, kwargs)
                else:
                    # 通用批量预测
                    from .container_client import BatchPredictionRequest
                    request = BatchPredictionRequest(
                        model_type=model_type,
                        sequences=sequences,
                        task_type=task_type,
                        parameters=kwargs
                    )
                    result = client.predict_batch(request)
            elif model_type == ModelType.LUCAONE:
                if isinstance(client, LucaOneClient):
                    if task_type == "prediction":
                        result = client.predict_batch_sequences(sequences, kwargs)
                    elif task_type == "classification":
                        result = client.classify_batch_sequences(sequences, kwargs)
                    elif task_type == "annotation":
                        result = client.annotate_batch_sequences(sequences, kwargs)
                    else:
                        # 默认预测
                        result = client.predict_batch_sequences(sequences, kwargs)
                else:
                    # 通用批量预测
                    from .container_client import BatchPredictionRequest
                    request = BatchPredictionRequest(
                        model_type=model_type,
                        sequences=sequences,
                        task_type=task_type,
                        parameters=kwargs
                    )
                    result = client.predict_batch(request)
            else:
                # 通用批量预测
                from .container_client import BatchPredictionRequest
                request = BatchPredictionRequest(
                    model_type=model_type,
                    sequences=sequences,
                    task_type=task_type,
                    parameters=kwargs
                )
                result = client.predict_batch(request)
            
            # 转换结果格式
            if hasattr(result, 'success') and hasattr(result, 'data'):
                if result.success:
                    return {
                        "success": True,
                        "results": result.data,
                        "model_name": model_name,
                        "model_type": model_type.value,
                        "batch_size": len(sequences),
                        "processing_time": result.processing_time or (time.time() - start_time),
                        "container_model": True
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error,
                        "model_name": model_name,
                        "model_type": model_type.value,
                        "batch_size": len(sequences),
                        "processing_time": result.processing_time or (time.time() - start_time),
                        "container_model": True
                    }
            else:
                # 假设结果是字典格式
                return {
                    "success": True,
                    "results": result,
                    "model_name": model_name,
                    "model_type": model_type.value,
                    "batch_size": len(sequences),
                    "processing_time": time.time() - start_time,
                    "container_model": True
                }
                
        except Exception as e:
            self.logger.error(f"容器模型批量预测失败: 模型={model_name}, 错误={str(e)}")
            metrics_collector.add_model_metric(model_name, "batch_prediction_error", 1)
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "model_type": model_type.value if 'model_type' in locals() else "unknown",
                "batch_size": len(sequences),
                "processing_time": time.time() - start_time,
                "container_model": True
            }
    
    def _log_resource_usage(self, model_name: str) -> None:
        """记录模型加载后的资源使用情况"""
        # 获取系统资源信息
        cpu_memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        self.logger.info(
            "系统资源使用情况",
            model_name=model_name,
            cpu_percent=cpu_usage,
            memory_total=cpu_memory.total,
            memory_used=cpu_memory.used,
            memory_percent=cpu_memory.percent
        )
        
        # GPU资源信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                memory_info = self.gpu_scheduler.get_device_memory_info(device)
                if "error" not in memory_info:
                    self.logger.info(
                        "GPU资源使用情况",
                        model_name=model_name,
                        gpu_id=i,
                        gpu_memory_total=memory_info["total"],
                        gpu_memory_allocated=memory_info["allocated"],
                        gpu_memory_available=memory_info["available"]
                    )
        
        # PyTorch GPU内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)   # GB
            
            self.logger.info(
                "PyTorch GPU内存使用情况",
                model_name=model_name,
                memory_allocated_gb=allocated,
                memory_reserved_gb=reserved
            )
    
    def _check_memory(self, model: BaseModel) -> bool:
        """检查是否有足够的内存加载模型"""
        # 获取可用内存
        available_memory = psutil.virtual_memory().available // (1024 * 1024)  # MB
        
        # 计算已加载模型的内存使用
        loaded_memory = sum(m.get_memory_usage() for m in self.loaded_models.values())
        
        # 检查是否有足够内存
        return available_memory - loaded_memory >= model.memory_requirement
    
    def _unload_least_used_model(self):
        """卸载最少使用的模型以释放内存"""
        if not self.loaded_models:
            return
            
        # 简单实现：卸载第一个加载的模型
        # 在实际应用中，可以根据使用频率等信息来决定
        model_name = next(iter(self.loaded_models))
        self.unload_model(model_name)
    
    def _save_performance_metric(self, metric: PerformanceMetric) -> None:
        """
        保存性能指标到文件
        
        Args:
            metric: 性能指标
        """
        try:
            # 每天保存一次指标数据到CSV文件
            today = datetime.now().strftime("%Y%m%d")
            csv_path = f"performance_data/metrics_{today}.csv"
            
            # 检查文件是否存在，如果不存在则创建并写入头部
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'model_name', 'deployment_type', 'metric_type',
                    'value', 'unit', 'metadata'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'timestamp': metric.timestamp.isoformat(),
                    'model_name': metric.model_name,
                    'deployment_type': metric.deployment_type,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': json.dumps(metric.metadata) if metric.metadata else ''
                })
        except Exception as e:
            self.logger.error(f"保存性能指标失败: {str(e)}")
    
    def collect_inference_time(
        self,
        model_name: str,
        deployment_type: str,
        inference_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        收集推理时间指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            inference_time_ms: 推理时间（毫秒）
            metadata: 元数据
        """
        self.performance_collector.add_inference_time(
            model_name=model_name,
            deployment_type=deployment_type,
            inference_time_ms=inference_time_ms,
            metadata=metadata
        )
    
    def collect_throughput(
        self,
        model_name: str,
        deployment_type: str,
        requests_per_sec: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        收集吞吐量指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            requests_per_sec: 每秒请求数
            metadata: 元数据
        """
        self.performance_collector.add_throughput(
            model_name=model_name,
            deployment_type=deployment_type,
            requests_per_sec=requests_per_sec,
            metadata=metadata
        )
    
    def collect_memory_usage(
        self,
        model_name: str,
        deployment_type: str,
        memory_mb: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        收集内存使用指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            memory_mb: 内存使用量（MB）
            metadata: 元数据
        """
        self.performance_collector.add_memory_usage(
            model_name=model_name,
            deployment_type=deployment_type,
            memory_mb=memory_mb,
            metadata=metadata
        )
    
    def collect_cpu_usage(
        self,
        model_name: str,
        deployment_type: str,
        cpu_percent: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        收集CPU使用指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            cpu_percent: CPU使用率（百分比）
            metadata: 元数据
        """
        self.performance_collector.add_cpu_usage(
            model_name=model_name,
            deployment_type=deployment_type,
            cpu_percent=cpu_percent,
            metadata=metadata
        )
    
    def get_performance_metrics(
        self,
        model_name: Optional[str] = None,
        deployment_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        获取性能指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            metric_type: 指标类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: 性能指标列表
        """
        metrics = self.performance_collector.get_metrics(
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time
        )
        
        return [m.to_dict() for m in metrics]
    
    def generate_performance_report(
        self,
        model_name: str,
        deployment_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        生成性能报告
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, Any]: 性能报告
        """
        report = self.performance_collector.generate_report(
            model_name=model_name,
            deployment_type=deployment_type,
            start_time=start_time,
            end_time=end_time
        )
        
        return report.to_dict()
    
    def compare_model_performance(
        self,
        model_names: List[str],
        metric_type: str,
        deployment_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        比较模型性能
        
        Args:
            model_names: 模型名称列表
            metric_type: 指标类型
            deployment_type: 部署类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        return self.performance_collector.compare_models(
            model_names=model_names,
            metric_type=metric_type,
            deployment_type=deployment_type,
            start_time=start_time,
            end_time=end_time
        )
    
    def save_performance_metrics(self, file_path: Optional[str] = None) -> str:
        """
        保存性能指标到文件
        
        Args:
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        return self.performance_collector.save_metrics_to_csv(file_path)
    
    def save_performance_report(
        self,
        model_name: str,
        deployment_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        file_path: Optional[str] = None
    ) -> str:
        """
        保存性能报告到文件
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            start_time: 开始时间
            end_time: 结束时间
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        report = self.performance_collector.generate_report(
            model_name=model_name,
            deployment_type=deployment_type,
            start_time=start_time,
            end_time=end_time
        )
        
        return self.performance_collector.save_report_to_json(report, file_path)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能收集器摘要
        
        Returns:
            Dict[str, Any]: 摘要信息
        """
        return self.performance_collector.get_summary()
    
    def _handle_container_alert(self, alert) -> None:
        """
        处理容器告警
        
        Args:
            alert: 容器告警对象
        """
        if alert.resolved:
            self.logger.info(f"容器告警已解决: {alert.name} - {alert.message}")
        else:
            self.logger.warning(f"容器告警: {alert.name} - {alert.message}")
            
            # 根据告警类型采取行动
            if alert.alert_type == "container_down" and alert.severity == "critical":
                # 尝试重启容器
                try:
                    self.logger.info(f"尝试重启容器: {alert.name}")
                    self.restart_container_model(alert.name)
                except Exception as e:
                    self.logger.error(f"重启容器失败: {alert.name}, 错误: {str(e)}")
    
    def start_container_monitoring(self, container_names: Optional[List[str]] = None) -> None:
        """
        启动容器监控
        
        Args:
            container_names: 要监控的容器名称列表，如果为None则监控所有容器
        """
        self.container_monitor.start_monitoring(container_names)
        self.logger.info("容器监控已启动")
    
    def stop_container_monitoring(self) -> None:
        """停止容器监控"""
        self.container_monitor.stop_monitoring()
        self.logger.info("容器监控已停止")
    
    def get_container_metrics(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取容器当前指标
        
        Args:
            name: 容器名称
            
        Returns:
            Optional[Dict[str, Any]]: 容器指标，如果不存在则返回None
        """
        metrics = self.container_monitor.get_container_metrics(name)
        if metrics:
            return metrics.to_dict()
        return None
    
    def get_container_metrics_history(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        获取容器指标历史记录
        
        Args:
            name: 容器名称
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: 指标历史记录
        """
        history = self.container_monitor.get_container_metrics_history(name, start_time, end_time)
        return [m.to_dict() for m in history]
    
    def get_container_alerts(self, name: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        获取容器告警
        
        Args:
            name: 容器名称，如果为None则返回所有容器的告警
            active_only: 是否只返回活动告警
            
        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        if active_only:
            alerts = self.container_monitor.get_active_alerts(name)
        else:
            alerts = self.container_monitor.get_alert_history(name)
        
        return [a.to_dict() for a in alerts]
    
    def set_container_monitoring_thresholds(
        self,
        cpu_threshold: Optional[float] = None,
        memory_threshold: Optional[float] = None,
        disk_threshold: Optional[float] = None
    ) -> None:
        """
        设置容器监控告警阈值
        
        Args:
            cpu_threshold: CPU使用率阈值（百分比）
            memory_threshold: 内存使用率阈值（百分比）
            disk_threshold: 磁盘使用率阈值（百分比）
        """
        self.container_monitor.set_thresholds(cpu_threshold, memory_threshold, disk_threshold)
        self.logger.info(f"容器监控告警阈值已更新")
    
    def get_container_monitoring_summary(self) -> Dict[str, Any]:
        """
        获取容器监控摘要
        
        Returns:
            Dict[str, Any]: 监控摘要
        """
        return self.container_monitor.get_summary()
    
    def shutdown(self) -> None:
        """关闭模型管理器，清理资源"""
        self.logger.info("正在关闭模型管理器...")
        
        # 停止容器监控
        self.stop_container_monitoring()
        
        # 停止所有容器模型
        for name in list(self.container_clients.keys()):
            try:
                self.stop_container_model(name)
                self.logger.info(f"已停止容器模型: {name}")
            except Exception as e:
                self.logger.error(f"停止容器模型失败: {name}, 错误: {str(e)}")
        
        # 卸载所有本地模型
        for name in list(self.loaded_models.keys()):
            try:
                self.unload_model(name)
                self.logger.info(f"已卸载模型: {name}")
            except Exception as e:
                self.logger.error(f"卸载模型失败: {name}, 错误: {str(e)}")
        
        self.logger.info("模型管理器已关闭")


class GPUScheduler:
    """GPU调度器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.device_usage = {}  # 记录每个GPU设备的使用情况
        self.model_to_gpu_map = {}  # 记录模型与GPU的关联关系
        self.gpu_to_models_map = {}  # 记录每个GPU上运行的模型
        
    def track_model_gpu_usage(self, model_name: str, device: str) -> None:
        """
        跟踪模型与GPU的使用关系
        
        Args:
            model_name: 模型名称
            device: GPU设备，如"cuda:0"
        """
        if device.startswith("cuda:"):
            # 记录模型使用的GPU
            self.model_to_gpu_map[model_name] = device
            
            # 记录GPU上运行的模型
            if device not in self.gpu_to_models_map:
                self.gpu_to_models_map[device] = set()
            self.gpu_to_models_map[device].add(model_name)
            
            self.logger.info(f"模型 {model_name} 正在使用设备 {device}")
    
    def untrack_model_gpu_usage(self, model_name: str) -> None:
        """
        取消跟踪模型与GPU的使用关系
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.model_to_gpu_map:
            device = self.model_to_gpu_map[model_name]
            
            # 从GPU到模型的映射中移除
            if device in self.gpu_to_models_map and model_name in self.gpu_to_models_map[device]:
                self.gpu_to_models_map[device].remove(model_name)
                # 如果GPU上没有模型了，清理映射
                if not self.gpu_to_models_map[device]:
                    del self.gpu_to_models_map[device]
            
            # 移除模型到GPU的映射
            del self.model_to_gpu_map[model_name]
            
            self.logger.info(f"取消跟踪模型 {model_name} 的GPU使用")
    
    def get_model_gpu_info(self, model_name: str) -> Optional[str]:
        """
        获取模型使用的GPU信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Optional[str]: GPU设备名称，如果模型未使用GPU则返回None
        """
        return self.model_to_gpu_map.get(model_name)
    
    def get_gpu_models_info(self, device: str) -> List[str]:
        """
        获取在指定GPU上运行的所有模型
        
        Args:
            device: GPU设备名称，如"cuda:0"
            
        Returns:
            List[str]: 在该GPU上运行的模型名称列表
        """
        return list(self.gpu_to_models_map.get(device, set()))
    
    @monitor_performance(model_name="gpu_scheduler")
    def get_available_device(self) -> Optional[str]:
        """
        获取可用的GPU设备
        
        Returns:
            Optional[str]: 可用的GPU设备，如"cuda:0"，如果没有可用GPU则返回None
        """
        if not torch.cuda.is_available():
            self.logger.info("没有可用的CUDA设备")
            return None
            
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        self.logger.info("检测到GPU设备", gpu_count=gpu_count)
        
        # 检查每个GPU的内存使用情况
        best_device = None
        max_available_memory = 0
        
        for i in range(gpu_count):
            device = f"cuda:{i}"
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            available_memory = total_memory - allocated_memory
            
            self.logger.info(
                "GPU设备内存情况",
                gpu_id=i,
                total_memory_gb=total_memory / (1024**3),
                allocated_memory_gb=allocated_memory / (1024**3),
                available_memory_gb=available_memory / (1024**3)
            )
            
            if available_memory > max_available_memory:
                max_available_memory = available_memory
                best_device = device
                
        if best_device:
            self.logger.info("选择最佳GPU设备", device=best_device, available_memory_gb=max_available_memory / (1024**3))
        else:
            self.logger.warning("没有找到可用的GPU设备")
            
        return best_device
    
    @monitor_performance(model_name="gpu_scheduler")
    def get_device_memory_info(self, device: str) -> Dict[str, int]:
        """
        获取设备内存信息
        
        Args:
            device: 设备名称，如"cuda:0"
            
        Returns:
            Dict[str, int]: 内存信息，包含total(总内存), allocated(已分配), available(可用)
        """
        if not device.startswith("cuda:"):
            return {"error": "仅支持CUDA设备"}
            
        try:
            device_id = int(device.split(":")[1])
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            available_memory = total_memory - allocated_memory
            
            result = {
                "total": total_memory // (1024 * 1024),  # MB
                "allocated": allocated_memory // (1024 * 1024),  # MB
                "available": available_memory // (1024 * 1024)  # MB
            }
            
            self.logger.info(
                "设备内存信息",
                device=device,
                total_memory_mb=result["total"],
                allocated_memory_mb=result["allocated"],
                available_memory_mb=result["available"]
            )
            
            return result
        except Exception as e:
            self.logger.error("获取设备内存信息失败", device=device, error=str(e))
            return {"error": str(e)}