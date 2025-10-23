"""
生物序列模型管理器
"""
import os
import json
import logging
import re
from typing import Dict, List, Optional, Union, Any
import torch
import psutil
from .models.base_model import BaseModel
from .task_types import TaskType, SequenceType, ModelStatus, DeviceType
from .monitoring import get_logger, monitor_performance, monitor_context, metrics_collector


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
        self.config = {}
        self.gpu_scheduler = GPUScheduler()
        
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
            
        # 加载模型
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
    
    def list_models(self) -> List[str]:
        """列出所有注册的模型"""
        return list(self.models.keys())
    
    def list_loaded_models(self) -> List[str]:
        """列出所有已加载的模型"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
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
            "is_loaded": model.is_loaded()
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


class GPUScheduler:
    """GPU调度器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.device_usage = {}  # 记录每个GPU设备的使用情况
        
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