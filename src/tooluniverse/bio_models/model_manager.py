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


class ModelManager:
    """生物序列模型管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
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
    
    def register_model(self, model: BaseModel):
        """
        注册模型
        
        Args:
            model: 模型实例
        """
        self.models[model.model_name] = model
        self.logger.info(f"已注册模型: {model.model_name}")
    
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
            self.logger.error(f"模型不存在: {model_name}")
            return False
            
        model = self.models[model_name]
        
        # 检查模型是否已加载
        if model.is_loaded():
            self.logger.info(f"模型已加载: {model_name}")
            return True
            
        # 检查内存是否足够
        if not self._check_memory(model):
            self.logger.warning(f"内存不足，尝试卸载其他模型: {model_name}")
            self._unload_least_used_model()
            
        # 加载模型
        try:
            success = model.load_model(device)
            if success:
                self.loaded_models[model_name] = model
                self.logger.info(f"模型加载成功: {model_name}")
            return success
        except Exception as e:
            self.logger.error(f"模型加载失败: {model_name}, 错误: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否卸载成功
        """
        if model_name not in self.loaded_models:
            self.logger.warning(f"模型未加载: {model_name}")
            return False
            
        model = self.loaded_models[model_name]
        try:
            success = model.unload_model()
            if success:
                del self.loaded_models[model_name]
                self.logger.info(f"模型卸载成功: {model_name}")
            return success
        except Exception as e:
            self.logger.error(f"模型卸载失败: {model_name}, 错误: {e}")
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
            result = model.predict(sequences, task_type, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
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
        self.logger = logging.getLogger(__name__)
        self.device_usage = {}  # 记录每个GPU设备的使用情况
        
    def get_available_device(self) -> Optional[str]:
        """
        获取可用的GPU设备
        
        Returns:
            Optional[str]: 可用的GPU设备，如"cuda:0"，如果没有可用GPU则返回None
        """
        if not torch.cuda.is_available():
            return None
            
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        
        # 检查每个GPU的内存使用情况
        best_device = None
        max_available_memory = 0
        
        for i in range(gpu_count):
            device = f"cuda:{i}"
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            available_memory = total_memory - allocated_memory
            
            if available_memory > max_available_memory:
                max_available_memory = available_memory
                best_device = device
                
        return best_device
    
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
            
            return {
                "total": total_memory // (1024 * 1024),  # MB
                "allocated": allocated_memory // (1024 * 1024),  # MB
                "available": available_memory // (1024 * 1024)  # MB
            }
        except Exception as e:
            self.logger.error(f"获取设备内存信息失败: {e}")
            return {"error": str(e)}