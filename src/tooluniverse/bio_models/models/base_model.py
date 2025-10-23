"""
生物序列模型基础类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import torch
from ..task_types import TaskType, SequenceType, ModelStatus, DeviceType


class BaseModel(ABC):
    """生物序列模型基础类"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            config: 模型配置
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.status = ModelStatus.UNLOADED
        self.device = None
        self.memory_usage = 0
        
        # 从配置中获取支持的序列类型和任务
        self.supported_sequences = config.get("supported_sequences", [])
        self.supported_tasks = config.get("supported_tasks", [])
        self.memory_requirement = config.get("memory_requirement", 4096)  # MB
        
    @abstractmethod
    def load_model(self, device: Union[str, DeviceType] = DeviceType.AUTO) -> bool:
        """
        加载模型
        
        Args:
            device: 设备类型
            
        Returns:
            bool: 是否加载成功
        """
        pass
        
    @abstractmethod
    def unload_model(self) -> bool:
        """
        卸载模型，释放资源
        
        Returns:
            bool: 是否卸载成功
        """
        pass
        
    @abstractmethod
    def predict(self, sequences: List[str], task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            sequences: 输入序列列表
            task_type: 任务类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        pass
        
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.status == ModelStatus.LOADED
        
    def get_memory_usage(self) -> int:
        """获取模型内存使用量(MB)"""
        return self.memory_usage
        
    def supports_sequence_type(self, sequence_type: SequenceType) -> bool:
        """检查模型是否支持指定的序列类型"""
        return sequence_type.value in self.supported_sequences
        
    def supports_task(self, task_type: TaskType) -> bool:
        """检查模型是否支持指定的任务类型"""
        return task_type.value in self.supported_tasks
        
    def get_device(self) -> Optional[str]:
        """获取模型所在设备"""
        return self.device
        
    def _set_device(self, device: Union[str, DeviceType]) -> str:
        """
        设置设备
        
        Args:
            device: 设备类型
            
        Returns:
            str: 实际使用的设备
        """
        if isinstance(device, DeviceType):
            device = device.value
            
        if device == DeviceType.AUTO.value:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        self.device = device
        return device
        
    def _validate_sequences(self, sequences: List[str], sequence_type: Optional[SequenceType] = None) -> bool:
        """
        验证输入序列
        
        Args:
            sequences: 输入序列列表
            sequence_type: 序列类型
            
        Returns:
            bool: 是否有效
        """
        if not sequences:
            return False
            
        # 如果指定了序列类型，检查模型是否支持
        if sequence_type and not self.supports_sequence_type(sequence_type):
            return False
            
        return True