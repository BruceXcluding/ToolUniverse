"""
生物序列分析统一接口工具
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType, DeviceType


class BioSequenceAnalysisTool:
    """生物序列分析统一接口工具"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化工具
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager(config_path)
        
    def analyze(
        self, 
        sequences: Union[str, List[str]], 
        task_type: Union[str, TaskType],
        model_name: Optional[str] = None,
        sequence_type: Optional[Union[str, SequenceType]] = None,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析生物序列
        
        Args:
            sequences: 输入序列，可以是单个字符串或字符串列表
            task_type: 任务类型
            model_name: 指定模型名称
            sequence_type: 序列类型
            device: 设备类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 转换输入格式
        if isinstance(sequences, str):
            sequences = [sequences]
            
        # 转换任务类型
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                return {"error": f"不支持的任务类型: {task_type}"}
                
        # 转换序列类型
        if sequence_type and isinstance(sequence_type, str):
            try:
                sequence_type = SequenceType(sequence_type)
            except ValueError:
                return {"error": f"不支持的序列类型: {sequence_type}"}
                
        # 自动检测序列类型（如果未指定）
        if not sequence_type and sequences:
            sequence_type = self._detect_sequence_type(sequences[0])
            
        # 执行预测
        result = self.model_manager.predict(
            sequences=sequences,
            task_type=task_type,
            model_name=model_name,
            sequence_type=sequence_type,
            **kwargs
        )
        
        # 添加元数据
        if "error" not in result:
            result["metadata"] = {
                "task_type": task_type.value,
                "sequence_type": sequence_type.value if sequence_type else None,
                "model_name": model_name,
                "sequence_count": len(sequences)
            }
            
        return result
        
    def _detect_sequence_type(self, sequence: str) -> Optional[SequenceType]:
        """
        自动检测序列类型
        
        Args:
            sequence: 序列字符串
            
        Returns:
            Optional[SequenceType]: 检测到的序列类型
        """
        # 转换为大写
        seq_upper = sequence.upper()
        
        # 检查是否为DNA序列 (只包含A, T, G, C)
        if all(base in "ATGC" for base in seq_upper):
            return SequenceType.DNA
            
        # 检查是否为RNA序列 (只包含A, U, G, C)
        if all(base in "AUGC" for base in seq_upper):
            return SequenceType.RNA
            
        # 检查是否为蛋白质序列 (包含标准氨基酸)
        amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if all(base in amino_acids for base in seq_upper):
            return SequenceType.protein
            
        # 无法确定
        return None
        
    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        return self.model_manager.list_models()
        
    def list_loaded_models(self) -> List[str]:
        """列出所有已加载的模型"""
        return self.model_manager.list_loaded_models()
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_manager.get_model_info(model_name)
        
    def get_best_model(self, task_type: Union[str, TaskType], sequence_type: Optional[Union[str, SequenceType]] = None) -> Optional[str]:
        """获取最佳模型"""
        # 转换任务类型
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                self.logger.error(f"不支持的任务类型: {task_type}")
                return None
                
        # 转换序列类型
        if sequence_type and isinstance(sequence_type, str):
            try:
                sequence_type = SequenceType(sequence_type)
            except ValueError:
                self.logger.error(f"不支持的序列类型: {sequence_type}")
                return None
                
        return self.model_manager.get_best_model(task_type, sequence_type)
        
    def load_model(self, model_name: str, device: Union[str, DeviceType] = DeviceType.AUTO) -> bool:
        """加载指定模型"""
        return self.model_manager.load_model(model_name, device)
        
    def unload_model(self, model_name: str) -> bool:
        """卸载指定模型"""
        return self.model_manager.unload_model(model_name)
        
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [task.value for task in TaskType]
        
    def get_supported_sequence_types(self) -> List[str]:
        """获取支持的序列类型"""
        return [seq.value for seq in SequenceType]