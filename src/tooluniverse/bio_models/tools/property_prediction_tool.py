"""
生物序列属性预测工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class PropertyPredictionTool:
    """生物序列属性预测工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化属性预测工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.PROPERTY_PREDICTION
    
    def predict_properties(
        self,
        sequences: Union[str, List[str]],
        properties: List[str],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的多种属性
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            properties: 要预测的属性列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "properties": properties,
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行属性预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_stability(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的稳定性
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含稳定性预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "property": "stability",
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行稳定性预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_expression(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        expression_type: str = "general",
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的表达水平
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            expression_type: 表达类型 (general, tissue_specific, etc.)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含表达水平预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "property": "expression",
            "expression_type": expression_type,
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行表达水平预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_binding_affinity(
        self,
        sequences: Union[str, List[str]],
        target_molecules: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列与靶分子的结合亲和力
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            target_molecules: 靶分子，可以是单个分子或分子列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含结合亲和力预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        if isinstance(target_molecules, str):
            target_molecules = [target_molecules]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "target_molecules": target_molecules,
            "property": "binding_affinity",
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行结合亲和力预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_solubility(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质的溶解度
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (通常为PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含溶解度预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "property": "solubility",
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行溶解度预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_subcellular_location(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质的亚细胞定位
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (通常为PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含亚细胞定位预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "property": "subcellular_location",
            "sequence_type": sequence_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行亚细胞定位预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def get_supported_properties(
        self,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None
    ) -> List[str]:
        """
        获取模型支持的属性列表
        
        Args:
            sequence_type: 序列类型
            model_name: 模型名称
            
        Returns:
            支持的属性列表
        """
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name)
        
        # 获取模型支持的属性
        model = self.model_manager.get_model(model_name)
        return model.get_supported_properties(sequence_type)
    
    def get_supported_models(
        self,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA
    ) -> List[str]:
        """
        获取支持属性预测任务的模型列表
        
        Args:
            sequence_type: 序列类型
            
        Returns:
            支持的模型名称列表
        """
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
            
        return self.model_manager.get_models_for_task(
            task_type=self.task_type,
            sequence_type=sequence_type
        )