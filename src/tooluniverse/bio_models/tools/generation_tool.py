"""
生物序列生成工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class GenerationTool:
    """生物序列生成工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化生成工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.GENERATION
    
    def generate_sequences(
        self,
        prompt: Optional[str] = None,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        num_sequences: int = 5,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成生物序列
        
        Args:
            prompt: 提示序列，如果为None则无条件生成
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            num_sequences: 生成序列数量
            max_length: 生成序列最大长度
            temperature: 采样温度，控制随机性
            top_p: 核采样参数
            top_k: Top-K采样参数
            device: 计算设备
            **kwargs: 其他生成参数
            
        Returns:
            包含生成序列的字典
        """
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
            "prompt": prompt,
            "sequence_type": sequence_type.value,
            "num_sequences": num_sequences,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            **kwargs
        }
        
        # 执行生成
        results = self.model_manager.generate_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def mutate_sequence(
        self,
        sequence: str,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        num_mutations: int = 5,
        mutation_rate: float = 0.1,
        preserve_function: bool = True,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        对序列进行突变生成
        
        Args:
            sequence: 原始序列
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            num_mutations: 生成突变序列数量
            mutation_rate: 突变率
            preserve_function: 是否保持功能
            device: 计算设备
            **kwargs: 其他参数
            
        Returns:
            包含突变序列的字典
        """
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=TaskType.MUTATION_ANALYSIS,
                sequence_type=sequence_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence": sequence,
            "sequence_type": sequence_type.value,
            "num_mutations": num_mutations,
            "mutation_rate": mutation_rate,
            "preserve_function": preserve_function,
            **kwargs
        }
        
        # 执行突变生成
        results = self.model_manager.mutate_with_model(
            model_name=model_name,
            task_type=TaskType.MUTATION_ANALYSIS,
            **params
        )
        
        return results
    
    def optimize_sequence(
        self,
        sequence: str,
        target_property: str,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        optimization_steps: int = 10,
        optimization_method: str = "gradient_ascent",
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        优化序列以增强特定属性
        
        Args:
            sequence: 原始序列
            target_property: 目标属性 (如: stability, expression, binding_affinity)
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            optimization_steps: 优化步数
            optimization_method: 优化方法
            device: 计算设备
            **kwargs: 其他参数
            
        Returns:
            包含优化序列的字典
        """
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
            "sequence": sequence,
            "target_property": target_property,
            "sequence_type": sequence_type.value,
            "optimization_steps": optimization_steps,
            "optimization_method": optimization_method,
            **kwargs
        }
        
        # 执行序列优化
        results = self.model_manager.optimize_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def design_sequence(
        self,
        constraints: Dict[str, Any],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        num_sequences: int = 5,
        design_method: str = "constrained_generation",
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        根据约束条件设计序列
        
        Args:
            constraints: 设计约束条件
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            num_sequences: 设计序列数量
            design_method: 设计方法
            device: 计算设备
            **kwargs: 其他参数
            
        Returns:
            包含设计序列的字典
        """
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
            "constraints": constraints,
            "sequence_type": sequence_type.value,
            "num_sequences": num_sequences,
            "design_method": design_method,
            **kwargs
        }
        
        # 执行序列设计
        results = self.model_manager.design_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def get_supported_models(
        self,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA
    ) -> List[str]:
        """
        获取支持生成任务的模型列表
        
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
    
    def evaluate_generated_sequences(
        self,
        sequences: List[str],
        reference_sequence: Optional[str] = None,
        evaluation_metrics: List[str] = None,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        评估生成序列的质量
        
        Args:
            sequences: 生成的序列列表
            reference_sequence: 参考序列（可选）
            evaluation_metrics: 评估指标列表
            sequence_type: 序列类型
            model_name: 模型名称
            device: 计算设备
            
        Returns:
            包含评估结果的字典
        """
        if evaluation_metrics is None:
            evaluation_metrics = ["diversity", "novelty", "validity"]
        
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
            "reference_sequence": reference_sequence,
            "evaluation_metrics": evaluation_metrics,
            "sequence_type": sequence_type.value
        }
        
        # 执行评估
        results = self.model_manager.evaluate_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results