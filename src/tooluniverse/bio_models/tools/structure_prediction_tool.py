"""
生物序列结构预测工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class StructurePredictionTool:
    """生物序列结构预测工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化结构预测工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.STRUCTURE_PREDICTION
    
    def predict_structure(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        structure_type: str = "3d",
        confidence_threshold: float = 0.7,
        batch_size: int = 1,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的结构
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            structure_type: 结构类型 (2d, 3d, secondary)
            confidence_threshold: 置信度阈值
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含结构预测结果的字典
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
            "sequence_type": sequence_type.value,
            "structure_type": structure_type,
            "confidence_threshold": confidence_threshold,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_secondary_structure(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的二级结构
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含二级结构预测结果的字典
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
            "sequence_type": sequence_type.value,
            "structure_type": "secondary",
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行二级结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_3d_structure(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        use_templates: bool = True,
        confidence_threshold: float = 0.7,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的3D结构
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            use_templates: 是否使用模板
            confidence_threshold: 置信度阈值
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含3D结构预测结果的字典
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
            "sequence_type": sequence_type.value,
            "structure_type": "3d",
            "use_templates": use_templates,
            "confidence_threshold": confidence_threshold,
            "batch_size": 1,  # 3D结构预测通常一次只能处理一个序列
            **kwargs
        }
        
        # 执行3D结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_rna_structure(
        self,
        sequences: Union[str, List[str]],
        structure_type: str = "secondary",
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测RNA序列的结构
        
        Args:
            sequences: 待预测的RNA序列，可以是单个序列或序列列表
            structure_type: 结构类型 (secondary, 3d)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含RNA结构预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.RNA
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "sequence_type": SequenceType.RNA.value,
            "structure_type": structure_type,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行RNA结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_protein_structure(
        self,
        sequences: Union[str, List[str]],
        structure_type: str = "3d",
        model_name: Optional[str] = None,
        use_msa: bool = True,
        use_templates: bool = True,
        confidence_threshold: float = 0.7,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质序列的结构
        
        Args:
            sequences: 待预测的蛋白质序列，可以是单个序列或序列列表
            structure_type: 结构类型 (secondary, 3d)
            model_name: 指定模型名称，如果为None则自动选择
            use_msa: 是否使用多序列比对
            use_templates: 是否使用模板
            confidence_threshold: 置信度阈值
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含蛋白质结构预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.PROTEIN
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "sequence_type": SequenceType.PROTEIN.value,
            "structure_type": structure_type,
            "use_msa": use_msa,
            "use_templates": use_templates,
            "confidence_threshold": confidence_threshold,
            "batch_size": 1 if structure_type == "3d" else 8,
            **kwargs
        }
        
        # 执行蛋白质结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_dna_structure(
        self,
        sequences: Union[str, List[str]],
        structure_type: str = "secondary",
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测DNA序列的结构
        
        Args:
            sequences: 待预测的DNA序列，可以是单个序列或序列列表
            structure_type: 结构类型 (secondary)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含DNA结构预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.DNA
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "sequence_type": SequenceType.DNA.value,
            "structure_type": structure_type,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行DNA结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_complex_structure(
        self,
        sequences: Dict[str, str],
        complex_type: str = "protein_protein",
        model_name: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物分子复合物的结构
        
        Args:
            sequences: 复合物中各分子的序列字典，键为分子名称，值为序列
            complex_type: 复合物类型 (protein_protein, protein_rna, protein_dna, rna_rna)
            model_name: 指定模型名称，如果为None则自动选择
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含复合物结构预测结果的字典
        """
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.PROTEIN  # 默认使用蛋白质模型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequences": sequences,
            "complex_type": complex_type,
            "batch_size": 1,  # 复合物结构预测通常一次只能处理一个复合物
            **kwargs
        }
        
        # 执行复合物结构预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def get_supported_models(
        self,
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN
    ) -> List[str]:
        """
        获取支持结构预测任务的模型列表
        
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
    
    def evaluate_structure_prediction(
        self,
        predicted_structures: List[Dict[str, Any]],
        reference_structures: List[Dict[str, Any]],
        evaluation_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估结构预测的质量
        
        Args:
            predicted_structures: 预测的结构列表
            reference_structures: 参考结构列表
            evaluation_metrics: 评估指标列表
            
        Returns:
            包含评估结果的字典
        """
        if evaluation_metrics is None:
            evaluation_metrics = ["rmsd", "tm_score", "gdt_ts"]
        
        # 这里应该实现结构评估逻辑
        # 简化版实现
        results = {
            "evaluation_metrics": evaluation_metrics,
            "num_structures": len(predicted_structures),
            "scores": {}
        }
        
        for metric in evaluation_metrics:
            # 简化版：生成随机分数
            results["scores"][metric] = np.random.uniform(0.5, 1.0, len(predicted_structures)).tolist()
        
        return results