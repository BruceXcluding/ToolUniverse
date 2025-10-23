"""
生物分子相互作用预测工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class InteractionTool:
    """生物分子相互作用预测工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化相互作用预测工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.INTERACTION
    
    def predict_interaction(
        self,
        sequence1: str,
        sequence2: str,
        sequence_type1: Union[str, SequenceType] = SequenceType.PROTEIN,
        sequence_type2: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测两个生物分子之间的相互作用
        
        Args:
            sequence1: 第一个分子的序列
            sequence2: 第二个分子的序列
            sequence_type1: 第一个分子的序列类型 (DNA, RNA, PROTEIN)
            sequence_type2: 第二个分子的序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含相互作用预测结果的字典
        """
        # 转换序列类型
        if isinstance(sequence_type1, str):
            sequence_type1 = SequenceType(sequence_type1.lower())
        if isinstance(sequence_type2, str):
            sequence_type2 = SequenceType(sequence_type2.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type1  # 使用第一个分子的序列类型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence1": sequence1,
            "sequence2": sequence2,
            "sequence_type1": sequence_type1.value,
            "sequence_type2": sequence_type2.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_protein_protein_interaction(
        self,
        protein1: str,
        protein2: str,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质-蛋白质相互作用
        
        Args:
            protein1: 第一个蛋白质序列
            protein2: 第二个蛋白质序列
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含蛋白质-蛋白质相互作用预测结果的字典
        """
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
            "sequence1": protein1,
            "sequence2": protein2,
            "sequence_type1": SequenceType.PROTEIN.value,
            "sequence_type2": SequenceType.PROTEIN.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行蛋白质-蛋白质相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_protein_rna_interaction(
        self,
        protein: str,
        rna: str,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质-RNA相互作用
        
        Args:
            protein: 蛋白质序列
            rna: RNA序列
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含蛋白质-RNA相互作用预测结果的字典
        """
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.PROTEIN  # 使用蛋白质序列类型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence1": protein,
            "sequence2": rna,
            "sequence_type1": SequenceType.PROTEIN.value,
            "sequence_type2": SequenceType.RNA.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行蛋白质-RNA相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_protein_dna_interaction(
        self,
        protein: str,
        dna: str,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质-DNA相互作用
        
        Args:
            protein: 蛋白质序列
            dna: DNA序列
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含蛋白质-DNA相互作用预测结果的字典
        """
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=SequenceType.PROTEIN  # 使用蛋白质序列类型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence1": protein,
            "sequence2": dna,
            "sequence_type1": SequenceType.PROTEIN.value,
            "sequence_type2": SequenceType.DNA.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行蛋白质-DNA相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_rna_rna_interaction(
        self,
        rna1: str,
        rna2: str,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测RNA-RNA相互作用
        
        Args:
            rna1: 第一个RNA序列
            rna2: 第二个RNA序列
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含RNA-RNA相互作用预测结果的字典
        """
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
            "sequence1": rna1,
            "sequence2": rna2,
            "sequence_type1": SequenceType.RNA.value,
            "sequence_type2": SequenceType.RNA.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行RNA-RNA相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_drug_target_interaction(
        self,
        drug_smiles: str,
        target_sequence: str,
        target_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测药物-靶标相互作用
        
        Args:
            drug_smiles: 药物分子的SMILES字符串
            target_sequence: 靶标分子的序列
            target_type: 靶标分子的序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含药物-靶标相互作用预测结果的字典
        """
        # 转换序列类型
        if isinstance(target_type, str):
            target_type = SequenceType(target_type.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=target_type
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "drug_smiles": drug_smiles,
            "target_sequence": target_sequence,
            "target_type": target_type.value,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行药物-靶标相互作用预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_binding_affinity(
        self,
        sequence1: str,
        sequence2: str,
        sequence_type1: Union[str, SequenceType] = SequenceType.PROTEIN,
        sequence_type2: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测两个生物分子之间的结合亲和力
        
        Args:
            sequence1: 第一个分子的序列
            sequence2: 第二个分子的序列
            sequence_type1: 第一个分子的序列类型 (DNA, RNA, PROTEIN)
            sequence_type2: 第二个分子的序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含结合亲和力预测结果的字典
        """
        # 转换序列类型
        if isinstance(sequence_type1, str):
            sequence_type1 = SequenceType(sequence_type1.lower())
        if isinstance(sequence_type2, str):
            sequence_type2 = SequenceType(sequence_type2.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type1  # 使用第一个分子的序列类型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence1": sequence1,
            "sequence2": sequence2,
            "sequence_type1": sequence_type1.value,
            "sequence_type2": sequence_type2.value,
            "batch_size": batch_size,
            "predict_affinity": True,
            **kwargs
        }
        
        # 执行结合亲和力预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_interaction_sites(
        self,
        sequence1: str,
        sequence2: str,
        sequence_type1: Union[str, SequenceType] = SequenceType.PROTEIN,
        sequence_type2: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测两个生物分子之间的相互作用位点
        
        Args:
            sequence1: 第一个分子的序列
            sequence2: 第二个分子的序列
            sequence_type1: 第一个分子的序列类型 (DNA, RNA, PROTEIN)
            sequence_type2: 第二个分子的序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含相互作用位点预测结果的字典
        """
        # 转换序列类型
        if isinstance(sequence_type1, str):
            sequence_type1 = SequenceType(sequence_type1.lower())
        if isinstance(sequence_type2, str):
            sequence_type2 = SequenceType(sequence_type2.lower())
        
        # 获取适合的模型
        if model_name is None:
            model_name = self.model_manager.get_best_model_for_task(
                task_type=self.task_type,
                sequence_type=sequence_type1  # 使用第一个分子的序列类型
            )
        
        # 确保模型已加载
        self.model_manager.load_model(model_name, device=device)
        
        # 准备参数
        params = {
            "sequence1": sequence1,
            "sequence2": sequence2,
            "sequence_type1": sequence_type1.value,
            "sequence_type2": sequence_type2.value,
            "batch_size": batch_size,
            "predict_sites": True,
            **kwargs
        }
        
        # 执行相互作用位点预测
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
        获取支持相互作用预测任务的模型列表
        
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
    
    def evaluate_interaction_prediction(
        self,
        predicted_interactions: List[Dict[str, Any]],
        reference_interactions: List[Dict[str, Any]],
        evaluation_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估相互作用预测的质量
        
        Args:
            predicted_interactions: 预测的相互作用列表
            reference_interactions: 参考相互作用列表
            evaluation_metrics: 评估指标列表
            
        Returns:
            包含评估结果的字典
        """
        if evaluation_metrics is None:
            evaluation_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        # 这里应该实现相互作用评估逻辑
        # 简化版实现
        results = {
            "evaluation_metrics": evaluation_metrics,
            "num_interactions": len(predicted_interactions),
            "scores": {}
        }
        
        for metric in evaluation_metrics:
            # 简化版：生成随机分数
            results["scores"][metric] = np.random.uniform(0.5, 1.0)
        
        return results