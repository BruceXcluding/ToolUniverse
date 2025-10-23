"""
生物序列分类工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class ClassificationTool:
    """生物序列分类工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化分类工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.CLASSIFICATION
    
    def classify_sequences(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        batch_size: int = 8,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        对生物序列进行分类
        
        Args:
            sequences: 待分类的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            labels: 分类标签列表，如果为None则使用模型默认标签
            threshold: 分类阈值，默认0.5
            batch_size: 批处理大小
            device: 计算设备
            
        Returns:
            包含分类结果的字典，包括预测标签、概率等
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
            "batch_size": batch_size,
            "labels": labels,
            "threshold": threshold
        }
        
        # 执行分类
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_probabilities(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        batch_size: int = 8,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        获取分类概率
        
        Args:
            sequences: 待分类的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            labels: 分类标签列表，如果为None则使用模型默认标签
            batch_size: 批处理大小
            device: 计算设备
            
        Returns:
            包含分类概率的字典
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
            "batch_size": batch_size,
            "labels": labels,
            "return_probabilities": True
        }
        
        # 执行分类
        results = self.model_manager.predict_with_model(
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
        获取支持分类任务的模型列表
        
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
    
    def get_model_labels(
        self,
        model_name: str,
        sequence_type: Union[str, SequenceType] = SequenceType.DNA
    ) -> List[str]:
        """
        获取模型支持的分类标签
        
        Args:
            model_name: 模型名称
            sequence_type: 序列类型
            
        Returns:
            分类标签列表
        """
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
            
        # 确保模型已加载
        self.model_manager.load_model(model_name)
        
        # 获取模型标签
        model = self.model_manager.get_model(model_name)
        return model.get_labels(sequence_type)
    
    def evaluate(
        self,
        sequences: List[str],
        true_labels: List[str],
        sequence_type: Union[str, SequenceType] = SequenceType.DNA,
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        评估分类模型性能
        
        Args:
            sequences: 测试序列列表
            true_labels: 真实标签列表
            sequence_type: 序列类型
            model_name: 模型名称
            batch_size: 批处理大小
            device: 计算设备
            
        Returns:
            包含评估指标的字典
        """
        if len(sequences) != len(true_labels):
            raise ValueError("序列数量与标签数量不匹配")
        
        # 获取预测结果
        results = self.classify_sequences(
            sequences=sequences,
            sequence_type=sequence_type,
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
        
        # 计算评估指标
        predicted_labels = results.get("predictions", [])
        probabilities = results.get("probabilities", [])
        
        # 计算准确率
        correct = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        
        # 计算其他指标（简化版）
        unique_labels = list(set(true_labels + predicted_labels))
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        
        for true, pred in zip(true_labels, predicted_labels):
            confusion_matrix[label_to_index[true], label_to_index[pred]] += 1
        
        # 计算精确率、召回率和F1分数
        precision = {}
        recall = {}
        f1 = {}
        
        for i, label in enumerate(unique_labels):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
        
        # 宏平均
        macro_precision = sum(precision.values()) / len(precision)
        macro_recall = sum(recall.values()) / len(recall)
        macro_f1 = sum(f1.values()) / len(f1)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "confusion_matrix": confusion_matrix.tolist(),
            "labels": unique_labels
        }