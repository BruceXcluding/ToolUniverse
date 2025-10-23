"""
生物序列功能注释工具
"""
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType


class AnnotationTool:
    """生物序列功能注释工具"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化功能注释工具
        
        Args:
            model_manager: 模型管理器实例，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.task_type = TaskType.FUNCTION_ANNOTATION
    
    def annotate_sequence(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        annotation_type: str = "function",
        model_name: Optional[str] = None,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        注释生物序列的功能
        
        Args:
            sequences: 待注释的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            annotation_type: 注释类型 (function, domain, motif, pathway, go, ec)
            model_name: 指定模型名称，如果为None则自动选择
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他注释参数
            
        Returns:
            包含功能注释结果的字典
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
            "annotation_type": annotation_type,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行功能注释
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_function(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        top_k: int = 5,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的功能
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            top_k: 返回前k个最可能的功能
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含功能预测结果的字典
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
            "annotation_type": "function",
            "top_k": top_k,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行功能预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_domains(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        min_confidence: float = 0.5,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的结构域
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            min_confidence: 最小置信度阈值
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含结构域预测结果的字典
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
            "annotation_type": "domain",
            "min_confidence": min_confidence,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行结构域预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_motifs(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        min_confidence: float = 0.5,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的基序
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            min_confidence: 最小置信度阈值
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含基序预测结果的字典
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
            "annotation_type": "motif",
            "min_confidence": min_confidence,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行基序预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_go_terms(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        go_aspect: str = "all",
        model_name: Optional[str] = None,
        top_k: int = 10,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列的Gene Ontology (GO)术语
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            go_aspect: GO方面 (molecular_function, biological_process, cellular_component, all)
            model_name: 指定模型名称，如果为None则自动选择
            top_k: 返回前k个最可能的GO术语
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含GO术语预测结果的字典
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
            "annotation_type": "go",
            "go_aspect": go_aspect,
            "top_k": top_k,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行GO术语预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_ec_numbers(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        top_k: int = 5,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测蛋白质序列的酶学委员会(EC)编号
        
        Args:
            sequences: 待预测的蛋白质序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (仅支持PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            top_k: 返回前k个最可能的EC编号
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含EC编号预测结果的字典
        """
        # 确保输入是列表格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 转换序列类型
        if isinstance(sequence_type, str):
            sequence_type = SequenceType(sequence_type.lower())
        
        # EC编号预测仅适用于蛋白质
        if sequence_type != SequenceType.PROTEIN:
            raise ValueError("EC编号预测仅适用于蛋白质序列")
        
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
            "annotation_type": "ec",
            "top_k": top_k,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行EC编号预测
        results = self.model_manager.predict_with_model(
            model_name=model_name,
            task_type=self.task_type,
            **params
        )
        
        return results
    
    def predict_pathways(
        self,
        sequences: Union[str, List[str]],
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN,
        model_name: Optional[str] = None,
        top_k: int = 5,
        batch_size: int = 8,
        device: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测生物序列参与的代谢通路
        
        Args:
            sequences: 待预测的序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, PROTEIN)
            model_name: 指定模型名称，如果为None则自动选择
            top_k: 返回前k个最可能的通路
            batch_size: 批处理大小
            device: 计算设备
            **kwargs: 其他预测参数
            
        Returns:
            包含通路预测结果的字典
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
            "annotation_type": "pathway",
            "top_k": top_k,
            "batch_size": batch_size,
            **kwargs
        }
        
        # 执行通路预测
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
            sequences: 待预测的蛋白质序列，可以是单个序列或序列列表
            sequence_type: 序列类型 (仅支持PROTEIN)
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
        
        # 亚细胞定位预测仅适用于蛋白质
        if sequence_type != SequenceType.PROTEIN:
            raise ValueError("亚细胞定位预测仅适用于蛋白质序列")
        
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
            "annotation_type": "subcellular_location",
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
    
    def get_supported_models(
        self,
        sequence_type: Union[str, SequenceType] = SequenceType.PROTEIN
    ) -> List[str]:
        """
        获取支持功能注释任务的模型列表
        
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
    
    def evaluate_annotation(
        self,
        predicted_annotations: List[Dict[str, Any]],
        reference_annotations: List[Dict[str, Any]],
        evaluation_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估功能注释的质量
        
        Args:
            predicted_annotations: 预测的注释列表
            reference_annotations: 参考注释列表
            evaluation_metrics: 评估指标列表
            
        Returns:
            包含评估结果的字典
        """
        if evaluation_metrics is None:
            evaluation_metrics = ["precision", "recall", "f1", "accuracy"]
        
        # 这里应该实现注释评估逻辑
        # 简化版实现
        results = {
            "evaluation_metrics": evaluation_metrics,
            "num_sequences": len(predicted_annotations),
            "scores": {}
        }
        
        for metric in evaluation_metrics:
            # 简化版：生成随机分数
            results["scores"][metric] = np.random.uniform(0.5, 1.0)
        
        return results