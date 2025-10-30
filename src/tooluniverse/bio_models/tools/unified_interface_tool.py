"""
生物序列分析统一接口工具
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union
from ...base_tool import ToolError, ToolServerError, ToolValidationError, ToolConfigError, ToolDependencyError, ToolAuthError, ToolRateLimitError, ToolUnavailableError
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
            
        # 转换任务类型，支持大小写不敏感
        if isinstance(task_type, str):
            try:
                # 转为小写以支持大小写不敏感
                task_type = TaskType(task_type.lower())
            except ValueError:
                return {"error": f"不支持的任务类型: {task_type}"}
                
        # 转换序列类型，支持大小写不敏感
        if sequence_type and isinstance(sequence_type, str):
            try:
                # 转为小写以支持大小写不敏感
                sequence_type = SequenceType(sequence_type.lower())
            except ValueError:
                # 尝试一些常见的缩写
                seq_lower = sequence_type.lower()
                if seq_lower in ['prot', 'protein']:
                    sequence_type = SequenceType.PROTEIN
                elif seq_lower in ['dna', 'deoxyribonucleic']:
                    sequence_type = SequenceType.DNA
                elif seq_lower in ['rna', 'ribonucleic']:
                    sequence_type = SequenceType.RNA
                else:
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
            return SequenceType.PROTEIN
            
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
        
    def handle_error(self, exception: Exception) -> ToolError:
        """
        处理工具执行过程中的错误
        
        Args:
            exception: 异常对象
            
        Returns:
            ToolError: 结构化的工具错误
        """
        error_str = str(exception).lower()
        
        if any(keyword in error_str for keyword in ["model", "模型", "加载", "load"]):
            return ToolConfigError(f"模型配置错误: {exception}")
        elif any(keyword in error_str for keyword in ["sequence", "序列", "invalid", "无效"]):
            return ToolValidationError(f"序列验证错误: {exception}")
        elif any(keyword in error_str for keyword in ["task", "任务", "type", "类型"]):
            return ToolValidationError(f"任务类型错误: {exception}")
        elif any(keyword in error_str for keyword in ["device", "设备", "cuda", "cpu"]):
            return ToolConfigError(f"设备配置错误: {exception}")
        elif any(keyword in error_str for keyword in ["memory", "内存", "gpu", "显存"]):
            return ToolUnavailableError(f"资源不足: {exception}")
        else:
            return ToolServerError(f"生物序列分析工具错误: {exception}")
            
    def run(self, arguments=None, stream_callback=None, use_cache=False, validate=True, sequences=None, task_type=None, model_name=None, sequence_type=None, device=None, **kwargs):
        """
        执行生物序列分析
        
        Args:
            arguments: 包含分析参数的字典，符合ToolUniverse标准格式（可选）
            stream_callback: 流式回调函数（可选）
            use_cache: 是否使用缓存（可选）
            validate: 是否验证参数（可选）
            sequences: 要分析的序列（可选，直接参数）
            task_type: 任务类型（可选，直接参数）
            model_name: 模型名称（可选，直接参数）
            sequence_type: 序列类型（可选，直接参数）
            device: 设备类型（可选，直接参数）
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 处理两种参数传递方式
        if arguments is None:
            arguments = {}
        
        # 如果直接参数提供了值，使用它们；否则从arguments字典中获取
        sequences = sequences if sequences is not None else arguments.get("sequences")
        task_type = task_type if task_type is not None else arguments.get("task_type")
        model_name = model_name if model_name is not None else arguments.get("model_name")
        sequence_type = sequence_type if sequence_type is not None else arguments.get("sequence_type")
        device = device if device is not None else arguments.get("device", DeviceType.AUTO)
        
        # 调用analyze方法
        return self.analyze(
            sequences=sequences,
            task_type=task_type,
            model_name=model_name,
            sequence_type=sequence_type,
            device=device
        )