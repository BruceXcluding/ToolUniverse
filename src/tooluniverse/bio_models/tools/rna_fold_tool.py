"""
RNA折叠工具 - 用于RNA二级结构预测
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from ..task_types import TaskType, SequenceType, DeviceType

# 尝试导入RNA库
try:
    import RNA
    RNA_FOLD_AVAILABLE = True
    RNAFOLD_AVAILABLE = True  # 兼容两种变量名
except ImportError:
    RNA_FOLD_AVAILABLE = False
    RNAFOLD_AVAILABLE = False  # 兼容两种变量名
    logging.warning("无法导入RNA库，请确保Vienna RNA Package已安装")


class RNAFoldTool:
    """RNA折叠工具 - 用于RNA二级结构预测"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化RNA折叠工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        self.tool_name = "rna_fold"
        
        # 设置RNA库参数（如果可用）
        if RNA_FOLD_AVAILABLE:
            RNA.cvar.temperature = 37.0  # 设置默认温度为37°C
    
    def analyze(
        self,
        sequences: Union[str, List[str]],
        task_type: str,
        model_name: Optional[str] = None,
        device: DeviceType = DeviceType.CPU,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析RNA序列并预测二级结构
        
        Args:
            sequences: RNA序列或序列列表
            task_type: 任务类型，应该是'rna_fold'
            model_name: 模型名称（不使用）
            device: 设备类型（不使用）
            **kwargs: 额外参数
                - temperature: 温度（摄氏度）
                - batch_size: 批处理大小
        
        Returns:
            Dict[str, Any]: 包含预测结果的字典
        """
        # 参数验证与标准化
        sequences = [sequences] if isinstance(sequences, str) else sequences
        
        # 检查RNA库可用性
        if not RNA_FOLD_AVAILABLE:
            return {
                "status": "error",
                "error": "RNA折叠工具不可用，请确保Vienna RNA Package已正确安装",
                "results": []
            }
        
        # 解析参数
        temperature = kwargs.get("temperature", 37.0)
        batch_size = kwargs.get("batch_size", 10)
        
        self.logger.info(f"开始RNA折叠分析，温度: {temperature}°C")
        
        try:
            # 设置温度
            RNA.cvar.temperature = temperature
            
            # 批量处理序列
            results = []
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                for seq in batch:
                    try:
                        # 预处理序列
                        seq = seq.strip().upper()
                        
                        # 预测二级结构和MFE
                        (structure, mfe) = RNA.fold(seq)
                        
                        results.append({
                            "sequence": seq,
                            "structure": structure,
                            "mfe": float(mfe),
                            "temperature": temperature,
                            "status": "success"
                        })
                    except Exception as inner_e:
                        self.logger.error(f"处理序列时出错: {str(inner_e)}")
                        results.append({
                            "sequence": seq,
                            "error": f"处理序列时出错: {str(inner_e)}",
                            "status": "error"
                        })
            
            return {
                "status": "success",
                "num_sequences": len(sequences),
                "results": results
            }
        except Exception as e:
            self.logger.error(f"RNA折叠分析过程中出错: {str(e)}")
            return {
                "status": "error",
                "error": f"RNA折叠分析过程中出错: {str(e)}",
                "results": []
            }
    
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [TaskType.STRUCTURE_PREDICTION.value]
    
    def get_supported_sequence_types(self) -> List[str]:
        """获取支持的序列类型"""
        return [SequenceType.RNA.value]
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return RNAFOLD_AVAILABLE
        
    def run(self, arguments=None, stream_callback=None, use_cache=False, validate=True):
        """
        提供run方法接口，与ToolUniverse系统兼容
        
        Args:
            arguments: 包含分析参数的字典
            stream_callback: 流式回调函数（未使用）
            use_cache: 是否使用缓存（未使用）
            validate: 是否验证参数（未使用）
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 从arguments中提取参数
        if arguments is None:
            arguments = {}
            
        sequences = arguments.get("sequences")
        task_type = arguments.get("task_type", TaskType.RNA_FOLDING)
        model_name = arguments.get("model_name")
        sequence_type = arguments.get("sequence_type")
        device = arguments.get("device", "cpu")
        
        # 提取kwargs
        kwargs = arguments.copy()
        for key in ["sequences", "task_type", "model_name", "sequence_type", "device"]:
            if key in kwargs:
                del kwargs[key]
                
        # 调用analyze方法执行实际分析
        return self.analyze(
            sequences=sequences,
            task_type=task_type,
            model_name=model_name,
            sequence_type=sequence_type,
            device=device,
            **kwargs
        )

# 如果作为主程序运行，提供简单的测试
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化工具
    tool = RNAFoldTool()
    
    # 测试序列
    test_sequence = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGG"
    
    # 执行预测
    result = tool.analyze(test_sequence, "structure_prediction")
    
    # 打印结果
    import json
    print(json.dumps(result, indent=2))