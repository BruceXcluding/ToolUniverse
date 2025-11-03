""" 序列比较工具 - 用于序列比对与相似度计算 """
import os
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from ...base_tool import BaseTool
from ...exceptions import (
    ToolError, 
    ToolConfigError, 
    ToolValidationError, 
    ToolUnavailableError,
    ToolServerError
)
from ..task_types import TaskType, SequenceType, DeviceType

# 检查Biopython是否可用
try:
    import Bio
    SEQ_COMPARE_AVAILABLE = True
except ImportError:
    SEQ_COMPARE_AVAILABLE = False
    logging.warning("无法导入Biopython库，请确保Biopython已安装")


class SeqCompareTool(BaseTool):
    """序列比较工具 - 用于序列相似度比较"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化序列比较工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 初始化BaseTool
        super().__init__({
            "name": "seq_compare",
            "description": "序列相似度比较工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的序列列表（必须为2条序列）"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "任务类型",
                        "default": "sequence_similarity"
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "比对算法 (global, local, semi-global)",
                        "default": "global"
                    },
                    "match_score": {
                        "type": "integer",
                        "description": "匹配得分",
                        "default": 2
                    },
                    "mismatch_penalty": {
                        "type": "integer",
                        "description": "错配罚分",
                        "default": -1
                    },
                    "gap_open_penalty": {
                        "type": "integer",
                        "description": "空位开放罚分",
                        "default": -2
                    },
                    "gap_extend_penalty": {
                        "type": "integer",
                        "description": "空位扩展罚分",
                        "default": -0.5
                    }
                },
                "required": ["sequences"]
            }
        })
        
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        
        # 缓存实例字典，根据参数组合缓存配置
        self.compare_configs = {}
    
    def handle_error(self, exception: Exception) -> ToolError:
        """
        处理工具执行过程中的错误
        
        Args:
            exception: 异常对象
            
        Returns:
            ToolError: 结构化的工具错误
        """
        error_str = str(exception).lower()
        
        if any(keyword in error_str for keyword in ["import", "module", "library", "biopython", "bio"]):
            return ToolConfigError(f"序列比较库配置错误: {exception}")
        elif any(keyword in error_str for keyword in ["sequence", "序列", "invalid", "无效"]):
            return ToolValidationError(f"序列验证错误: {exception}")
        elif any(keyword in error_str for keyword in ["compare", "比较", "similarity", "相似度"]):
            return ToolServerError(f"序列比较错误: {exception}")
        elif any(keyword in error_str for keyword in ["alignment", "比对", "score", "得分"]):
            return ToolServerError(f"序列比对错误: {exception}")
        else:
            return ToolServerError(f"序列比较工具错误: {exception}")
    
    def analyze(
        self,
        sequences: Union[str, List[str]],
        task_type: str,
        model_name: Optional[str] = None,
        device: DeviceType = DeviceType.CPU,
        **kwargs
    ) -> Dict[str, Any]:
        """
        比较两条序列的相似度
        
        Args:
            sequences: 包含两条序列的列表
            task_type: 任务类型，应该是'seq_compare'
            model_name: 模型名称（不使用）
            device: 设备类型（不使用）
            **kwargs: 额外参数
                - case_sensitive: 是否区分大小写
        
        Returns:
            Dict[str, Any]: 包含比较结果的字典
        """
        # 参数验证与标准化
        if isinstance(sequences, str):
            return {
                "status": "error",
                "error": "序列比较需要提供两条序列",
                "results": []
            }
        
        if len(sequences) != 2:
            return {
                "status": "error",
                "error": "序列比较需要恰好两条序列",
                "results": []
            }
        
        # 检查工具可用性
        if not SEQ_COMPARE_AVAILABLE:
            return {
                "status": "error",
                "error": "序列比较工具不可用",
                "results": []
            }
        
        # 解析参数
        case_sensitive = kwargs.get("case_sensitive", False)
        
        self.logger.info(f"开始序列比较分析，区分大小写: {case_sensitive}")
        
        try:
            # 直接执行序列比较
            seq1, seq2 = sequences[0], sequences[1]
            
            # 预处理序列
            if not case_sensitive:
                seq1 = seq1.upper()
                seq2 = seq2.upper()
            
            # 验证序列长度
            if len(seq1) != len(seq2):
                return {
                    "status": "error",
                    "error": "两条序列长度不相等，无法进行比较",
                    "sequence1": sequences[0],
                    "sequence2": sequences[1],
                    "length1": len(seq1),
                    "length2": len(seq2)
                }
            
            # 计算匹配情况
            matches = []
            match_count = 0
            mismatch_count = 0
            
            for i in range(len(seq1)):
                if seq1[i] == seq2[i]:
                    match_count += 1
                    matches.append("|")
                else:
                    mismatch_count += 1
                    matches.append(" ")
            
            # 计算相似度
            similarity = (match_count / len(seq1)) * 100 if len(seq1) > 0 else 0
            
            # 构建结果
            result = {
                "match_count": match_count,
                "mismatch_count": mismatch_count,
                "total_length": len(seq1),
                "similarity": similarity,
                "alignment": {
                    "sequence1": seq1,
                    "match_line": ''.join(matches),
                    "sequence2": seq2
                }
            }
            
            # 构建响应
            return {
                "status": "success",
                "sequence1": sequences[0],
                "sequence2": sequences[1],
                "comparison": result,
                "parameters": {
                    "case_sensitive": case_sensitive
                }
            }
        except Exception as e:
            self.logger.error(f"序列比较失败: {str(e)}")
            return {
                "error": f"序列比较失败: {str(e)}",
                "status": "error"
            }
    
    def compare_single_pair(
        self,
        seq1: str,
        seq2: str,
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        便捷方法：比较单对序列
        
        Args:
            seq1: 第一条序列
            seq2: 第二条序列
            case_sensitive: 是否区分大小写
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        return self.analyze([seq1, seq2], "seq_compare", case_sensitive=case_sensitive)
    
    def batch_compare(
        self,
        sequence_pairs: List[List[str]],
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        便捷方法：批量比较多对序列
        
        Args:
            sequence_pairs: 序列对列表 [[seq1, seq2], [seq3, seq4], ...]
            case_sensitive: 是否区分大小写
            
        Returns:
            Dict[str, Any]: 比较结果字典列表
        """
        results = []
        for pair in sequence_pairs:
            if len(pair) == 2:
                result = self.compare_single_pair(pair[0], pair[1], case_sensitive)
                results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_pairs": len(sequence_pairs)
        }
    
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [TaskType.PREDICTION.value]
    
    def get_supported_sequence_types(self) -> List[str]:
        """
        获取支持的序列类型
        序列比较工具支持DNA、RNA和蛋白质序列
        """
        return [SequenceType.DNA.value, SequenceType.RNA.value, SequenceType.PROTEIN.value]
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return SEQ_COMPARE_AVAILABLE
    
    def run(self, arguments=None, stream_callback=None, use_cache=False, validate=True):
        """Execute the sequence comparison tool."""
        if arguments is None:
            arguments = {}
        
        # Extract required parameters
        seq1 = arguments.get("seq1", "")
        seq2 = arguments.get("seq2", "")
        
        # Validate parameters
        if not seq1 or not seq2:
            return {"error": "Both seq1 and seq2 must be provided"}
        
        # Perform the sequence comparison
        result = self.compare_single_pair(seq1, seq2)
        
        return result


# 如果作为主程序运行，提供简单的测试
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化工具
    tool = SeqCompareTool()
    
    # 测试序列对
    test_seq1 = "GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA"
    test_seq2 = "GCTGGATACTCGATGGCCTTGTTTCTTGCCCCTTGACCCTCCCCCCATCCCCTCCTCCCCTTCCTGCAACCGTACCCCCGTGTTCTTTGAATAAAGTCTGTGTGGGCGACA"
    
    # 执行比较
    result = tool.compare_single_pair(test_seq1, test_seq2, show_result=True)
    
    # 打印结果
    import json
    print(json.dumps(result, indent=2))