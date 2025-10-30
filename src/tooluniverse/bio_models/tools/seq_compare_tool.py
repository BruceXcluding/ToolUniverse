""" 序列比较工具 - 用于序列比对与相似度计算 """
import os
import logging
from typing import Dict, List, Any, Optional, Union
from ..task_types import TaskType, SequenceType, DeviceType

# 直接实现序列比较功能，不需要额外导入
SEQ_COMPARE_AVAILABLE = True


class SeqCompareTool:
    """序列比较工具 - 用于计算两条序列的相似度"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化序列比较工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        self.tool_name = "seq_compare"
        self.logger.info("序列比较工具初始化完成")
    
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