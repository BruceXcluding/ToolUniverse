"""
JASPAR扫描工具 - 用于转录因子结合位点扫描
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from ...base_tool import BaseTool
from ...exceptions import (
    ToolError, 
    ToolConfigError, 
    ToolValidationError, 
    ToolUnavailableError,
    ToolServerError
)
from ..task_types import TaskType, SequenceType, DeviceType

# 检查JASPAR库是否可用
try:
    import pyjaspar
    JASPAR_AVAILABLE = True
except ImportError:
    JASPAR_AVAILABLE = False
    logging.warning("无法导入JASPAR库，请确保JASPAR已安装")


class JasparScanTool(BaseTool):
    """JASPAR扫描工具 - 用于转录因子结合位点扫描"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化JASPAR扫描工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 初始化BaseTool
        super().__init__({
            "name": "jaspar_scan",
            "description": "JASPAR转录因子结合位点扫描工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要扫描的DNA序列列表"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "任务类型",
                        "default": "tf_binding_site"
                    },
                    "collection": {
                        "type": "string",
                        "description": "JASPAR集合 (CORE, CNE, PBM等)",
                        "default": "CORE"
                    },
                    "tax_group": {
                        "type": "string",
                        "description": "分类群 (vertebrates, plants, insects等)",
                        "default": "vertebrates"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "最小匹配分数阈值",
                        "default": 0.8
                    },
                    "pvalue_threshold": {
                        "type": "number",
                        "description": "p值阈值",
                        "default": 0.001
                    }
                },
                "required": ["sequences"]
            }
        })
        
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        
        # 缓存实例字典，根据参数组合缓存配置
        self.jaspar_configs = {}
    
    def handle_error(self, exception: Exception) -> ToolError:
        """
        处理工具执行过程中的错误
        
        Args:
            exception: 异常对象
            
        Returns:
            ToolError: 结构化的工具错误
        """
        error_str = str(exception).lower()
        
        if any(keyword in error_str for keyword in ["import", "module", "library", "jaspar", "bio"]):
            return ToolConfigError(f"JASPAR库配置错误: {exception}")
        elif any(keyword in error_str for keyword in ["sequence", "序列", "invalid", "无效"]):
            return ToolValidationError(f"序列验证错误: {exception}")
        elif any(keyword in error_str for keyword in ["motif", "profile", "矩阵", "matrix"]):
            return ToolValidationError(f"转录因子矩阵错误: {exception}")
        elif any(keyword in error_str for keyword in ["network", "网络", "connection", "连接", "timeout", "超时"]):
            return ToolUnavailableError(f"网络连接错误: {exception}")
        elif any(keyword in error_str for keyword in ["scan", "scanning", "扫描"]):
            return ToolServerError(f"JASPAR扫描错误: {exception}")
        else:
            return ToolServerError(f"JASPAR扫描工具错误: {exception}")
    
    def analyze(
        self,
        sequences: Union[str, List[str]],
        task_type: str,
        model_name: Optional[str] = None,
        device: DeviceType = DeviceType.CPU,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用JASPAR数据库扫描转录因子结合位点
        
        Args:
            sequences: DNA序列（单个或列表）
            task_type: 任务类型，应该是'jaspar_scan'
            model_name: 模型名称（不使用）
            device: 设备类型（不使用）
            **kwargs: 额外参数
                - tf_names: 转录因子名称列表（可选）
                - collection: JASPAR集合（如CORE, CNE等）
                - species: 物种过滤（如9606为人类）
                - min_score: 最小匹配分数阈值（0.0-1.0）
                - pvalue: p值阈值
        
        Returns:
            Dict[str, Any]: 包含扫描结果的字典
        """
        # 参数验证与标准化
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 检查工具可用性
        if not JASPAR_AVAILABLE:
            return {
                "status": "error",
                "error": "JASPAR工具不可用，请安装pyjaspar库",
                "results": []
            }
        
        # 解析参数
        tf_names = kwargs.get("tf_names", [])
        collection = kwargs.get("collection", "CORE")
        tax_group = kwargs.get("tax_group", "vertebrates")
        species = kwargs.get("species", None)
        min_score = kwargs.get("min_score", 0.8)
        pvalue = kwargs.get("pvalue", 0.001)
        
        self.logger.info(f"开始JASPAR扫描，序列数量: {len(sequences)}")
        
        try:
            # 导入JASPAR库
            from pyjaspar import jaspardb
            
            # 初始化JASPAR数据库连接
            jdb = jaspardb()
            
            # 获取转录因子矩阵
            if tf_names:
                # 如果指定了转录因子名称，只获取这些因子
                matrices = []
                for tf_name in tf_names:
                    try:
                        matrices.extend(jdb.fetch_motifs_by_name(tf_name))
                    except Exception as e:
                        self.logger.warning(f"无法获取转录因子 {tf_name}: {str(e)}")
            else:
                # 否则获取所有转录因子
                matrices = jdb.fetch_motifs(collection=collection, tax_group=tax_group)
            
            if not matrices:
                return {
                    "status": "error",
                    "error": "未找到匹配的转录因子矩阵",
                    "results": []
                }
            
            self.logger.info(f"获取到 {len(matrices)} 个转录因子矩阵")
            
            # 扫描每个序列
            results = []
            for seq_idx, sequence in enumerate(sequences):
                seq_results = {
                    "sequence_id": seq_idx,
                    "sequence": sequence,
                    "length": len(sequence),
                    "tfbs": []  # 转录因子结合位点
                }
                
                # 对每个转录因子矩阵进行扫描
                for matrix in matrices:
                    try:
                        # 尝试使用pyjaspar的search方法
                        hits = []
                        try:
                            hits = matrix.search(sequence, pvalue=pvalue)
                            
                            # 过滤低分命中
                            filtered_hits = [
                                {
                                    "tf_name": matrix.name,
                                    "tf_id": matrix.matrix_id,
                                    "tf_family": getattr(matrix, 'family', 'unknown'),
                                    "start": hit.start,
                                    "end": hit.end,
                                    "strand": hit.strand,
                                    "score": hit.score,
                                    "relative_score": hit.score / matrix.max_score,
                                    "matched_sequence": hit.matched_sequence
                                }
                                for hit in hits
                                if hit.score / matrix.max_score >= min_score
                            ]
                        except AttributeError:
                            # 如果search方法不可用，使用Biopython的motif模块
                            from Bio import motifs
                            from Bio.Seq import Seq
                            
                            # 转换为Biopython的motif对象
                            motif = motifs.Motif(counts=matrix.counts)
                            
                            # 计算序列与motif的匹配分数
                            seq_obj = Seq(sequence)
                            
                            # 使用Biopython的search方法
                            try:
                                # 尝试使用search方法
                                positions = motif.search(seq_obj, threshold=min_score)
                                
                                filtered_hits = []
                                for pos in positions:
                                    # 获取匹配的序列片段
                                    match_seq = sequence[pos:pos+len(matrix)]
                                    
                                    filtered_hits.append({
                                        "tf_name": matrix.name,
                                        "tf_id": matrix.matrix_id,
                                        "tf_family": getattr(matrix, 'family', 'unknown'),
                                        "start": pos,
                                        "end": pos + len(matrix),
                                        "strand": "+",
                                        "score": 1.0,  # 默认分数
                                        "relative_score": 1.0,
                                        "matched_sequence": match_seq
                                    })
                            except AttributeError:
                                # 如果search方法也不可用，使用简单的滑动窗口方法
                                filtered_hits = []
                                motif_length = len(matrix)
                                
                                # 滑动窗口计算匹配分数
                                for pos in range(len(sequence) - motif_length + 1):
                                    sub_seq = sequence[pos:pos+motif_length]
                                    
                                    # 计算匹配分数（简化版本）
                                    score = 0.0
                                    for i, nucleotide in enumerate(sub_seq):
                                        if nucleotide.upper() in 'ACGT':
                                            # 获取该位置的核苷酸频率
                                            if i < len(matrix):
                                                # 获取该位置所有核苷酸的总数
                                                total_counts = sum(matrix.counts[base][i] for base in 'ACGT')
                                                if total_counts > 0:
                                                    # 获取当前核苷酸的频率
                                                    count_value = matrix.counts[nucleotide.upper()][i]
                                                    score += count_value / total_counts
                                    
                                    # 归一化分数
                                    normalized_score = score / motif_length
                                    
                                    if normalized_score >= min_score:
                                        filtered_hits.append({
                                            "tf_name": matrix.name,
                                            "tf_id": matrix.matrix_id,
                                            "tf_family": getattr(matrix, 'family', 'unknown'),
                                            "start": pos,
                                            "end": pos + motif_length,
                                            "strand": "+",
                                            "score": float(normalized_score),
                                            "relative_score": float(normalized_score),
                                            "matched_sequence": sub_seq
                                        })
                        
                        # 按相对分数排序
                        filtered_hits.sort(key=lambda x: x["relative_score"], reverse=True)
                        
                        # 添加到结果
                        seq_results["tfbs"].extend(filtered_hits)
                        
                    except Exception as e:
                        self.logger.warning(f"矩阵 {matrix.name} 扫描失败: {str(e)}")
                
                # 按相对分数排序所有TFBS
                seq_results["tfbs"].sort(key=lambda x: x["relative_score"], reverse=True)
                
                # 添加统计信息
                seq_results["tfbs_count"] = len(seq_results["tfbs"])
                seq_results["unique_tfs"] = len(set(hit["tf_name"] for hit in seq_results["tfbs"]))
                
                results.append(seq_results)
            
            # 汇总统计
            total_tfbs = sum(result["tfbs_count"] for result in results)
            unique_tfs = set()
            for result in results:
                unique_tfs.update(hit["tf_name"] for hit in result["tfbs"])
            
            # 构建响应
            return {
                "status": "success",
                "parameters": {
                    "collection": collection,
                    "species": species,
                    "min_score": min_score,
                    "pvalue": pvalue,
                    "tf_names": tf_names
                },
                "summary": {
                    "total_sequences": len(sequences),
                    "total_tfbs": total_tfbs,
                    "unique_tfs": len(unique_tfs),
                    "matrices_used": len(matrices)
                },
                "results": results
            }
        except Exception as e:
            self.logger.error(f"JASPAR扫描失败: {str(e)}")
            return {
                "error": f"JASPAR扫描失败: {str(e)}",
                "status": "error"
            }
    
    def scan_single_sequence(
        self,
        sequence: str,
        tf_names: Optional[List[str]] = None,
        collection: str = "CORE",
        species: Optional[int] = None,
        min_score: float = 0.8,
        pvalue: float = 0.001
    ) -> Dict[str, Any]:
        """
        便捷方法：扫描单个序列
        
        Args:
            sequence: DNA序列
            tf_names: 转录因子名称列表（可选）
            collection: JASPAR集合
            species: 物种过滤
            min_score: 最小匹配分数阈值
            pvalue: p值阈值
            
        Returns:
            Dict[str, Any]: 扫描结果
        """
        return self.analyze(
            sequence, 
            "jaspar_scan", 
            tf_names=tf_names,
            collection=collection,
            species=species,
            min_score=min_score,
            pvalue=pvalue
        )
    
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [TaskType.PREDICTION.value]
    
    def get_supported_sequence_types(self) -> List[str]:
        """
        获取支持的序列类型
        JASPAR扫描工具只支持DNA序列
        """
        return [SequenceType.DNA.value]
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return JASPAR_AVAILABLE

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
        task_type = arguments.get("task_type", TaskType.JASPAR_SCAN)
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
    tool = JasparScanTool()
    
    # 测试序列
    test_seq = "GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA"
    
    # 执行扫描
    result = tool.scan_single_sequence(test_seq)
    
    # 打印结果
    import json
    print(json.dumps(result, indent=2))