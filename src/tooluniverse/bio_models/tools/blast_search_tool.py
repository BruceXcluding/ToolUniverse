"""
BLAST搜索工具 - 用于NCBI序列比对搜索
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from Bio.Blast import NCBIWWW, NCBIXML
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
    BLAST_AVAILABLE = True
except ImportError:
    BLAST_AVAILABLE = False
    logging.warning("无法导入Biopython库，请确保Biopython已安装")


class BlastSearchTool(BaseTool):
    """BLAST搜索工具 - 用于NCBI序列比对搜索"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化BLAST搜索工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 初始化BaseTool
        super().__init__({
            "name": "blast_search",
            "description": "BLAST序列比对搜索工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要搜索的序列列表"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "任务类型",
                        "default": "function_annotation"
                    },
                    "program": {
                        "type": "string",
                        "description": "BLAST程序类型 (blastn, blastp, blastx, tblastn, tblastx)",
                        "default": "blastn"
                    },
                    "database": {
                        "type": "string",
                        "description": "搜索数据库 (nt, nr, refseq_rna等)",
                        "default": "nt"
                    },
                    "email": {
                        "type": "string",
                        "description": "NCBI邮箱地址（必需）"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回的最大结果数",
                        "default": 10
                    },
                    "expect_threshold": {
                        "type": "number",
                        "description": "E值阈值",
                        "default": 0.001
                    }
                },
                "required": ["sequences", "email"]
            }
        })
        
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        
        # 缓存实例字典，根据参数组合缓存配置
        self.blast_configs = {}
    
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
            return ToolConfigError(f"BLAST库配置错误: {exception}")
        elif any(keyword in error_str for keyword in ["sequence", "序列", "invalid", "无效"]):
            return ToolValidationError(f"序列验证错误: {exception}")
        elif any(keyword in error_str for keyword in ["network", "网络", "connection", "连接", "timeout", "超时"]):
            return ToolUnavailableError(f"网络连接错误: {exception}")
        elif any(keyword in error_str for keyword in ["blast", "search", "搜索"]):
            return ToolServerError(f"BLAST搜索错误: {exception}")
        else:
            return ToolServerError(f"BLAST搜索工具错误: {exception}")
    
    def analyze(
        self,
        sequences: Union[str, List[str]],
        task_type: Union[str, TaskType] = TaskType.FUNCTION_ANNOTATION,
        model_name: Optional[str] = None,
        sequence_type: Optional[Union[str, SequenceType]] = None,
        device: Union[str, DeviceType] = DeviceType.CPU,  # BLAST主要使用CPU
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行BLAST序列比对搜索
        
        Args:
            sequences: 输入序列（单个字符串或列表）
            task_type: 任务类型，默认为功能注释
            model_name: 模型名称（这里不使用，保留为兼容性）
            sequence_type: 序列类型（可选，会自动检测）
            device: 设备类型，默认为CPU
            **kwargs: 其他参数
                program: BLAST程序类型 (blastn, blastp, blastx, tblastn, tblastx)
                database: 搜索数据库 (nt, nr, refseq_rna等)
                email: NCBI邮箱地址（必需）
                max_results: 返回的最大结果数
                expect_threshold: E值阈值
                
        Returns:
            Dict[str, Any]: 包含BLAST搜索结果的字典
        """
        # 验证BLAST工具是否可用
        if not BLAST_AVAILABLE:
            return {
                "error": "BLAST搜索工具不可用，请确保正确安装了Biopython",
                "status": "error"
            }
        
        # 确保任务类型正确
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        
        # 转换输入序列格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 获取参数，设置默认值
        program = kwargs.get("program", "blastn")
        database = kwargs.get("database", "nt")
        email = kwargs.get("email")
        max_results = kwargs.get("max_results", 10)
        expect_threshold = kwargs.get("expect_threshold", 0.001)
        
        # 验证必需参数
        if not email:
            return {
                "error": "缺少必需参数: email - NCBI要求提供邮箱地址",
                "status": "error"
            }
        
        try:
            # 设置NCBI邮箱
            NCBIWWW.email = email
            
            # 批量处理序列
            results = []
            for i, seq in enumerate(sequences):
                self.logger.info(f"正在处理序列 {i+1}/{len(sequences)}")
                
                # 执行BLAST搜索
                try:
                    self.logger.info(f"正在提交 BLAST 请求 ({program} vs {database})...")
                    
                    result_handle = NCBIWWW.qblast(
                        program=program,
                        database=database,
                        sequence=seq,
                        expect=expect_threshold
                    )
                    
                    # 保存原始 XML 结果到临时文件
                    xml_content = result_handle.read()
                    xml_file = f"blast_result_{i}.xml"
                    with open(xml_file, "w") as f:
                        f.write(xml_content)
                    
                    # 解析结果
                    blast_results = []
                    with open(xml_file) as f:
                        blast_records = NCBIXML.parse(f)
                        
                        for blast_record in blast_records:
                            for alignment in blast_record.alignments[:max_results]:
                                for hsp in alignment.hsps[:1]:  # 只取每个 alignment 的最佳 HSP
                                    blast_results.append({
                                        "query_id": blast_record.query_id,
                                        "query_length": blast_record.query_length,
                                        "subject_title": alignment.title,
                                        "subject_length": alignment.length,
                                        "align_length": hsp.align_length,
                                        "identities": hsp.identities,
                                        "expect": float(hsp.expect),
                                        "score": float(hsp.score),
                                        "bits": float(hsp.bits),
                                        "query_seq": hsp.query,
                                        "match_seq": hsp.match,
                                        "sbjct_seq": hsp.sbjct
                                    })
                    
                    # 清理临时文件
                    if os.path.exists(xml_file):
                        os.remove(xml_file)
                    
                    # 排序并限制结果数量
                    blast_results = sorted(blast_results, key=lambda x: x["expect"])[:max_results]
                    
                    results.append({
                        "sequence": seq,
                        "sequence_length": len(seq),
                        "matches": blast_results,
                        "match_count": len(blast_results),
                        "parameters": {
                            "program": program,
                            "database": database,
                            "max_results": max_results,
                            "expect_threshold": expect_threshold
                        },
                        "status": "success"
                    })
                    
                except Exception as inner_e:
                    self.logger.error(f"处理序列时出错: {str(inner_e)}")
                    results.append({
                        "sequence": seq,
                        "error": f"处理序列时出错: {str(inner_e)}",
                        "status": "error"
                    })
            
            # 构建最终响应
            response = {
                "success": True,
                "results": results,
                "metadata": {
                    "tool": self.tool_config.get("name"),
                    "task_type": task_type.value,
                    "sequence_count": len(sequences),
                    "device": "cpu",
                    "parameters": {
                        "program": program,
                        "database": database
                    }
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"BLAST搜索失败: {str(e)}")
            return {
                "error": f"BLAST搜索失败: {str(e)}",
                "status": "error"
            }
    
    def parse_xml_result(
        self,
        xml_file: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        解析已保存的BLAST XML结果文件
        
        Args:
            xml_file: XML文件路径
            max_results: 返回的最大结果数
            
        Returns:
            Dict[str, Any]: 解析结果
        """
        if not BLAST_AVAILABLE:
            return {
                "error": "BLAST搜索工具不可用",
                "status": "error"
            }
        
        try:
            # 使用默认参数创建临时搜索器
            searcher = BlastSearcher(program="blastn", database="nt", email="user@example.com")
            results = searcher.parse_xml_result(xml_file, max_results)
            
            return {
                "success": True,
                "results": results,
                "match_count": len(results),
                "metadata": {
                    "tool": self.tool_config.get("name"),
                    "xml_file": xml_file,
                    "max_results": max_results
                }
            }
            
        except Exception as e:
            self.logger.error(f"解析XML结果失败: {str(e)}")
            return {
                "error": f"解析XML结果失败: {str(e)}",
                "status": "error"
            }
    
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [TaskType.FUNCTION_ANNOTATION.value]
    
    def get_supported_sequence_types(self) -> List[str]:
        """获取支持的序列类型"""
        return [SequenceType.DNA.value, SequenceType.RNA.value, SequenceType.PROTEIN.value]
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return BLAST_AVAILABLE
    
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
        task_type = arguments.get("task_type", TaskType.FUNCTION_ANNOTATION)
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
    tool = BlastSearchTool()
    
    # 测试序列 - 使用一个短序列进行快速测试
    test_sequence = "ATCGATCGATCGATCGATCG"
    
    # 执行搜索 - 注意这里需要提供真实的邮箱地址
    result = tool.analyze(
        test_sequence,
        email="test@example.com",  # 使用测试邮箱
        max_results=5,
        program="blastn",
        database="nt"
    )
    
    # 打印结果
    import json
    print(json.dumps(result, indent=2))