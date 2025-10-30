"""
JASPAR扫描工具 - 用于转录因子结合位点MOTIF扫描
"""
import logging
from typing import Dict, List, Any, Optional, Union
from Bio.Seq import Seq
from ..task_types import TaskType, SequenceType, DeviceType

# 尝试导入pyjaspar库
try:
    from pyjaspar import jaspardb
    JASPAR_AVAILABLE = True
except ImportError:
    JASPAR_AVAILABLE = False
    logging.warning("无法导入pyjaspar库，请确保pyjaspar和Biopython已安装")


class JasparScanTool:
    """JASPAR扫描工具 - 用于转录因子结合位点预测"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化JASPAR扫描工具
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 直接初始化必要的组件
        self.logger = logging.getLogger(__name__)
        self.tool_name = "jaspar_scan"
        # 添加一个标志来控制是否输出额外信息
        self.quiet_mode = False
        
        # 已知会导致问题的motif
        self.problematic_motifs = ['MA1651.2', 'MA1975.2']
        
        # JASPAR数据库连接
        self.jdb = None
        self.motifs = []
        self.initialized = False
        self.error_message = None
        
        if JASPAR_AVAILABLE:
            try:
                # 初始化JASPAR数据库连接
                self.jdb = jaspardb(release="JASPAR2024")
                # 加载默认motifs
                self._load_default_motifs()
                self.initialized = True
            except Exception as e:
                self.error_message = f"初始化JASPAR数据库失败: {str(e)}"
                if not self.quiet_mode:
                    self.logger.error(self.error_message)
    
    def _load_default_motifs(self):
        """加载默认的人类CORE集合motifs"""
        try:
            self.motifs = self.jdb.fetch_motifs(
                collection="CORE",
                tax_group="vertebrates",
                species=[9606]  # 人类
            )
        except Exception as e:
            self.logger.error(f"加载默认motifs失败: {str(e)}")
    
    def analyze(
        self,
        sequences: Union[str, List[str]],
        task_type: Union[str, TaskType] = TaskType.MOTIF_DETECTION,
        model_name: Optional[str] = None,
        sequence_type: Optional[Union[str, SequenceType]] = None,
        device: Union[str, DeviceType] = DeviceType.CPU,
        **kwargs
    ) -> Dict[str, Any]:
        """
        扫描DNA序列中的转录因子结合位点
        
        Args:
            sequences: DNA序列（单个字符串或列表）
            task_type: 任务类型，默认为motif检测
            model_name: 模型名称（这里不使用，保留为兼容性）
            sequence_type: 序列类型，默认为DNA
            device: 设备类型，默认为CPU
            **kwargs: 其他参数
                threshold_ratio: 阈值比例（0.0-1.0）
                pseudocount: 伪计数
                top_n: 返回的顶部结果数
                
        Returns:
            Dict[str, Any]: 包含motif扫描结果的字典
        """
        # 验证JASPAR工具是否可用
        if not JASPAR_AVAILABLE:
            return {
                "error": "JASPAR扫描工具不可用，请确保正确安装了pyjaspar和Biopython",
                "status": "error"
            }
        
        # 检查是否初始化成功
        if not self.initialized:
            error_msg = self.error_message or "JASPAR数据库未初始化"
            return {
                "error": error_msg,
                "status": "error"
            }
        
        # 检查是否有motif数据
        if not self.motifs:
            return {
                "error": "未加载到任何motif数据，请检查网络连接或JASPAR数据库访问权限",
                "status": "error"
            }
        
        # 确保任务类型为motif检测
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        
        if task_type != TaskType.MOTIF_DETECTION:
            return {
                "error": f"JASPAR扫描工具只支持motif检测任务，不支持 {task_type.value}",
                "status": "error"
            }
        
        # 设置默认序列类型为DNA
        if not sequence_type:
            sequence_type = SequenceType.DNA
        
        # 获取参数，设置默认值
        threshold_ratio = kwargs.get("threshold_ratio", 0.85)
        pseudocount = kwargs.get("pseudocount", 0.5)
        top_n = kwargs.get("top_n", 10)
        
        # 转换输入序列格式
        if isinstance(sequences, str):
            sequences = [sequences]
        
        try:
            # 批量处理序列
            results = []
            for seq in sequences:
                try:
                    # 验证序列是否为DNA（只包含ATGC，支持U转T）
                    seq_processed = seq.upper().replace("U", "T")
                    if not all(base in "ATGC" for base in seq_processed):
                        results.append({
                            "sequence": seq,
                            "error": "输入不是有效的DNA序列，请确保只包含A、T、G、C字符",
                            "status": "error"
                        })
                        continue
                    
                    # 执行motif扫描
                    motif_hits = self._scan_sequence(seq_processed, threshold_ratio, pseudocount)
                    
                    # 按分数排序并限制数量
                    motif_hits = sorted(motif_hits, key=lambda x: x["score"], reverse=True)[:top_n]
                    
                    results.append({
                        "sequence": seq,
                        "sequence_length": len(seq),
                        "motif_hits": motif_hits,
                        "hit_count": len(motif_hits),
                        "parameters": {
                            "threshold_ratio": threshold_ratio,
                            "pseudocount": pseudocount,
                            "top_n": top_n
                        },
                        "status": "success"
                    })
                except Exception as inner_e:
                    # 在JSON模式下完全抑制日志输出
                    pass
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
                    "tool": self.tool_name,
                    "task_type": task_type.value,
                    "sequence_type": sequence_type.value,
                    "sequence_count": len(sequences),
                    "device": "cpu",
                    "jaspar_version": "JASPAR2024",
                    "species": "human (9606)"
                }
            }
            
            return response
            
        except Exception as e:
                # 在JSON模式下完全抑制日志输出
                return {
                "error": f"JASPAR扫描失败: {str(e)}",
                "status": "error"
            }
    
    def _scan_sequence(self, seq_str: str, threshold_ratio: float, pseudocount: float) -> List[Dict[str, Any]]:
        """
        扫描单个DNA序列，查找motif匹配
        
        Args:
            seq_str: 处理后的DNA序列字符串
            threshold_ratio: 阈值比例
            pseudocount: 伪计数
            
        Returns:
            List[Dict[str, Any]]: 找到的motif匹配列表
        """
        hits = []
        seq = Seq(seq_str)
        
        for motif in self.motifs:
            try:
                # 跳过已知问题motif
                if motif.matrix_id in self.problematic_motifs:
                    continue
                
                # 按照客户实现的简洁方式处理
                pwm = motif.counts.normalize(pseudocounts=pseudocount)
                pssm = pwm.log_odds()
                threshold = pssm.max * threshold_ratio
                
                # 正链搜索
                try:
                    for pos, score in pssm.search(seq, threshold=threshold):
                        hits.append({
                            "matrix_id": motif.matrix_id,
                            "name": motif.name,
                            "start": int(pos) + 1,  # 转换为1-based索引
                            "end": int(pos) + motif.length,
                            "strand": "+",
                            "score": float(score),
                            "sequence": str(seq[pos:pos+motif.length]),
                            "relative_score": float(score) / pssm.max
                        })
                except Exception:
                    # 静默跳过错误，继续处理下一个motif
                    pass
                
                # 反向互补链搜索
                try:
                    rev_comp_seq = seq.reverse_complement()
                    for pos, score in pssm.search(rev_comp_seq, threshold=threshold):
                        hits.append({
                            "matrix_id": motif.matrix_id,
                            "name": motif.name,
                            "start": int(len(seq) - pos - motif.length) + 1,  # 转换为1-based索引
                            "end": int(len(seq) - pos),
                            "strand": "-",
                            "score": float(score),
                            "sequence": str(rev_comp_seq[pos:pos+motif.length]),
                            "relative_score": float(score) / pssm.max
                        })
                except Exception:
                    # 静默跳过错误，继续处理下一个motif
                    pass
            except Exception:
                # 捕获所有异常，确保程序继续运行
                continue
        
        return hits
    
    def get_supported_tasks(self) -> List[str]:
        """获取支持的任务类型"""
        return [TaskType.MOTIF_DETECTION.value]
    
    def get_supported_sequence_types(self) -> List[str]:
        """
        获取支持的序列类型
        JASPAR主要用于DNA序列，但也支持RNA序列（会自动转换U为T）
        """
        return [SequenceType.DNA.value, SequenceType.RNA.value]
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return JASPAR_AVAILABLE
    
    def get_available_motifs(self) -> Dict[str, Any]:
        """
        获取可用的motif信息
        
        Returns:
            Dict[str, Any]: 包含可用motif信息的字典
        """
        if not JASPAR_AVAILABLE:
            return {
                "error": "JASPAR扫描工具不可用",
                "status": "error"
            }
        
        # 检查是否初始化成功
        if not self.initialized:
            error_msg = self.error_message or "JASPAR数据库未初始化"
            return {
                "error": error_msg,
                "status": "error"
            }
        
        try:
            # 获取motif信息
            motifs_info = []
            for motif in self.motifs:
                motifs_info.append({
                    "matrix_id": motif.matrix_id,
                    "name": motif.name,
                    "length": motif.length,
                    "release": "JASPAR2024",
                    "species": "human (9606)"
                })
            
            return {
                "success": True,
                "motif_count": len(motifs_info),
                "motifs": motifs_info,
                "metadata": {
                    "release": "JASPAR2024",
                    "species": "human (9606)",
                    "collection": "CORE"
                }
            }
            
        except Exception as e:
                # 在JSON模式下完全抑制日志输出
                return {
                "error": f"获取motif信息失败: {str(e)}",
                "status": "error"
            }


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
        
        # 设置静默模式
        self.quiet_mode = arguments.get('quiet_mode', False)
            
        sequences = arguments.get("sequences")
        task_type = arguments.get("task_type", TaskType.MOTIF_DETECTION)
        model_name = arguments.get("model_name")
        sequence_type = arguments.get("sequence_type")
        device = arguments.get("device", "cpu")
        
        # 提取kwargs
        kwargs = arguments.copy()
        for key in ["sequences", "task_type", "model_name", "sequence_type", "device", "quiet_mode"]:
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
    test_sequence = "GGATGGCCATGTAGCTAATCTTAAAGCAAGTTAAGCAGATGTAGCTAACG"
    
    # 执行扫描
    result = tool.analyze(test_sequence, threshold_ratio=0.8, top_n=5)
    
    # 打印结果
    import json
    print(json.dumps(result, indent=2))