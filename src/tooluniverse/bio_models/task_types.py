"""
生物序列分析任务类型定义
"""
from enum import Enum


class TaskType(Enum):
    """生物序列分析任务类型"""
    EMBEDDING = "embedding"                    # 序列嵌入
    CLASSIFICATION = "classification"          # 分类任务
    STRUCTURE_PREDICTION = "structure_prediction"  # 结构预测
    MOTIF_DETECTION = "motif_detection"        # 基序检测
    MUTATION_ANALYSIS = "mutation_analysis"    # 突变分析
    GENERATION = "generation"                  # 序列生成
    FUNCTION_ANNOTATION = "function_annotation" # 功能注释


class SequenceType(Enum):
    """序列类型"""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


class ModelStatus(Enum):
    """模型状态"""
    UNLOADED = "unloaded"      # 未加载
    LOADING = "loading"        # 加载中
    LOADED = "loaded"          # 已加载
    ERROR = "error"            # 错误
    UNLOADING = "unloading"    # 卸载中


class DeviceType(Enum):
    """设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"              # 自动选择