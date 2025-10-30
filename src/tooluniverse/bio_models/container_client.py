"""
ToolUniverse 模型容器统一API客户端
提供与模型容器交互的统一接口
"""

import json
import time
import uuid
import requests
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """支持的任务类型"""
    PROMOTER_PREDICTION = "promoter_prediction"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    PROPERTY_PREDICTION = "property_prediction"


@dataclass
class PredictionResult:
    """预测结果"""
    prediction: Union[float, List[float], Dict[str, Any]]
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    sequence_length: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str
    description: str
    tasks: List[str]
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    max_sequence_length: int
    supported_species: Optional[List[str]] = None


@dataclass
class HealthStatus:
    """健康状态"""
    status: str
    model: str
    version: str
    uptime: int
    gpu_available: bool
    memory_usage: str


class ModelContainerClient:
    """模型容器客户端"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        初始化客户端
        
        Args:
            base_url: 模型容器的基础URL，例如 "http://localhost:8000"
            timeout: 请求超时时间（秒）
        """
        import logging
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应数据
            
        Raises:
            Exception: 请求失败时抛出异常
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求失败: {str(e)}")
    
    def check_health(self) -> HealthStatus:
        """
        检查模型服务健康状态
        
        Returns:
            健康状态信息
        """
        response = self._make_request('GET', '/health')
        
        # 适配不同的健康检查响应格式
        if 'success' in response and not response.get('success', False):
            raise Exception(f"健康检查失败: {response.get('message', '未知错误')}")
        
        # 处理标准格式
        if 'data' in response:
            data = response.get('data', {})
        else:
            # 处理DNABERT2和LucaOne容器的实际返回格式
            data = response
        
        # 确保有status字段
        status = data.get('status', 'unknown')
        
        return HealthStatus(
            status=status,
            model=data.get('model', 'unknown'),
            version=data.get('version', 'unknown'),
            uptime=data.get('uptime', 0),
            gpu_available=data.get('gpu_available', False),
            memory_usage=str(data.get('memory_usage', '0'))
        )
    
    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        import os
        import json
        
        # 尝试从URL推断模型名称
        model_name = self._infer_model_name()
        
        # 尝试加载model_config.json配置文件
        config_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'config', 'bio_models', 'model_config.json'
        )
        
        model_info = None
        
        # 尝试从配置文件加载模型信息
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 从配置中获取模型信息
                models_config = config.get('models', {})
                
                # 尝试找到匹配的模型配置
                matched_config = None
                for name, model_config in models_config.items():
                    # 模糊匹配模型名称
                    if name.lower() == model_name.lower() or \
                       model_name.lower() in name.lower() or \
                       name.lower() in model_name.lower():
                        # 确保使用dnabert2而不是dnabert_2
                        if name.lower() == "dnabert_2":
                            model_name = "dnabert2"
                        else:
                            model_name = name
                        matched_config = model_config
                        break
                
                # 如果找到匹配的配置，构建ModelInfo
                if matched_config:
                    model_info = ModelInfo(
                        name=model_name,
                        version="1.0.0",  # 配置文件中没有版本信息，使用默认值
                        description=matched_config.get('description', f"{model_name} 模型"),
                        tasks=matched_config.get('supported_tasks', []),
                        input_format={
                            "sequences": "序列列表", 
                            "task_type": "任务类型",
                            "supported_sequences": matched_config.get('supported_sequences', [])
                        },
                        output_format={
                            "results": "预测结果列表", 
                            "processing_time": "处理时间"
                        },
                        max_sequence_length=matched_config.get('max_sequence_length', 512),
                        supported_species=None  # 配置文件中没有提供具体的物种信息
                    )
                    self.logger.debug(f"从配置文件加载了 {model_name} 的模型信息")
            except Exception as e:
                self.logger.warning(f"加载模型配置文件失败: {str(e)}")
        
        # 如果没有从配置文件获取到信息，使用默认策略
        if model_info is None:
            self.logger.debug(f"未找到 {model_name} 的配置，使用默认信息")
            # 使用基于模型名称的默认信息
            if "dnabert2" in model_name.lower():
                model_info = ModelInfo(
                    name="dnabert2",
                    version="1.0.0",
                    description="DNABERT-2 with BPE tokenization and ALiBi position encoding",
                    tasks=["embedding", "classification", "annotation", "property_prediction"],
                    input_format={"sequences": "DNA序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=512,
                    supported_species=None
                )
            elif "luca" in model_name.lower():
                model_info = ModelInfo(
                    name="lucaone",
                    version="1.0.0",
                    description="Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language",
                    tasks=["embedding", "classification", "property_prediction", "interaction", "annotation", "generation"],
                    input_format={"sequences": "序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=1024,
                    supported_species=None
                )
            elif "rna-fm" in model_name.lower() or "rna" in model_name.lower():
                model_info = ModelInfo(
                    name="rna-fm",
                    version="1.0.0",
                    description="RNA Foundation Model trained on 23+ million non-coding RNA sequences",
                    tasks=["embedding", "classification", "structure_prediction", "generation", "property_prediction"],
                    input_format={"sequences": "RNA序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=1024,
                    supported_species=None
                )
            elif "utr" in model_name.lower():
                model_info = ModelInfo(
                    name=model_name,
                    version="1.0.0",
                    description="UTR序列表示学习模型",
                    tasks=["embedding", "classification", "annotation", "translation", "property_prediction"],
                    input_format={"sequences": "UTR序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=512,
                    supported_species=None
                )
            elif "codon" in model_name.lower():
                model_info = ModelInfo(
                    name="codonbert",
                    version="1.0.0",
                    description="BERT model for codon optimization and mRNA design",
                    tasks=["embedding", "generation", "property_prediction"],
                    input_format={"sequences": "DNA序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=1024,
                    supported_species=None
                )
            else:
                # 默认通用模型信息
                model_info = ModelInfo(
                    name=model_name,
                    version="1.0.0",
                    description="生物序列表示学习模型",
                    tasks=["embedding", "classification"],
                    input_format={"sequences": "序列列表", "task_type": "任务类型"},
                    output_format={"results": "预测结果列表", "processing_time": "处理时间"},
                    max_sequence_length=512,
                    supported_species=None
                )
        
        return model_info
    
    def _infer_model_name(self) -> str:
        """
        从基础URL推断模型名称
        
        Returns:
            推断的模型名称
        """
        # 首先尝试从健康检查获取模型名称
        try:
            health_info = self.check_health()
            if hasattr(health_info, 'model') and health_info.model:
                return health_info.model.lower()
        except:
            # 健康检查失败，继续从URL推断
            pass
        
        # 从URL推断模型名称
        url_lower = self.base_url.lower()
        
        if "dnabert" in url_lower:
            return "dnabert2"
        elif "luca" in url_lower:
            return "lucaone"
        elif "rna-fm" in url_lower:
            return "rna-fm"
        elif "rna" in url_lower:
            return "rna-fm"  # 默认RNA模型
        elif "utr" in url_lower and "3" in url_lower:
            return "3utrbert"
        elif "utr" in url_lower:
            return "utr-lm"
        elif "codon" in url_lower:
            return "codonbert"
        elif "alphafold" in url_lower:
            return "alphafold"
        else:
            return "unknown"

    
    def predict(
        self,
        sequence: str,
        task_type: Union[str, TaskType],
        options: Optional[Dict] = None
    ) -> PredictionResult:
        """
        执行单个预测
        
        Args:
            sequence: 输入序列
            task_type: 任务类型
            options: 额外选项
            
        Returns:
            预测结果
        """
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        
        # 适配DNABERT2和LucaOne容器的API格式
        data = {
            "sequences": [sequence],
            "task_type": task_type
        }
        
        # 添加可选参数
        if options:
            if "max_sequence_length" in options:
                data["max_sequence_length"] = options["max_sequence_length"]
            if "batch_size" in options:
                data["batch_size"] = options["batch_size"]
        
        response = self._make_request('POST', '/predict', data)
        
        # 适配不同的响应格式
        if 'success' in response and not response.get('success', False):
            raise Exception(f"预测失败: {response.get('message', '未知错误')}")
        
        # 处理DNABERT2和LucaOne容器的实际返回格式
        if 'success' in response and response.get('success'):
            results = response.get('results', [])
            if results and len(results) > 0:
                result = results[0]
                # 对于嵌入任务，使用embedding字段作为prediction
                prediction = result.get('prediction')
                if task_type == "embedding" and 'embedding' in result:
                    prediction = result.get('embedding')
                
                return PredictionResult(
                    prediction=prediction,
                    confidence=result.get('confidence'),
                    processing_time=response.get('processing_time'),
                    sequence_length=result.get('sequence_length', len(sequence)),
                    details=result.get('details')
                )
        
        # 处理没有success字段的返回格式（可能是旧版本或不同实现）
        if 'results' in response:
            results = response.get('results', [])
            if results and len(results) > 0:
                result = results[0]
                # 对于嵌入任务，使用embedding字段作为prediction
                prediction = result.get('prediction')
                if task_type == "embedding" and 'embedding' in result:
                    prediction = result.get('embedding')
                
                return PredictionResult(
                    prediction=prediction,
                    confidence=result.get('confidence'),
                    processing_time=response.get('processing_time'),
                    sequence_length=result.get('sequence_length', len(sequence)),
                    details=result.get('details')
                )
        
        # 如果格式不匹配，尝试标准格式
        if 'data' in response:
            result_data = response.get('data', {})
            return PredictionResult(
                prediction=result_data.get('prediction'),
                confidence=result_data.get('confidence'),
                processing_time=result_data.get('processing_time'),
                sequence_length=result_data.get('sequence_length', len(sequence)),
                details=result_data.get('details')
            )
        
        raise Exception(f"无法解析预测结果: {response}")
    
    def predict_batch(
        self,
        sequences: List[str],
        task_type: Union[str, TaskType],
        options: Optional[Dict] = None
    ) -> List[PredictionResult]:
        """
        执行批量预测
        
        Args:
            sequences: 输入序列列表
            task_type: 任务类型
            options: 额外选项
            
        Returns:
            预测结果列表
        """
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        
        # 使用DNABERT2服务端支持的/predict端点，而不是/predict/batch
        data = {
            "sequences": sequences,
            "task_type": task_type
        }
        
        # 添加可选参数
        if options:
            if "max_sequence_length" in options:
                data["max_sequence_length"] = options["max_sequence_length"]
            if "batch_size" in options:
                data["batch_size"] = options["batch_size"]
        
        response = self._make_request('POST', '/predict', data)
        
        if 'success' in response and not response.get('success', False):
            raise Exception(f"批量预测失败: {response.get('message', '未知错误')}")
        
        # 处理DNABERT2服务端的响应格式
        api_results = response.get('results', [])
        prediction_results = []
        
        for i, result in enumerate(api_results):
            # 对于嵌入任务，使用embedding字段作为prediction
            prediction = result.get('prediction')
            if task_type == "embedding" and 'embedding' in result:
                prediction = result.get('embedding')
            # 对于分类任务，使用predicted_label或其他适当字段
            elif task_type == "classification":
                prediction = result.get('predicted_label')
            
            prediction_results.append(PredictionResult(
                prediction=prediction,
                confidence=result.get('scores'),  # 分类任务使用scores作为置信度
                sequence_length=len(sequences[i]),
                details=result
            ))
        
        return prediction_results
    
    def get_metrics(self) -> Dict:
        """
        获取模型指标
        
        Returns:
            模型指标数据
        """
        response = self._make_request('GET', '/metrics')
        
        if not response.get('success', False):
            raise Exception(f"获取指标失败: {response.get('message', '未知错误')}")
        
        return response.get('data', {})


class DNABERT2Client(ModelContainerClient):
    """DNABERT2模型客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        super().__init__(base_url, timeout)
    
    def get_embedding(
        self,
        sequence: str,
        embedding_dimension: int = 768
    ) -> PredictionResult:
        """
        获取DNA序列嵌入
        
        Args:
            sequence: DNA序列
            embedding_dimension: 嵌入维度
            
        Returns:
            嵌入向量
        """
        options = {"embedding_dimension": embedding_dimension}
        return self.predict(sequence, TaskType.EMBEDDING, options)
    
    def classify_sequence(
        self,
        sequence: str,
        return_confidence: bool = True
    ) -> PredictionResult:
        """
        DNA序列分类
        
        Args:
            sequence: DNA序列
            return_confidence: 是否返回置信度
            
        Returns:
            分类结果
        """
        options = {"return_confidence": return_confidence}
        return self.predict(sequence, TaskType.CLASSIFICATION, options)
    
    def predict_promoter(
        self,
        sequence: str,
        return_confidence: bool = True
    ) -> PredictionResult:
        """
        预测启动子活性
        
        Args:
            sequence: DNA序列
            return_confidence: 是否返回置信度
            
        Returns:
            预测结果
        """
        options = {"return_confidence": return_confidence}
        return self.predict(sequence, TaskType.PROMOTER_PREDICTION, options)
    
    def predict_promoter_batch(
        self,
        sequences: List[str],
        return_confidence: bool = True
    ) -> List[PredictionResult]:
        """
        批量预测启动子活性
        
        Args:
            sequences: DNA序列列表
            return_confidence: 是否返回置信度
            
        Returns:
            预测结果列表
        """
        options = {"return_confidence": return_confidence}
        return self.predict_batch(sequences, TaskType.PROMOTER_PREDICTION, options)
    
    def classify_batch(
        self,
        sequences: List[str],
        kwargs: Dict = None
    ) -> List[PredictionResult]:
        """
        批量分类DNA序列
        
        Args:
            sequences: DNA序列列表
            kwargs: 其他参数
            
        Returns:
            分类结果列表
        """
        if kwargs is None:
            kwargs = {}
        options = {"return_confidence": kwargs.get("return_confidence", True)}
        return self.predict_batch(sequences, TaskType.CLASSIFICATION, options)


class LucaOneClient(ModelContainerClient):
    """LucaOne模型客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8002", timeout: int = 30):
        super().__init__(base_url, timeout)
    
    def get_embedding(
        self,
        sequence: str,
        embedding_dimension: int = 768
    ) -> PredictionResult:
        """
        获取序列嵌入
        
        Args:
            sequence: DNA/RNA/蛋白质序列
            embedding_dimension: 嵌入维度
            
        Returns:
            嵌入向量
        """
        options = {"embedding_dimension": embedding_dimension}
        return self.predict(sequence, TaskType.EMBEDDING, options)
    
    def classify_sequence(
        self,
        sequence: str,
        return_confidence: bool = True
    ) -> PredictionResult:
        """
        序列分类
        
        Args:
            sequence: DNA/RNA/蛋白质序列
            return_confidence: 是否返回置信度
            
        Returns:
            分类结果
        """
        options = {"return_confidence": return_confidence}
        return self.predict(sequence, TaskType.CLASSIFICATION, options)
    
    def predict_property(
        self,
        sequence: str,
        return_confidence: bool = True
    ) -> PredictionResult:
        """
        属性预测
        
        Args:
            sequence: DNA/RNA/蛋白质序列
            return_confidence: 是否返回置信度
            
        Returns:
            属性预测结果
        """
        options = {"return_confidence": return_confidence}
        return self.predict(sequence, TaskType.PROPERTY_PREDICTION, options)


# 使用示例
if __name__ == "__main__":
    # DNABERT2示例
    dnabert2 = DNABERT2Client()
    
    # 检查健康状态
    try:
        health = dnabert2.check_health()
        print(f"DNABERT2健康状态: {health.status}")
    except Exception as e:
        print(f"健康检查失败: {e}")
    
    # 预测启动子活性
    try:
        sequence = "ATGCGTACGTAGCTAGCTAGCTAGC"
        result = dnabert2.predict_promoter(sequence)
        print(f"启动子活性预测: {result.prediction}, 置信度: {result.confidence}")
    except Exception as e:
        print(f"预测失败: {e}")
    
    # LucaOne示例
    lucaone = LucaOneClient()
    
    # 获取序列嵌入
    try:
        embedding_result = lucaone.get_embedding(sequence)
        print(f"序列嵌入维度: {len(embedding_result.prediction)}")
    except Exception as e:
        print(f"获取嵌入失败: {e}")