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
        
        if not response.get('success', False):
            raise Exception(f"健康检查失败: {response.get('message', '未知错误')}")
        
        data = response.get('data', {})
        return HealthStatus(
            status=data.get('status'),
            model=data.get('model'),
            version=data.get('version'),
            uptime=data.get('uptime', 0),
            gpu_available=data.get('gpu_available', False),
            memory_usage=data.get('memory_usage', '0')
        )
    
    def get_model_info(self) -> ModelInfo:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        response = self._make_request('GET', '/model/info')
        
        if not response.get('success', False):
            raise Exception(f"获取模型信息失败: {response.get('message', '未知错误')}")
        
        data = response.get('data', {})
        return ModelInfo(
            name=data.get('name'),
            version=data.get('version'),
            description=data.get('description'),
            tasks=data.get('tasks', []),
            input_format=data.get('input_format', {}),
            output_format=data.get('output_format', {}),
            max_sequence_length=data.get('max_sequence_length', 0),
            supported_species=data.get('supported_species')
        )
    
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
        
        data = {
            "sequence": sequence,
            "task_type": task_type,
            "options": options or {}
        }
        
        response = self._make_request('POST', '/predict', data)
        
        if not response.get('success', False):
            raise Exception(f"预测失败: {response.get('message', '未知错误')}")
        
        result_data = response.get('data', {})
        return PredictionResult(
            prediction=result_data.get('prediction'),
            confidence=result_data.get('confidence'),
            processing_time=result_data.get('processing_time'),
            sequence_length=result_data.get('sequence_length'),
            details=result_data.get('details')
        )
    
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
        
        data = {
            "sequences": sequences,
            "task_type": task_type,
            "options": options or {}
        }
        
        response = self._make_request('POST', '/predict/batch', data)
        
        if not response.get('success', False):
            raise Exception(f"批量预测失败: {response.get('message', '未知错误')}")
        
        result_data = response.get('data', {})
        predictions = result_data.get('predictions', [])
        
        results = []
        for pred in predictions:
            results.append(PredictionResult(
                prediction=pred.get('prediction'),
                confidence=pred.get('confidence'),
                sequence_length=len(pred.get('sequence', '')),
                details=pred.get('details')
            ))
        
        return results
    
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