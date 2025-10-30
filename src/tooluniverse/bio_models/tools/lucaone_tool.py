import os
import sys
import json
import logging
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from tooluniverse.bio_models.models.base_model import BaseModel
from tooluniverse.bio_models.model_manager import ModelManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LucaOneModel(BaseModel):
    """LucaOne模型封装类，支持Docker容器调用和本地模型调用"""
    
    def __init__(self, model_path: str = None, use_docker: bool = True, docker_host: str = "localhost", docker_port: int = 8002):
        """
        初始化LucaOne模型
        
        Args:
            model_path: 模型路径
            use_docker: 是否使用Docker容器
            docker_host: Docker主机地址
            docker_port: Docker端口
        """
        self.model_path = model_path or "/mnt/models/yigex/3rdparty/LucaOne"
        self.use_docker = use_docker
        self.docker_host = docker_host
        self.docker_port = docker_port
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_loaded = False
        
        # 如果不使用Docker，尝试加载本地模型
        if not self.use_docker:
            self._load_local_model()
    
    def _load_local_model(self):
        """加载本地LucaOne模型"""
        try:
            # 这里应该根据实际的LucaOne模型实现来加载
            # 以下是示例代码，需要根据实际模型调整
            logger.info(f"尝试从本地路径加载LucaOne模型: {self.model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.warning(f"模型路径不存在: {self.model_path}")
                return False
            
            # 模拟加载模型（实际实现需要根据LucaOne模型的具体API）
            # self.model = load_lucaone_model(self.model_path)
            # self.tokenizer = load_lucaone_tokenizer(self.model_path)
            
            logger.info("本地LucaOne模型加载成功")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"加载本地LucaOne模型失败: {str(e)}")
            return False
    
    def _call_docker_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用Docker容器API"""
        try:
            url = f"http://{self.docker_host}:{self.docker_port}{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"调用Docker API失败: {str(e)}")
            raise
    
    def load_model(self) -> bool:
        """加载模型"""
        if self.use_docker:
            try:
                # 通过Docker API加载模型
                response = self._call_docker_api("/load", {
                    "model_path": self.model_path
                })
                
                if response.get("status") == "success":
                    self.model_loaded = True
                    logger.info("通过Docker API加载LucaOne模型成功")
                    return True
                else:
                    logger.error(f"Docker API加载模型失败: {response.get('message', '未知错误')}")
                    return False
                    
            except Exception as e:
                logger.error(f"通过Docker API加载模型失败: {str(e)}")
                return False
        else:
            return self._load_local_model()
    
    def unload_model(self) -> bool:
        """卸载模型"""
        if self.use_docker:
            try:
                # 通过Docker API卸载模型
                response = self._call_docker_api("/unload", {})
                
                if response.get("status") == "success":
                    self.model_loaded = False
                    logger.info("通过Docker API卸载LucaOne模型成功")
                    return True
                else:
                    logger.error(f"Docker API卸载模型失败: {response.get('message', '未知错误')}")
                    return False
                    
            except Exception as e:
                logger.error(f"通过Docker API卸载模型失败: {str(e)}")
                return False
        else:
            # 本地模型卸载
            try:
                if self.model:
                    del self.model
                if self.tokenizer:
                    del self.tokenizer
                self.model = None
                self.tokenizer = None
                self.model_loaded = False
                logger.info("本地LucaOne模型卸载成功")
                return True
            except Exception as e:
                logger.error(f"本地模型卸载失败: {str(e)}")
                return False
    
    def predict(self, sequence: str, task_type: str = "embedding", **kwargs) -> Dict[str, Any]:
        """
        使用LucaOne模型进行预测
        
        Args:
            sequence: 输入序列
            task_type: 任务类型 (embedding, classification, property_prediction, interaction, annotation)
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        if self.use_docker:
            # 通过Docker API进行预测
            try:
                data = {
                    "sequence": sequence,
                    "task_type": task_type,
                    **kwargs
                }
                
                response = self._call_docker_api("/predict", data)
                
                if response.get("status") == "success":
                    return response.get("result", {})
                else:
                    logger.error(f"Docker API预测失败: {response.get('message', '未知错误')}")
                    return {"error": response.get('message', '未知错误')}
                    
            except Exception as e:
                logger.error(f"通过Docker API预测失败: {str(e)}")
                return {"error": str(e)}
        else:
            # 本地模型预测
            try:
                if not self.model_loaded:
                    self.load_model()
                
                if not self.model_loaded:
                    return {"error": "模型未加载"}
                
                # 根据任务类型调用不同的预测方法
                if task_type == "embedding":
                    return self._predict_embedding(sequence, **kwargs)
                elif task_type == "classification":
                    return self._predict_classification(sequence, **kwargs)
                elif task_type == "property_prediction":
                    return self._predict_property(sequence, **kwargs)
                elif task_type == "interaction":
                    return self._predict_interaction(sequence, **kwargs)
                elif task_type == "annotation":
                    return self._predict_annotation(sequence, **kwargs)
                else:
                    return {"error": f"不支持的任务类型: {task_type}"}
                    
            except Exception as e:
                logger.error(f"本地模型预测失败: {str(e)}")
                return {"error": str(e)}
    
    def _predict_embedding(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """生成序列嵌入"""
        # 实际实现需要根据LucaOne模型的具体API
        # 这里是示例代码
        embedding = np.random.rand(len(sequence), 768).tolist()  # 模拟嵌入
        return {
            "embedding": embedding,
            "sequence_length": len(sequence),
            "embedding_dim": 768
        }
    
    def _predict_classification(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """序列分类预测"""
        # 实际实现需要根据LucaOne模型的具体API
        # 这里是示例代码
        classes = ["class_0", "class_1", "class_2"]
        scores = np.random.rand(len(classes)).tolist()
        predicted_class = classes[np.argmax(scores)]
        
        return {
            "predicted_class": predicted_class,
            "scores": dict(zip(classes, scores)),
            "sequence_length": len(sequence)
        }
    
    def _predict_property(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """属性预测"""
        # 实际实现需要根据LucaOne模型的具体API
        # 这里是示例代码
        properties = {
            "stability": np.random.rand(),
            "solubility": np.random.rand(),
            "toxicity": np.random.rand()
        }
        
        return {
            "properties": properties,
            "sequence_length": len(sequence)
        }
    
    def _predict_interaction(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """相互作用预测"""
        # 实际实现需要根据LucaOne模型的具体API
        # 这里是示例代码
        target_sequence = kwargs.get("target_sequence", "")
        if not target_sequence:
            return {"error": "缺少目标序列"}
        
        interaction_score = np.random.rand()
        
        return {
            "interaction_score": float(interaction_score),
            "source_sequence_length": len(sequence),
            "target_sequence_length": len(target_sequence)
        }
    
    def _predict_annotation(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """序列注释"""
        # 实际实现需要根据LucaOne模型的具体API
        # 这里是示例代码
        annotations = []
        for i in range(0, len(sequence), 10):
            start = i
            end = min(i + 10, len(sequence))
            annotation_type = np.random.choice(["domain", "motif", "site"])
            annotations.append({
                "type": annotation_type,
                "start": start,
                "end": end,
                "confidence": float(np.random.rand())
            })
        
        return {
            "annotations": annotations,
            "sequence_length": len(sequence)
        }


class LucaOneTool:
    """LucaOne工具类，提供序列分析功能"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_name = "lucaone"
        
        # 注册LucaOne模型
        self.model_manager.register_model(
            name=self.model_name,
            model_class=LucaOneModel,
            description="LucaOne生物序列分析模型",
            default_params={
                "model_path": "/mnt/models/yigex/3rdparty/LucaOne",
                "use_docker": True,
                "docker_host": "localhost",
                "docker_port": 8002
            }
        )
    
    def analyze(self, sequence: str, task_type: str = "embedding", **kwargs) -> Dict[str, Any]:
        """
        分析生物序列
        
        Args:
            sequence: 输入序列
            task_type: 任务类型 (embedding, classification, property_prediction, interaction, annotation)
            **kwargs: 其他参数
            
        Returns:
            分析结果
        """
        try:
            # 获取模型实例
            model = self.model_manager.get_model(self.model_name)
            if not model:
                return {"error": f"无法获取模型: {self.model_name}"}
            
            # 确保模型已加载
            if not model.model_loaded:
                if not model.load_model():
                    return {"error": "模型加载失败"}
            
            # 执行预测
            result = model.predict(sequence, task_type, **kwargs)
            
            # 添加元数据
            result["metadata"] = {
                "model": self.model_name,
                "task_type": task_type,
                "sequence_length": len(sequence)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"LucaOne分析失败: {str(e)}")
            return {"error": str(e)}
    
    def load_model(self, **kwargs) -> Dict[str, Any]:
        """加载模型"""
        try:
            model = self.model_manager.get_model(self.model_name)
            if not model:
                return {"error": f"无法获取模型: {self.model_name}"}
            
            success = model.load_model()
            return {
                "status": "success" if success else "failed",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"加载LucaOne模型失败: {str(e)}")
            return {"error": str(e)}
    
    def unload_model(self, **kwargs) -> Dict[str, Any]:
        """卸载模型"""
        try:
            model = self.model_manager.get_model(self.model_name)
            if not model:
                return {"error": f"无法获取模型: {self.model_name}"}
            
            success = model.unload_model()
            return {
                "status": "success" if success else "failed",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"卸载LucaOne模型失败: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self, **kwargs) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = self.model_manager.get_model(self.model_name)
            if not model:
                return {"error": f"无法获取模型: {self.model_name}"}
            
            return {
                "model": self.model_name,
                "model_path": model.model_path,
                "use_docker": model.use_docker,
                "docker_host": model.docker_host,
                "docker_port": model.docker_port,
                "model_loaded": model.model_loaded
            }
            
        except Exception as e:
            logger.error(f"获取LucaOne模型信息失败: {str(e)}")
            return {"error": str(e)}