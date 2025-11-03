"""
任务特定工具，根据任务类型和序列类型自动选择合适的模型
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path

# 导入所有模型工具
from .lucaoneapp_tool import LucaOneAppTool
from .lucaonetasks_tool import LucaOneTasksTool
from .three_utrbert_tool import ThreeUTRBERTTool
from .alphafold_tool import AlphaFoldTool
from .codonbert_tool import CodonBERTTool
from .dnabert2_tool import DNABERT2Tool
from .rnafm_tool import RNAFMTool
from .utrlm_tool import UTRLMTool


class TaskSpecificTool:
    """
    任务特定工具类，根据任务类型和序列类型自动选择合适的模型
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化任务特定工具
        
        Args:
            config_path: 模型配置文件路径，默认为项目配置文件
        """
        if config_path is None:
            # 默认配置文件路径
            current_dir = Path(__file__).parent.parent.parent.parent
            config_path = os.path.join(current_dir, "config", "bio_models", "model_config.json")
        
        # 检查配置文件是否存在，如果不存在，尝试其他可能的路径
        if not os.path.exists(config_path):
            # 尝试从ToolUniverse根目录开始
            current_dir = Path(__file__).parent.parent.parent.parent.parent
            config_path = os.path.join(current_dir, "config", "bio_models", "model_config.json")
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化所有模型工具
        self.model_tools = {
            "lucaone": LucaOneAppTool(),  # 使用LucaOneApp进行嵌入
            "lucaoneapp": LucaOneAppTool(),
            "lucaonetasks": LucaOneTasksTool(),
            "3utrbert": ThreeUTRBERTTool(model_manager=self),
            "alphafold": AlphaFoldTool(),
            "codonbert": CodonBERTTool(),
            "dnabert_2": DNABERT2Tool(),
            "rna-fm": RNAFMTool(),
            "utr-lm": UTRLMTool(model_manager=self)
        }
        
        # 记录已加载的模型
        self.loaded_models = set()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载模型配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"无法加载配置文件 {self.config_path}: {str(e)}")
    
    def _select_model(self, task_type: str, sequence_type: str) -> str:
        """
        根据任务类型和序列类型选择合适的模型
        
        Args:
            task_type: 任务类型
            sequence_type: 序列类型 (DNA, RNA, protein)
            
        Returns:
            选择的模型名称
        """
        task_mapping = self.config.get("task_mapping", {})
        
        # 首先检查是否有针对特定序列类型的模型
        if task_type in task_mapping and sequence_type in task_mapping[task_type]:
            return task_mapping[task_type][sequence_type]
        
        # 如果没有，使用默认模型
        if task_type in task_mapping and "default_model" in task_mapping[task_type]:
            return task_mapping[task_type]["default_model"]
        
        raise ValueError(f"无法找到适合任务类型 {task_type} 和序列类型 {sequence_type} 的模型")
    
    def _ensure_model_loaded(self, model_name: str):
        """确保模型已加载"""
        if model_name not in self.loaded_models:
            model_config = self.config["models"][model_name]
            model_path = model_config["model_path"]
            
            try:
                self.model_tools[model_name].load_model(model_path)
                self.loaded_models.add(model_name)
                self.logger.info(f"成功加载模型: {model_name}")
            except Exception as e:
                self.logger.error(f"加载模型 {model_name} 失败: {str(e)}")
                raise
    
    def get_embedding(self, sequences: Union[str, List[str]], sequence_type: str) -> Dict[str, Any]:
        """
        生成序列嵌入
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            
        Returns:
            嵌入结果字典
        """
        model_name = self._select_model("embedding", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].get_embedding(sequences)
    
    def classify(self, sequences: Union[str, List[str]], sequence_type: str, 
                 labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        序列分类
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            labels: 可选的标签列表
            
        Returns:
            分类结果字典
        """
        model_name = self._select_model("classification", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].classify(sequences, labels)
    
    def predict_property(self, sequences: Union[str, List[str]], sequence_type: str,
                        property_name: str) -> Dict[str, Any]:
        """
        预测序列属性
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            property_name: 属性名称
            
        Returns:
            属性预测结果字典
        """
        model_name = self._select_model("property_prediction", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].predict_property(sequences, property_name)
    
    def predict_structure(self, sequences: Union[str, List[str]], sequence_type: str) -> Dict[str, Any]:
        """
        预测序列结构
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (RNA, protein)
            
        Returns:
            结构预测结果字典
        """
        model_name = self._select_model("structure_prediction", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].predict_structure(sequences)
    
    def predict_interaction(self, sequences: Union[str, List[str]], sequence_type: str) -> Dict[str, Any]:
        """
        预测序列相互作用
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            
        Returns:
            相互作用预测结果字典
        """
        model_name = self._select_model("interaction", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].predict_interaction(sequences)
    
    def annotate(self, sequences: Union[str, List[str]], sequence_type: str) -> Dict[str, Any]:
        """
        序列注释
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            
        Returns:
            注释结果字典
        """
        model_name = self._select_model("annotation", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].annotate(sequences)
    
    def generate(self, sequences: Union[str, List[str]], sequence_type: str,
                generation_type: str = "optimization") -> Dict[str, Any]:
        """
        序列生成
        
        Args:
            sequences: 序列或序列列表
            sequence_type: 序列类型 (DNA, RNA, protein)
            generation_type: 生成类型 (optimization, design, etc.)
            
        Returns:
            生成结果字典
        """
        model_name = self._select_model("generation", sequence_type)
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].generate(sequences, generation_type)
    
    def predict_translation(self, sequences: Union[str, List[str]]) -> Dict[str, Any]:
        """
        预测翻译效率
        
        Args:
            sequences: RNA序列或序列列表
            
        Returns:
            翻译效率预测结果字典
        """
        model_name = self._select_model("translation", "RNA")
        self._ensure_model_loaded(model_name)
        
        return self.model_tools[model_name].predict_translation(sequences)
    
    def get_supported_tasks(self) -> Dict[str, List[str]]:
        """
        获取支持的任务类型
        
        Returns:
            支持的任务类型字典
        """
        return self.config.get("task_mapping", {})
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称，如果为None则返回所有模型信息
            
        Returns:
            模型信息字典
        """
        if model_name is None:
            return self.config.get("models", {})
        
        if model_name not in self.config.get("models", {}):
            raise ValueError(f"未找到模型: {model_name}")
        
        return self.config["models"][model_name]
    
    def unload_model(self, model_name: str):
        """
        卸载模型
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.loaded_models:
            self.model_tools[model_name].unload_model()
            self.loaded_models.remove(model_name)
            self.logger.info(f"已卸载模型: {model_name}")
    
    def unload_all_models(self):
        """卸载所有已加载的模型"""
        for model_name in list(self.loaded_models):
            self.unload_model(model_name)
    
    def get_loaded_models(self) -> List[str]:
        """
        获取已加载的模型列表
        
        Returns:
            已加载的模型名称列表
        """
        return list(self.loaded_models)
    
    def __del__(self):
        """析构函数，确保所有模型都被卸载"""
        self.unload_all_models()