"""
生物序列嵌入任务工具
"""
import logging
from typing import Dict, List, Any, Optional, Union
from ..model_manager import ModelManager
from ..task_types import TaskType, SequenceType, DeviceType


class EmbeddingTool:
    """生物序列嵌入任务工具"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化嵌入工具
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager(config_path)
        
    def embed(
        self, 
        sequences: Union[str, List[str]], 
        model_name: Optional[str] = None,
        sequence_type: Optional[Union[str, SequenceType]] = None,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成生物序列的嵌入向量
        
        Args:
            sequences: 输入序列，可以是单个字符串或字符串列表
            model_name: 指定模型名称
            sequence_type: 序列类型
            device: 设备类型
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 嵌入结果
        """
        # 转换输入格式
        if isinstance(sequences, str):
            sequences = [sequences]
            
        # 转换序列类型
        if sequence_type and isinstance(sequence_type, str):
            try:
                sequence_type = SequenceType(sequence_type)
            except ValueError:
                return {"error": f"不支持的序列类型: {sequence_type}"}
                
        # 使用模型管理器进行嵌入
        result = self.model_manager.predict(
            sequences=sequences,
            task_type=TaskType.EMBEDDING,
            model_name=model_name,
            sequence_type=sequence_type,
            **kwargs
        )
        
        # 添加嵌入任务特定的元数据
        if "error" not in result:
            result["metadata"]["task"] = "embedding"
            # 计算嵌入向量的统计信息
            if "results" in result and result["results"]:
                embeddings = [r.get("embedding", []) for r in result["results"] if "embedding" in r]
                if embeddings:
                    # 计算嵌入维度
                    embedding_dim = len(embeddings[0])
                    result["metadata"]["embedding_dimension"] = embedding_dim
                    
        return result
        
    def get_best_model(self, sequence_type: Optional[Union[str, SequenceType]] = None) -> Optional[str]:
        """
        获取嵌入任务的最佳模型
        
        Args:
            sequence_type: 序列类型
            
        Returns:
            Optional[str]: 最佳模型名称
        """
        # 转换序列类型
        if sequence_type and isinstance(sequence_type, str):
            try:
                sequence_type = SequenceType(sequence_type)
            except ValueError:
                self.logger.error(f"不支持的序列类型: {sequence_type}")
                return None
                
        return self.model_manager.get_best_model(TaskType.EMBEDDING, sequence_type)
        
    def compare_embeddings(
        self, 
        sequences: List[str], 
        model_name: Optional[str] = None,
        sequence_type: Optional[Union[str, SequenceType]] = None,
        method: str = "cosine"
    ) -> Dict[str, Any]:
        """
        比较多个序列的嵌入向量
        
        Args:
            sequences: 输入序列列表
            model_name: 指定模型名称
            sequence_type: 序列类型
            method: 比较方法，支持"cosine"(余弦相似度)或"euclidean"(欧氏距离)
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        if len(sequences) < 2:
            return {"error": "至少需要两个序列进行比较"}
            
        # 生成嵌入向量
        embedding_result = self.embed(
            sequences=sequences,
            model_name=model_name,
            sequence_type=sequence_type
        )
        
        if "error" in embedding_result:
            return embedding_result
            
        # 提取嵌入向量
        embeddings = []
        for result in embedding_result.get("results", []):
            if "embedding" in result:
                embeddings.append(result["embedding"])
                
        if len(embeddings) < 2:
            return {"error": "无法生成有效的嵌入向量"}
            
        # 计算相似度/距离矩阵
        import numpy as np
        
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0 if method == "cosine" else 0.0
                else:
                    if method == "cosine":
                        # 计算余弦相似度
                        vec_i = np.array(embeddings[i])
                        vec_j = np.array(embeddings[j])
                        dot_product = np.dot(vec_i, vec_j)
                        norm_i = np.linalg.norm(vec_i)
                        norm_j = np.linalg.norm(vec_j)
                        similarity = dot_product / (norm_i * norm_j)
                        similarity_matrix[i][j] = similarity
                    elif method == "euclidean":
                        # 计算欧氏距离
                        vec_i = np.array(embeddings[i])
                        vec_j = np.array(embeddings[j])
                        distance = np.linalg.norm(vec_i - vec_j)
                        similarity_matrix[i][j] = distance
                    else:
                        return {"error": f"不支持的比较方法: {method}"}
                        
        # 找出最相似的序列对
        if method == "cosine":
            # 对于余弦相似度，找最大值
            max_sim = 0
            max_pair = (0, 0)
            for i in range(n):
                for j in range(i+1, n):
                    if similarity_matrix[i][j] > max_sim:
                        max_sim = similarity_matrix[i][j]
                        max_pair = (i, j)
            most_similar = {
                "pair": max_pair,
                "sequences": [sequences[max_pair[0]], sequences[max_pair[1]]],
                "similarity": max_sim
            }
        else:
            # 对于欧氏距离，找最小值
            min_dist = float('inf')
            min_pair = (0, 0)
            for i in range(n):
                for j in range(i+1, n):
                    if similarity_matrix[i][j] < min_dist:
                        min_dist = similarity_matrix[i][j]
                        min_pair = (i, j)
            most_similar = {
                "pair": min_pair,
                "sequences": [sequences[min_pair[0]], sequences[min_pair[1]]],
                "distance": min_dist
            }
            
        return {
            "sequences": sequences,
            "method": method,
            "similarity_matrix": similarity_matrix.tolist(),
            "most_similar": most_similar,
            "metadata": embedding_result.get("metadata", {})
        }