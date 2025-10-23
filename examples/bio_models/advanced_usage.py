#!/usr/bin/env python3
"""
高级用法示例 - BioModels模块

本示例展示BioModels模块的高级功能，包括:
1. 自定义模型集成
2. 模型性能调优
3. 工作流管道构建
4. 结果可视化
5. 模型组合使用
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from tooluniverse.bio_models import BioSequenceAnalysisTool
from tooluniverse.bio_models.models.base_model import BaseModel
from tooluniverse.bio_models.task_types import TaskType, SequenceType


class CustomProteinModel(BaseModel):
    """自定义蛋白质模型示例"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.model_name = "custom_protein_model"
        self.supported_tasks = [TaskType.EMBEDDING, TaskType.CLASSIFICATION]
        self.supported_sequences = [SequenceType.PROTEIN]
        self._is_loaded = False
        
    def load(self) -> bool:
        """加载模型"""
        try:
            print(f"正在加载自定义蛋白质模型: {self.model_path}")
            # 模拟模型加载过程
            time.sleep(1)
            self._is_loaded = True
            print("自定义蛋白质模型加载成功")
            return True
        except Exception as e:
            print(f"自定义蛋白质模型加载失败: {str(e)}")
            return False
    
    def unload(self) -> bool:
        """卸载模型"""
        try:
            print("正在卸载自定义蛋白质模型")
            self._is_loaded = False
            print("自定义蛋白质模型卸载成功")
            return True
        except Exception as e:
            print(f"自定义蛋白质模型卸载失败: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "type": "custom_protein_model",
            "supported_tasks": [task.value for task in self.supported_tasks],
            "supported_sequences": [seq.value for seq in self.supported_sequences],
            "is_loaded": self._is_loaded,
            "description": "自定义蛋白质分析模型，用于演示模型集成"
        }
    
    def process(self, task_type: TaskType, sequences: List[str], **kwargs) -> Dict[str, Any]:
        """处理序列"""
        if not self._is_loaded:
            raise ValueError("模型未加载，请先调用load()方法")
        
        if task_type == TaskType.EMBEDDING:
            return self._generate_embeddings(sequences)
        elif task_type == TaskType.CLASSIFICATION:
            return self._classify_sequences(sequences, **kwargs)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    
    def _generate_embeddings(self, sequences: List[str]) -> Dict[str, Any]:
        """生成嵌入向量"""
        print(f"正在为{len(sequences)}个蛋白质序列生成嵌入向量...")
        time.sleep(0.5)
        
        # 模拟生成嵌入向量
        embeddings = []
        for seq in sequences:
            # 创建基于序列长度的简单嵌入向量
            embedding = np.random.rand(128).tolist()
            embeddings.append(embedding)
        
        return {
            "task": "embedding",
            "embeddings": embeddings,
            "embedding_dim": 128,
            "processing_time": 0.5
        }
    
    def _classify_sequences(self, sequences: List[str], **kwargs) -> Dict[str, Any]:
        """分类序列"""
        print(f"正在分类{len(sequences)}个蛋白质序列...")
        time.sleep(0.3)
        
        # 模拟分类结果
        classes = ["enzyme", "structural", "regulatory", "transport"]
        results = []
        
        for seq in sequences:
            # 基于序列长度随机分配类别
            class_idx = len(seq) % len(classes)
            confidence = 0.6 + (len(seq) % 40) / 100.0
            results.append({
                "sequence": seq,
                "class": classes[class_idx],
                "confidence": confidence
            })
        
        return {
            "task": "classification",
            "results": results,
            "processing_time": 0.3
        }


def register_custom_model(tool: BioSequenceAnalysisTool, model_config: Dict[str, Any]) -> bool:
    """注册自定义模型"""
    try:
        print("正在注册自定义模型...")
        custom_model = CustomProteinModel(
            model_path=model_config.get("model_path", ""),
            config=model_config
        )
        
        # 注册模型
        success = tool.register_model("custom_protein", custom_model)
        if success:
            print("自定义模型注册成功")
            return True
        else:
            print("自定义模型注册失败")
            return False
    except Exception as e:
        print(f"注册自定义模型时出错: {str(e)}")
        return False


def create_analysis_pipeline(tool: BioSequenceAnalysisTool) -> Dict[str, Any]:
    """创建分析管道"""
    print("\n=== 创建分析管道 ===")
    
    # 示例蛋白质序列
    protein_sequences = [
        "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSHG",
        "GATLPVLAAELSGQNLKELGAFATKRGVVTYELDPLRDIILTVGAP",
        "MTKLLILAVAVFASVTLASAQAGAVAAAGGKSTLLRLIAEVNRHLGDNVT"
    ]
    
    # 步骤1: 生成嵌入向量
    print("\n步骤1: 生成嵌入向量")
    embedding_result = tool.analyze_sequences(
        sequences=protein_sequences,
        task_type="embedding",
        sequence_type="protein",
        model_name="lucaone"
    )
    
    if embedding_result["success"]:
        embeddings = embedding_result["result"]["embeddings"]
        print(f"成功生成{len(embeddings)}个嵌入向量，维度: {len(embeddings[0])}")
    else:
        print(f"嵌入向量生成失败: {embedding_result['error']}")
        return {"success": False, "error": embedding_result.get("error")}
    
    # 步骤2: 序列分类
    print("\n步骤2: 序列分类")
    classification_result = tool.analyze_sequences(
        sequences=protein_sequences,
        task_type="classification",
        sequence_type="protein",
        model_name="lucaone"
    )
    
    if classification_result["success"]:
        classifications = classification_result["result"]["results"]
        print(f"成功分类{len(classifications)}个序列")
    else:
        print(f"序列分类失败: {classification_result['error']}")
        return {"success": False, "error": classification_result.get("error")}
    
    # 步骤3: 组合结果
    print("\n步骤3: 组合分析结果")
    combined_results = []
    
    for i, seq in enumerate(protein_sequences):
        result = {
            "sequence": seq,
            "embedding": embeddings[i] if i < len(embeddings) else None,
            "classification": classifications[i] if i < len(classifications) else None
        }
        combined_results.append(result)
    
    return {
        "success": True,
        "pipeline_results": combined_results,
        "summary": {
            "total_sequences": len(protein_sequences),
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "classification_classes": list(set(r["class"] for r in classifications))
        }
    }


def visualize_embeddings(embeddings: List[List[float]], labels: List[str], output_path: str = None):
    """可视化嵌入向量"""
    print("\n=== 可视化嵌入向量 ===")
    
    try:
        # 转换为numpy数组
        embeddings_array = np.array(embeddings)
        
        # 使用PCA降维到2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # 创建可视化
        plt.figure(figsize=(10, 8))
        
        # 为每个类别创建散点图
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(
                embeddings_2d[indices, 0], 
                embeddings_2d[indices, 1],
                c=[colors[i]], 
                label=label, 
                alpha=0.7
            )
        
        plt.title('蛋白质序列嵌入向量可视化 (PCA降维)')
        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {output_path}")
        else:
            plt.show()
        
        return True
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        return False


def compare_models(tool: BioSequenceAnalysisTool, sequences: List[str], models: List[str]) -> Dict[str, Any]:
    """比较不同模型的性能"""
    print("\n=== 模型性能比较 ===")
    
    comparison_results = {}
    
    for model_name in models:
        print(f"\n测试模型: {model_name}")
        
        # 检查模型是否支持蛋白质序列
        model_info = tool.get_model_info(model_name)
        if not model_info["success"] or SequenceType.PROTEIN.value not in model_info["result"]["supported_sequences"]:
            print(f"模型 {model_name} 不支持蛋白质序列，跳过")
            continue
        
        # 加载模型
        load_result = tool.load_model(model_name)
        if not load_result["success"]:
            print(f"加载模型 {model_name} 失败: {load_result['error']}")
            continue
        
        # 测试嵌入生成
        start_time = time.time()
        embedding_result = tool.analyze_sequences(
            sequences=sequences,
            task_type="embedding",
            sequence_type="protein",
            model_name=model_name
        )
        embedding_time = time.time() - start_time
        
        if embedding_result["success"]:
            embeddings = embedding_result["result"]["embeddings"]
            embedding_dim = len(embeddings[0]) if embeddings else 0
            
            comparison_results[model_name] = {
                "success": True,
                "embedding_time": embedding_time,
                "embedding_dim": embedding_dim,
                "avg_time_per_sequence": embedding_time / len(sequences),
                "model_info": model_info["result"]
            }
            
            print(f"  嵌入生成成功")
            print(f"  处理时间: {embedding_time:.2f}秒")
            print(f"  平均每个序列: {embedding_time/len(sequences):.2f}秒")
            print(f"  嵌入维度: {embedding_dim}")
        else:
            print(f"  嵌入生成失败: {embedding_result['error']}")
            comparison_results[model_name] = {
                "success": False,
                "error": embedding_result["error"]
            }
        
        # 卸载模型以释放内存
        tool.unload_model(model_name)
    
    return {
        "comparison_results": comparison_results,
        "summary": {
            "total_models_tested": len(models),
            "successful_models": sum(1 for r in comparison_results.values() if r["success"]),
            "sequences_tested": len(sequences)
        }
    }


def save_results_to_file(results: Dict[str, Any], output_path: str):
    """将结果保存到文件"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
        return False


def main():
    """主函数"""
    print("=== BioModels高级用法示例 ===")
    
    # 初始化工具
    tool = BioSequenceAnalysisTool()
    
    # 示例1: 注册自定义模型
    print("\n=== 示例1: 注册自定义模型 ===")
    custom_model_config = {
        "model_path": "/path/to/custom/model",
        "description": "自定义蛋白质分析模型"
    }
    register_custom_model(tool, custom_model_config)
    
    # 列出所有可用模型
    models = tool.list_models()
    print(f"可用模型: {models['result']}")
    
    # 示例2: 创建分析管道
    pipeline_results = create_analysis_pipeline(tool)
    
    # 示例3: 可视化嵌入向量
    if pipeline_results["success"]:
        embeddings = [r["embedding"] for r in pipeline_results["pipeline_results"] if r["embedding"]]
        labels = [r["classification"]["class"] for r in pipeline_results["pipeline_results"] if r["classification"]]
        
        if embeddings and labels:
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # 可视化嵌入向量
            viz_path = os.path.join(output_dir, "embeddings_visualization.png")
            visualize_embeddings(embeddings, labels, viz_path)
    
    # 示例4: 比较不同模型性能
    protein_sequences = [
        "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSHG",
        "GATLPVLAAELSGQNLKELGAFATKRGVVTYELDPLRDIILTVGAP",
        "MTKLLILAVAVFASVTLASAQAGAVAAAGGKSTLLRLIAEVNRHLGDNVT",
        "AVVTYPAPAGEGQVNLFVTDVQKASLVAVDGNKDIIVTLGAGVTG",
        "GVLTDFFKVDLGVKGKVNDEVVVGGMVTLYAAKQLAGKLDIGL"
    ]
    
    # 选择要比较的模型
    models_to_compare = ["lucaone", "protbert", "esm2_t6"]
    
    comparison_results = compare_models(tool, protein_sequences, models_to_compare)
    
    # 保存比较结果
    if comparison_results:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, "model_comparison.json")
        save_results_to_file(comparison_results, results_path)
    
    # 示例5: 使用自定义模型
    print("\n=== 示例5: 使用自定义模型 ===")
    custom_model_result = tool.analyze_sequences(
        sequences=protein_sequences[:2],
        task_type="embedding",
        sequence_type="protein",
        model_name="custom_protein"
    )
    
    if custom_model_result["success"]:
        print("自定义模型使用成功")
        embeddings = custom_model_result["result"]["embeddings"]
        print(f"生成了{len(embeddings)}个嵌入向量，维度: {len(embeddings[0])}")
    else:
        print(f"自定义模型使用失败: {custom_model_result['error']}")
    
    print("\n=== 高级用法示例完成 ===")


if __name__ == "__main__":
    main()