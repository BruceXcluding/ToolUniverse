"""
生物序列模型特定工具使用示例
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.tooluniverse.bio_models.tools.lucaone_tool import LucaOneTool
from src.tooluniverse.bio_models.tools.embedding_tool import EmbeddingTool
from src.tooluniverse.bio_models.task_types import TaskType, SequenceType


def main():
    """主函数"""
    print("=== 生物序列模型特定工具使用示例 ===\n")
    
    # 示例序列
    dna_sequence = "ATCGATCGATCGATCG"
    rna_sequence = "AUCGAUCGAUCGAUCG"
    protein_sequence = "MKVILLFVLAVFATSASVAP"
    
    # 1. 使用LucaOne模型工具
    print("1. 使用LucaOne模型工具:")
    lucaone_tool = LucaOneTool()
    
    # 1.1 获取模型信息
    print("   1.1 LucaOne模型信息:")
    model_info = lucaone_tool.get_model_info()
    print(f"   模型名称: {model_info.get('name')}")
    print(f"   支持的任务: {', '.join(model_info.get('supported_tasks', []))}")
    print(f"   支持的序列类型: {', '.join(model_info.get('supported_sequences', []))}")
    print()
    
    # 1.2 加载模型
    print("   1.2 加载LucaOne模型:")
    success = lucaone_tool.load_model()
    print(f"   加载结果: {'成功' if success else '失败'}")
    print()
    
    # 1.3 使用LucaOne进行不同任务
    print("   1.3 使用LucaOne进行不同任务:")
    
    # 嵌入任务
    print("   - DNA序列嵌入:")
    result = lucaone_tool.analyze(
        sequences=dna_sequence,
        task_type=TaskType.EMBEDDING
    )
    print(f"     结果: {result}")
    
    # 分类任务
    print("   - 蛋白质序列分类:")
    result = lucaone_tool.analyze(
        sequences=protein_sequence,
        task_type=TaskType.CLASSIFICATION
    )
    print(f"     结果: {result}")
    
    # 属性预测
    print("   - RNA序列属性预测:")
    result = lucaone_tool.analyze(
        sequences=rna_sequence,
        task_type=TaskType.PROPERTY_PREDICTION
    )
    print(f"     结果: {result}")
    print()
    
    # 2. 使用嵌入任务特定工具
    print("2. 使用嵌入任务特定工具:")
    embedding_tool = EmbeddingTool()
    
    # 2.1 获取最佳模型
    print("   2.1 不同序列类型的最佳嵌入模型:")
    for seq_type in [SequenceType.DNA, SequenceType.RNA, SequenceType.protein]:
        best_model = embedding_tool.get_best_model(seq_type)
        print(f"   {seq_type.value}: {best_model}")
    print()
    
    # 2.2 生成嵌入向量
    print("   2.2 生成嵌入向量:")
    result = embedding_tool.embed(
        sequences=protein_sequence,
        sequence_type=SequenceType.protein
    )
    print(f"   结果: {result}")
    print()
    
    # 2.3 比较多个序列的嵌入向量
    print("   2.3 比较多个序列的嵌入向量:")
    sequences = [dna_sequence, rna_sequence, protein_sequence]
    result = embedding_tool.compare_embeddings(
        sequences=sequences,
        method="cosine"
    )
    print(f"   相似度矩阵: {result.get('similarity_matrix')}")
    print(f"   最相似的序列对: {result.get('most_similar')}")
    print()
    
    # 3. 卸载模型
    print("3. 卸载模型:")
    success = lucaone_tool.unload_model()
    print(f"   卸载结果: {'成功' if success else '失败'}")
    print()


if __name__ == "__main__":
    main()