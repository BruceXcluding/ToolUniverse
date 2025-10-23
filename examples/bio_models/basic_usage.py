"""
生物序列模型基本使用示例
本示例展示如何使用ToolUniverse生物序列模型进行基本操作
"""
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from src.tooluniverse.bio_models.task_types import TaskType, SequenceType


def main():
    """主函数 - 基本使用示例"""
    print("=== ToolUniverse生物序列模型基本使用示例 ===\n")
    
    # 1. 初始化工具
    print("1. 初始化生物序列分析工具")
    tool = BioSequenceAnalysisTool()
    print("   ✓ 工具初始化成功\n")
    
    # 2. 准备示例序列
    print("2. 准备示例序列")
    dna_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
    rna_sequence = "AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG"
    protein_sequence = "MKVILLFVLAVFATSASVAPSKQIKVALL"
    
    print(f"   DNA序列: {dna_sequence[:20]}...")
    print(f"   RNA序列: {rna_sequence[:20]}...")
    print(f"   蛋白质序列: {protein_sequence[:20]}...\n")
    
    # 3. 列出可用模型
    print("3. 列出可用模型")
    models = tool.list_models()
    print(f"   共有 {len(models)} 个可用模型:")
    for i, model in enumerate(models[:5]):  # 只显示前5个
        print(f"   {i+1}. {model}")
    if len(models) > 5:
        print(f"   ... 还有 {len(models)-5} 个模型")
    print()
    
    # 4. 基本序列分析 - 嵌入
    print("4. 基本序列分析 - 生成嵌入向量")
    print("   正在为DNA序列生成嵌入向量...")
    result = tool.analyze(
        sequences=dna_sequence,
        task_type=TaskType.EMBEDDING,
        sequence_type=SequenceType.DNA
    )
    
    if result.get('success'):
        embeddings = result.get('results', [])
        if embeddings:
            embedding = embeddings[0].get('embedding', [])
            print(f"   ✓ DNA序列嵌入向量生成成功")
            print(f"   向量维度: {len(embedding)}")
            print(f"   向量前5个值: {embedding[:5]}")
    else:
        print(f"   ✗ 嵌入向量生成失败: {result.get('error', '未知错误')}")
    print()
    
    # 5. 基本序列分析 - 分类
    print("5. 基本序列分析 - 序列分类")
    print("   正在对蛋白质序列进行分类...")
    result = tool.analyze(
        sequences=protein_sequence,
        task_type=TaskType.CLASSIFICATION,
        sequence_type=SequenceType.protein
    )
    
    if result.get('success'):
        classifications = result.get('results', [])
        if classifications:
            classification = classifications[0]
            print(f"   ✓ 蛋白质序列分类成功")
            print(f"   预测类别: {classification.get('predicted_class', '未知')}")
            print(f"   置信度: {classification.get('confidence', 0):.2f}")
    else:
        print(f"   ✗ 序列分类失败: {result.get('error', '未知错误')}")
    print()
    
    # 6. 获取模型信息
    print("6. 获取模型信息")
    if models:
        model_name = models[0]
        model_info = tool.get_model_info(model_name)
        print(f"   模型名称: {model_info.get('name', model_name)}")
        print(f"   描述: {model_info.get('description', '无描述')}")
        print(f"   支持的任务: {', '.join(model_info.get('supported_tasks', []))}")
        print(f"   支持的序列类型: {', '.join(model_info.get('supported_sequences', []))}")
        print(f"   内存需求: {model_info.get('memory_requirement', 0)} MB")
    print()
    
    # 7. 自动选择最佳模型
    print("7. 自动选择最佳模型")
    print("   为不同任务和序列类型选择最佳模型:")
    
    tasks = [TaskType.EMBEDDING, TaskType.CLASSIFICATION, TaskType.GENERATION]
    seq_types = [SequenceType.DNA, SequenceType.RNA, SequenceType.protein]
    
    for task in tasks:
        for seq_type in seq_types:
            best_model = tool.get_best_model(task, seq_type)
            print(f"   {task.value} + {seq_type.value}: {best_model}")
    print()
    
    print("=== 基本使用示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()