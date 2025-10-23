"""
生物序列模型统一接口使用示例
"""
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# 直接导入模块
from src.tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from src.tooluniverse.bio_models.task_types import TaskType, SequenceType


def main():
    """主函数"""
    # 初始化工具
    tool = BioSequenceAnalysisTool()
    
    # 示例序列
    dna_sequence = "ATCGATCGATCGATCG"
    rna_sequence = "AUCGAUCGAUCGAUCG"
    protein_sequence = "MKVILLFVLAVFATSASVAP"
    
    print("=== 生物序列模型统一接口使用示例 ===\n")
    
    # 1. 列出所有可用模型
    print("1. 可用模型列表:")
    models = tool.list_models()
    for model in models:
        print(f"   - {model}")
    print()
    
    # 2. 获取模型信息
    if models:
        print("2. 模型信息:")
        model_info = tool.get_model_info(models[0])
        print(f"   模型名称: {model_info.get('name')}")
        print(f"   支持的任务: {', '.join(model_info.get('supported_tasks', []))}")
        print(f"   支持的序列类型: {', '.join(model_info.get('supported_sequences', []))}")
        print(f"   内存需求: {model_info.get('memory_requirement')} MB")
        print()
    
    # 3. 获取最佳模型
    print("3. 不同任务和序列类型的最佳模型:")
    for task in [TaskType.EMBEDDING, TaskType.CLASSIFICATION, TaskType.GENERATION]:
        for seq_type in [SequenceType.DNA, SequenceType.RNA, SequenceType.protein]:
            best_model = tool.get_best_model(task, seq_type)
            print(f"   {task.value} + {seq_type.value}: {best_model}")
    print()
    
    # 4. 执行分析任务
    print("4. 执行分析任务:")
    
    # 4.1 DNA序列嵌入
    print("   4.1 DNA序列嵌入:")
    result = tool.analyze(
        sequences=dna_sequence,
        task_type=TaskType.EMBEDDING,
        sequence_type=SequenceType.DNA
    )
    print(f"   结果: {result}")
    print()
    
    # 4.2 蛋白质序列分类
    print("   4.2 蛋白质序列分类:")
    result = tool.analyze(
        sequences=protein_sequence,
        task_type=TaskType.CLASSIFICATION,
        sequence_type=SequenceType.protein
    )
    print(f"   结果: {result}")
    print()
    
    # 4.3 RNA序列注释
    print("   4.3 RNA序列注释:")
    result = tool.analyze(
        sequences=rna_sequence,
        task_type=TaskType.ANNOTATION,
        sequence_type=SequenceType.RNA
    )
    print(f"   结果: {result}")
    print()
    
    # 5. 批量处理
    print("5. 批量处理多个序列:")
    sequences = [dna_sequence, rna_sequence, protein_sequence]
    result = tool.analyze(
        sequences=sequences,
        task_type=TaskType.EMBEDDING
    )
    print(f"   处理的序列数量: {len(sequences)}")
    print(f"   结果数量: {len(result.get('results', []))}")
    print()


if __name__ == "__main__":
    main()