"""
生物序列模型批处理示例
本示例展示如何使用ToolUniverse生物序列模型进行批量处理
"""
import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Any

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from src.tooluniverse.bio_models.task_types import TaskType, SequenceType


def generate_test_sequences(num_sequences: int = 100, seq_type: str = "DNA") -> List[str]:
    """生成测试序列"""
    sequences = []
    
    if seq_type.upper() == "DNA":
        bases = ["A", "T", "C", "G"]
        for i in range(num_sequences):
            seq = "".join(np.random.choice(bases, size=np.random.randint(50, 200)))
            sequences.append(seq)
    elif seq_type.upper() == "RNA":
        bases = ["A", "U", "C", "G"]
        for i in range(num_sequences):
            seq = "".join(np.random.choice(bases, size=np.random.randint(50, 200)))
            sequences.append(seq)
    elif seq_type.upper() == "PROTEIN":
        amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        for i in range(num_sequences):
            seq = "".join(np.random.choice(amino_acids, size=np.random.randint(30, 100)))
            sequences.append(seq)
    
    return sequences


def batch_embedding_analysis(tool: BioSequenceAnalysisTool, sequences: List[str], 
                           sequence_type: SequenceType, batch_sizes: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
    """批量嵌入分析"""
    results = {
        "sequence_type": sequence_type.value,
        "total_sequences": len(sequences),
        "batch_results": []
    }
    
    for batch_size in batch_sizes:
        print(f"   批处理大小: {batch_size}")
        
        # 计算需要的批次数
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        start_time = time.time()
        all_embeddings = []
        failed_batches = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # 执行批处理
            result = tool.analyze(
                sequences=batch_sequences,
                task_type=TaskType.EMBEDDING,
                sequence_type=sequence_type
            )
            
            if result.get('success'):
                batch_embeddings = [r.get('embedding', []) for r in result.get('results', [])]
                all_embeddings.extend(batch_embeddings)
            else:
                failed_batches += 1
                print(f"     批次 {i+1} 失败: {result.get('error', '未知错误')}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        batch_result = {
            "batch_size": batch_size,
            "num_batches": num_batches,
            "successful_batches": num_batches - failed_batches,
            "failed_batches": failed_batches,
            "total_embeddings": len(all_embeddings),
            "processing_time": processing_time,
            "avg_time_per_sequence": processing_time / len(sequences) if sequences else 0,
            "throughput": len(sequences) / processing_time if processing_time > 0 else 0
        }
        
        results["batch_results"].append(batch_result)
        print(f"     处理时间: {processing_time:.2f}秒")
        print(f"     吞吐量: {batch_result['throughput']:.2f} 序列/秒")
        print(f"     成功批次: {batch_result['successful_batches']}/{num_batches}")
        print()
    
    return results


def batch_classification_analysis(tool: BioSequenceAnalysisTool, sequences: List[str], 
                                sequence_type: SequenceType, batch_size: int = 10) -> Dict[str, Any]:
    """批量分类分析"""
    print(f"   批处理大小: {batch_size}")
    
    # 计算需要的批次数
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    start_time = time.time()
    all_classifications = []
    class_counts = {}
    failed_batches = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # 执行批处理
        result = tool.analyze(
            sequences=batch_sequences,
            task_type=TaskType.CLASSIFICATION,
            sequence_type=sequence_type
        )
        
        if result.get('success'):
            batch_classifications = result.get('results', [])
            all_classifications.extend(batch_classifications)
            
            # 统计类别
            for classification in batch_classifications:
                predicted_class = classification.get('predicted_class', '未知')
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
        else:
            failed_batches += 1
            print(f"     批次 {i+1} 失败: {result.get('error', '未知错误')}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    results = {
        "sequence_type": sequence_type.value,
        "total_sequences": len(sequences),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "successful_batches": num_batches - failed_batches,
        "failed_batches": failed_batches,
        "total_classifications": len(all_classifications),
        "processing_time": processing_time,
        "avg_time_per_sequence": processing_time / len(sequences) if sequences else 0,
        "throughput": len(sequences) / processing_time if processing_time > 0 else 0,
        "class_distribution": class_counts
    }
    
    print(f"     处理时间: {processing_time:.2f}秒")
    print(f"     吞吐量: {results['throughput']:.2f} 序列/秒")
    print(f"     成功批次: {results['successful_batches']}/{num_batches}")
    print(f"     类别分布: {class_counts}")
    print()
    
    return results


def compare_models_for_task(tool: BioSequenceAnalysisTool, sequences: List[str], 
                           task_type: TaskType, sequence_type: SequenceType, 
                           models: List[str]) -> Dict[str, Any]:
    """比较不同模型在同一任务上的性能"""
    results = {
        "task_type": task_type.value,
        "sequence_type": sequence_type.value,
        "model_results": {}
    }
    
    for model_name in models:
        print(f"   测试模型: {model_name}")
        
        start_time = time.time()
        successful_results = 0
        
        # 使用指定模型执行任务
        result = tool.analyze(
            sequences=sequences,
            task_type=task_type,
            sequence_type=sequence_type,
            model_name=model_name
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.get('success'):
            successful_results = len(result.get('results', []))
        
        model_result = {
            "processing_time": processing_time,
            "successful_results": successful_results,
            "total_sequences": len(sequences),
            "success_rate": successful_results / len(sequences) if sequences else 0,
            "throughput": len(sequences) / processing_time if processing_time > 0 else 0
        }
        
        results["model_results"][model_name] = model_result
        print(f"     处理时间: {processing_time:.2f}秒")
        print(f"     成功处理: {successful_results}/{len(sequences)}")
        print(f"     吞吐量: {model_result['throughput']:.2f} 序列/秒")
        print()
    
    return results


def main():
    """主函数 - 批处理示例"""
    print("=== ToolUniverse生物序列模型批处理示例 ===\n")
    
    # 1. 初始化工具
    print("1. 初始化生物序列分析工具")
    tool = BioSequenceAnalysisTool()
    print("   ✓ 工具初始化成功\n")
    
    # 2. 获取可用模型
    print("2. 获取可用模型")
    models = tool.list_models()
    print(f"   共有 {len(models)} 个可用模型")
    
    # 选择支持嵌入的模型
    embedding_models = []
    for model in models:
        model_info = tool.get_model_info(model)
        if "embedding" in model_info.get("supported_tasks", []):
            embedding_models.append(model)
    
    print(f"   支持嵌入的模型: {', '.join(embedding_models[:3])}...")
    print()
    
    # 3. 生成测试数据
    print("3. 生成测试数据")
    num_sequences = 50  # 使用较少的序列以便快速演示
    dna_sequences = generate_test_sequences(num_sequences, "DNA")
    rna_sequences = generate_test_sequences(num_sequences, "RNA")
    protein_sequences = generate_test_sequences(num_sequences, "PROTEIN")
    
    print(f"   生成 {num_sequences} 个DNA序列")
    print(f"   生成 {num_sequences} 个RNA序列")
    print(f"   生成 {num_sequences} 个蛋白质序列")
    print()
    
    # 4. 批量嵌入分析
    print("4. 批量嵌入分析")
    print("   4.1 DNA序列嵌入")
    dna_embedding_results = batch_embedding_analysis(tool, dna_sequences, SequenceType.DNA)
    
    print("   4.2 RNA序列嵌入")
    rna_embedding_results = batch_embedding_analysis(tool, rna_sequences, SequenceType.RNA)
    
    print("   4.3 蛋白质序列嵌入")
    protein_embedding_results = batch_embedding_analysis(tool, protein_sequences, SequenceType.protein)
    
    # 5. 批量分类分析
    print("5. 批量分类分析")
    print("   5.1 DNA序列分类")
    dna_classification_results = batch_classification_analysis(tool, dna_sequences, SequenceType.DNA)
    
    print("   5.2 RNA序列分类")
    rna_classification_results = batch_classification_analysis(tool, rna_sequences, SequenceType.RNA)
    
    # 6. 模型比较
    print("6. 模型性能比较")
    test_sequences = dna_sequences[:20]  # 使用较少序列进行比较
    comparison_models = embedding_models[:2] if len(embedding_models) >= 2 else embedding_models
    
    if comparison_models:
        print("   6.1 嵌入任务模型比较")
        embedding_comparison = compare_models_for_task(
            tool, test_sequences, TaskType.EMBEDDING, SequenceType.DNA, comparison_models
        )
        
        print("   6.2 分类任务模型比较")
        classification_comparison = compare_models_for_task(
            tool, test_sequences, TaskType.CLASSIFICATION, SequenceType.DNA, comparison_models
        )
    
    # 7. 保存结果
    print("7. 保存结果")
    results = {
        "dna_embedding": dna_embedding_results,
        "rna_embedding": rna_embedding_results,
        "protein_embedding": protein_embedding_results,
        "dna_classification": dna_classification_results,
        "rna_classification": rna_classification_results
    }
    
    if comparison_models:
        results["embedding_comparison"] = embedding_comparison
        results["classification_comparison"] = classification_comparison
    
    results_file = os.path.join(os.path.dirname(__file__), "batch_processing_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   结果已保存到: {results_file}")
    print()
    
    # 8. 性能总结
    print("8. 性能总结")
    print("   最佳批处理大小 (基于吞吐量):")
    
    for seq_type, result_name in [
        (SequenceType.DNA, "dna_embedding"),
        (SequenceType.RNA, "rna_embedding"),
        (SequenceType.protein, "protein_embedding")
    ]:
        embedding_result = results.get(result_name, {})
        batch_results = embedding_result.get("batch_results", [])
        
        if batch_results:
            best_batch = max(batch_results, key=lambda x: x.get("throughput", 0))
            print(f"   {seq_type.value}: 批处理大小 {best_batch.get('batch_size')}, "
                  f"吞吐量 {best_batch.get('throughput', 0):.2f} 序列/秒")
    
    print("\n=== 批处理示例完成 ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()