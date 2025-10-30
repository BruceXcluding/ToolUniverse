#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNABERT2 CLI工具

本工具提供命令行接口，用于使用ToolUniverse中的DNABERT2工具进行DNA序列嵌入分析。
DNABERT2是一个基于Transformer的预训练模型，专门为DNA序列设计，
可生成768维的嵌入向量，这些向量可用于各种下游生物信息学任务。

使用方法:
  python dnabert2_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding [--json]
  python dnabert2_cli.py --sequence-file sequences.txt --task embedding [--json]
  python dnabert2_cli.py --sequences "ATCGATCG" "GCTAGCTA" --task embedding [--json]

参数说明:
  --sequence      单个DNA序列
  --sequences     多个DNA序列(空格分隔)
  --sequence-file 包含DNA序列的文件路径(每行一个序列)
  --task          任务类型(当前支持'embedding')
  --json          以JSON格式输出结果
  --output        输出文件路径(默认输出到终端)
  --use-docker    是否使用Docker模式(默认:true)
"""

import os
import sys
import json
import logging
import time
import numpy as np
import argparse

# 设置日志级别和格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 禁用特定第三方库的DEBUG日志
for logger_name in ['ToolRegistry', 'urllib3', 'transformers', 'torch']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)

# 添加项目根目录到Python路径，确保能正确导入tooluniverse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))


def save_results_to_json(results, output_file="dnabert2_results.json"):
    """
    将结果保存到JSON文件
    
    Args:
        results: 要保存的结果字典
        output_file: 输出文件路径
    """
    try:
        # 创建一个可序列化的结果副本
        serializable_results = {}
        for name, data in results.items():
            if isinstance(data, dict):
                serializable_data = {}
                for key, value in data.items():
                    # 处理numpy数组和特殊对象
                    if isinstance(value, np.ndarray):
                        serializable_data[key] = value.tolist()
                    elif hasattr(value, 'tolist'):
                        serializable_data[key] = value.tolist()
                    elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                        serializable_data[key] = value
                    else:
                        serializable_data[key] = str(value)
                serializable_results[name] = serializable_data
            else:
                serializable_results[name] = str(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger = logging.getLogger("dnabert2_example")
        logger.error(f"保存结果到文件时出错: {str(e)}")
        return False

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DNABERT2 CLI工具')
    
    # 输入选项 (互斥组)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequence', type=str, help='单个DNA序列')
    input_group.add_argument('--sequences', nargs='+', help='多个DNA序列(空格分隔)')
    input_group.add_argument('--sequence-file', type=str, help='包含DNA序列的文件路径(每行一个序列)')
    
    # 其他选项
    parser.add_argument('--task', type=str, default='embedding', choices=['embedding', 'classification'], help='任务类型')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出结果')
    parser.add_argument('--output', type=str, help='输出文件路径(默认输出到终端)')
    parser.add_argument('--use-docker', type=bool, default=True, help='是否使用Docker模式')
    
    return parser.parse_args()



def read_sequences_from_file(file_path):
    """
    从文件中读取DNA序列
    
    Args:
        file_path: 文件路径
    
    Returns:
        序列列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip() for line in f if line.strip()]
        return sequences
    except Exception as e:
        logger = logging.getLogger("dnabert2_cli")
        logger.error(f"读取序列文件时出错: {str(e)}")
        raise ValueError(f"无法读取序列文件: {str(e)}")

def format_json_output(results):
    """
    格式化JSON输出
    
    Args:
        results: 结果字典
    
    Returns:
        可序列化的结果字典
    """
    # 创建一个可序列化的结果副本
    serializable_results = {}
    for name, data in results.items():
        if isinstance(data, dict):
            serializable_data = {}
            for key, value in data.items():
                # 处理numpy数组和特殊对象
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                elif hasattr(value, 'tolist'):
                    serializable_data[key] = value.tolist()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serializable_data[key] = value
                else:
                    serializable_data[key] = str(value)
            serializable_results[name] = serializable_data
        else:
            serializable_results[name] = str(data)
    return serializable_results

def main():
    """
    DNABERT2 CLI工具主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    logger = logging.getLogger("dnabert2_cli")
    logger.info("开始DNABERT2 CLI工具")
    
    try:
        # 导入DNABERT2工具（尝试不同的导入路径）
        try:
            from tooluniverse.bio_models.dnabert2_tool import DNABERT2Tool
            logger.info("成功从tooluniverse.bio_models导入DNABERT2Tool")
        except ImportError:
            try:
                from tooluniverse.bio_models.tools.dnabert2_tool import DNABERT2Tool
                logger.info("成功从tooluniverse.bio_models.tools导入DNABERT2Tool")
            except ImportError as e:
                logger.error(f"无法导入DNABERT2Tool: {e}")
                raise
        
        # 初始化工具实例
        use_docker = args.use_docker
        logger.info(f"初始化DNABERT2Tool实例 (use_docker={use_docker})")
        
        tool = DNABERT2Tool(use_docker=use_docker)
        logger.info("DNABERT2Tool实例初始化成功")
        
        # 获取序列
        sequences = []
        if args.sequence:
            sequences = [args.sequence]
        elif args.sequences:
            sequences = args.sequences
        elif args.sequence_file:
            sequences = read_sequences_from_file(args.sequence_file)
        
        task_type = args.task
        logger.info(f"处理任务类型: {task_type}, 序列数量: {len(sequences)}")
        
        # 处理序列
        results = {}
        
        # 记录总开始时间
        total_start_time = time.time()
        
        # 处理序列（使用真实的DNABERT2服务）
        try:
            if len(sequences) == 1:
                sequence = sequences[0]
                logger.info(f"处理序列: 长度={len(sequence)}nt")
                
                start_time = time.time()
                result = tool.analyze(sequences=[sequence], task_type=task_type)
                processing_time = time.time() - start_time
                
                # 解析结果
                parsed_result = parse_result(result, sequence, processing_time, task_type)
                results["sequence"] = parsed_result
                
                logger.info(f"序列处理完成，耗时: {processing_time:.4f}秒")
            else:
                # 批量处理多个序列
                logger.info("执行批量序列处理")
                start_time = time.time()
                
                batch_result = tool.analyze(sequences=sequences, task_type=task_type)
                batch_time = time.time() - start_time
                
                # 处理批量结果
                sequence_count = len(sequences)
                avg_time = batch_time / sequence_count if sequence_count > 0 else 0
                
                # 解析批量结果
                batch_results = parse_batch_result(batch_result, sequences, batch_time, task_type)
                results = batch_results
                
                logger.info(f"批量处理完成，处理了{sequence_count}个序列，总耗时: {batch_time:.4f}秒")
        except Exception as e:
            logger.error(f"处理过程中发生意外错误: {str(e)}")
            results["error"] = str(e)
            results["sequence_count"] = len(sequences)
        
        total_time = time.time() - total_start_time
        results["metadata"] = {
            "total_processing_time": total_time,
            "sequence_count": len(sequences),
            "task_type": task_type,
            "use_docker": use_docker
        }
        
        # 输出结果
        if args.json:
            # 以JSON格式输出
            output_content = json.dumps(format_json_output(results), ensure_ascii=False, indent=2)
        else:
            # 以人类可读格式输出
            output_content = format_human_readable_output(results)
        
        # 写入文件或输出到终端
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"✅ 结果已保存到: {args.output}")
        else:
            print(output_content)
            
    except ImportError as e:
        logger.error(f"导入错误: {str(e)}")
        error_msg = f"❌ 无法导入DNABERT2Tool。请确保:\n  1. ToolUniverse已正确安装\n  2. 环境变量已设置正确\n  3. 相关依赖已安装"
        if args.json:
            print(json.dumps({"error": error_msg}, ensure_ascii=False, indent=2))
        else:
            print(error_msg)
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        else:
            print(f"\n❌ 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

def parse_result(result, sequence, processing_time, task_type='embedding'):
    """
    解析单个序列的结果
    
    Args:
        result: API返回的结果
        sequence: 输入的序列
        processing_time: 处理时间
        task_type: 任务类型，用于区分不同的结果格式
    
    Returns:
        解析后的结果字典
    """
    try:
        # 处理实际API返回的结果格式
        if isinstance(result, dict):
                    if result.get('success') is True:
                        # 根据任务类型处理不同的结果格式
                        if task_type == 'embedding':
                            # 嵌入任务 - 寻找embedding相关字段
                            if isinstance(result.get('results'), list) and result['results']:
                                first_result = result['results'][0]
                                if isinstance(first_result, dict) and 'embedding' in first_result:
                                    embedding = first_result['embedding']
                                    return extract_embedding_info(embedding, sequence, processing_time, result)
                                elif isinstance(first_result, dict):
                                    return {
                                        "sequence": sequence,
                                        "length": len(sequence),
                                        "processing_time": processing_time,
                                        "error": f"结果中不包含embedding字段，可用字段: {list(first_result.keys())}",
                                        "raw_result": result
                                    }
                                else:
                                    return {
                                        "sequence": sequence,
                                        "length": len(sequence),
                                        "processing_time": processing_time,
                                        "error": f"结果项类型不是字典: {type(first_result).__name__}",
                                        "raw_result": result
                                    }
                            elif 'embedding' in result:
                                # 某些API可能直接返回embedding字段
                                return extract_embedding_info(result['embedding'], sequence, processing_time, result)
                            else:
                                return {
                                    "sequence": sequence,
                                    "length": len(sequence),
                                    "processing_time": processing_time,
                                    "error": f"结果中找不到有效嵌入信息，可用字段: {list(result.keys())}",
                                    "raw_result": result
                                }
                        elif task_type in ['prediction', 'classification']:
                            # 预测或分类任务
                            if isinstance(result.get('results'), list) and result['results']:
                                first_result = result['results'][0]
                                if isinstance(first_result, dict):
                                    # 提取预测或分类结果
                                    return {
                                        "sequence": sequence,
                                        "length": len(sequence),
                                        "processing_time": processing_time,
                                        "prediction": first_result,
                                        "raw_result": result
                                    }
                                else:
                                    return {
                                        "sequence": sequence,
                                        "length": len(sequence),
                                        "processing_time": processing_time,
                                        "error": f"结果项类型不是字典: {type(first_result).__name__}",
                                        "raw_result": result
                                    }
                            else:
                                return {
                                    "sequence": sequence,
                                    "length": len(sequence),
                                    "processing_time": processing_time,
                                    "error": f"预测结果格式不符合预期，可用字段: {list(result.keys())}",
                                    "raw_result": result
                                }
                        else:
                            # 其他任务类型
                            return {
                                "sequence": sequence,
                                "length": len(sequence),
                                "processing_time": processing_time,
                                "result": result,
                                "raw_result": result
                            }
                    else:
                        # 失败的情况
                        error_msg = result.get('error', '未知错误')
                        # 特殊处理常见错误
                        if "'DNABERT2Client' object has no attribute" in str(error_msg):
                            return {
                                "sequence": sequence,
                                "length": len(sequence),
                                "processing_time": processing_time,
                                "error": f"功能不支持: {error_msg}\n提示: 当前DNABERT2实现可能不支持所选的'{task_type}'任务",
                                "raw_result": result
                            }
                        else:
                            return {
                                "sequence": sequence,
                                "length": len(sequence),
                                "processing_time": processing_time,
                                "error": f"成功标志为False，错误信息: {error_msg}",
                                "raw_result": result
                            }
        else:
            return {
                "sequence": sequence,
                "length": len(sequence),
                "processing_time": processing_time,
                "error": f"返回类型不是字典: {type(result).__name__}",
                "raw_result": result
            }
    except Exception as e:
        return {
            "sequence": sequence,
            "length": len(sequence),
            "processing_time": processing_time,
            "error": f"处理结果时发生错误: {str(e)}",
            "raw_result": result
        }

def parse_batch_result(batch_result, sequences, processing_time, task_type='embedding'):
    """
    解析批量序列的结果
    
    Args:
        batch_result: API返回的批量结果
        sequences: 输入的序列列表
        processing_time: 总处理时间
        task_type: 任务类型，用于区分不同的结果格式
    
    Returns:
        解析后的结果字典
    """
    results = {
        "total_processing_time": processing_time,
        "sequence_count": len(sequences),
        "avg_processing_time": processing_time / len(sequences) if sequences else 0
    }
    
    try:
        if isinstance(batch_result, dict):
            if batch_result.get('success') is True:
                if 'results' in batch_result and isinstance(batch_result['results'], list):
                    # 为每个序列创建结果
                    sequence_results = []
                    for i, sequence in enumerate(sequences):
                        seq_result = {
                            "sequence": sequence,
                            "length": len(sequence),
                            "index": i
                        }
                        
                        # 尝试获取对应的结果
                        if i < len(batch_result['results']):
                            result_item = batch_result['results'][i]
                            seq_processing_time = processing_time / len(sequences)
                            
                            # 根据任务类型处理不同的结果格式
                            if isinstance(result_item, dict):
                                if task_type == 'embedding' and 'embedding' in result_item:
                                    # 嵌入任务
                                    embedding_info = extract_embedding_info(
                                        result_item['embedding'], 
                                        sequence, 
                                        seq_processing_time, 
                                        result_item
                                    )
                                    seq_result.update(embedding_info)
                                elif task_type in ['prediction', 'classification']:
                                    # 预测或分类任务
                                    seq_result.update({
                                        "processing_time": seq_processing_time,
                                        "prediction": result_item
                                    })
                                else:
                                    # 其他任务类型
                                    seq_result.update({
                                        "processing_time": seq_processing_time,
                                        "result": result_item
                                    })
                            else:
                                seq_result['error'] = f"无效的结果格式: {type(result_item).__name__}"
                        else:
                            seq_result['error'] = "没有对应的结果"
                        
                        sequence_results.append(seq_result)
                    
                    results["sequences"] = sequence_results
                    # 根据任务类型设置结果信息
                    if task_type == 'embedding':
                        results["result_info"] = f"成功处理 {len(batch_result['results'])} 个序列的嵌入向量"
                    elif task_type in ['prediction', 'classification']:
                        results["result_info"] = f"成功处理 {len(batch_result['results'])} 个序列的预测结果"
                    else:
                        results["result_info"] = f"成功处理 {len(batch_result['results'])} 个序列"
                else:
                    results["error"] = "成功标志为True，但缺少有效results字段"
                    results["raw_result"] = batch_result
            else:
                # 失败的情况，特殊处理常见错误
                error_msg = batch_result.get('error', '未知错误')
                if "'DNABERT2Client' object has no attribute" in str(error_msg):
                    results["error"] = f"功能不支持: {error_msg}\n提示: 当前DNABERT2实现可能不支持所选的'{task_type}'任务"
                else:
                    results["error"] = f"处理失败: {error_msg}"
                results["raw_result"] = batch_result
        else:
            results["error"] = f"返回类型: {type(batch_result).__name__}"
            results["raw_result"] = batch_result
    except Exception as e:
        results["error"] = f"解析批量结果时出错: {str(e)}"
        results["raw_result"] = batch_result
    
    return results

def extract_embedding_info(embedding, sequence, processing_time, raw_result):
    """
    提取嵌入向量信息
    
    Args:
        embedding: 嵌入向量
        sequence: 输入的序列
        processing_time: 处理时间
        raw_result: 原始结果
    
    Returns:
        嵌入向量信息字典
    """
    try:
        # 提取嵌入向量的维度信息
        if hasattr(embedding, 'shape'):
            embedding_dim = str(embedding.shape)
        elif isinstance(embedding, list):
            if embedding and isinstance(embedding[0], (list, np.ndarray)):
                embedding_dim = f"({len(embedding)}, {len(embedding[0])})" if embedding[0] else "(未知, 0)"
            else:
                embedding_dim = str(len(embedding))
        else:
            embedding_dim = "标量值"
        
        # 提取嵌入向量样本（显示前5个值）
        if hasattr(embedding, 'shape'):
            # NumPy数组或PyTorch张量
            if embedding.ndim > 0:
                if embedding.ndim > 1:
                    sample_values = embedding[0][:5].tolist() if hasattr(embedding[0][:5], 'tolist') else embedding[0][:5]
                else:
                    sample_values = embedding[:5].tolist() if hasattr(embedding[:5], 'tolist') else embedding[:5]
            else:
                sample_values = embedding.item() if hasattr(embedding, 'item') else embedding
        elif isinstance(embedding, list):
            if embedding and isinstance(embedding[0], (list, np.ndarray)):
                sample_values = embedding[0][:5] if embedding[0] else []
            else:
                sample_values = embedding[:5]
        else:
            sample_values = embedding
        
        return {
            "sequence": sequence,
            "length": len(sequence),
            "processing_time": processing_time,
            "embedding_dim": embedding_dim,
            "embedding_sample": sample_values,
            "raw_result": raw_result
        }
    except Exception as e:
        return {
            "sequence": sequence,
            "length": len(sequence),
            "processing_time": processing_time,
            "embedding_dim": "未知",
            "error": f"解析嵌入向量时出错: {str(e)}",
            "raw_result": raw_result
        }

def format_human_readable_output(results):
    """
    格式化人类可读的输出
    
    Args:
        results: 结果字典
    
    Returns:
        格式化的字符串
    """
    output_lines = []
    output_lines.append("\n" + "="*80)
    output_lines.append("DNABERT2 CLI工具 - 结果输出")
    output_lines.append("="*80)
    
    # 输出元数据
    if "metadata" in results:
        metadata = results["metadata"]
        output_lines.append(f"\n任务信息:")
        output_lines.append(f"  任务类型: {metadata.get('task_type', '未知')}")
        output_lines.append(f"  序列数量: {metadata.get('sequence_count', 0)}")
        output_lines.append(f"  总处理时间: {metadata.get('total_processing_time', 0):.4f}秒")
        output_lines.append(f"  使用Docker: {metadata.get('use_docker', True)}")
    
    # 输出单个序列结果
    if "sequence" in results:
        data = results["sequence"]
        output_lines.append(f"\n序列结果:")
        if "error" in data:
            output_lines.append(f"  ❌ 错误: {data['error']}")
        else:
            output_lines.append(f"  ✅ 成功")
            output_lines.append(f"  序列: {data['sequence'][:50]}{'...' if len(data['sequence']) > 50 else ''}")
            output_lines.append(f"  序列长度: {data['length']}nt")
            
            # 根据任务类型显示不同信息
            task_type = metadata.get('task_type', 'unknown') if 'metadata' in results else 'unknown'
            
            if task_type == 'embedding':
                output_lines.append(f"  处理时间: {data['processing_time']:.4f}秒")
                output_lines.append(f"  嵌入向量维度: {data.get('embedding_dim', '未知')}")
                output_lines.append(f"  嵌入向量样本: {data.get('embedding_sample', '未知')}")
            elif task_type == 'classification':
                output_lines.append(f"  预测标签: {data.get('predicted_label', '未知')}")
                output_lines.append(f"  置信度分数: {data.get('scores', '未知')}")
            else:
                # 通用信息
                if 'processing_time' in data:
                    output_lines.append(f"  处理时间: {data['processing_time']:.4f}秒")
    
    # 输出批量序列结果
    if "sequences" in results:
        sequences = results["sequences"]
        success_count = sum(1 for seq in sequences if "error" not in seq)
        output_lines.append(f"\n批量处理结果:")
        output_lines.append(f"  总序列数: {len(sequences)}")
        output_lines.append(f"  成功处理: {success_count}/{len(sequences)}")
        output_lines.append(f"  总处理时间: {results.get('total_processing_time', 0):.4f}秒")
        output_lines.append(f"  平均处理时间: {results.get('avg_processing_time', 0):.4f}秒/序列")
        
        # 输出前3个序列的详细信息
        output_lines.append("\n前3个序列的详细信息:")
        for i, seq in enumerate(sequences[:3]):
            output_lines.append(f"\n  序列 {i+1}:")
            if "error" in seq:
                output_lines.append(f"    ❌ 错误: {seq['error']}")
            else:
                output_lines.append(f"    ✅ 成功")
                output_lines.append(f"    序列: {seq['sequence'][:50]}{'...' if len(seq['sequence']) > 50 else ''}")
                output_lines.append(f"    序列长度: {seq['length']}nt")
                
                # 根据任务类型显示不同信息
                task_type = metadata.get('task_type', 'unknown') if 'metadata' in results else 'unknown'
                
                if task_type == 'embedding':
                    output_lines.append(f"    嵌入向量维度: {seq.get('embedding_dim', '未知')}")
                elif task_type == 'classification':
                    output_lines.append(f"    预测标签: {seq.get('predicted_label', '未知')}")
                    output_lines.append(f"    置信度分数: {seq.get('scores', '未知')}")
        
        # 如果有更多序列，提示用户
        if len(sequences) > 3:
            output_lines.append(f"\n  ... 还有 {len(sequences) - 3} 个序列的结果未显示 ...")
    
    # 如果有错误信息
    if "error" in results and "sequence" not in results and "sequences" not in results:
        output_lines.append(f"\n❌ 错误: {results['error']}")
    
    output_lines.append("\n" + "="*80)
    return "\n".join(output_lines)


if __name__ == "__main__":
    main()