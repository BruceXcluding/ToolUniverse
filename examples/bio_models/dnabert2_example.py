#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNABERT2 工具使用示例

本示例演示如何使用ToolUniverse中的DNABERT2工具进行DNA序列嵌入分析。
DNABERT2是一个基于Transformer的预训练模型，专门为DNA序列设计，
可生成768维的嵌入向量，这些向量可用于各种下游生物信息学任务，如：
- 序列分类和功能预测
- 调控元件识别
- 进化分析
- 序列相似性比较

使用说明:
1. 确保ToolUniverse环境已正确安装
2. 确保dnabert2 Docker容器正在运行 (`docker ps` 检查)
3. 直接运行此脚本：python dnabert2_example.py

注意事项:
- 示例包含短序列(20nt)、中等长度序列(70nt)和长序列(400nt)的测试
- DNABERT2模型在GPU上运行，确保Docker容器可以访问GPU
- 嵌入向量维度固定为768维
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
    parser = argparse.ArgumentParser(description='DNABERT2工具使用示例')
    parser.add_argument('--save-results', action='store_true', help='保存结果到JSON文件')
    parser.add_argument('--output-file', type=str, default='dnabert2_results.json', help='输出JSON文件路径')
    parser.add_argument('--use-docker', type=bool, default=True, help='是否使用Docker模式')
    return parser.parse_args()

def main():
    """
    DNABERT2工具使用主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    logger = logging.getLogger("dnabert2_example")
    logger.info("开始DNABERT2工具示例")
    
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
        
        # 初始化工具实例 - 使用Docker模式
        # 注意: 使用Docker模式需要确保dnabert2容器已启动
        use_docker = args.use_docker
        logger.info(f"初始化DNABERT2Tool实例 (use_docker={use_docker})")
        
        tool = DNABERT2Tool(use_docker=use_docker)
        logger.info("DNABERT2Tool实例初始化成功")
        
        # 准备测试数据 - 不同长度的DNA序列
        test_sequences = {
            "短序列": "ATCGATCGATCGATCGATCG",  # 20nt的短序列
            "中等长度序列": "ATCGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC",  # 70nt的中等长度序列
            "长序列": "ATCG" * 100  # 400nt的长序列
        }
        
        # 任务类型 - 当前支持'embedding'（生成嵌入向量）
        task_type = "embedding"
        
        # 处理每个测试序列
        results = {}
        for name, sequence in test_sequences.items():
            logger.info(f"处理{name}: 长度={len(sequence)}nt")
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 调用analyze方法进行分析
                # 注意：输入必须是序列列表
                result = tool.analyze(sequences=[sequence], task_type=task_type)
                
                # 计算处理时间
                processing_time = time.time() - start_time
                
                try:
                    # 处理实际API返回的结果格式
                    # 解析包含success和results字段的响应格式
                    if isinstance(result, dict):
                        if result.get('success') is True and isinstance(result.get('results'), list):
                            # 从results列表中获取第一个结果
                            if result['results'] and len(result['results']) > 0:
                                first_result = result['results'][0]
                                if isinstance(first_result, dict):
                                    # 提取嵌入向量信息
                                    if 'embedding' in first_result:
                                        embedding = first_result['embedding']
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
                                        except Exception as e:
                                            embedding_dim = "未知"
                                            sample_values = f"解析嵌入向量时出错: {str(e)}"
                                    else:
                                        embedding_dim = "未知"
                                        sample_values = f"结果中不包含embedding字段，可用字段: {list(first_result.keys())}"
                                else:
                                    embedding_dim = "未知"
                                    sample_values = f"结果项类型不是字典: {type(first_result).__name__}"
                            else:
                                embedding_dim = "未知"
                                sample_values = "结果列表为空"
                        else:
                            embedding_dim = "未知"
                            sample_values = f"成功标志为{result.get('success')}，错误信息: {result.get('error', '无')}"
                    else:
                        embedding_dim = "未知"
                        sample_values = f"返回类型不是字典: {type(result).__name__}"
                except Exception as e:
                    embedding_dim = "未知"
                    sample_values = f"处理结果时发生错误: {str(e)}"
                
                # 记录结果
                results[name] = {
                    "sequence": sequence,
                    "length": len(sequence),
                    "embedding_dim": embedding_dim,
                    "processing_time": processing_time,
                    "result_sample": sample_values,
                    "raw_result": result
                }
                
                logger.info(f"{name}处理完成，耗时: {processing_time:.4f}秒")
                
            except Exception as e:
                logger.error(f"处理{name}时出错: {str(e)}")
                results[name] = {"error": str(e)}
        
        # 批量处理示例
        logger.info("\n执行批量序列处理示例")
        batch_sequences = ["ATCG" * i for i in range(1, 6)]  # 生成5个不同长度的序列
        batch_start_time = time.time()
        
        try:
            batch_result = tool.analyze(sequences=batch_sequences, task_type=task_type)
            batch_time = time.time() - batch_start_time
            
            # 处理批量结果
            sequence_count = len(batch_sequences)
            avg_time = batch_time / sequence_count if sequence_count > 0 else 0
            
            # 检查结果格式
            try:
                if isinstance(batch_result, dict):
                    if batch_result.get('success') is True:
                        if 'results' in batch_result and isinstance(batch_result['results'], list):
                            result_count = len(batch_result['results'])
                            result_info = f"成功处理 {result_count} 个序列的嵌入向量"
                        else:
                            result_info = f"成功标志为True，但缺少有效results字段"
                    else:
                        result_info = f"处理失败: {batch_result.get('error', '未知错误')}"
                else:
                    result_info = f"返回类型: {type(batch_result).__name__}"
            except Exception as e:
                result_info = f"解析结果时出错: {str(e)}"
            
            results["批量处理"] = {
                "sequence_count": sequence_count,
                "total_processing_time": batch_time,
                "avg_processing_time": avg_time,
                "result_info": result_info,
                "raw_result": batch_result
            }
            
            logger.info(f"批量处理完成，处理了{sequence_count}个序列，总耗时: {batch_time:.4f}秒")
            
        except Exception as e:
            logger.error(f"批量处理出错: {str(e)}")
            results["批量处理"] = {"error": str(e)}
        
        # 打印结果摘要
        print("\n" + "="*80)
        print("DNABERT2 工具示例 - 结果摘要")
        print("="*80)
        
        # 统计成功处理的序列数
        success_count = sum(1 for data in results.values() if "error" not in data)
        total_count = len(results)
        
        print(f"\n总体统计:")
        print(f"  成功处理: {success_count}/{total_count}")
        
        for name, data in results.items():
            print(f"\n{name}:")
            if "error" in data:
                print(f"  ❌ 错误: {data['error']}")
            elif name == "批量处理":
                print(f"  ✅ 成功")
                print(f"  序列数量: {data['sequence_count']}")
                print(f"  总处理时间: {data['total_processing_time']:.4f}秒")
                print(f"  平均处理时间: {data['avg_processing_time']:.4f}秒/序列")
                print(f"  结果信息: {data['result_info']}")
            else:
                print(f"  ✅ 成功")
                print(f"  序列长度: {data['length']}nt")
                print(f"  处理时间: {data['processing_time']:.4f}秒")
                print(f"  嵌入向量维度: {data['embedding_dim']}")
                print(f"  嵌入向量样本: {data['result_sample']}")
        
        # 保存结果到文件
        if args.save_results:
            if save_results_to_json(results, args.output_file):
                print(f"\n✅ 结果已保存到: {args.output_file}")
            else:
                print(f"\n❌ 保存结果失败")
        
        print("\n" + "="*80)
        print("示例完成！")
        print("提示: 要查看容器内的GPU使用情况，请运行:")
        print("  docker logs dnabert2 | grep -i 'gpu\\|memory'")
        print("提示: 要保存结果到文件，请使用参数: --save-results --output-file your_results.json")
        
    except ImportError as e:
        logger.error(f"导入错误: {str(e)}")
        print("\n❌ 无法导入DNABERT2Tool。请确保:")
        print("  1. ToolUniverse已正确安装")
        print("  2. 环境变量已设置正确")
        print("  3. 相关依赖已安装")
    
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()