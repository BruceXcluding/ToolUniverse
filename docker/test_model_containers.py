#!/usr/bin/env python3
"""
ToolUniverse 模型容器API测试脚本
用于验证模型容器的API功能是否正常工作
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, List, Any

# 添加项目路径到sys.path
# 获取当前脚本所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.tooluniverse.bio_models.container_client import (
    DNABERT2Client,
    LucaOneClient,
    TaskType
)


def print_test_header(test_name: str):
    """打印测试标题"""
    print(f"\n{'='*50}")
    print(f"测试: {test_name}")
    print(f"{'='*50}")


def print_test_result(test_name: str, success: bool, message: str = ""):
    """打印测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{test_name}: {status}")
    if message:
        print(f"  详情: {message}")


def test_dnabert2_client(base_url: str) -> Dict[str, Any]:
    """测试DNABERT2客户端"""
    results = {"total": 0, "passed": 0, "failed": 0, "tests": []}
    
    print_test_header("DNABERT2模型API测试")
    
    # 创建客户端
    try:
        client = DNABERT2Client(base_url)
        print("✅ DNABERT2客户端创建成功")
    except Exception as e:
        print(f"❌ DNABERT2客户端创建失败: {e}")
        return results
    
    # 测试健康检查
    results["total"] += 1
    try:
        health = client.check_health()
        print(f"✅ 健康检查通过: 状态={health.status}, 模型={health.model}")
        results["passed"] += 1
        results["tests"].append({"name": "健康检查", "status": "passed"})
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "健康检查", "status": "failed", "error": str(e)})
    
    # 测试模型信息
    results["total"] += 1
    try:
        model_info = client.get_model_info()
        print(f"✅ 模型信息获取成功: 名称={model_info.name}, 版本={model_info.version}")
        results["passed"] += 1
        results["tests"].append({"name": "模型信息", "status": "passed"})
    except Exception as e:
        print(f"❌ 模型信息获取失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "模型信息", "status": "failed", "error": str(e)})
    
    # 测试单个预测
    results["total"] += 1
    try:
        sequence = "ATGCGTACGTAGCTAGCTAGCTAGC"
        prediction = client.predict_promoter(sequence)
        print(f"✅ 单个预测成功: 预测值={prediction.prediction}, 置信度={prediction.confidence}")
        results["passed"] += 1
        results["tests"].append({"name": "单个预测", "status": "passed"})
    except Exception as e:
        print(f"❌ 单个预测失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "单个预测", "status": "failed", "error": str(e)})
    
    # 测试批量预测
    results["total"] += 1
    try:
        sequences = [
            "ATGCGTACGTAGCTAGCTAGCTAGC",
            "GCTAGCTAGCTACGTACGCAT",
            "TTACGATCGATCGATCGATCGA"
        ]
        batch_predictions = client.predict_promoter_batch(sequences)
        print(f"✅ 批量预测成功: 预测了{len(batch_predictions)}个序列")
        results["passed"] += 1
        results["tests"].append({"name": "批量预测", "status": "passed"})
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "批量预测", "status": "failed", "error": str(e)})
    
    # 测试指标获取
    results["total"] += 1
    try:
        metrics = client.get_metrics()
        print(f"✅ 指标获取成功: 总请求数={metrics.get('model_metrics', {}).get('total_requests', 'N/A')}")
        results["passed"] += 1
        results["tests"].append({"name": "指标获取", "status": "passed"})
    except Exception as e:
        print(f"❌ 指标获取失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "指标获取", "status": "failed", "error": str(e)})
    
    return results


def test_lucaone_client(base_url: str) -> Dict[str, Any]:
    """测试LucaOne客户端"""
    results = {"total": 0, "passed": 0, "failed": 0, "tests": []}
    
    print_test_header("LucaOne模型API测试")
    
    # 创建客户端
    try:
        client = LucaOneClient(base_url)
        print("✅ LucaOne客户端创建成功")
    except Exception as e:
        print(f"❌ LucaOne客户端创建失败: {e}")
        return results
    
    # 测试健康检查
    results["total"] += 1
    try:
        health = client.check_health()
        print(f"✅ 健康检查通过: 状态={health.status}, 模型={health.model}")
        results["passed"] += 1
        results["tests"].append({"name": "健康检查", "status": "passed"})
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "健康检查", "status": "failed", "error": str(e)})
    
    # 测试模型信息
    results["total"] += 1
    try:
        model_info = client.get_model_info()
        print(f"✅ 模型信息获取成功: 名称={model_info.name}, 版本={model_info.version}")
        results["passed"] += 1
        results["tests"].append({"name": "模型信息", "status": "passed"})
    except Exception as e:
        print(f"❌ 模型信息获取失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "模型信息", "status": "failed", "error": str(e)})
    
    # 测试嵌入获取
    results["total"] += 1
    try:
        sequence = "ATGCGTACGTAGCTAGCTAGCTAGC"
        embedding = client.get_embedding(sequence)
        print(f"✅ 嵌入获取成功: 嵌入维度={len(embedding.prediction)}")
        results["passed"] += 1
        results["tests"].append({"name": "嵌入获取", "status": "passed"})
    except Exception as e:
        print(f"❌ 嵌入获取失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "嵌入获取", "status": "failed", "error": str(e)})
    
    # 测试序列分类
    results["total"] += 1
    try:
        classification = client.classify_sequence(sequence)
        print(f"✅ 序列分类成功: 预测={classification.prediction}, 置信度={classification.confidence}")
        results["passed"] += 1
        results["tests"].append({"name": "序列分类", "status": "passed"})
    except Exception as e:
        print(f"❌ 序列分类失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "序列分类", "status": "failed", "error": str(e)})
    
    # 测试属性预测
    results["total"] += 1
    try:
        property_pred = client.predict_property(sequence)
        print(f"✅ 属性预测成功: 预测={property_pred.prediction}, 置信度={property_pred.confidence}")
        results["passed"] += 1
        results["tests"].append({"name": "属性预测", "status": "passed"})
    except Exception as e:
        print(f"❌ 属性预测失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "属性预测", "status": "failed", "error": str(e)})
    
    # 测试指标获取
    results["total"] += 1
    try:
        metrics = client.get_metrics()
        print(f"✅ 指标获取成功: 总请求数={metrics.get('model_metrics', {}).get('total_requests', 'N/A')}")
        results["passed"] += 1
        results["tests"].append({"name": "指标获取", "status": "passed"})
    except Exception as e:
        print(f"❌ 指标获取失败: {e}")
        results["failed"] += 1
        results["tests"].append({"name": "指标获取", "status": "failed", "error": str(e)})
    
    return results


def test_performance(client, test_name: str, num_requests: int = 10) -> Dict[str, Any]:
    """测试API性能"""
    print_test_header(f"{test_name}性能测试")
    
    sequence = "ATGCGTACGTAGCTAGCTAGCTAGC"
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            if "DNABERT2" in test_name:
                client.predict_promoter(sequence)
            else:
                client.get_embedding(sequence)
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"请求 {i+1}: {response_time:.3f}秒")
        except Exception as e:
            print(f"请求 {i+1} 失败: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n性能统计:")
        print(f"  平均响应时间: {avg_time:.3f}秒")
        print(f"  最小响应时间: {min_time:.3f}秒")
        print(f"  最大响应时间: {max_time:.3f}秒")
        print(f"  成功率: {len(response_times)}/{num_requests} ({len(response_times)/num_requests*100:.1f}%)")
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "success_rate": len(response_times) / num_requests,
            "total_requests": num_requests
        }
    else:
        print("所有请求都失败了")
        return {
            "avg_time": 0,
            "min_time": 0,
            "max_time": 0,
            "success_rate": 0,
            "total_requests": num_requests
        }


def save_results(results: Dict[str, Any], output_file: str):
    """保存测试结果到文件"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n测试结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存测试结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ToolUniverse模型容器API测试")
    parser.add_argument("--dnabert2-url", default="http://localhost:8000", 
                        help="DNABERT2模型容器URL")
    parser.add_argument("--lucaone-url", default="http://localhost:8002", 
                        help="LucaOne模型容器URL")
    parser.add_argument("--performance", action="store_true", 
                        help="运行性能测试")
    parser.add_argument("--output", default="test_results.json", 
                        help="测试结果输出文件")
    parser.add_argument("--requests", type=int, default=10, 
                        help="性能测试请求数量")
    
    args = parser.parse_args()
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dnabert2": {},
        "lucaone": {},
        "performance": {}
    }
    
    # 测试DNABERT2
    dnabert2_results = test_dnabert2_client(args.dnabert2_url)
    all_results["dnabert2"] = dnabert2_results
    
    # 测试LucaOne
    lucaone_results = test_lucaone_client(args.lucaone_url)
    all_results["lucaone"] = lucaone_results
    
    # 性能测试
    if args.performance:
        try:
            dnabert2_client = DNABERT2Client(args.dnabert2_url)
            dnabert2_perf = test_performance(dnabert2_client, "DNABERT2", args.requests)
            all_results["performance"]["dnabert2"] = dnabert2_perf
        except Exception as e:
            print(f"DNABERT2性能测试失败: {e}")
            all_results["performance"]["dnabert2"] = {"error": str(e)}
        
        try:
            lucaone_client = LucaOneClient(args.lucaone_url)
            lucaone_perf = test_performance(lucaone_client, "LucaOne", args.requests)
            all_results["performance"]["lucaone"] = lucaone_perf
        except Exception as e:
            print(f"LucaOne性能测试失败: {e}")
            all_results["performance"]["lucaone"] = {"error": str(e)}
    
    # 打印总结
    print_test_header("测试总结")
    total_tests = dnabert2_results["total"] + lucaone_results["total"]
    total_passed = dnabert2_results["passed"] + lucaone_results["passed"]
    total_failed = dnabert2_results["failed"] + lucaone_results["failed"]
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"成功率: {total_passed/total_tests*100:.1f}%")
    
    # 保存结果
    save_results(all_results, args.output)
    
    # 返回适当的退出码
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())