#!/usr/bin/env python3
"""
测试Agent API的脚本
"""

import requests
import json
import time

# API基础URL
BASE_URL = "http://localhost:5000"

def test_health():
    """测试健康检查"""
    print("=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_list_models():
    """测试列出模型"""
    print("=== 测试列出模型 ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_list_loaded_models():
    """测试列出已加载模型"""
    print("=== 测试列出已加载模型 ===")
    response = requests.get(f"{BASE_URL}/models/loaded")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_get_best_model():
    """测试获取最佳模型"""
    print("=== 测试获取最佳模型 ===")
    data = {
        "task_type": "classification",
        "sequence_type": "DNA"
    }
    response = requests.post(f"{BASE_URL}/models/best", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_analyze_sequences():
    """测试序列分析"""
    print("=== 测试序列分析 ===")
    data = {
        "sequences": ["ATCGATCGATCG", "GCTAGCTAGCTA"],
        "task_type": "classification",
        "model_name": "dnabert2-container"
    }
    response = requests.post(f"{BASE_URL}/analyze", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_supported_tasks():
    """测试获取支持的任务类型"""
    print("=== 测试获取支持的任务类型 ===")
    response = requests.get(f"{BASE_URL}/tasks")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_get_supported_sequence_types():
    """测试获取支持的序列类型"""
    print("=== 测试获取支持的序列类型 ===")
    response = requests.get(f"{BASE_URL}/sequence-types")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_model_info():
    """测试获取模型信息"""
    print("=== 测试获取模型信息 ===")
    response = requests.get(f"{BASE_URL}/model/dnabert2-container/info")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def main():
    """主函数"""
    print("开始测试ToolUniverse Agent API...")
    print(f"API地址: {BASE_URL}")
    print()
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(5)
    
    # 运行测试
    try:
        test_health()
        test_list_models()
        test_list_loaded_models()
        test_get_best_model()
        test_get_supported_tasks()
        test_get_supported_sequence_types()
        test_model_info()
        test_analyze_sequences()
        
        print("所有测试完成!")
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main()