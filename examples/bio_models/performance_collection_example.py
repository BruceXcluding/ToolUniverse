#!/usr/bin/env python3
"""
性能数据收集示例脚本
演示如何使用ModelManager的性能收集功能
"""

import os
import sys
import time
import random
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "..", "..", "src"))

def main():
    """主函数"""
    print("性能数据收集示例")
    print("=" * 50)
    
    try:
        # 导入必要模块
        from tooluniverse.bio_models.model_manager import ModelManager, ModelType, DeploymentType, ContainerModelConfig
        from tooluniverse.bio_models.performance_collector import PerformanceMetric
        
        # 创建模型管理器
        print("\n1. 初始化模型管理器...")
        manager = ModelManager()
        
        # 配置容器模型
        print("\n2. 配置容器模型...")
        container_config = ContainerModelConfig(
            name="dnabert2-container",
            model_type=ModelType.DNABERT2,
            image="dnabert2:latest",
            ports={"5000": 5000},
            environment={"MODEL_PATH": "/models/dnabert2"},
            memory_limit="4g",
            gpu_enabled=False
        )
        
        # 注册容器模型
        manager.register_container_model(container_config)
        
        # 启动容器模型
        print("\n3. 启动容器模型...")
        success = manager.start_container_model("dnabert2-container")
        if not success:
            print("启动容器模型失败，使用模拟数据继续演示")
        else:
            print("容器模型启动成功")
        
        # 启动性能收集
        print("\n4. 启动性能收集...")
        manager.start_performance_collection(interval=5)  # 每5秒收集一次
        
        # 模拟一些推理操作并收集性能数据
        print("\n5. 模拟推理操作并收集性能数据...")
        for i in range(10):
            # 模拟推理时间
            inference_time = random.uniform(0.1, 0.5)
            manager.collect_inference_time("dnabert2-container", inference_time)
            
            # 模拟吞吐量
            throughput = random.uniform(10, 50)
            manager.collect_throughput("dnabert2-container", throughput)
            
            # 模拟内存使用
            memory_usage = random.uniform(1.0, 4.0)
            manager.collect_memory_usage("dnabert2-container", memory_usage)
            
            # 模拟CPU使用
            cpu_usage = random.uniform(10, 80)
            manager.collect_cpu_usage("dnabert2-container", cpu_usage)
            
            print(f"  已收集第 {i+1} 批性能数据")
            time.sleep(2)  # 等待2秒
        
        # 获取性能指标
        print("\n6. 获取性能指标...")
        metrics = manager.get_performance_metrics("dnabert2-container")
        print(f"  共收集到 {len(metrics)} 条性能指标")
        
        # 生成性能报告
        print("\n7. 生成性能报告...")
        report = manager.generate_performance_report("dnabert2-container")
        print(f"  报告生成时间: {report.timestamp}")
        print(f"  平均推理时间: {report.avg_inference_time:.4f}s")
        print(f"  平均吞吐量: {report.avg_throughput:.2f} req/s")
        print(f"  平均内存使用: {report.avg_memory_usage:.2f}GB")
        print(f"  平均CPU使用: {report.avg_cpu_usage:.2f}%")
        
        # 保存性能报告
        print("\n8. 保存性能报告...")
        report_path = manager.save_performance_report("dnabert2-container")
        print(f"  报告已保存到: {report_path}")
        
        # 获取性能摘要
        print("\n9. 获取性能摘要...")
        summary = manager.get_performance_summary()
        print(f"  摘要生成时间: {summary['timestamp']}")
        print(f"  监控的模型数量: {summary['model_count']}")
        for model_name, model_summary in summary['models'].items():
            print(f"  模型 {model_name}:")
            print(f"    总请求数: {model_summary['total_requests']}")
            print(f"    平均推理时间: {model_summary['avg_inference_time']:.4f}s")
            print(f"    平均吞吐量: {model_summary['avg_throughput']:.2f} req/s")
        
        # 停止性能收集
        print("\n10. 停止性能收集...")
        manager.stop_performance_collection()
        
        # 停止容器模型
        print("\n11. 停止容器模型...")
        manager.stop_container_model("dnabert2-container")
        
        print("\n性能数据收集示例完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()