#!/usr/bin/env python3
"""
容器模型管理示例

这个示例展示了如何使用扩展后的ModelManager来管理容器模型和本地模型。
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tooluniverse.bio_models.model_manager import (
    ModelManager, 
    ModelType, 
    TaskType, 
    SequenceType,
    DeploymentType,
    ContainerModelConfig
)
from tooluniverse.utils.logger import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 创建模型管理器
    manager = ModelManager()
    
    try:
        # 1. 注册容器模型
        print("\n=== 注册容器模型 ===")
        dnabert2_config = ContainerModelConfig(
            name="dnabert2-container",
            model_type=ModelType.DNABERT2,
            image="tooluniverse/dnabert2:latest",
            ports={"8000": "8000"},
            environment={"MODEL_PATH": "/models/dnabert2"},
            description="DNABERT2容器模型",
            version="1.0.0",
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.EXTRACTION],
            supported_sequences=[SequenceType.DNA, SequenceType.RNA]
        )
        
        lucaone_config = ContainerModelConfig(
            name="lucaone-container",
            model_type=ModelType.LUCAONE,
            image="tooluniverse/lucaone:latest",
            ports={"8001": "8000"},
            environment={"MODEL_PATH": "/models/lucaone"},
            description="LucaOne容器模型",
            version="1.0.0",
            supported_tasks=[TaskType.PREDICTION, TaskType.ANNOTATION],
            supported_sequences=[SequenceType.DNA, SequenceType.PROTEIN]
        )
        
        # 注册容器模型
        manager.register_container_model(dnabert2_config)
        manager.register_container_model(lucaone_config)
        
        # 2. 启动容器模型
        print("\n=== 启动容器模型 ===")
        manager.start_container_model("dnabert2-container")
        manager.start_container_model("lucaone-container")
        
        # 等待容器启动
        print("等待容器启动...")
        time.sleep(10)
        
        # 3. 列出所有模型
        print("\n=== 列出所有模型 ===")
        models = manager.list_models()
        print(f"本地模型数量: {models['total_local']}")
        print(f"容器模型数量: {models['total_container']}")
        print(f"总模型数量: {models['total']}")
        
        # 4. 列出已加载/运行中的模型
        print("\n=== 列出已加载/运行中的模型 ===")
        loaded = manager.list_loaded_models()
        print(f"本地已加载模型数量: {loaded['total_local']}")
        print(f"容器运行中模型数量: {loaded['total_container']}")
        print(f"总运行中模型数量: {loaded['total']}")
        
        # 5. 获取最佳模型
        print("\n=== 获取最佳模型 ===")
        best_model = manager.get_best_model(TaskType.CLASSIFICATION, SequenceType.DNA)
        print(f"分类任务最佳模型: {best_model}")
        
        best_model = manager.get_best_model(TaskType.PREDICTION, SequenceType.PROTEIN)
        print(f"预测任务最佳模型: {best_model}")
        
        # 6. 使用容器模型进行预测
        print("\n=== 使用容器模型进行预测 ===")
        test_sequence = "ATCGATCGATCG"
        
        # 使用DNABERT2容器模型进行分类
        result = manager.predict(
            model_name="dnabert2-container",
            sequence=test_sequence,
            task_type="classification"
        )
        print(f"DNABERT2分类结果: {result}")
        
        # 使用LucaOne容器模型进行预测
        result = manager.predict(
            model_name="lucaone-container",
            sequence=test_sequence,
            task_type="prediction"
        )
        print(f"LucaOne预测结果: {result}")
        
        # 7. 批量预测
        print("\n=== 批量预测 ===")
        test_sequences = [
            "ATCGATCGATCG",
            "GCTAGCTAGCTA",
            "TTTTAAAACCCCGGGG"
        ]
        
        # 使用DNABERT2容器模型进行批量分类
        result = manager.predict_batch(
            model_name="dnabert2-container",
            sequences=test_sequences,
            task_type="classification"
        )
        print(f"DNABERT2批量分类结果: {result}")
        
        # 8. 获取容器模型信息
        print("\n=== 获取容器模型信息 ===")
        info = manager.get_container_model_info("dnabert2-container")
        print(f"DNABERT2容器信息: {info}")
        
        # 9. 获取容器模型指标
        print("\n=== 获取容器模型指标 ===")
        metrics = manager.get_container_model_metrics("dnabert2-container")
        print(f"DNABERT2容器指标: {metrics}")
        
        # 10. 健康检查
        print("\n=== 健康检查 ===")
        health = manager.check_container_model_health("dnabert2-container")
        print(f"DNABERT2健康状态: {health}")
        
        # 11. 启动容器监控
        print("\n=== 启动容器监控 ===")
        manager.start_container_monitoring()
        
        # 12. 设置监控阈值
        print("\n=== 设置容器监控阈值 ===")
        manager.set_container_monitoring_thresholds(
            cpu_threshold=80.0,  # CPU使用率超过80%时告警
            memory_threshold=85.0,  # 内存使用率超过85%时告警
            disk_threshold=90.0  # 磁盘使用率超过90%时告警
        )
        
        # 13. 获取容器监控摘要
        print("\n=== 获取容器监控摘要 ===")
        monitoring_summary = manager.get_container_monitoring_summary()
        print(f"监控摘要: {monitoring_summary}")
        
        # 14. 获取容器指标
        print("\n=== 获取容器指标 ===")
        for model_name in ["dnabert2-container", "lucaone-container"]:
            metrics = manager.get_container_metrics(model_name)
            if metrics:
                print(f"{model_name} 指标: {metrics}")
        
        # 15. 等待一段时间，让监控收集数据
        print("\n=== 等待监控收集数据 ===")
        time.sleep(5)
        
        # 16. 获取容器告警
        print("\n=== 获取容器告警 ===")
        alerts = manager.get_container_alerts()
        if alerts:
            print(f"活动告警: {alerts}")
        else:
            print("当前没有活动告警")
        
        # 17. 获取容器指标历史
        print("\n=== 获取容器指标历史 ===")
        for model_name in ["dnabert2-container", "lucaone-container"]:
            history = manager.get_container_metrics_history(model_name)
            if history:
                print(f"{model_name} 指标历史记录数: {len(history)}")
                if history:
                    print(f"最新指标: {history[-1]}")
        
        # 18. 停止容器监控
        print("\n=== 停止容器监控 ===")
        manager.stop_container_monitoring()
        
    except Exception as e:
        logger.error(f"示例运行出错: {str(e)}")
        
    finally:
        # 19. 清理资源
        print("\n=== 清理资源 ===")
        try:
            # 停止容器模型
            manager.stop_container_model("dnabert2-container")
            manager.stop_container_model("lucaone-container")
            
            # 删除容器
            manager.remove_container_model("dnabert2-container")
            manager.remove_container_model("lucaone-container")
            
            print("容器模型已停止并删除")
        except Exception as e:
            logger.error(f"清理资源出错: {str(e)}")
        
        # 关闭模型管理器
        manager.shutdown()

if __name__ == "__main__":
    main()