#!/usr/bin/env python3
"""
ToolUniverse监控系统测试脚本
"""

import sys
import os
import time
import random
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tooluniverse.bio_models.monitoring import get_logger, monitor_performance, monitor_context, metrics_collector, log_io_shapes
from src.tooluniverse.bio_models.model_manager import ModelManager
from src.tooluniverse.bio_models.tools.dnabert2_tool import DNABERT2Model
from src.tooluniverse.bio_models.tools.utrlm_tool import UTRLMModel
from src.tooluniverse.bio_models.task_types import DeviceType


def test_logging():
    """测试日志系统"""
    print("测试日志系统...")
    logger = get_logger("test")
    
    logger.info("测试信息日志", test_id="logging_test", step=1)
    logger.warning("测试警告日志", test_id="logging_test", step=2)
    logger.error("测试错误日志", test_id="logging_test", step=3)
    
    print("日志系统测试完成")


@monitor_performance(model_name="test_model", task_type="test_task")
def test_performance_monitoring():
    """测试性能监控"""
    print("测试性能监控...")
    
    with monitor_context("test_operation", param1="value1", param2=42):
        # 模拟一些工作
        time.sleep(0.1)
        
        # 记录输入输出
        log_io_shapes(
            input_data={"param1": "value1", "param2": 42},
            output_data={"result": "success"},
            task_type="test_task"
        )
        
        # 记录指标
        metrics_collector.record_metric("test_operation_success", 1, {"model": "test_model"})
        
        print("性能监控测试完成")


def test_model_manager():
    """测试ModelManager的监控功能"""
    print("测试ModelManager监控功能...")
    
    # 创建ModelManager实例
    manager = ModelManager()
    
    # 测试模型注册
    print("注册模型...")
    from src.tooluniverse.bio_models.task_types import TaskType, SequenceType, ModelConfig
    
    # 创建一个虚拟模型配置
    model_config = ModelConfig(
        model_name="test_dnabert2",
        model_class="DNABERT2Model",
        model_path="/test/path",
        tasks=[TaskType.PROMOTER_PREDICTION],
        sequence_types=[SequenceType.DNA],
        device_types=[DeviceType.CPU]
    )
    
    # 注册模型
    manager.register_model(model_config)
    
    print("ModelManager监控功能测试完成")


def test_tool_models():
    """测试工具模型的监控功能"""
    print("测试工具模型监控功能...")
    
    # 测试DNABERT2模型
    print("测试DNABERT2模型...")
    dnabert2 = DNABERT2Model("/test/path", DeviceType.CPU)
    
    # 加载模型
    dnabert2.load_model()
    
    # 预测启动子活性
    dna_sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
    results = dnabert2.predict_promoter_activity(dna_sequences)
    
    # 卸载模型
    dnabert2.unload_model()
    
    # 测试UTRLM模型
    print("测试UTRLM模型...")
    utrlm = UTRLMModel("/test/path", DeviceType.CPU)
    
    # 加载模型
    utrlm.load_model()
    
    # 预测翻译效率
    utr_sequences = ["AUGCUAGCUAGC", "UCGAUCGAUCGA"]
    results = utrlm.predict_translation_efficiency(utr_sequences)
    
    # 卸载模型
    utrlm.unload_model()
    
    print("工具模型监控功能测试完成")


def test_metrics_collection():
    """测试指标收集"""
    print("测试指标收集...")
    
    # 记录一些测试指标
    for i in range(5):
        metrics_collector.record_metric(
            "test_metric", 
            random.uniform(0.1, 1.0), 
            {"iteration": i, "type": "test"}
        )
    
    # 获取指标数据
    metrics_data = metrics_collector.get_metrics_data()
    print(f"收集到 {len(metrics_data)} 种指标")
    
    print("指标收集测试完成")


def main():
    """主函数"""
    print("开始ToolUniverse监控系统测试...")
    print("=" * 50)
    
    try:
        # 测试日志系统
        test_logging()
        print()
        
        # 测试性能监控
        test_performance_monitoring()
        print()
        
        # 测试指标收集
        test_metrics_collection()
        print()
        
        # 测试ModelManager
        test_model_manager()
        print()
        
        # 测试工具模型
        test_tool_models()
        print()
        
        print("=" * 50)
        print("所有测试完成！监控系统运行正常。")
        
        # 提示用户如何查看监控仪表板
        print("\n要查看监控仪表板，请运行:")
        print("python start_dashboard.py")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())