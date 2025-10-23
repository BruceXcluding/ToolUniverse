#!/usr/bin/env python3
"""
容器监控示例脚本
演示如何使用容器监控功能
"""

import time
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tooluniverse.bio_models.model_manager import ModelManager, ModelType, ContainerModelConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("=== 容器监控示例 ===")
    
    # 创建模型管理器
    manager = ModelManager()
    
    try:
        # 1. 注册并启动容器模型
        print("\n1. 注册并启动容器模型...")
        
        # 注册DNABERT2容器模型
        dnabert2_config = ContainerModelConfig(
            image="bioinformatics/dnabert2:latest",
            ports={"8000": 8000},
            environment={"MODEL_PATH": "/models/dnabert2"}
        )
        manager.register_container_model("dnabert2-monitor", ModelType.DNABERT2, dnabert2_config)
        manager.start_container_model("dnabert2-monitor")
        
        # 注册LucaOne容器模型
        lucaone_config = ContainerModelConfig(
            image="bioinformatics/lucaone:latest",
            ports={"8001": 8001},
            environment={"MODEL_PATH": "/models/lucaone"}
        )
        manager.register_container_model("lucaone-monitor", ModelType.LUCAONE, lucaone_config)
        manager.start_container_model("lucaone-monitor")
        
        # 2. 启动容器监控
        print("\n2. 启动容器监控...")
        manager.start_container_monitoring()
        
        # 3. 设置监控阈值
        print("\n3. 设置容器监控阈值...")
        manager.set_container_monitoring_thresholds(
            cpu_threshold=70.0,  # CPU使用率超过70%时告警
            memory_threshold=80.0,  # 内存使用率超过80%时告警
            disk_threshold=90.0  # 磁盘使用率超过90%时告警
        )
        
        # 4. 获取容器监控摘要
        print("\n4. 获取容器监控摘要...")
        monitoring_summary = manager.get_container_monitoring_summary()
        print(f"监控摘要: {json.dumps(monitoring_summary, indent=2, default=str)}")
        
        # 5. 模拟运行一段时间，让监控收集数据
        print("\n5. 运行一段时间，收集监控数据...")
        print("将运行30秒，期间会定期显示容器指标...")
        
        for i in range(6):
            time.sleep(5)
            print(f"\n--- 第 {i+1} 次检查 ---")
            
            # 获取容器指标
            for model_name in ["dnabert2-monitor", "lucaone-monitor"]:
                metrics = manager.get_container_metrics(model_name)
                if metrics:
                    print(f"{model_name} CPU使用率: {metrics['cpu_usage']:.2f}%")
                    print(f"{model_name} 内存使用率: {metrics['memory_usage']:.2f}%")
                    print(f"{model_name} 磁盘使用率: {metrics['disk_usage']:.2f}%")
                    print(f"{model_name} 网络IO: {metrics['network_io']}")
                    print(f"{model_name} 状态: {metrics['status']}")
            
            # 检查告警
            alerts = manager.get_container_alerts()
            if alerts:
                print(f"活动告警数量: {len(alerts)}")
                for alert in alerts:
                    print(f"  - {alert['name']}: {alert['message']} (严重程度: {alert['severity']})")
            else:
                print("当前没有活动告警")
        
        # 6. 获取容器指标历史
        print("\n6. 获取容器指标历史...")
        for model_name in ["dnabert2-monitor", "lucaone-monitor"]:
            history = manager.get_container_metrics_history(model_name)
            if history:
                print(f"{model_name} 指标历史记录数: {len(history)}")
                
                # 计算平均CPU和内存使用率
                avg_cpu = sum(m['cpu_usage'] for m in history) / len(history)
                avg_memory = sum(m['memory_usage'] for m in history) / len(history)
                max_cpu = max(m['cpu_usage'] for m in history)
                max_memory = max(m['memory_usage'] for m in history)
                
                print(f"  平均CPU使用率: {avg_cpu:.2f}%")
                print(f"  平均内存使用率: {avg_memory:.2f}%")
                print(f"  最大CPU使用率: {max_cpu:.2f}%")
                print(f"  最大内存使用率: {max_memory:.2f}%")
        
        # 7. 获取特定时间范围的指标历史
        print("\n7. 获取最近10秒的指标历史...")
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=10)
        
        for model_name in ["dnabert2-monitor", "lucaone-monitor"]:
            recent_history = manager.get_container_metrics_history(
                model_name, start_time=start_time, end_time=end_time
            )
            if recent_history:
                print(f"{model_name} 最近10秒记录数: {len(recent_history)}")
        
        # 8. 获取所有告警历史
        print("\n8. 获取所有告警历史...")
        all_alerts = manager.get_container_alerts(active_only=False)
        if all_alerts:
            print(f"总告警数: {len(all_alerts)}")
            for alert in all_alerts:
                status = "已解决" if alert['resolved'] else "活动"
                print(f"  - {alert['name']} [{status}]: {alert['message']}")
        else:
            print("没有告警历史")
        
        # 9. 调整监控阈值
        print("\n9. 调整监控阈值...")
        manager.set_container_monitoring_thresholds(
            cpu_threshold=60.0,  # 降低CPU阈值
            memory_threshold=75.0,  # 降低内存阈值
            disk_threshold=85.0  # 降低磁盘阈值
        )
        print("已调整监控阈值")
        
        # 10. 再运行一段时间，观察新阈值下的告警
        print("\n10. 使用新阈值运行10秒...")
        time.sleep(10)
        
        # 检查新阈值下的告警
        alerts = manager.get_container_alerts()
        if alerts:
            print(f"新阈值下的活动告警数量: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert['name']}: {alert['message']} (严重程度: {alert['severity']})")
        else:
            print("新阈值下没有活动告警")
        
        # 11. 停止容器监控
        print("\n11. 停止容器监控...")
        manager.stop_container_monitoring()
        
    except Exception as e:
        logger.error(f"示例运行出错: {str(e)}")
        
    finally:
        # 12. 清理资源
        print("\n12. 清理资源...")
        try:
            manager.shutdown()
        except Exception as e:
            logger.error(f"清理资源时出错: {str(e)}")
    
    print("\n容器监控示例完成!")

if __name__ == "__main__":
    main()