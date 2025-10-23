"""
性能数据收集模块
用于收集和分析容器化模型的性能数据
"""

import time
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import statistics


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    model_name: str
    deployment_type: str  # "local" 或 "container"
    metric_type: str  # "inference_time", "throughput", "memory_usage", "cpu_usage", etc.
    value: float
    unit: str  # "ms", "requests/sec", "MB", "%", etc.
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """性能报告数据类"""
    model_name: str
    deployment_type: str
    start_time: datetime
    end_time: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, Any]


class PerformanceCollector:
    """性能数据收集器"""
    
    def __init__(self, output_dir: str = "performance_data"):
        """
        初始化性能数据收集器
        
        Args:
            output_dir: 输出目录
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 存储性能指标
        self.metrics: List[PerformanceMetric] = []
        self.metrics_lock = threading.Lock()
        
        # 存储性能报告
        self.reports: List[PerformanceReport] = []
        
        # 自动收集标志
        self.auto_collect = False
        self.collect_thread = None
        self.collect_interval = 60  # 默认60秒收集一次
        
        # 性能回调函数
        self.performance_callbacks: List[Callable[[PerformanceMetric], None]] = []
    
    def add_metric(
        self,
        model_name: str,
        deployment_type: str,
        metric_type: str,
        value: float,
        unit: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加性能指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            metric_type: 指标类型
            value: 指标值
            unit: 单位
            timestamp: 时间戳，如果为None则使用当前时间
            metadata: 元数据
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type=metric_type,
            value=value,
            unit=unit,
            metadata=metadata
        )
        
        with self.metrics_lock:
            self.metrics.append(metric)
        
        # 调用性能回调
        for callback in self.performance_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"性能回调执行失败: {str(e)}")
    
    def add_inference_time(
        self,
        model_name: str,
        deployment_type: str,
        inference_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加推理时间指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            inference_time_ms: 推理时间（毫秒）
            metadata: 元数据
        """
        self.add_metric(
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type="inference_time",
            value=inference_time_ms,
            unit="ms",
            metadata=metadata
        )
    
    def add_throughput(
        self,
        model_name: str,
        deployment_type: str,
        requests_per_sec: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加吞吐量指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            requests_per_sec: 每秒请求数
            metadata: 元数据
        """
        self.add_metric(
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type="throughput",
            value=requests_per_sec,
            unit="requests/sec",
            metadata=metadata
        )
    
    def add_memory_usage(
        self,
        model_name: str,
        deployment_type: str,
        memory_mb: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加内存使用指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            memory_mb: 内存使用量（MB）
            metadata: 元数据
        """
        self.add_metric(
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type="memory_usage",
            value=memory_mb,
            unit="MB",
            metadata=metadata
        )
    
    def add_cpu_usage(
        self,
        model_name: str,
        deployment_type: str,
        cpu_percent: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加CPU使用指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            cpu_percent: CPU使用率（百分比）
            metadata: 元数据
        """
        self.add_metric(
            model_name=model_name,
            deployment_type=deployment_type,
            metric_type="cpu_usage",
            value=cpu_percent,
            unit="%",
            metadata=metadata
        )
    
    def get_metrics(
        self,
        model_name: Optional[str] = None,
        deployment_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceMetric]:
        """
        获取性能指标
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            metric_type: 指标类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[PerformanceMetric]: 符合条件的性能指标列表
        """
        with self.metrics_lock:
            filtered_metrics = self.metrics.copy()
        
        # 应用过滤条件
        if model_name:
            filtered_metrics = [m for m in filtered_metrics if m.model_name == model_name]
        
        if deployment_type:
            filtered_metrics = [m for m in filtered_metrics if m.deployment_type == deployment_type]
        
        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
        
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        return filtered_metrics
    
    def generate_report(
        self,
        model_name: str,
        deployment_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> PerformanceReport:
        """
        生成性能报告
        
        Args:
            model_name: 模型名称
            deployment_type: 部署类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            PerformanceReport: 性能报告
        """
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # 默认使用最近24小时的数据
            start_time = end_time - timedelta(hours=24)
        
        # 获取指标数据
        metrics = self.get_metrics(
            model_name=model_name,
            deployment_type=deployment_type,
            start_time=start_time,
            end_time=end_time
        )
        
        # 按指标类型分组
        metrics_by_type = {}
        for metric in metrics:
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # 计算摘要统计
        summary = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[metric_type] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # 创建报告
        report = PerformanceReport(
            model_name=model_name,
            deployment_type=deployment_type,
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            summary=summary
        )
        
        # 存储报告
        self.reports.append(report)
        
        return report
    
    def compare_models(
        self,
        model_names: List[str],
        metric_type: str,
        deployment_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        比较模型性能
        
        Args:
            model_names: 模型名称列表
            metric_type: 指标类型
            deployment_type: 部署类型
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison = {}
        
        for model_name in model_names:
            metrics = self.get_metrics(
                model_name=model_name,
                deployment_type=deployment_type,
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time
            )
            
            if metrics:
                values = [m.value for m in metrics]
                comparison[model_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0
                }
            else:
                comparison[model_name] = {
                    "count": 0,
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "stdev": 0
                }
        
        return comparison
    
    def save_metrics_to_csv(self, file_path: Optional[str] = None) -> str:
        """
        将指标保存到CSV文件
        
        Args:
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.output_dir / f"metrics_{timestamp}.csv")
        
        with self.metrics_lock:
            metrics = self.metrics.copy()
        
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'model_name', 'deployment_type', 'metric_type',
                'value', 'unit', 'metadata'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in metrics:
                writer.writerow({
                    'timestamp': metric.timestamp.isoformat(),
                    'model_name': metric.model_name,
                    'deployment_type': metric.deployment_type,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': json.dumps(metric.metadata) if metric.metadata else ''
                })
        
        self.logger.info(f"指标已保存到: {file_path}")
        return file_path
    
    def save_report_to_json(self, report: PerformanceReport, file_path: Optional[str] = None) -> str:
        """
        将报告保存到JSON文件
        
        Args:
            report: 性能报告
            file_path: 文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.output_dir / f"report_{report.model_name}_{timestamp}.json")
        
        # 转换为可序列化的字典
        report_dict = asdict(report)
        report_dict['start_time'] = report.start_time.isoformat()
        report_dict['end_time'] = report.end_time.isoformat()
        
        # 转换指标列表
        metrics_list = []
        for metric in report.metrics:
            metric_dict = asdict(metric)
            metric_dict['timestamp'] = metric.timestamp.isoformat()
            metrics_list.append(metric_dict)
        
        report_dict['metrics'] = metrics_list
        
        with open(file_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"报告已保存到: {file_path}")
        return file_path
    
    def register_performance_callback(self, callback: Callable[[PerformanceMetric], None]) -> None:
        """
        注册性能回调函数
        
        Args:
            callback: 回调函数，接收PerformanceMetric参数
        """
        self.performance_callbacks.append(callback)
    
    def start_auto_collect(self, interval: int = 60) -> None:
        """
        启动自动收集
        
        Args:
            interval: 收集间隔（秒）
        """
        if self.auto_collect:
            self.logger.warning("自动收集已在运行")
            return
        
        self.collect_interval = interval
        self.auto_collect = True
        
        def collect_loop():
            while self.auto_collect:
                try:
                    # 这里可以添加自动收集逻辑
                    # 例如从监控系统获取指标
                    time.sleep(self.collect_interval)
                except Exception as e:
                    self.logger.error(f"自动收集出错: {str(e)}")
                    time.sleep(self.collect_interval)
        
        self.collect_thread = threading.Thread(target=collect_loop, daemon=True)
        self.collect_thread.start()
        
        self.logger.info(f"自动收集已启动，间隔: {interval}秒")
    
    def stop_auto_collect(self) -> None:
        """停止自动收集"""
        if not self.auto_collect:
            self.logger.warning("自动收集未在运行")
            return
        
        self.auto_collect = False
        
        if self.collect_thread and self.collect_thread.is_alive():
            self.collect_thread.join(timeout=5)
        
        self.logger.info("自动收集已停止")
    
    def clear_metrics(self) -> None:
        """清除所有指标"""
        with self.metrics_lock:
            self.metrics.clear()
        
        self.logger.info("所有指标已清除")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取收集器摘要
        
        Returns:
            Dict[str, Any]: 摘要信息
        """
        with self.metrics_lock:
            metrics_count = len(self.metrics)
            reports_count = len(self.reports)
        
        # 获取模型列表
        models = set()
        for metric in self.metrics:
            models.add(metric.model_name)
        
        # 获取指标类型列表
        metric_types = set()
        for metric in self.metrics:
            metric_types.add(metric.metric_type)
        
        return {
            "metrics_count": metrics_count,
            "reports_count": reports_count,
            "models": list(models),
            "metric_types": list(metric_types),
            "auto_collect": self.auto_collect,
            "collect_interval": self.collect_interval if self.auto_collect else None
        }