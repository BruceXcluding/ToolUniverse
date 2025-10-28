#!/usr/bin/env python3
"""
容器状态监控模块

该模块提供容器状态监控功能，集成到现有的监控系统中。
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import psutil

from .container_runtime import ContainerRuntime, ContainerStatus, ContainerInfo
from .monitoring import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class ContainerMetrics:
    """容器指标数据类"""
    name: str
    status: str
    cpu_percent: float
    memory_usage: float
    memory_limit: float
    memory_percent: float
    network_io: Dict[str, int]
    block_io: Dict[str, int]
    pids: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

@dataclass
class ContainerAlert:
    """容器告警数据类"""
    name: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.resolved_timestamp:
            data["resolved_timestamp"] = self.resolved_timestamp.isoformat()
        return data

class ContainerMonitor:
    """容器监控器"""
    
    def __init__(
        self,
        container_runtime: ContainerRuntime,
        metrics_collector: MetricsCollector,
        check_interval: int = 30,
        metrics_history_size: int = 1000
    ):
        """
        初始化容器监控器
        
        Args:
            container_runtime: 容器运行时实例
            metrics_collector: 指标收集器实例
            check_interval: 检查间隔（秒）
            metrics_history_size: 指标历史记录大小
        """
        self.container_runtime = container_runtime
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        self.metrics_history_size = metrics_history_size
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread = None
        
        # 指标存储
        self.metrics_history: Dict[str, List[ContainerMetrics]] = {}
        self.current_metrics: Dict[str, ContainerMetrics] = {}
        
        # 告警存储
        self.active_alerts: Dict[str, List[ContainerAlert]] = {}
        self.alert_history: List[ContainerAlert] = []
        
        # 告警阈值
        self.cpu_threshold = 80.0  # CPU使用率阈值（百分比）
        self.memory_threshold = 80.0  # 内存使用率阈值（百分比）
        self.disk_threshold = 80.0  # 磁盘使用率阈值（百分比）
        
        # 告警回调
        self.alert_callbacks: List[Callable[[ContainerAlert], None]] = []
        
        logger.info("容器监控器初始化完成")
    
    def start_monitoring(self, container_names: Optional[List[str]] = None) -> None:
        """
        开始监控
        
        Args:
            container_names: 要监控的容器名称列表，如果为None则监控所有容器
        """
        if self._monitoring:
            logger.warning("容器监控已在运行中")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(container_names,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("容器监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self._monitoring:
            logger.warning("容器监控未在运行")
            return
        
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("容器监控已停止")
    
    def _monitor_loop(self, container_names: Optional[List[str]] = None) -> None:
        """监控循环"""
        logger.info("容器监控循环已启动")
        
        while self._monitoring:
            try:
                # 获取要监控的容器列表
                if container_names:
                    containers = []
                    for name in container_names:
                        info = self.container_runtime.get_container_info(name)
                        if info:
                            containers.append(info)
                else:
                    containers = self.container_runtime.list_containers()
                
                # 收集每个容器的指标
                for container in containers:
                    self._collect_container_metrics(container)
                
                # 检查告警
                self._check_alerts()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"容器监控循环出错: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("容器监控循环已退出")
    
    def _collect_container_metrics(self, container: ContainerInfo) -> None:
        """
        收集容器指标
        
        Args:
            container: 容器信息
        """
        try:
            # 获取容器统计信息
            stats = self.container_runtime.get_container_stats(container.name)
            if not stats:
                return
            
            # 解析统计信息
            cpu_percent = stats.get("cpu_percent", 0.0)
            memory_usage = stats.get("memory_usage", 0)
            memory_limit = stats.get("memory_limit", 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0.0
            network_io = stats.get("network_io", {})
            block_io = stats.get("block_io", {})
            pids = stats.get("pids", 0)
            
            # 创建指标对象
            metrics = ContainerMetrics(
                name=container.name,
                status=container.status.value,
                cpu_percent=cpu_percent,
                memory_usage=memory_usage,
                memory_limit=memory_limit,
                memory_percent=memory_percent,
                network_io=network_io,
                block_io=block_io,
                pids=pids,
                timestamp=datetime.now()
            )
            
            # 存储当前指标
            self.current_metrics[container.name] = metrics
            
            # 添加到历史记录
            if container.name not in self.metrics_history:
                self.metrics_history[container.name] = []
            
            self.metrics_history[container.name].append(metrics)
            
            # 限制历史记录大小
            if len(self.metrics_history[container.name]) > self.metrics_history_size:
                self.metrics_history[container.name].pop(0)
            
            # 添加到指标收集器
            self.metrics_collector.add_container_metric(
                container.name, 
                "cpu_percent", 
                cpu_percent
            )
            self.metrics_collector.add_container_metric(
                container.name, 
                "memory_percent", 
                memory_percent
            )
            self.metrics_collector.add_container_metric(
                container.name, 
                "pids", 
                pids
            )
            
            # 记录日志
            logger.debug(f"收集容器指标: {container.name}, CPU: {cpu_percent:.2f}%, 内存: {memory_percent:.2f}%")
            
        except Exception as e:
            logger.error(f"收集容器指标失败: {container.name}, 错误: {str(e)}")
    
    def _check_alerts(self) -> None:
        """检查告警条件"""
        for name, metrics in self.current_metrics.items():
            # 检查CPU使用率
            if metrics.cpu_percent > self.cpu_threshold:
                self._create_alert(
                    name=name,
                    alert_type="cpu_high",
                    severity="warning",
                    message=f"容器CPU使用率过高: {metrics.cpu_percent:.2f}%"
                )
            else:
                self._resolve_alert(
                    name=name,
                    alert_type="cpu_high"
                )
            
            # 检查内存使用率
            if metrics.memory_percent > self.memory_threshold:
                self._create_alert(
                    name=name,
                    alert_type="memory_high",
                    severity="warning",
                    message=f"容器内存使用率过高: {metrics.memory_percent:.2f}%"
                )
            else:
                self._resolve_alert(
                    name=name,
                    alert_type="memory_high"
                )
            
            # 检查容器状态
            if metrics.status != ContainerStatus.RUNNING.value:
                self._create_alert(
                    name=name,
                    alert_type="container_down",
                    severity="critical",
                    message=f"容器状态异常: {metrics.status}"
                )
            else:
                self._resolve_alert(
                    name=name,
                    alert_type="container_down"
                )
    
    def _create_alert(
        self,
        name: str,
        alert_type: str,
        severity: str,
        message: str
    ) -> None:
        """
        创建告警
        
        Args:
            name: 容器名称
            alert_type: 告警类型
            severity: 严重程度
            message: 告警消息
        """
        # 检查是否已有相同类型的未解决告警
        if name in self.active_alerts:
            for alert in self.active_alerts[name]:
                if alert.alert_type == alert_type and not alert.resolved:
                    return  # 已有相同类型的未解决告警
        
        # 创建新告警
        alert = ContainerAlert(
            name=name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        
        # 添加到活动告警
        if name not in self.active_alerts:
            self.active_alerts[name] = []
        self.active_alerts[name].append(alert)
        
        # 添加到历史记录
        self.alert_history.append(alert)
        
        # 记录日志
        logger.warning(f"容器告警: {name} - {message}")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {str(e)}")
    
    def _resolve_alert(self, name: str, alert_type: str) -> None:
        """
        解决告警
        
        Args:
            name: 容器名称
            alert_type: 告警类型
        """
        if name not in self.active_alerts:
            return
        
        for alert in self.active_alerts[name]:
            if alert.alert_type == alert_type and not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = datetime.now()
                
                # 记录日志
                logger.info(f"容器告警已解决: {name} - {alert.message}")
                
                # 调用告警回调
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调执行失败: {str(e)}")
    
    def get_container_metrics(self, name: str) -> Optional[ContainerMetrics]:
        """
        获取容器当前指标
        
        Args:
            name: 容器名称
            
        Returns:
            Optional[ContainerMetrics]: 容器指标，如果不存在则返回None
        """
        return self.current_metrics.get(name)
    
    def get_container_metrics_history(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ContainerMetrics]:
        """
        获取容器指标历史记录
        
        Args:
            name: 容器名称
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[ContainerMetrics]: 指标历史记录
        """
        if name not in self.metrics_history:
            return []
        
        history = self.metrics_history[name]
        
        # 过滤时间范围
        if start_time or end_time:
            filtered = []
            for metrics in history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                filtered.append(metrics)
            return filtered
        
        return history
    
    def get_active_alerts(self, name: Optional[str] = None) -> List[ContainerAlert]:
        """
        获取活动告警
        
        Args:
            name: 容器名称，如果为None则返回所有容器的活动告警
            
        Returns:
            List[ContainerAlert]: 活动告警列表
        """
        alerts = []
        
        if name:
            if name in self.active_alerts:
                alerts.extend([a for a in self.active_alerts[name] if not a.resolved])
        else:
            for container_alerts in self.active_alerts.values():
                alerts.extend([a for a in container_alerts if not a.resolved])
        
        return alerts
    
    def get_alert_history(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ContainerAlert]:
        """
        获取告警历史记录
        
        Args:
            name: 容器名称，如果为None则返回所有容器的告警历史
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[ContainerAlert]: 告警历史记录
        """
        history = self.alert_history
        
        # 过滤容器名称
        if name:
            history = [a for a in history if a.name == name]
        
        # 过滤时间范围
        if start_time or end_time:
            filtered = []
            for alert in history:
                if start_time and alert.timestamp < start_time:
                    continue
                if end_time and alert.timestamp > end_time:
                    continue
                filtered.append(alert)
            history = filtered
        
        return history
    
    def add_alert_callback(self, callback: Callable[[ContainerAlert], None]) -> None:
        """
        添加告警回调函数
        
        Args:
            callback: 回调函数，接收ContainerAlert参数
        """
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[ContainerAlert], None]) -> None:
        """
        移除告警回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def set_thresholds(
        self,
        cpu_threshold: Optional[float] = None,
        memory_threshold: Optional[float] = None,
        disk_threshold: Optional[float] = None
    ) -> None:
        """
        设置告警阈值
        
        Args:
            cpu_threshold: CPU使用率阈值（百分比）
            memory_threshold: 内存使用率阈值（百分比）
            disk_threshold: 磁盘使用率阈值（百分比）
        """
        if cpu_threshold is not None:
            self.cpu_threshold = cpu_threshold
        if memory_threshold is not None:
            self.memory_threshold = memory_threshold
        if disk_threshold is not None:
            self.disk_threshold = disk_threshold
        
        logger.info(f"告警阈值已更新: CPU={self.cpu_threshold}%, 内存={self.memory_threshold}%, 磁盘={self.disk_threshold}%")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取监控摘要
        
        Returns:
            Dict[str, Any]: 监控摘要
        """
        total_containers = len(self.current_metrics)
        running_containers = sum(
            1 for m in self.current_metrics.values()
            if m.status == ContainerStatus.RUNNING.value
        )
        active_alerts = len(self.get_active_alerts())
        
        return {
            "total_containers": total_containers,
            "running_containers": running_containers,
            "active_alerts": active_alerts,
            "monitoring": self._monitoring,
            "check_interval": self.check_interval,
            "thresholds": {
                "cpu": self.cpu_threshold,
                "memory": self.memory_threshold,
                "disk": self.disk_threshold
            }
        }