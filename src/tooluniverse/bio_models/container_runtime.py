"""
容器运行时管理模块
提供对模型容器的管理功能，包括启动、停止、监控等
"""

import os
import json
import time
import logging
import docker
import requests
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .monitoring import get_logger, monitor_performance, metrics_collector


class ContainerStatus(Enum):
    """容器状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class ContainerConfig:
    """容器配置"""
    name: str
    image: str
    ports: Dict[str, int]  # 容器端口到主机端口的映射
    environment: Dict[str, str] = None
    volumes: Dict[str, Dict[str, str]] = None  # 主机路径到容器路径的映射
    gpu_enabled: bool = False
    memory_limit: str = "8g"  # 内存限制，如"8g"表示8GB
    restart_policy: str = "unless-stopped"
    healthcheck: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.volumes is None:
            self.volumes = {}
        if self.healthcheck is None:
            self.healthcheck = {}


@dataclass
class ContainerInfo:
    """容器信息"""
    id: str
    name: str
    status: ContainerStatus
    image: str
    ports: Dict[str, int]
    created_at: str
    started_at: Optional[str] = None
    health_status: Optional[str] = None
    api_endpoint: Optional[str] = None
    metrics_endpoint: Optional[str] = None


class ContainerRuntime:
    """容器运行时管理器"""
    
    def __init__(self):
        """初始化容器运行时管理器"""
        self.logger = get_logger(__name__)
        self.containers: Dict[str, ContainerInfo] = {}
        
        try:
            self.client = docker.from_env()
            self.client.ping()
            self.logger.info("Docker客户端初始化成功")
        except Exception as e:
            self.logger.error(f"Docker客户端初始化失败: {e}")
            self.client = None
    
    @monitor_performance(model_name="container_runtime")
    def start_container(self, config: ContainerConfig) -> Optional[ContainerInfo]:
        """
        启动容器
        
        Args:
            config: 容器配置
            
        Returns:
            Optional[ContainerInfo]: 容器信息，如果启动失败则返回None
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return None
        
        try:
            # 检查容器是否已存在
            existing_container = None
            try:
                existing_container = self.client.containers.get(config.name)
                if existing_container.status == "running":
                    self.logger.info(f"容器已在运行: {config.name}")
                    return self._get_container_info(existing_container)
                else:
                    self.logger.info(f"停止并删除现有容器: {config.name}")
                    existing_container.stop()
                    existing_container.remove()
            except docker.errors.NotFound:
                pass
            
            # 准备端口映射
            port_bindings = {}
            for container_port, host_port in config.ports.items():
                port_bindings[f"{container_port}/tcp"] = host_port
            
            # 准备卷映射
            volumes = {}
            volume_bindings = {}
            for host_path, volume_config in config.volumes.items():
                container_path = volume_config.get("bind", host_path)
                volumes[container_path] = {"bind": container_path, "mode": volume_config.get("mode", "rw")}
                volume_bindings[host_path] = volume_config
            
            # 准备设备请求（GPU）
            device_requests = None
            if config.gpu_enabled:
                device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
            
            # 准备健康检查
            healthcheck = None
            if config.healthcheck:
                test = config.healthcheck.get("test", ["CMD", "curl", "-f", "http://localhost:8000/health"])
                interval = config.healthcheck.get("interval", 30)
                timeout = config.healthcheck.get("timeout", 30)
                retries = config.healthcheck.get("retries", 3)
                start_period = config.healthcheck.get("start_period", 40)
                
                healthcheck = docker.types.Healthcheck(
                    test=test,
                    interval=interval,
                    timeout=timeout,
                    retries=retries,
                    start_period=start_period
                )
            
            # 创建并启动容器
            self.logger.info(f"启动容器: {config.name}")
            container = self.client.containers.run(
                image=config.image,
                name=config.name,
                ports=port_bindings,
                environment=config.environment,
                volumes=volumes,
                device_requests=device_requests,
                mem_limit=config.memory_limit,
                restart_policy={"Name": config.restart_policy},
                healthcheck=healthcheck,
                detach=True
            )
            
            # 等待容器启动
            time.sleep(2)
            
            # 获取容器信息
            container_info = self._get_container_info(container)
            if container_info:
                self.containers[config.name] = container_info
                
                # 记录容器启动指标
                metrics_collector.add_container_metric(config.name, "start", 1)
                
                self.logger.info(f"容器启动成功: {config.name}")
            
            return container_info
            
        except Exception as e:
            self.logger.error(f"容器启动失败: {config.name}, 错误: {e}")
            # 记录容器启动失败指标
            metrics_collector.add_container_metric(config.name, "start_failure", 1)
            return None
    
    @monitor_performance(model_name="container_runtime")
    def stop_container(self, container_name: str) -> bool:
        """
        停止容器
        
        Args:
            container_name: 容器名称
            
        Returns:
            bool: 是否停止成功
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return False
        
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            
            # 更新容器信息
            if container_name in self.containers:
                self.containers[container_name].status = ContainerStatus.EXITED
            
            # 记录容器停止指标
            metrics_collector.add_container_metric(container_name, "stop", 1)
            
            self.logger.info(f"容器停止成功: {container_name}")
            return True
            
        except docker.errors.NotFound:
            self.logger.warning(f"容器不存在: {container_name}")
            return False
        except Exception as e:
            self.logger.error(f"容器停止失败: {container_name}, 错误: {e}")
            # 记录容器停止失败指标
            metrics_collector.add_container_metric(container_name, "stop_failure", 1)
            return False
    
    @monitor_performance(model_name="container_runtime")
    def remove_container(self, container_name: str) -> bool:
        """
        删除容器
        
        Args:
            container_name: 容器名称
            
        Returns:
            bool: 是否删除成功
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return False
        
        try:
            container = self.client.containers.get(container_name)
            
            # 如果容器正在运行，先停止它
            if container.status == "running":
                container.stop()
            
            container.remove()
            
            # 从内存中移除容器信息
            if container_name in self.containers:
                del self.containers[container_name]
            
            # 记录容器删除指标
            metrics_collector.add_container_metric(container_name, "remove", 1)
            
            self.logger.info(f"容器删除成功: {container_name}")
            return True
            
        except docker.errors.NotFound:
            self.logger.warning(f"容器不存在: {container_name}")
            return False
        except Exception as e:
            self.logger.error(f"容器删除失败: {container_name}, 错误: {e}")
            # 记录容器删除失败指标
            metrics_collector.add_container_metric(container_name, "remove_failure", 1)
            return False
    
    @monitor_performance(model_name="container_runtime")
    def restart_container(self, container_name: str) -> bool:
        """
        重启容器
        
        Args:
            container_name: 容器名称
            
        Returns:
            bool: 是否重启成功
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return False
        
        try:
            container = self.client.containers.get(container_name)
            container.restart()
            
            # 等待容器重启
            time.sleep(2)
            
            # 更新容器信息
            container_info = self._get_container_info(container)
            if container_info:
                self.containers[container_name] = container_info
            
            # 记录容器重启指标
            metrics_collector.add_container_metric(container_name, "restart", 1)
            
            self.logger.info(f"容器重启成功: {container_name}")
            return True
            
        except docker.errors.NotFound:
            self.logger.error(f"容器不存在: {container_name}")
            return False
        except Exception as e:
            self.logger.error(f"容器重启失败: {container_name}, 错误: {e}")
            # 记录容器重启失败指标
            metrics_collector.add_container_metric(container_name, "restart_failure", 1)
            return False
    
    def get_container_info(self, container_name: str) -> Optional[ContainerInfo]:
        """
        获取容器信息
        
        Args:
            container_name: 容器名称
            
        Returns:
            Optional[ContainerInfo]: 容器信息，如果容器不存在则返回None
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return None
        
        try:
            container = self.client.containers.get(container_name)
            return self._get_container_info(container)
        except docker.errors.NotFound:
            self.logger.warning(f"容器不存在: {container_name}")
            return None
        except Exception as e:
            self.logger.error(f"获取容器信息失败: {container_name}, 错误: {e}")
            return None
    
    def list_containers(self, all_containers: bool = False) -> List[ContainerInfo]:
        """
        列出容器
        
        Args:
            all_containers: 是否列出所有容器（包括已停止的）
            
        Returns:
            List[ContainerInfo]: 容器信息列表
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return []
        
        try:
            containers = self.client.containers.list(all=all_containers)
            return [self._get_container_info(container) for container in containers]
        except Exception as e:
            self.logger.error(f"列出容器失败: {e}")
            return []
    
    def check_container_health(self, container_name: str) -> Optional[str]:
        """
        检查容器健康状态
        
        Args:
            container_name: 容器名称
            
        Returns:
            Optional[str]: 健康状态，如果容器不存在或无法获取健康状态则返回None
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return None
        
        try:
            container = self.client.containers.get(container_name)
            health = container.attrs.get("State", {}).get("Health", {})
            
            if health:
                status = health.get("Status")
                return status
            else:
                # 如果没有健康检查，根据容器状态返回
                if container.status == "running":
                    return "healthy"  # 假设运行中的容器是健康的
                else:
                    return "unhealthy"
                    
        except docker.errors.NotFound:
            self.logger.warning(f"容器不存在: {container_name}")
            return None
        except Exception as e:
            self.logger.error(f"检查容器健康状态失败: {container_name}, 错误: {e}")
            return None
    
    def get_container_logs(self, container_name: str, lines: int = 100) -> Optional[str]:
        """
        获取容器日志
        
        Args:
            container_name: 容器名称
            lines: 获取的日志行数
            
        Returns:
            Optional[str]: 容器日志，如果获取失败则返回None
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return None
        
        try:
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines).decode("utf-8")
            return logs
        except docker.errors.NotFound:
            self.logger.warning(f"容器不存在: {container_name}")
            return None
        except Exception as e:
            self.logger.error(f"获取容器日志失败: {container_name}, 错误: {e}")
            return None
    
    def _get_container_info(self, container) -> Optional[ContainerInfo]:
        """
        从Docker容器对象获取容器信息
        
        Args:
            container: Docker容器对象
            
        Returns:
            Optional[ContainerInfo]: 容器信息
        """
        try:
            # 获取端口映射
            ports = {}
            for port_spec, port_bindings in container.ports.items():
                if port_bindings:
                    ports[port_spec.split("/")[0]] = port_bindings[0]["HostPort"]
            
            # 获取健康状态
            health_status = None
            health = container.attrs.get("State", {}).get("Health", {})
            if health:
                health_status = health.get("Status")
            
            # 获取API端点
            api_endpoint = None
            metrics_endpoint = None
            
            # 检查常见的API端口和监控端口
            # 支持多种端口配置：8000/8001, 8001/8011, 8002/8012等
            api_ports = ["8000", "8001", "8002", "8003", "8004", "8005", "8006"]
            metrics_ports = ["8010", "8011", "8012", "8013", "8014", "8015", "8016"]
            
            # 查找API端点
            for port in api_ports:
                if port in ports:
                    api_endpoint = f"http://localhost:{ports[port]}"
                    break
            
            # 查找监控端点
            for port in metrics_ports:
                if port in ports:
                    metrics_endpoint = f"http://localhost:{ports[port]}"
                    break
            
            # 获取创建和启动时间
            created_at = container.attrs.get("Created", "")
            started_at = container.attrs.get("State", {}).get("StartedAt", "")
            
            return ContainerInfo(
                id=container.id,
                name=container.name,
                status=ContainerStatus(container.status),
                image=container.image.tags[0] if container.image.tags else "",
                ports=ports,
                created_at=created_at,
                started_at=started_at if started_at else None,
                health_status=health_status,
                api_endpoint=api_endpoint,
                metrics_endpoint=metrics_endpoint
            )
        except Exception as e:
            self.logger.error(f"获取容器信息失败: {e}")
            return None
    
    def check_api_health(self, container_name: str) -> bool:
        """
        检查容器API健康状态
        
        Args:
            container_name: 容器名称
            
        Returns:
            bool: API是否健康
        """
        if container_name not in self.containers:
            return False
        
        container_info = self.containers[container_name]
        if not container_info.api_endpoint:
            return False
        
        try:
            response = requests.get(f"{container_info.api_endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"检查API健康状态失败: {container_name}, 错误: {e}")
            return False
    
    def get_container_metrics(self, container_name: str) -> Optional[Dict[str, Any]]:
        """
        获取容器指标
        
        Args:
            container_name: 容器名称
            
        Returns:
            Optional[Dict[str, Any]]: 容器指标，如果获取失败则返回None
        """
        if not self.client:
            self.logger.error("Docker客户端未初始化")
            return None
        
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # 计算CPU使用率
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            
            cpu_usage = 0
            if cpu_stats and precpu_stats:
                cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                           precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
                system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                              precpu_stats.get("system_cpu_usage", 0)
                
                if system_delta > 0:
                    cpu_count = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", []))
                    cpu_usage = (cpu_delta / system_delta) * cpu_count * 100
            
            # 计算内存使用
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 0)
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # 网络IO
            network_stats = stats.get("networks", {})
            network_rx = 0
            network_tx = 0
            for interface, data in network_stats.items():
                network_rx += data.get("rx_bytes", 0)
                network_tx += data.get("tx_bytes", 0)
            
            return {
                "cpu_percent": round(cpu_usage, 2),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "memory_limit_mb": round(memory_limit / (1024 * 1024), 2),
                "memory_percent": round(memory_percent, 2),
                "network_rx_mb": round(network_rx / (1024 * 1024), 2),
                "network_tx_mb": round(network_tx / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"获取容器指标失败: {container_name}, 错误: {e}")
            return None