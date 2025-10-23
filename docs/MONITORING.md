# ToolUniverse 监控系统

## 概述

ToolUniverse监控系统为生物模型工具提供了全面的性能监控、日志记录和可视化功能。该系统基于structlog和Prometheus构建，提供结构化日志记录、性能指标收集和实时监控仪表板。

## 功能特性

### 1. 结构化日志记录
- 使用structlog库提供结构化日志记录
- 支持彩色控制台输出和JSON格式日志
- 记录模型加载、卸载和推理过程中的关键事件
- 包含详细的上下文信息和错误追踪

### 2. 性能监控
- 监控模型加载时间和推理时间
- 跟踪CPU和GPU内存使用情况
- 记录输入输出数据形状和大小
- 提供性能统计和分析

### 3. 指标收集
- 基于Prometheus的指标收集系统
- 记录模型加载/卸载成功/失败次数
- 跟踪推理请求和响应时间
- 监控系统资源使用情况

### 4. 可视化仪表板
- 基于Streamlit的实时监控仪表板
- 系统资源监控（CPU、内存、GPU）
- 模型性能指标展示
- 实时日志查看

## 安装和使用

### 1. 安装依赖

```bash
cd /Users/yigex/Documents/trae_projects/deployment/ToolUniverse
pip install -e .
```

### 2. 启动监控仪表板

```bash
python start_dashboard.py
```

或者指定端口和主机：

```bash
python start_dashboard.py --port 8502 --host 0.0.0.0
```

### 3. 访问仪表板

在浏览器中打开 `http://localhost:8501` 查看监控仪表板。

## 集成到现有代码

监控系统已集成到以下组件中：

### 1. ModelManager
- 模型注册、加载和卸载过程的监控
- 预测请求的性能跟踪
- 资源使用情况记录

### 2. DNABERT2工具
- 启动子活性预测监控
- 增强子活性预测监控
- 输入输出数据记录

### 3. GPUScheduler
- GPU设备检测和内存监控
- 设备分配和释放跟踪

## 添加监控到新组件

要在新的组件中添加监控功能，请按照以下步骤：

### 1. 导入监控模块

```python
from ..monitoring import get_logger, monitor_performance, monitor_context, metrics_collector, log_io_shapes
```

### 2. 替换日志记录器

```python
# 替换
self.logger = logging.getLogger(__name__)

# 为
self.logger = get_logger(__name__)
```

### 3. 添加性能监控装饰器

```python
@monitor_performance(model_name="your_model_name", task_type="your_task_type")
def your_method(self, params):
    # 方法实现
```

### 4. 使用监控上下文管理器

```python
with monitor_context("operation_name", param1=value1, param2=value2):
    # 操作代码
```

### 5. 记录指标

```python
# 成功指标
metrics_collector.record_metric("operation_success", 1, {"param": "value"})

# 失败指标
metrics_collector.record_metric("operation_failure", 1, {"param": "value"})
```

### 6. 记录输入输出

```python
log_io_shapes(
    input_data={"param1": value1, "param2": value2},
    output_data={"result": result},
    task_type="your_task_type"
)
```

## 监控仪表板功能

### 1. 系统资源标签页
- CPU使用率实时图表
- 内存使用情况图表
- GPU内存使用情况（如果有GPU）
- 系统资源历史趋势

### 2. 模型性能标签页
- 模型加载时间统计
- 推理时间分布
- 模型使用频率
- 错误率统计

### 3. 日志标签页
- 实时日志流
- 日志级别过滤
- 关键词搜索
- 日志详情查看

## 故障排除

### 1. 依赖问题

如果遇到依赖问题，请确保安装了以下包：

```bash
pip install structlog>=23.1.0 colorama>=0.4.6 pynvml>=11.5.0 prometheus-client>=0.17.0 streamlit plotly pandas
```

### 2. GPU监控问题

如果没有NVIDIA GPU或驱动问题，GPU监控将显示为"不可用"。这是正常行为，不会影响其他监控功能。

### 3. 仪表板访问问题

如果无法访问仪表板，请检查：
- 端口是否被占用
- 防火墙设置
- 网络连接

## 扩展和定制

### 1. 添加新指标

在`monitoring/__init__.py`中的`MetricsCollector`类中添加新指标：

```python
def record_custom_metric(self, value, labels=None):
    """记录自定义指标"""
    self.metrics["custom_metric"].labels(**(labels or {})).observe(value)
```

### 2. 自定义日志格式

修改`monitoring/__init__.py`中的`configure_logging`函数来自定义日志格式。

### 3. 扩展仪表板

在`monitoring/dashboard.py`中添加新的标签页和可视化组件。

## 性能影响

监控系统设计为轻量级，对系统性能的影响最小：
- 日志记录是异步的
- 指标收集使用高效的Prometheus客户端
- 仪表板是独立进程，不影响主应用

## 安全注意事项

- 监控仪表板默认只在本地主机上运行
- 不要在生产环境中暴露仪表板到公网
- 日志可能包含敏感信息，请妥善处理