# ToolUniverse 生物序列模型容器化架构与部署指南

本文档介绍了ToolUniverse生物序列模型的容器化架构、性能监控功能和部署指南。

## 目录结构

```
docker/
├── dnabert2/          # DNABERT2模型容器配置
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── build.sh
├── lucaone/           # LucaOne模型容器配置
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── build.sh
├── docker-compose.yml # Docker Compose配置
├── prometheus.yml     # Prometheus监控配置
├── start_bio_models.sh # 生物序列模型启动脚本
└── README.md          # 本文件
```

## 架构概述

ToolUniverse生物序列模型容器化架构包含以下核心组件：

1. **容器运行时管理** (`container_runtime.py`) - 负责Docker容器的生命周期管理
2. **容器客户端** (`container_client.py`) - 提供与容器化模型交互的接口
3. **容器监控** (`container_monitor.py`) - 监控容器的资源使用和健康状态
4. **性能收集器** (`performance_collector.py`) - 收集和分析模型性能数据
5. **模型管理器** (`model_manager.py`) - 统一管理本地和容器化模型

## 主要功能

### 1. 容器化模型管理

- 支持DNABERT2、LucaOne等生物序列模型的容器化部署
- 提供容器模型的注册、启动、停止和卸载功能
- 支持容器资源限制和健康检查配置
- 自动处理容器网络和存储卷配置

### 2. 容器监控

- 实时监控容器CPU、内存、网络和磁盘使用情况
- 支持自定义监控阈值和告警机制
- 提供容器状态历史记录和趋势分析
- 支持容器健康检查和自动重启策略

### 3. 性能数据收集

- 收集模型推理时间、吞吐量、资源使用等性能指标
- 支持性能数据的实时分析和报告生成
- 提供模型性能比较和基准测试功能
- 支持性能数据的持久化存储和可视化

## 快速开始

### 前置要求

- Docker 20.10+
- Python 3.8+
- 足够的计算资源（根据模型需求）

### 使用Docker Compose部署所有服务

```bash
# 进入docker目录
cd docker

# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止所有服务
docker-compose down
```

### 使用启动脚本部署生物序列模型

```bash
# 进入docker目录
cd docker

# 设置环境变量（可选）
export MODEL_BASE_PATH="/path/to/your/models"
export PORT=8000
export HOST="0.0.0.0"

# 运行启动脚本
./start_bio_models.sh
```

启动脚本会自动检查Python环境和必要的依赖包，然后启动ToolUniverse生物序列模型服务。

### 单独构建和运行模型容器

#### DNABERT2模型

```bash
# 进入dnabert2目录
cd docker/dnabert2

# 构建镜像
./build.sh build

# 运行容器
./build.sh run

# 查看日志
./build.sh logs

# 停止容器
./build.sh stop
```

#### LucaOne模型

```bash
# 进入lucaone目录
cd docker/lucaone

# 构建镜像
./build.sh build

# 运行容器
./build.sh run

# 查看日志
./build.sh logs

# 停止容器
./build.sh stop
```

## 服务端点

### API端点

- DNABERT2 API: http://localhost:8000
- LucaOne API: http://localhost:8002

### 监控端点

- DNABERT2监控: http://localhost:8001/metrics
- LucaOne监控: http://localhost:8003/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## API规范

### 基础信息

- 基础URL: `http://localhost:{port}`
- 内容类型: `application/json`
- 字符编码: `UTF-8`

### 通用响应格式

所有API响应都遵循以下统一格式：

```json
{
  "success": true,
  "message": "操作成功",
  "data": {},
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

### 错误响应格式

```json
{
  "success": false,
  "message": "错误描述",
  "error": {
    "code": "ERROR_CODE",
    "details": "详细错误信息"
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

### API端点

#### 1. 健康检查

检查模型服务的健康状态。

**端点**: `GET /health`

**请求参数**: 无

**响应示例**:
```json
{
  "success": true,
  "message": "服务正常",
  "data": {
    "status": "healthy",
    "model": "DNABERT2",
    "version": "1.0.0",
    "uptime": 3600,
    "gpu_available": true,
    "memory_usage": "4.2GB"
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

#### 2. 模型信息

获取模型详细信息。

**端点**: `GET /model/info`

**请求参数**: 无

**响应示例**:
```json
{
  "success": true,
  "message": "模型信息获取成功",
  "data": {
    "name": "DNABERT2",
    "version": "1.0.0",
    "description": "DNA序列分析模型",
    "tasks": ["promoter_prediction"],
    "input_format": {
      "type": "string",
      "description": "DNA序列",
      "example": "ATGCGTACGTAGCTAGCTAGCTAGC"
    },
    "output_format": {
      "type": "float",
      "description": "启动子活性概率",
      "range": [0, 1]
    },
    "max_sequence_length": 512,
    "supported_species": ["human", "mouse", "fruitfly"]
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

#### 3. 模型预测

执行模型预测。

**端点**: `POST /predict`

**请求参数**:
```json
{
  "sequence": "ATGCGTACGTAGCTAGCTAGCTAGC",
  "task_type": "promoter_prediction",
  "options": {
    "return_confidence": true,
    "batch_size": 1
  }
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "预测完成",
  "data": {
    "prediction": 0.85,
    "confidence": 0.92,
    "processing_time": 0.15,
    "sequence_length": 24,
    "details": {
      "class": "promoter",
      "probability": 0.85
    }
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

#### 4. 批量预测

执行批量模型预测。

**端点**: `POST /predict/batch`

**请求参数**:
```json
{
  "sequences": [
    "ATGCGTACGTAGCTAGCTAGCTAGC",
    "GCTAGCTAGCTACGTACGCAT"
  ],
  "task_type": "promoter_prediction",
  "options": {
    "return_confidence": true,
    "batch_size": 10
  }
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "批量预测完成",
  "data": {
    "predictions": [
      {
        "sequence": "ATGCGTACGTAGCTAGCTAGCTAGC",
        "prediction": 0.85,
        "confidence": 0.92
      },
      {
        "sequence": "GCTAGCTAGCTACGTACGCAT",
        "prediction": 0.12,
        "confidence": 0.78
      }
    ],
    "processing_time": 0.35,
    "total_sequences": 2
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

#### 5. 模型指标

获取模型性能指标。

**端点**: `GET /metrics`

**请求参数**: 无

**响应示例**:
```json
{
  "success": true,
  "message": "指标获取成功",
  "data": {
    "model_metrics": {
      "total_requests": 1250,
      "successful_requests": 1245,
      "failed_requests": 5,
      "average_response_time": 0.18,
      "gpu_utilization": 0.75,
      "memory_usage": "4.2GB"
    },
    "system_metrics": {
      "cpu_usage": 0.25,
      "memory_usage": "5.8GB",
      "disk_usage": "12.3GB",
      "network_io": "120MB/s"
    }
  },
  "timestamp": "2023-11-20T12:00:00Z",
  "request_id": "uuid"
}
```

### 模型特定API

#### DNABERT2特定参数

**task_type** 支持的值:
- `promoter_prediction`: 启动子预测

**options** 支持的参数:
- `return_confidence`: 是否返回置信度 (默认: true)
- `batch_size`: 批处理大小 (默认: 1)

#### LucaOne特定参数

**task_type** 支持的值:
- `embedding`: 序列嵌入
- `classification`: 序列分类
- `property_prediction`: 属性预测

**options** 支持的参数:
- `return_confidence`: 是否返回置信度 (默认: true)
- `batch_size`: 批处理大小 (默认: 1)
- `embedding_dimension`: 嵌入维度 (默认: 768)

### 错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| INVALID_REQUEST | 400 | 请求参数无效 |
| INVALID_SEQUENCE | 400 | 序列格式无效 |
| SEQUENCE_TOO_LONG | 400 | 序列长度超过限制 |
| MODEL_NOT_LOADED | 503 | 模型未加载 |
| PREDICTION_FAILED | 500 | 预测失败 |
| INTERNAL_ERROR | 500 | 内部服务器错误 |

### 限流和配额

- 每个IP每分钟最多100个请求
- 单次批量预测最多100个序列
- 单个序列最大长度: 5120个字符

## API使用示例

### DNABERT2 API

```bash
# 健康检查
curl http://localhost:8000/health

# 预测启动子活性
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGTACGTAGCTAGCTAGCTAGC"}'
```

### LucaOne API

```bash
# 健康检查
curl http://localhost:8002/health

# 预测多任务
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ATGCGTACGTAGCTAGCTAGCTAGC", "task_type": "embedding"}'
```

### Python客户端示例

```python
import requests
import json

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 预测
data = {
    "sequence": "ATGCGTACGTAGCTAGCTAGCTAGC",
    "task_type": "promoter_prediction"
}
response = requests.post(
    "http://localhost:8000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)
print(response.json())
```

### JavaScript客户端示例

```javascript
// 健康检查
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// 预测
const data = {
  sequence: "ATGCGTACGTAGCTAGCTAGCTAGC",
  task_type: "promoter_prediction"
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

## 编程接口使用

### 容器模型管理

```python
from tooluniverse.bio_models.model_manager import ModelManager, ModelType, ContainerModelConfig

# 创建模型管理器
manager = ModelManager()

# 配置容器模型
container_config = ContainerModelConfig(
    name="dnabert2-container",
    model_type=ModelType.DNABERT2,
    image="dnabert2:latest",
    ports={"5000": 5000},
    environment={"MODEL_PATH": "/models/dnabert2"},
    memory_limit="4g",
    gpu_enabled=False
)

# 注册并启动容器模型
manager.register_container_model(container_config)
manager.start_container_model("dnabert2-container")

# 使用模型进行预测
sequence = "ATCGATCGATCG"
result = manager.predict_with_container("dnabert2-container", sequence)

# 停止容器模型
manager.stop_container_model("dnabert2-container")
```

### 容器监控

```python
# 启动容器监控
manager.start_container_monitoring()

# 设置监控阈值
manager.set_container_monitoring_thresholds(
    cpu_threshold=80.0,
    memory_threshold=90.0,
    disk_threshold=85.0
)

# 获取容器指标
metrics = manager.get_container_metrics("dnabert2-container")
print(f"CPU使用率: {metrics.cpu_percent}%")
print(f"内存使用: {metrics.memory_usage_mb}MB")

# 获取监控摘要
summary = manager.get_container_monitoring_summary()
print(f"监控的容器数量: {summary['total_containers']}")
print(f"告警数量: {summary['total_alerts']}")

# 停止容器监控
manager.stop_container_monitoring()
```

### 性能数据收集

```python
# 启动性能收集
manager.start_performance_collection(interval=5)  # 每5秒收集一次

# 手动收集性能数据
manager.collect_inference_time("dnabert2-container", 0.25)
manager.collect_throughput("dnabert2-container", 40.5)
manager.collect_memory_usage("dnabert2-container", 2.1)
manager.collect_cpu_usage("dnabert2-container", 65.2)

# 生成性能报告
report = manager.generate_performance_report("dnabert2-container")
print(f"平均推理时间: {report.avg_inference_time:.4f}s")
print(f"平均吞吐量: {report.avg_throughput:.2f} req/s")

# 保存性能报告
report_path = manager.save_performance_report("dnabert2-container")
print(f"报告已保存到: {report_path}")

# 停止性能收集
manager.stop_performance_collection()
```

## 示例脚本

项目提供了以下示例脚本，帮助您快速上手：

1. `examples/bio_models/container_model_example.py` - 容器模型管理示例
2. `examples/bio_models/container_monitoring_example.py` - 容器监控示例
3. `examples/bio_models/performance_collection_example.py` - 性能数据收集示例

## 部署指南

### 部署步骤

1. 构建或获取模型容器镜像
2. 配置容器模型参数
3. 启动模型管理器和容器监控
4. 注册并启动容器模型
5. 配置性能收集和告警阈值

### 配置文件

模型配置文件位于 `config/bio_models/model_config.json`，包含以下配置项：

- 模型路径和参数
- 容器资源配置
- 监控阈值设置
- 性能收集配置

## 性能优化建议

1. **资源配置**：根据模型需求合理配置CPU和内存限制
2. **存储优化**：使用SSD存储提高I/O性能
3. **网络配置**：优化容器网络设置减少延迟
4. **监控调优**：根据实际情况调整监控频率和阈值

## 注意事项

1. **GPU支持**: 确保宿主机已安装NVIDIA Docker运行时，以支持GPU加速
2. **模型文件**: 模型文件需要挂载到容器的/app/models目录
3. **资源限制**: 根据实际硬件配置调整Docker Compose中的资源限制
4. **网络配置**: 确保容器间的网络通信正常，特别是Prometheus和模型容器之间

## 故障排除

### 常见问题

1. **容器启动失败**
   - 检查Docker日志: `docker logs <container_name>`
   - 确认GPU驱动和NVIDIA Docker运行时已正确安装

2. **API请求失败**
   - 确认容器正在运行且健康检查通过
   - 检查端口是否被占用

3. **监控数据缺失**
   - 确认Prometheus配置正确
   - 检查网络连接和防火墙设置

4. **容器启动失败**：检查镜像是否存在，资源配置是否足够
5. **监控数据缺失**：确认监控服务是否正常运行
6. **性能数据异常**：检查模型负载和系统资源状态

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs dnabert2
docker-compose logs lucaone
docker-compose logs prometheus
docker-compose logs grafana
```

其他日志查看方式：
- 模型管理器日志：查看应用日志文件
- 容器日志：使用 `docker logs` 命令查看
- 系统日志：查看系统日志获取更多信息

## 扩展开发

### 添加新的容器模型

1. 在 `container_client.py` 中实现新的客户端类
2. 在 `ModelType` 枚举中添加新的模型类型
3. 在模型管理器中注册新的模型类型

### 自定义监控指标

1. 扩展 `ContainerMetrics` 类添加新的指标
2. 在 `ContainerMonitor` 中实现指标收集逻辑
3. 更新告警阈值配置

### 性能数据扩展

1. 扩展 `PerformanceMetric` 类添加新的指标类型
2. 在 `PerformanceCollector` 中实现数据收集逻辑
3. 更新报告生成模板

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。在提交代码前，请确保：

1. 代码符合项目编码规范
2. 添加了必要的测试用例
3. 更新了相关文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。