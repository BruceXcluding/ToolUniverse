# ToolUniverse 生物序列模型模块

## 📚 目录

- [1. 架构概览](#1-架构概览)
- [2. 核心组件](#2-核心组件)
  - [2.1 基础模型 (BaseModel)](#21-基础模型-basemodel)
  - [2.2 模型管理器 (ModelManager)](#22-模型管理器-modelmanager)
  - [2.3 任务与序列类型](#23-任务与序列类型)
  - [2.4 容器运行时](#24-容器运行时)
- [3. 工具接口](#3-工具接口)
  - [3.1 统一接口工具](#31-统一接口工具)
  - [3.2 具体工具实现](#32-具体工具实现)
- [4. 工具注册指南](#4-工具注册指南)
  - [4.1 创建自定义工具的步骤](#41-创建自定义工具的步骤)
  - [4.2 注册到系统](#42-注册到系统)
- [5. 配置说明](#5-配置说明)
- [6. 部署方式](#6-部署方式)
  - [6.1 本地部署](#61-本地部署)
  - [6.2 容器部署](#62-容器部署)
- [7. 性能监控](#7-性能监控)
- [8. 错误处理](#8-错误处理)
- [9. 非模型工具集成指南](#9-非模型工具集成指南)
- [10. 示例](#10-示例)

## 1. 架构概览

ToolUniverse 生物序列模型模块提供了一个统一、灵活的框架，用于管理和使用各种生物序列分析模型。该模块采用分层设计，支持多种部署方式（本地和容器化），并提供标准化的接口以便于客户集成自己的工具。

```
┌─────────────────────────┐
│     工具接口层          │
│  (BioSequenceAnalysisTool)│
└─────────────┬───────────┘
              │
┌─────────────▼───────────┐
│     模型管理层          │
│      (ModelManager)     │
└─────────────┬───────────┘
              │
┌─────────────┴───────────┐
│                         │
▼                         ▼
┌─────────────────┐  ┌─────────────────┐
│  本地模型实现    │  │  容器模型实现    │
│  (BaseModel子类) │  │(ContainerRuntime)│
└─────────────────┘  └─────────────────┘
```

## 2. 核心组件

### 2.1 基础模型 (BaseModel)

`BaseModel` 是所有生物序列模型的抽象基类，定义了模型必须实现的核心接口：

```python
class BaseModel(ABC):
    def load_model(self, device: Union[str, DeviceType] = DeviceType.AUTO) -> bool: ...
    def unload_model(self) -> bool: ...
    def predict(self, sequences: List[str], task_type: TaskType, **kwargs) -> Dict[str, Any]: ...
```

**主要功能**：
- 模型加载与卸载
- 序列预测
- 设备管理（CPU/CUDA自动选择）
- 序列类型验证
- 内存使用监控

### 2.2 模型管理器 (ModelManager)

`ModelManager` 负责管理所有模型的生命周期、调度和资源分配：

**主要功能**：
- 模型注册与管理
- GPU资源调度
- 容器化模型管理
- 性能监控和指标收集
- 自动发现运行中的容器

### 2.3 任务与序列类型

**支持的任务类型**：
- `EMBEDDING`: 序列嵌入
- `CLASSIFICATION`: 分类任务
- `PREDICTION`: 预测任务
- `FINE_TUNING`: 微调任务
- `STRUCTURE_PREDICTION`: 结构预测
- `MOTIF_DETECTION`: 基序检测
- `MUTATION_ANALYSIS`: 突变分析
- `GENERATION`: 序列生成
- `FUNCTION_ANNOTATION`: 功能注释

**支持的序列类型**：
- `DNA`: DNA序列
- `RNA`: RNA序列
- `PROTEIN`: 蛋白质序列

### 2.4 容器运行时

`ContainerRuntime` 提供了容器化模型的管理功能：

**主要功能**：
- 容器生命周期管理（启动、停止、删除）
- 容器状态监控
- 容器健康检查
- 容器资源限制管理

## 3. 工具接口

### 3.1 统一接口工具

`BioSequenceAnalysisTool` 是所有生物序列分析工具的统一入口，提供了标准化的API：

```python
def analyze(
    self, 
    sequences: Union[str, List[str]], 
    task_type: Union[str, TaskType],
    model_name: Optional[str] = None,
    sequence_type: Optional[Union[str, SequenceType]] = None,
    device: Union[str, DeviceType] = DeviceType.AUTO,
    **kwargs
) -> Dict[str, Any]: ...
```

**主要方法**：
- `analyze`: 执行序列分析
- `list_models`: 列出所有可用模型
- `list_loaded_models`: 列出已加载的模型
- `get_model_info`: 获取模型详细信息
- `get_best_model`: 获取最适合特定任务的模型

### 3.2 具体工具实现

系统内置了多种专门的工具实现，包括但不限于：

- `LucaOneTool`: LUCA-ONE模型工具
- `DNABERT2Tool`: DNABERT2模型工具
- `RNAFMTool`: RNA-FM模型工具
- `EmbeddingTool`: 通用嵌入工具
- `ClassificationTool`: 通用分类工具
- `StructurePredictionTool`: 结构预测工具
- `AnnotationTool`: 功能注释工具
- `RNAFoldTool`: RNA折叠工具
- `BlastSearchTool`: BLAST搜索工具

## 4. 工具注册指南

### 4.1 创建自定义工具的步骤

要注册自己的生物序列分析工具，请遵循以下步骤：

#### 步骤1: 创建模型实现类

继承 `BaseModel` 并实现必要的方法：

```python
from tooluniverse.bio_models.models.base_model import BaseModel
from tooluniverse.bio_models.task_types import TaskType, SequenceType, ModelStatus, DeviceType

class YourCustomModel(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        # 初始化自定义属性
        
    def load_model(self, device: Union[str, DeviceType] = DeviceType.AUTO) -> bool:
        # 实现模型加载逻辑
        try:
            # 设置设备
            self._set_device(device)
            # 加载模型和tokenizer
            # self.model = your_model_loading_code()
            # self.tokenizer = your_tokenizer_loading_code()
            self.status = ModelStatus.LOADED
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            self.status = ModelStatus.ERROR
            return False
    
    def unload_model(self) -> bool:
        # 实现模型卸载逻辑
        try:
            self.model = None
            self.tokenizer = None
            self.status = ModelStatus.UNLOADED
            return True
        except Exception as e:
            self.logger.error(f"卸载模型失败: {str(e)}")
            return False
    
    def predict(self, sequences: List[str], task_type: TaskType, **kwargs) -> Dict[str, Any]:
        # 实现预测逻辑
        try:
            # 验证序列
            if not self._validate_sequences(sequences):
                return {"error": "无效的输入序列"}
            
            # 执行预测
            # results = your_prediction_code(sequences, task_type, **kwargs)
            
            return {
                "predictions": ["example_result" for _ in sequences],
                "task_type": task_type.value,
                "model_name": self.model_name
            }
        except Exception as e:
            return {"error": str(e)}
```

#### 步骤2: 创建工具类

创建一个工具类，可以直接使用统一接口或扩展它：

```python
from tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool

class YourCustomTool(BioSequenceAnalysisTool):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        # 初始化自定义属性
    
    # 可以重写或扩展统一接口的方法
    def analyze(self, sequences, task_type, **kwargs):
        # 可以添加自定义预处理或后处理
        result = super().analyze(sequences, task_type, **kwargs)
        # 添加自定义处理
        return result
```

### 4.2 注册到系统

要将自定义工具注册到系统中，请将其添加到 `__init__.py` 文件中：

```python
# 在 tooluniverse/bio_models/tools/__init__.py 中添加
from .your_custom_tool import YourCustomTool

__all__ = [
    # 现有工具...
    "YourCustomTool"
]
```

同时，还需要在主模块的 `__init__.py` 中导出：

```python
# 在 tooluniverse/bio_models/__init__.py 中添加
from .tools import YourCustomTool

__all__ = [
    # 现有导出...
    "YourCustomTool"
]
```

#### 配置模型

创建或修改配置文件，添加你的模型配置：

```json
{
  "models": [
    {
      "name": "your_custom_model",
      "model_type": "your_model_type",
      "model_path": "/path/to/your/model",
      "supported_sequences": ["dna", "rna"],
      "supported_tasks": ["embedding", "classification"],
      "memory_requirement": 8192
    }
  ]
}
```

## 5. 配置说明

模型配置文件（默认为 `config/bio_models/model_config.json`）包含以下字段：

- `models`: 模型配置列表
  - `name`: 模型名称
  - `model_type`: 模型类型
  - `model_path`: 模型路径
  - `supported_sequences`: 支持的序列类型
  - `supported_tasks`: 支持的任务类型
  - `memory_requirement`: 内存需求（MB）
  - `container_config` (可选): 容器配置

## 6. 部署方式

### 6.1 本地部署

```python
from tooluniverse.bio_models.model_manager import ModelManager

# 初始化模型管理器
manager = ModelManager("path/to/config.json")

# 加载模型
manager.load_model("your_model_name", "cuda")

# 执行预测
result = manager.predict(
    sequences=["ATGCATGC"],
    task_type="embedding",
    model_name="your_model_name"
)
```

### 6.2 容器部署

容器化模型可以通过Docker部署：

```python
from tooluniverse.bio_models.container_runtime import ContainerRuntime, ContainerConfig

# 初始化容器运行时
runtime = ContainerRuntime()

# 配置容器
config = ContainerConfig(
    name="your_model_container",
    image="your_model_image:latest",
    ports={"8000": 8000},
    gpu_enabled=True,
    memory_limit="16g"
)

# 启动容器
container = runtime.start_container(config)
```

## 7. 性能监控

系统内置了性能监控功能：

- 内存使用监控
- GPU使用率监控
- 容器健康检查
- 预测性能指标收集

可以通过 `monitoring` 模块访问监控功能：

```python
from tooluniverse.bio_models.monitoring import get_logger, monitor_performance

# 监控函数性能
@monitor_performance(model_name="your_model")
def your_function():
    # 函数实现
    pass
```

## 8. 错误处理

系统提供了统一的错误处理机制：

```python
from tooluniverse.base_tool import (
    ToolError, 
    ToolValidationError, 
    ToolServerError,
    ToolConfigError,
    ToolDependencyError,
    ToolAuthError,
    ToolRateLimitError,
    ToolUnavailableError
)
```

预测结果中的错误格式：

```json
{
  "error": "错误描述信息"
}
```

## 9. 非模型工具集成指南

除了依赖深度学习模型的工具外，ToolUniverse也支持集成不需要模型的生物信息学工具，如RNA折叠预测、序列比较、BLAST搜索和JASPAR扫描等。

### 9.1 非模型工具特点

- **独立实现**：不依赖BaseModel类和模型加载机制
- **轻量级**：直接调用相应的生物信息学库或算法
- **统一接口**：保持与模型工具相同的`analyze()`方法签名，确保API一致性

### 9.2 创建非模型工具步骤

1. **创建工具类**：
   ```python
   class YourToolName:
       def __init__(self, config_path=None):
           # 初始化必要组件
           self.logger = logging.getLogger(__name__)
           self.tool_name = "your_tool_name"
           # 检查依赖库可用性
           self.available = self._check_availability()
       
       def _check_availability(self):
           # 检查必要依赖是否可用
           try:
               # 导入必要库
               return True
           except ImportError:
               return False
       
       def analyze(self, sequences, task_type, model_name=None, device=DeviceType.CPU, **kwargs):
           # 实现分析逻辑
           # 返回标准化的结果格式
           pass
   ```

2. **在tools/__init__.py中导出**：
   ```python
   from .your_tool_file import YourToolName
   
   __all__ = [
       # 现有工具...
       "YourToolName"
   ]
   ```

3. **创建CLI接口**（可选）：
   在examples目录下创建命令行入口，如rna_tools_cli.py所示。

### 9.3 工具可用性检查

所有非模型工具都应实现可用性检查机制：

```python
# 示例：检查RNA库可用性
try:
    import RNA
    RNA_FOLD_AVAILABLE = True
except ImportError:
    RNA_FOLD_AVAILABLE = False
    logging.warning("无法导入RNA库")
```

### 9.4 结果格式规范

保持结果格式统一，包含以下关键字段：
- `status`: 成功/失败状态
- `error`: 错误信息（如有）
- `results`: 结果数据列表

## 10. 示例

### 使用统一接口进行序列分析

```python
from tooluniverse.bio_models.tools import BioSequenceAnalysisTool
from tooluniverse.bio_models.task_types import TaskType

# 初始化工具
tool = BioSequenceAnalysisTool()

# 执行嵌入任务
result = tool.analyze(
    sequences=["ATGCATGC", "CGTACGTA"],
    task_type=TaskType.EMBEDDING,
    model_name="dnabert2",
    sequence_type="dna"
)

# 查看结果
print(result)
```

### 使用特定工具

```python
from tooluniverse.bio_models.tools import LucaOneTool

# 初始化特定工具
tool = LucaOneTool()

# 执行特定任务
result = tool.analyze(
    sequences=["ATGCATGC"],
    task_type="classification"
)
```

### 注册自定义模型

```python
from tooluniverse.bio_models.model_manager import ModelManager
from your_custom_model import YourCustomModel

# 初始化模型管理器
manager = ModelManager()

# 注册自定义模型
manager.register_model(
    model_name="your_custom_model",
    model_class=YourCustomModel,
    config={
        "supported_sequences": ["dna"],
        "supported_tasks": ["embedding"],
        "memory_requirement": 4096
    }
)
```

## 11. 部署说明

模型可以通过Docker容器或本地安装两种方式部署。详细配置见`config/bio_models/`目录。

## 📝 注意事项

1. 自定义工具必须实现标准接口以确保兼容性
2. 模型配置中的内存需求应根据实际情况设置
3. 容器化模型需要提供标准API接口
4. 建议为自定义工具添加完整的日志和错误处理
5. 如需GPU支持，请确保正确配置设备类型