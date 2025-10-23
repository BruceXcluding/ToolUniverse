# 生物序列模型配置说明

## 环境变量配置

模型配置文件现在支持使用环境变量，以便在不同部署环境中灵活配置模型路径。

### 配置格式

配置文件使用 `${VAR_NAME:default_value}` 格式来指定环境变量，其中：
- `VAR_NAME` 是环境变量名称
- `default_value` 是当环境变量未设置时的默认值

### 示例

```json
{
  "model_base_path": "${MODEL_BASE_PATH:./models}",
  "models": {
    "lucaone": {
      "model_path": "${MODEL_BASE_PATH:./models}/LucaOne",
      "supported_tasks": ["embedding", "classification"],
      "supported_sequences": ["DNA", "RNA", "protein"]
    }
  }
}
```

### 环境变量设置

#### Linux/macOS

```bash
# 设置模型基础路径
export MODEL_BASE_PATH=/path/to/your/models

# 启动应用
python your_app.py
```

#### Windows

```cmd
# 设置模型基础路径
set MODEL_BASE_PATH=C:\path\to\your\models

# 启动应用
python your_app.py
```

#### Docker

```dockerfile
ENV MODEL_BASE_PATH=/app/models
```

或使用 docker-compose:

```yaml
services:
  your-app:
    environment:
      - MODEL_BASE_PATH=/app/models
```

#### Kubernetes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: your-app
    image: your-image
    env:
    - name: MODEL_BASE_PATH
      value: "/app/models"
```

### 部署建议

1. **开发环境**：可以使用默认值 `./models`，将模型文件放在应用目录下的 models 文件夹中

2. **生产环境**：设置 `MODEL_BASE_PATH` 环境变量，指向实际的模型存储路径

3. **容器化部署**：通过环境变量或配置映射(ConfigMap)来设置模型路径

4. **多模型共享**：可以将所有模型存储在共享存储中，通过环境变量统一配置

### 配置验证

系统启动时会自动解析环境变量，如果环境变量未设置且没有提供默认值，将使用空字符串。

建议在部署前验证模型路径是否正确：

```python
import os
from tooluniverse.bio_models.model_manager import ModelManager

# 检查环境变量
model_base_path = os.getenv("MODEL_BASE_PATH", "./models")
print(f"模型基础路径: {model_base_path}")

# 初始化模型管理器
manager = ModelManager()

# 检查模型配置
for model_name, model_config in manager.config.get("models", {}).items():
    model_path = model_config.get("model_path")
    print(f"模型 {model_name} 路径: {model_path}")
    if not os.path.exists(model_path):
        print(f"警告: 模型路径不存在: {model_path}")
```

### 常见问题

1. **路径不存在**：确保设置的路径在目标环境中存在且可访问
2. **权限问题**：确保应用有权限访问模型文件
3. **相对路径**：使用相对路径时，确保工作目录正确
4. **容器挂载**：在容器环境中，确保模型文件正确挂载到容器内部

通过使用环境变量配置，您可以在不同的部署环境中灵活调整模型路径，而无需修改配置文件。