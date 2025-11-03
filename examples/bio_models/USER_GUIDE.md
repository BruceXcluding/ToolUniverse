# 🧬 ToolUniverse 生物信息学工具使用指南

<div align="center">
  <img src="https://via.placeholder.com/400x100.png?text=ToolUniverse+Logo" alt="ToolUniverse Logo" />
  <p><em>强大的生物信息学工具管理中台</em></p>
</div>

## 📑 目录

- [整体架构概述](#整体架构概述)
- [准备工作](#准备工作)
- [部署步骤](#部署步骤)
- [使用生物信息学工具](#使用生物信息学工具)
- [高级用法](#高级用法)
- [故障排除](#故障排除)
- [性能优化](#性能优化)
- [总结](#总结)

## 🔍 整体架构概述

ToolUniverse 是一个强大的工具管理中台，专为管理和调用各种生物信息学工具而设计。其核心架构包含以下几个关键组件：

### 🧩 核心组件

- **Docker 容器化模型服务** 🐳 - 为计算密集型任务（如 DNABERT2）提供隔离的运行环境
- **工具注册与发现系统** 🔍 - 统一管理和发现各种生物信息学工具
- **模型计算协议 (MCP) 服务** 📡 - 作为工具调用的中间层，处理请求转发和结果聚合
- **命令行界面工具** 💻 - 提供简单直观的用户接口，方便调用各种功能

### 📊 架构流程图

```mermaid

flowchart TB
%% ======= 样式定义 =======
classDef bg fill:#FAFAFA,stroke:#BDBDBD,stroke-width:2px,color:#424242
classDef user fill:#E3F2FD,stroke:#1565C0,stroke-width:3px,color:#0D47A1,font-weight:bold
classDef mcp fill:#FFF3E0,stroke:#EF6C00,stroke-width:4px,color:#E65100,font-weight:bold
classDef runtime fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px,color:#1B5E20,font-weight:bold
classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:3px,color:#4A148C,font-weight:bold

%% ======= 外层背景分组（整体背景板） =======
subgraph ALL["🧩 TU 推理与计算中台架构"]

direction TB

    %% ======= 内部节点 =======
    A[🧑‍💻 用户应用 / CLI 工具<br>（命令行接口 · API 调用）]
    B[🔮 ToolUniverse MCP<br>服务注册 · 请求路由 · 负载均衡 · 工具编排]
    C[🐳 Docker 容器化服务<br>环境隔离 · 任务调度 · 资源管理]
    D1[🧬 DNABERT2 模型服务]
    D2[🧪 RNA 工具套件]
    D3[🔬 其他生物信息学工具]

    %% 连接结构
    A ==> B
    B ==> C
    C ==> D1
    C ==> D2
    C ==> D3

    %% 样式关联
    class A user
    class B mcp
    class C runtime
    class D1,D2,D3 model

end

%% 外层背景样式
class ALL bg


```

## 🔧 准备工作

### 📋 环境要求

- ✅ **Docker 和 Docker Compose** - 用于容器化服务的部署和管理
- ✅ **Python 3.8 或更高版本** - 运行 ToolUniverse 及其工具
- ✅ **足够的磁盘空间** - 用于存储 Docker 镜像和模型
- ✅ **8GB+ 内存** - 推荐配置，确保 Docker 容器稳定运行

## 🚀 部署步骤

### 🐳 启动 Docker 模型服务

首先，确保您的 Docker 环境已正确安装并正在运行。然后，使用以下命令启动生物信息学模型服务：

```bash
# 📁 进入 ToolUniverse 目录
cd /mnt/models/yigex/ToolUniverse

# ⚙️ 复制环境配置文件
cp .env.template .env

# 🚀 启动 Docker 容器服务
docker-compose up -d
```

这将启动包括 DNABERT2 在内的多个生物信息学模型服务。您可以使用以下命令检查服务状态：

```bash
# 📊 检查服务状态
docker-compose ps
```

### 🐍 配置 Python 虚拟环境并安装 ToolUniverse

接下来，设置 Python 虚拟环境并安装 ToolUniverse：

```bash
# 🌱 创建 Python 虚拟环境
python3 -m venv tooluniverse_env

# 🧩 激活虚拟环境
source tooluniverse_env/bin/activate

# 📦 安装 ToolUniverse
pip install -e .

# 📝 安装额外依赖
pip install -r requirements.txt
```

### 📡 启动 ToolUniverse MCP 服务

MCP (Model Computing Protocol) 服务是 ToolUniverse 的核心组件，负责管理和转发工具调用请求：

```bash
# 📁 进入生物信息学工具目录
cd /mnt/models/yigex/ToolUniverse/examples/bio_models

# 🚀 启动专用生物模型 MCP 服务
python bio_mcp_server.py
```

服务默认在本地 8080 端口启动。您可以通过修改配置文件来调整端口和其他参数。

## 🧬 使用生物信息学工具

ToolUniverse 提供了多个命令行工具，用于调用各种生物信息学功能。

### 🧬 使用 DNABERT2 工具

DNABERT2 是一个用于 DNA 序列分析的预训练语言模型，支持嵌入向量生成和序列分类任务。

#### 📝 基本用法（直接实例调用）

```bash
# 📁 进入生物信息学工具示例目录
cd /mnt/models/yigex/ToolUniverse/examples/bio_models

# 🔬 使用虚拟环境中的 Python 运行工具
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding
```

#### 📡 基本用法（通过MCP调用）

```bash
# 📁 进入生物信息学工具示例目录
cd /mnt/models/yigex/ToolUniverse/examples/bio_models

# 🔮 通过MCP服务调用工具
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口调用工具
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding --use-mcp
```

#### ⚙️ 可用参数

| 参数 | 描述 |
|------|------|
| `--sequence` | 指定单个 DNA 序列 |
| `--sequences` | 指定多个 DNA 序列（空格分隔） |
| `--sequence-file` | 从文件中读取序列（每行一个） |
| `--task` | 任务类型，可选值为 `embedding` 或 `classification` |
| `--json` | 以 JSON 格式输出结果 |
| `--output` | 输出文件路径 |
| `--use-docker` | 是否使用 Docker 模式（默认为 True） |
| `--use-mcp` | 是否通过 MCP 服务调用工具（默认为 False） |
| `--mcp-url` | MCP 服务 URL，默认为 http://localhost:8000 |

#### 🔍 示例：生成嵌入向量

```bash
# 📊 生成嵌入向量并以 JSON 格式输出（直接调用）
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding --json

# 📡 通过MCP生成嵌入向量
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口生成嵌入向量
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task embedding --use-mcp
```

#### 🔬 示例：分类任务

```bash
# 🧪 执行序列分类任务（直接调用）
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task classification

# 📡 通过MCP执行序列分类任务
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task classification --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口执行序列分类任务
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence "ATCGATCGATCGATCGATCG" --task classification --use-mcp
```

### 🧪 使用 RNA 工具包

RNA 工具包提供了多个用于 RNA 序列分析的工具，包括 RNA 二级结构预测、序列比较、Motif 扫描和 BLAST 搜索。

#### 📝 基本用法

RNA 工具包使用子命令结构，您可以通过以下命令查看所有可用工具：

```bash
# 📚 查看 RNA 工具包帮助
cd /mnt/models/yigex/ToolUniverse/examples/bio_models
../../tooluniverse_env/bin/python rna_tools_cli.py --help
```

所有 RNA 工具都支持通过 MCP 调用，只需添加 `--use-mcp` 和可选的 `--mcp-url` 参数。

#### ⚙️ MCP 调用参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--use-mcp` | 是否通过 MCP 服务调用工具 | False |
| `--mcp-url` | MCP 服务 URL | http://localhost:8080 |

#### 🧬 RNA 二级结构预测

```bash
# 🧬 预测 RNA 二级结构（直接调用）
../../tooluniverse_env/bin/python rna_tools_cli.py rna_fold --sequence "AUGGCCAUUGCAUUCAGAU" --json

# 📡 通过MCP预测 RNA 二级结构
../../tooluniverse_env/bin/python rna_tools_cli.py rna_fold --sequence "AUGGCCAUUGCAUUCAGAU" --json --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口预测RNA二级结构
../../tooluniverse_env/bin/python rna_tools_cli.py rna_fold --sequence "AUGGCCAUUGCAUUCAGAU" --json --use-mcp
```

#### 🔄 序列比较

```bash
# 🔄 比较两个 RNA 序列（直接调用）
../../tooluniverse_env/bin/python rna_tools_cli.py seq_compare --seq1 "AUGGCCAUU" --seq2 "AUGGUCACU"

# 📡 通过MCP比较两个 RNA 序列
../../tooluniverse_env/bin/python rna_tools_cli.py seq_compare --seq1 "AUGGCCAUU" --seq2 "AUGGUCACU" --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口比较序列
../../tooluniverse_env/bin/python rna_tools_cli.py seq_compare --seq1 "AUGGCCAUU" --seq2 "AUGGUCACU" --use-mcp
```

#### 🔍 JASPAR Motif 扫描

```bash
# 🔍 扫描序列中的 JASPAR Motif（直接调用）
../../tooluniverse_env/bin/python rna_tools_cli.py jaspar_scan --sequence "ATCGATCGATCGATCGATCG" --species 9606 --threshold 0.85 --top 10

# 📡 通过MCP扫描序列中的 JASPAR Motif
../../tooluniverse_env/bin/python rna_tools_cli.py jaspar_scan --sequence "ATCGATCGATCGATCGATCG" --species 9606 --threshold 0.85 --top 10 --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口扫描Motif
../../tooluniverse_env/bin/python rna_tools_cli.py jaspar_scan --sequence "ATCGATCGATCGATCGATCG" --species 9606 --threshold 0.85 --top 10 --use-mcp
```

#### 🧬 BLAST 搜索

```bash
# 🧬 执行 BLAST 序列搜索（直接调用）
../../tooluniverse_env/bin/python rna_tools_cli.py blast_search --sequence "ATCGATCGATCGATCGATCG" --program blastn --database nt --email user@example.com --max-results 10

# 📡 通过MCP执行 BLAST 序列搜索
../../tooluniverse_env/bin/python rna_tools_cli.py blast_search --sequence "ATCGATCGATCGATCGATCG" --program blastn --database nt --email user@example.com --max-results 10 --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口执行BLAST搜索
../../tooluniverse_env/bin/python rna_tools_cli.py blast_search --sequence "ATCGATCGATCGATCGATCG" --program blastn --database nt --email user@example.com --max-results 10 --use-mcp
```

#### 🔧 MCP 调用模式说明

RNA 工具包支持两种调用模式：

1. **直接调用模式**（默认）：
   - 直接实例化工具类并调用相应方法
   - 适合本地开发和测试
   - 不需要启动 MCP 服务

2. **MCP 调用模式**（使用 `--use-mcp` 参数）：
   - 通过 MCP 服务间接调用工具
   - 适合生产环境和分布式部署
   - 需要先启动 MCP 服务
   - 支持负载均衡和资源管理

MCP 调用模式的优势：
- 🔄 **负载均衡**：MCP 服务可以智能分配请求到不同的计算节点
- 🛡️ **错误处理**：提供统一的错误处理和重试机制
- 📊 **监控日志**：集中记录所有工具调用，便于监控和调试
- 🔧 **配置管理**：统一管理工具配置，无需在每个客户端重复配置

## ⚡ 高级用法

### 📊 批量处理序列

对于大量序列的处理，可以使用 `--sequence-file` 参数或 `--sequences` 参数：

```bash
# 📁 从文件批量处理序列（直接调用）
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence-file sequences.txt --task embedding --output results.json

# 📡 通过MCP从文件批量处理序列
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence-file sequences.txt --task embedding --output results.json --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口从文件批量处理序列
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequence-file sequences.txt --task embedding --output results.json --use-mcp

# 📋 直接指定多个序列（直接调用）
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequences "ATCGATCG" "GCTAGCTA" "CGATCGAT" --task classification

# 📡 通过MCP直接指定多个序列
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequences "ATCGATCG" "GCTAGCTA" "CGATCGAT" --task classification --use-mcp --mcp-url http://localhost:8080

# 📡 使用默认MCP端口直接指定多个序列
../../tooluniverse_env/bin/python dnabert2_tool_cli.py --sequences "ATCGATCG" "GCTAGCTA" "CGATCGAT" --task classification --use-mcp
```

### 🛠️ 自定义 Docker 配置

如果您需要自定义 Docker 服务配置，可以修改 `docker-compose.yml` 文件：

```bash
# 📁 进入 ToolUniverse 目录
cd /mnt/models/yigex/ToolUniverse

# 📝 编辑 Docker 配置文件
vi docker-compose.yml

# 🔄 修改后重启服务
docker-compose up -d
```

### 📋 查看日志

查看 ToolUniverse 和 Docker 容器的日志：

```bash
# 📊 查看 MCP 服务日志
tail -f /var/log/tooluniverse/mcp.log

# 🐳 查看 Docker 容器日志
docker-compose logs -f
```

## 🛠️ 故障排除

### ❓ 常见问题

- **Docker 服务启动失败** ❌ - 检查 Docker 是否正确安装，以及是否有足够的权限
- **工具调用超时** ⏱️ - 对于大型序列或复杂任务，可能需要增加超时设置
- **内存不足错误** 🧠 - 考虑增加 Docker 容器的内存限制

### 🔍 调试建议

- 使用 `--json` 参数输出详细的 JSON 格式结果，便于调试
- 检查 MCP 服务日志以获取详细的错误信息
- 确保所有 Docker 容器都处于运行状态

## ⚡ 性能优化

- ⚡ 对于批量处理，优先使用 `--sequence-file` 以减少命令行参数的复杂性
- 🔄 对于频繁调用，可以考虑使用 API 接口而不是命令行工具
- 💾 确保系统有足够的资源分配给 Docker 容器

## 📝 总结

ToolUniverse 提供了一个统一的平台，用于管理和调用各种生物信息学工具。通过 Docker 容器化和 MCP 服务，它实现了工具的隔离部署和统一访问，极大地简化了生物信息学分析的工作流程。

---

**⚠️ 注意**：本指南基于当前 ToolUniverse 的实现，随着项目的发展，某些命令和参数可能会有所变化。请参考最新的项目文档获取最准确的信息。

**📧 联系我们**：如果您在使用过程中遇到任何问题，请提交 Issue 或联系开发团队。

---

<p align="center">
  <strong>🔬 ToolUniverse - 释放生物信息学研究的无限可能 🔬</strong>
</p>