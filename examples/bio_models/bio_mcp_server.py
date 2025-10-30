#!/usr/bin/env python3
"""
生物序列分析工具的MCP服务器启动脚本
"""

import sys
import os

# 确保使用项目虚拟环境
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(project_root, "tooluniverse_env", "bin", "python")
if os.path.exists(venv_python) and sys.executable != venv_python:
    print(f"切换到虚拟环境: {venv_python}")
    os.execl(venv_python, venv_python, *sys.argv)

# 添加项目路径
sys.path.insert(0, project_root)

from tooluniverse import create_smcp_server
from tooluniverse.mcp_tool_registry import register_mcp_tool_from_config

# 导入所有生物模型工具
from tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from tooluniverse.bio_models.tools.alphafold_tool import AlphaFoldTool
from tooluniverse.bio_models.tools.annotation_tool import AnnotationTool
from tooluniverse.bio_models.tools.classification_tool import ClassificationTool
from tooluniverse.bio_models.tools.codonbert_tool import CodonBERTTool
from tooluniverse.bio_models.tools.dnabert2_tool import DNABERT2Tool
from tooluniverse.bio_models.tools.embedding_tool import EmbeddingTool
from tooluniverse.bio_models.tools.generation_tool import GenerationTool
from tooluniverse.bio_models.tools.interaction_tool import InteractionTool
from tooluniverse.bio_models.tools.lucaone_tool import LucaOneTool
from tooluniverse.bio_models.tools.lucaoneapp_tool import LucaOneAppTool
from tooluniverse.bio_models.tools.lucaonetasks_tool import LucaOneTasksTool
from tooluniverse.bio_models.tools.property_prediction_tool import PropertyPredictionTool
from tooluniverse.bio_models.tools.rnafm_tool import RNAFMTool
from tooluniverse.bio_models.tools.structure_prediction_tool import StructurePredictionTool
from tooluniverse.bio_models.tools.task_specific_tool import TaskSpecificTool
from tooluniverse.bio_models.tools.three_utrbert_tool import ThreeUTRBERTTool
from tooluniverse.bio_models.tools.utrlm_tool import UTRLMTool

def main():
    """启动生物序列分析MCP服务器"""
    print("🧬 启动生物序列分析MCP服务器...")
    
    # 创建ToolUniverse实例
    from tooluniverse import ToolUniverse
    tu = ToolUniverse(keep_default_tools=False)  # 不加载默认工具
    
    # 注册所有生物模型工具
    bio_tools = [
        (BioSequenceAnalysisTool, "bio_sequence_analysis", "生物序列分析统一接口工具"),
        (AlphaFoldTool, "alphafold", "AlphaFold蛋白质结构预测工具"),
        (AnnotationTool, "annotation", "生物序列注释工具"),
        (ClassificationTool, "classification", "生物序列分类工具"),
        (CodonBERTTool, "codonbert", "CodonBERT模型工具"),
        (DNABERT2Tool, "dnabert2", "DNABERT2模型工具"),
        (EmbeddingTool, "embedding", "生物序列嵌入工具"),
        (GenerationTool, "generation", "生物序列生成工具"),
        (InteractionTool, "interaction", "生物分子相互作用预测工具"),
        (LucaOneTool, "lucaone", "LucaOne多模态生物模型工具"),
        (LucaOneAppTool, "lucaoneapp", "LucaOne应用工具"),
        (LucaOneTasksTool, "lucaonetasks", "LucaOne任务特定工具"),
        (PropertyPredictionTool, "property_prediction", "生物分子属性预测工具"),
        (RNAFMTool, "rnafm", "RNA基础模型工具"),
        (StructurePredictionTool, "structure_prediction", "生物结构预测工具"),
        (TaskSpecificTool, "task_specific", "任务特定生物工具"),
        (ThreeUTRBERTTool, "three_utrbert", "3'UTR BERT模型工具"),
        (UTRLMTool, "utrlm", "UTR语言模型工具")
    ]
    
    for tool_class, tool_name, tool_desc in bio_tools:
        try:
            # 为bio_sequence_analysis工具使用特定的参数模式
            if tool_name == "bio_sequence_analysis":
                # 使用直接参数模式（与MCP客户端调用方式一致）
                # 更新参数模式以符合ToolUniverse标准格式
                parameter_schema = {
                    "type": "object",
                    "properties": {
                        "sequences": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "单个序列"
                                },
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "序列列表"
                                }
                            ],
                            "description": "要分析的序列"
                        },
                        "task_type": {
                            "type": "string",
                            "description": "任务类型",
                            "enum": ["embedding", "classification", "generation", "prediction", "analysis"]
                        },
                        "sequence_type": {
                            "type": "string",
                            "description": "序列类型",
                            "enum": ["DNA", "RNA", "protein", "peptide", "dna", "rna"]
                        },
                        "model_name": {
                            "type": "string",
                            "description": "模型名称",
                            "default": "default"
                        },
                        "device": {
                            "type": "string",
                            "description": "设备",
                            "default": "cpu"
                        },
                        "monitor_mode": {
                            "type": "boolean",
                            "description": "是否启用监控模式",
                            "default": False
                        }
                    },
                    "required": ["sequences", "task_type", "sequence_type"]
                }
                
                # 使用MCP工具注册器注册工具，并指定参数模式
                register_mcp_tool_from_config(tool_class, {
                    "name": tool_name,
                    "description": tool_desc,
                    "mcp_config": {
                        "server_name": f"生物模型工具 - {tool_name}",
                        "port": 8080,  # 所有工具使用同一个端口
                        "auto_start": False
                    },
                    "parameter_schema": parameter_schema
                })
            else:
                # 为dnabert2工具添加相同的参数模式
                if tool_name == "dnabert2":
                    register_mcp_tool_from_config(tool_class, {
                        "name": tool_name,
                        "description": tool_desc,
                        "mcp_config": {
                            "server_name": f"生物模型工具 - {tool_name}",
                            "port": 8080,  # 所有工具使用同一个端口
                            "auto_start": False
                        },
                        "parameter_schema": parameter_schema
                    })
                else:
                    # 使用MCP工具注册器注册工具
                    register_mcp_tool_from_config(tool_class, {
                        "name": tool_name,
                        "description": tool_desc,
                        "mcp_config": {
                            "server_name": f"生物模型工具 - {tool_name}",
                            "port": 8080,  # 所有工具使用同一个端口
                            "auto_start": False
                        }
                    })
            print(f"✅ 注册工具: {tool_name}")
        except Exception as e:
            print(f"❌ 注册工具 {tool_name} 失败: {str(e)}")
    
    # 使用SMCP创建服务器，自动暴露所有注册的工具
    server = create_smcp_server(
        name="生物序列分析MCP服务",
        tooluniverse_config=tu,
        search_enabled=False
    )
    
    print("🚀 启动MCP服务器，端口: 8080")
    print("📋 可用工具将通过MCP协议暴露")
    print("🔗 客户端可以通过以下方式连接:")
    print("   - HTTP: http://localhost:8080/mcp/")
    print("   - WebSocket: ws://localhost:8080/mcp/")
    
    try:
        # 启动服务器
        server.run_simple(transport="http", host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"❌ 服务器启动失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()