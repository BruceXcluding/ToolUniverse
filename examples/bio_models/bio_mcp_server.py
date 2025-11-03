#!/usr/bin/env python3
"""
生物序列分析MCP服务器 - 修复版本 v4
修复参数传递问题，提供统一的生物序列分析接口
"""

import sys
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tooluniverse import ToolUniverse
from tooluniverse.smcp import create_smcp_server
from tooluniverse.bio_models.tools import (
    BioSequenceAnalysisTool,
    AlphaFoldTool,
    AnnotationTool,
    ClassificationTool,
    CodonBERTTool,
    DNABERT2Tool,
    EmbeddingTool,
    GenerationTool,
    InteractionTool,
    LucaOneTool,
    LucaOneAppTool,
    LucaOneTasksTool,
    PropertyPredictionTool,
    RNAFMTool,
    StructurePredictionTool,
    TaskSpecificTool,
    ThreeUTRBERTTool,
    UTRLMTool,
    # RNA工具
    BlastSearchTool,
    JasparScanTool,
    RNAFoldTool,
    SeqCompareTool
)
from tooluniverse.bio_models.model_manager import ModelManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("🧬 启动生物序列分析MCP服务器 (修复版 v4)...")
    
    try:
        # 初始化ToolUniverse
        print("📦 初始化ToolUniverse...")
        tu = ToolUniverse()
        
        # 创建模型管理器实例
        print("🔧 创建模型管理器...")
        model_manager = ModelManager()
        
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
            (UTRLMTool, "utrlm", "UTR语言模型工具"),
            # RNA工具
            (BlastSearchTool, "blast_search", "BLAST序列搜索工具"),
            (JasparScanTool, "jaspar_scan", "JASPAR Motif扫描工具"),
            (RNAFoldTool, "rna_fold", "RNA二级结构预测工具"),
            (SeqCompareTool, "seq_compare", "序列比对与相似度计算工具")
        ]
        
        for tool_class, tool_name, tool_desc in bio_tools:
            try:
                # 为所有工具定义一致的参数模式
                if tool_name == "bio_sequence_analysis":
                    # 使用直接参数模式（与MCP客户端调用方式一致）
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
                    
                    # 使用ToolUniverse的register_custom_tool方法直接注册工具
                    tu.register_custom_tool(
                        tool_class=tool_class,
                        tool_name=tool_name,
                        tool_config={
                            "name": tool_name,
                            "type": tool_name,
                            "description": tool_desc,
                            "parameter": parameter_schema,
                            "category": "bio_models"
                        },
                        instantiate=True  # 立即实例化并缓存
                    )
                    
                elif tool_name == "dnabert2":
                    # 为dnabert2工具定义自己的参数模式
                    dnabert2_parameter_schema = {
                        "type": "object",
                        "properties": {
                            "sequences": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "要分析的序列列表（支持单个或多个序列）"
                            },
                            "task_type": {
                                "type": "string",
                                "description": "任务类型",
                                "enum": ["embedding", "classification"]
                            },
                            "sequence_type": {
                                "type": "string",
                                "description": "序列类型",
                                "enum": ["DNA", "RNA", "protein", "peptide", "dna", "rna"],
                                "default": "DNA"
                            },
                            "device": {
                                "type": "string",
                                "description": "设备",
                                "default": "AUTO"
                            },
                            "monitor_mode": {
                                "type": "boolean",
                                "description": "是否启用监控模式",
                                "default": False
                            }
                        },
                        "required": ["sequences", "task_type"]
                    }
                    
                    # 使用ToolUniverse的register_custom_tool方法直接注册工具
                    tu.register_custom_tool(
                        tool_class=tool_class,
                        tool_name=tool_name,
                        tool_config={
                            "name": tool_name,
                            "type": tool_name,
                            "description": tool_desc,
                            "parameter": dnabert2_parameter_schema,
                            "category": "bio_models"
                        },
                        instantiate=True  # 立即实例化并缓存
                    )
                elif tool_name in ["rna_fold", "blast_search", "jaspar_scan", "seq_compare"]:
                    # 为RNA工具定义特定的参数模式
                    if tool_name == "rna_fold":
                        rna_tool_parameter_schema = {
                            "type": "object",
                            "properties": {
                                "sequence": {
                                    "type": "string",
                                    "description": "RNA序列"
                                },
                                "task_type": {
                                    "type": "string",
                                    "description": "任务类型",
                                    "enum": ["fold", "structure"],
                                    "default": "fold"
                                }
                            },
                            "required": ["sequence"]
                        }
                    elif tool_name == "blast_search":
                        rna_tool_parameter_schema = {
                            "type": "object",
                            "properties": {
                                "sequence": {
                                    "type": "string",
                                    "description": "要搜索的序列"
                                },
                                "task_type": {
                                    "type": "string",
                                    "description": "任务类型",
                                    "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"],
                                    "default": "blastn"
                                },
                                "parse_xml": {
                                    "type": "boolean",
                                    "description": "是否解析XML结果",
                                    "default": False
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "最大结果数",
                                    "default": 10
                                }
                            },
                            "required": ["sequence"]
                        }
                    elif tool_name == "jaspar_scan":
                        rna_tool_parameter_schema = {
                            "type": "object",
                            "properties": {
                                "sequence": {
                                    "type": "string",
                                    "description": "要扫描的序列"
                                },
                                "task_type": {
                                    "type": "string",
                                    "description": "任务类型",
                                    "enum": ["scan", "search"],
                                    "default": "scan"
                                },
                                "species": {
                                    "type": "string",
                                    "description": "物种",
                                    "default": "human"
                                },
                                "threshold_ratio": {
                                    "type": "number",
                                    "description": "阈值比例",
                                    "default": 0.8
                                },
                                "top_n": {
                                    "type": "integer",
                                    "description": "返回前N个结果",
                                    "default": 5
                                },
                                "quiet_mode": {
                                    "type": "boolean",
                                    "description": "静默模式",
                                    "default": True
                                }
                            },
                            "required": ["sequence"]
                        }
                    elif tool_name == "seq_compare":
                        rna_tool_parameter_schema = {
                            "type": "object",
                            "properties": {
                                "seq1": {
                                    "type": "string",
                                    "description": "第一个序列"
                                },
                                "seq2": {
                                    "type": "string",
                                    "description": "第二个序列"
                                }
                            },
                            "required": ["seq1", "seq2"]
                        }
                    
                    # 使用ToolUniverse的register_custom_tool方法直接注册工具
                    tu.register_custom_tool(
                        tool_class=tool_class,
                        tool_name=tool_name,
                        tool_config={
                            "name": tool_name,
                            "type": tool_name,
                            "description": tool_desc,
                            "parameter": rna_tool_parameter_schema,
                            "category": "bio_models"
                        },
                        instantiate=True  # 立即实例化并缓存
                    )
                else:
                    # 为需要model_manager参数的工具提供该参数
                    if tool_name in ["lucaoneapp", "lucaonetasks", "three_utrbert", "utrlm", "task_specific"]:
                        # 先实例化工具，然后传递实例
                        tool_instance = tool_class(model_manager=model_manager)
                        tu.register_custom_tool(
                            tool_class=tool_class,
                            tool_name=tool_name,
                            tool_config={
                                "name": tool_name,
                                "type": tool_name,
                                "description": tool_desc,
                                "category": "bio_models"
                            },
                            tool_instance=tool_instance
                        )
                    else:
                        # 使用ToolUniverse的register_custom_tool方法直接注册工具
                        tu.register_custom_tool(
                            tool_class=tool_class,
                            tool_name=tool_name,
                            tool_config={
                                "name": tool_name,
                                "type": tool_name,
                                "description": tool_desc,
                                "category": "bio_models"
                            },
                            instantiate=True  # 立即实例化并缓存
                        )
                        
                print(f"✅ 注册工具: {tool_name}")
            except Exception as e:
                print(f"❌ 注册工具 {tool_name} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 使用SMCP创建服务器，自动暴露所有注册的工具
        server = create_smcp_server(
            name="生物序列分析MCP服务 (修复版 v4)",
            tooluniverse_config=tu,
            search_enabled=False,
            auto_expose_tools=True  # 确保自动暴露工具
        )
        
        print("🚀 启动MCP服务器，端口: 8080")
        print("📋 可用工具将通过MCP协议暴露")
        print("🔗 客户端可以通过以下方式连接:")
        print("   - HTTP: http://localhost:8080/mcp/")
        print("   - WebSocket: ws://localhost:8080/mcp/")
        
        # 打印已注册的工具列表
        if hasattr(tu, 'all_tools') and tu.all_tools:
            print("\n📋 已注册的工具列表:")
            for tool in tu.all_tools:
                if isinstance(tool, dict) and 'name' in tool:
                    print(f"   - {tool['name']}: {tool.get('description', '无描述')}")
        
        try:
            # 启动服务器
            server.run_simple(
                transport="http",
                host="0.0.0.0",
                port=8080,
                stateless_http=True  # Enable stateless mode for HTTP requests
            )
        except Exception as e:
            print(f"❌ 服务器启动失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()