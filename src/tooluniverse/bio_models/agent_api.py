#!/usr/bin/env python3
"""
Agent调用ToolUniverse生物工具的接口
提供RESTful API供上层agent调用
"""

from flask import Flask, request, jsonify
import os
import sys
import logging
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from tooluniverse.bio_models.model_manager import ModelManager
from tooluniverse.bio_models.container_client import DNABERT2Client, LucaOneClient

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化工具
bio_tool = BioSequenceAnalysisTool()
model_manager = ModelManager()

# 初始化容器客户端
dnabert2_client = None
lucaone_client = None

# 尝试连接到Docker容器中的模型服务
def init_container_clients():
    """初始化容器客户端"""
    global dnabert2_client, lucaone_client
    
    # 从环境变量获取容器服务地址，如果没有则使用默认值
    dnabert2_url = os.environ.get('DNABERT2_URL', 'http://dnabert2:8001')
    lucaone_url = os.environ.get('LUCAONE_URL', 'http://lucaone:8002')
    
    try:
        # 尝试连接DNABERT2容器
        dnabert2_client = DNABERT2Client(dnabert2_url)
        health = dnabert2_client.check_health()
        logger.info(f"成功连接到DNABERT2容器: {health.status}")
    except Exception as e:
        logger.warning(f"无法连接到DNABERT2容器: {str(e)}")
        dnabert2_client = None
    
    try:
        # 尝试连接LucaOne容器
        lucaone_client = LucaOneClient(lucaone_url)
        health = lucaone_client.check_health()
        logger.info(f"成功连接到LucaOne容器: {health.status}")
    except Exception as e:
        logger.warning(f"无法连接到LucaOne容器: {str(e)}")
        lucaone_client = None

# 初始化容器客户端
init_container_clients()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "ToolUniverse Bio Tools API",
        "container_clients": {
            "dnabert2": dnabert2_client is not None,
            "lucaone": lucaone_client is not None
        }
    })

@app.route('/models', methods=['GET'])
def list_models():
    """列出所有可用模型"""
    try:
        # 获取本地模型
        local_models = bio_tool.list_models()
        
        # 添加容器模型
        container_models = []
        if dnabert2_client:
            try:
                info = dnabert2_client.get_model_info()
                container_models.append({
                    "name": "dnabert2-container",
                    "type": "dnabert2",
                    "deployment_type": "container",
                    "status": "running",
                    "tasks": info.tasks,
                    "max_sequence_length": info.max_sequence_length
                })
            except Exception as e:
                logger.error(f"获取DNABERT2容器模型信息失败: {str(e)}")
        
        if lucaone_client:
            try:
                info = lucaone_client.get_model_info()
                container_models.append({
                    "name": "lucaone-container",
                    "type": "lucaone",
                    "deployment_type": "container",
                    "status": "running",
                    "tasks": info.tasks,
                    "max_sequence_length": info.max_sequence_length
                })
            except Exception as e:
                logger.error(f"获取LucaOne容器模型信息失败: {str(e)}")
        
        return jsonify({
            "success": True,
            "local_models": local_models,
            "container_models": container_models,
            "total_models": len(local_models) + len(container_models)
        })
    except Exception as e:
        logger.error(f"列出模型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models/loaded', methods=['GET'])
def list_loaded_models():
    """列出所有已加载的模型"""
    try:
        # 获取本地已加载模型
        local_loaded = bio_tool.list_loaded_models()
        
        # 添加运行中的容器模型
        container_loaded = []
        if dnabert2_client:
            container_loaded.append({
                "name": "dnabert2-container",
                "type": "dnabert2",
                "deployment_type": "container",
                "status": "running"
            })
        
        if lucaone_client:
            container_loaded.append({
                "name": "lucaone-container",
                "type": "lucaone",
                "deployment_type": "container",
                "status": "running"
            })
        
        return jsonify({
            "success": True,
            "local_loaded": local_loaded,
            "container_loaded": container_loaded,
            "total_loaded": len(local_loaded) + len(container_loaded)
        })
    except Exception as e:
        logger.error(f"列出已加载模型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models/best', methods=['POST'])
def get_best_model():
    """获取最佳模型"""
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        sequence_type = data.get('sequence_type')
        
        # 首先尝试从容器模型中选择
        if task_type in ['embedding', 'classification', 'property_prediction']:
            if sequence_type in ['DNA', 'RNA', 'protein'] and lucaone_client:
                return jsonify({
                    "success": True,
                    "best_model": "lucaone-container",
                    "deployment_type": "container"
                })
            elif sequence_type == 'DNA' and dnabert2_client:
                return jsonify({
                    "success": True,
                    "best_model": "dnabert2-container",
                    "deployment_type": "container"
                })
        
        # 如果没有合适的容器模型，使用本地模型
        model_name = bio_tool.get_best_model(task_type, sequence_type)
        return jsonify({
            "success": True,
            "best_model": model_name,
            "deployment_type": "local"
        })
    except Exception as e:
        logger.error(f"获取最佳模型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_sequences():
    """分析生物序列"""
    try:
        data = request.get_json()
        sequences = data.get('sequences')
        task_type = data.get('task_type')
        model_name = data.get('model_name')
        sequence_type = data.get('sequence_type')
        
        if not sequences or not task_type:
            return jsonify({
                "success": False,
                "error": "缺少必要参数: sequences, task_type"
            }), 400
        
        # 如果指定了容器模型，使用容器客户端
        if model_name == 'dnabert2-container' and dnabert2_client:
            return analyze_with_dnabert2_container(sequences, task_type, sequence_type)
        elif model_name == 'lucaone-container' and lucaone_client:
            return analyze_with_lucaone_container(sequences, task_type, sequence_type)
        
        # 否则使用本地模型
        result = bio_tool.analyze(
            sequences=sequences,
            task_type=task_type,
            model_name=model_name,
            sequence_type=sequence_type
        )
        
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        logger.error(f"序列分析失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def analyze_with_dnabert2_container(sequences, task_type, sequence_type=None):
    """使用DNABERT2容器进行分析"""
    try:
        if task_type == 'embedding':
            # 使用DNABERT2获取嵌入向量
            results = []
            for seq in sequences:
                result = dnabert2_client.get_embedding(seq)
                results.append({
                    "sequence": seq,
                    "embedding": result.prediction,
                    "sequence_length": result.sequence_length
                })
            
            return jsonify({
                "success": True,
                "result": {
                    "predictions": results,
                    "model_name": "dnabert2-container",
                    "deployment_type": "container"
                }
            })
        elif task_type == 'classification':
            # 使用DNABERT2进行分类
            results = []
            for seq in sequences:
                result = dnabert2_client.classify_sequence(seq)
                results.append({
                    "sequence": seq,
                    "classification": result.prediction,
                    "confidence": result.confidence,
                    "sequence_length": result.sequence_length
                })
            
            return jsonify({
                "success": True,
                "result": {
                    "predictions": results,
                    "model_name": "dnabert2-container",
                    "deployment_type": "container"
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": f"DNABERT2容器不支持任务类型: {task_type}"
            }), 400
    except Exception as e:
        logger.error(f"DNABERT2容器分析失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"DNABERT2容器分析失败: {str(e)}"
        }), 500

def analyze_with_lucaone_container(sequences, task_type, sequence_type=None):
    """使用LucaOne容器进行分析"""
    try:
        if task_type == 'embedding':
            # 使用LucaOne获取嵌入向量
            results = []
            for seq in sequences:
                result = lucaone_client.get_embedding(seq)
                results.append({
                    "sequence": seq,
                    "embedding": result.prediction,
                    "sequence_length": result.sequence_length
                })
            
            return jsonify({
                "success": True,
                "result": {
                    "predictions": results,
                    "model_name": "lucaone-container",
                    "deployment_type": "container"
                }
            })
        elif task_type == 'classification':
            # 使用LucaOne进行分类
            results = []
            for seq in sequences:
                result = lucaone_client.classify_sequence(seq)
                results.append({
                    "sequence": seq,
                    "classification": result.prediction,
                    "confidence": result.confidence,
                    "sequence_length": result.sequence_length
                })
            
            return jsonify({
                "success": True,
                "result": {
                    "predictions": results,
                    "model_name": "lucaone-container",
                    "deployment_type": "container"
                }
            })
        elif task_type == 'property_prediction':
            # 使用LucaOne进行属性预测
            results = []
            for seq in sequences:
                result = lucaone_client.predict_property(seq)
                results.append({
                    "sequence": seq,
                    "property": result.prediction,
                    "confidence": result.confidence,
                    "sequence_length": result.sequence_length
                })
            
            return jsonify({
                "success": True,
                "result": {
                    "predictions": results,
                    "model_name": "lucaone-container",
                    "deployment_type": "container"
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": f"LucaOne容器不支持任务类型: {task_type}"
            }), 400
    except Exception as e:
        logger.error(f"LucaOne容器分析失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"LucaOne容器分析失败: {str(e)}"
        }), 500

@app.route('/tasks', methods=['GET'])
def get_supported_tasks():
    """获取支持的任务类型"""
    try:
        tasks = bio_tool.get_supported_tasks()
        return jsonify({
            "success": True,
            "tasks": tasks
        })
    except Exception as e:
        logger.error(f"获取支持任务类型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/sequence-types', methods=['GET'])
def get_supported_sequence_types():
    """获取支持的序列类型"""
    try:
        sequence_types = bio_tool.get_supported_sequence_types()
        return jsonify({
            "success": True,
            "sequence_types": sequence_types
        })
    except Exception as e:
        logger.error(f"获取支持序列类型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/model/<model_name>/info', methods=['GET'])
def get_model_info(model_name):
    """获取模型信息"""
    try:
        if model_name == 'dnabert2-container' and dnabert2_client:
            info = dnabert2_client.get_model_info()
            return jsonify({
                "success": True,
                "model_info": {
                    "name": info.name,
                    "version": info.version,
                    "description": info.description,
                    "tasks": info.tasks,
                    "max_sequence_length": info.max_sequence_length,
                    "deployment_type": "container"
                }
            })
        elif model_name == 'lucaone-container' and lucaone_client:
            info = lucaone_client.get_model_info()
            return jsonify({
                "success": True,
                "model_info": {
                    "name": info.name,
                    "version": info.version,
                    "description": info.description,
                    "tasks": info.tasks,
                    "max_sequence_length": info.max_sequence_length,
                    "deployment_type": "container"
                }
            })
        else:
            info = bio_tool.get_model_info(model_name)
            return jsonify({
                "success": True,
                "model_info": info
            })
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/model/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """加载模型"""
    try:
        # 容器模型不需要加载
        if model_name in ['dnabert2-container', 'lucaone-container']:
            return jsonify({
                "success": True,
                "message": f"容器模型 {model_name} 已在运行中"
            })
        
        success = bio_tool.load_model(model_name)
        return jsonify({
            "success": success,
            "message": f"模型 {model_name} 加载{'成功' if success else '失败'}"
        })
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/model/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """卸载模型"""
    try:
        # 容器模型不需要卸载
        if model_name in ['dnabert2-container', 'lucaone-container']:
            return jsonify({
                "success": True,
                "message": f"容器模型 {model_name} 无需卸载"
            })
        
        success = bio_tool.unload_model(model_name)
        return jsonify({
            "success": success,
            "message": f"模型 {model_name} 卸载{'成功' if success else '失败'}"
        })
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/containers/reconnect', methods=['POST'])
def reconnect_containers():
    """重新连接容器服务"""
    try:
        init_container_clients()
        return jsonify({
            "success": True,
            "message": "容器服务重新连接完成",
            "container_clients": {
                "dnabert2": dnabert2_client is not None,
                "lucaone": lucaone_client is not None
            }
        })
    except Exception as e:
        logger.error(f"重新连接容器服务失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # 启动API服务
    app.run(host='0.0.0.0', port=5000, debug=True)