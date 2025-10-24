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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tooluniverse.bio_models.tools.unified_interface_tool import BioSequenceAnalysisTool
from tooluniverse.bio_models.model_manager import ModelManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化工具
bio_tool = BioSequenceAnalysisTool()
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "ToolUniverse Bio Tools API"
    })

@app.route('/models', methods=['GET'])
def list_models():
    """列出所有可用模型"""
    try:
        models = bio_tool.list_models()
        return jsonify({
            "success": True,
            "models": models
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
        models = bio_tool.list_loaded_models()
        return jsonify({
            "success": True,
            "loaded_models": models
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
        
        model_name = bio_tool.get_best_model(task_type, sequence_type)
        return jsonify({
            "success": True,
            "best_model": model_name
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

if __name__ == '__main__':
    # 启动API服务
    app.run(host='0.0.0.0', port=5000, debug=True)