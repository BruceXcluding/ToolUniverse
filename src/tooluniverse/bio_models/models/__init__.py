"""
生物序列模型实现
"""
from .base_model import BaseModel
import os
import importlib
import inspect
from typing import Dict, List, Type, Any

# 动态导入模型类
def _import_model_classes() -> Dict[str, Type[BaseModel]]:
    """动态导入所有模型类"""
    models = {}
    
    # 获取当前目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 遍历当前目录下的所有Python文件
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # 移除.py扩展名
            
            try:
                # 动态导入模块
                module = importlib.import_module(f'.{module_name}', package=__package__)
                
                # 查找模块中的所有类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 检查是否是BaseModel的子类，但不是BaseModel本身
                    if issubclass(obj, BaseModel) and obj is not BaseModel:
                        models[name] = obj
                        print(f"已导入模型类: {name}")
                        
            except ImportError as e:
                print(f"导入模型模块 {module_name} 失败: {str(e)}")
    
    return models

# 导入所有模型类
_model_classes = _import_model_classes()

# 将所有模型类添加到当前模块的命名空间
for name, cls in _model_classes.items():
    globals()[name] = cls

# 更新__all__列表
__all__ = ["BaseModel"] + list(_model_classes.keys())

# 提供获取所有可用模型类的函数
def get_available_models() -> Dict[str, Type[BaseModel]]:
    """获取所有可用的模型类"""
    return _model_classes.copy()

# 提供获取模型类的函数
def get_model_class(model_name: str) -> Type[BaseModel]:
    """根据名称获取模型类"""
    if model_name in _model_classes:
        return _model_classes[model_name]
    else:
        raise ValueError(f"未找到模型类: {model_name}")

# 提供注册新模型类的函数
def register_model_class(name: str, model_class: Type[BaseModel]) -> None:
    """注册新的模型类"""
    if not issubclass(model_class, BaseModel):
        raise ValueError(f"模型类必须继承自BaseModel")
    
    _model_classes[name] = model_class
    globals()[name] = model_class
    
    if name not in __all__:
        __all__.append(name)
    
    print(f"已注册模型类: {name}")