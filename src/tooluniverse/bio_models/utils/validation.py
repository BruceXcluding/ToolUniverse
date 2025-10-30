"""生物序列和参数验证功能"""
from typing import Dict, List, Any, Optional


def validate_sequences(sequences: List[str], sequence_type: str) -> Dict[str, Any]:
    """
    验证生物序列是否符合指定类型的格式要求
    
    Args:
        sequences: 要验证的序列列表
        sequence_type: 序列类型，支持 "DNA", "RNA", "PROTEIN" 等
    
    Returns:
        包含验证结果的字典，格式为 {"valid": bool, "errors": List[str]}
    """
    errors = []
    
    # 定义不同序列类型的有效字符集
    valid_chars = {
        "DNA": {'A', 'T', 'C', 'G', 'a', 't', 'c', 'g', 'N', 'n'},
        "RNA": {'A', 'U', 'C', 'G', 'a', 'u', 'c', 'g', 'N', 'n'},
        "PROTEIN": {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 
                    'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'B', 'Z', '*',
                    'a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 
                    'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'x', 'b', 'z'}
    }
    
    # 验证序列列表格式
    if not isinstance(sequences, list):
        return {"valid": False, "errors": ["输入必须是序列列表"]}
    
    if not sequences:
        return {"valid": False, "errors": ["序列列表不能为空"]}
    
    # 获取当前序列类型的有效字符集
    chars = valid_chars.get(sequence_type.upper(), valid_chars["DNA"])  # 默认使用DNA字符集
    
    # 验证每个序列
    for i, seq in enumerate(sequences):
        if not isinstance(seq, str):
            errors.append(f"序列{i}不是字符串类型")
        elif not seq:
            errors.append(f"序列{i}为空")
        elif not all(c in chars for c in seq):
            # 找出第一个无效字符
            invalid_chars = set(c for c in seq if c not in chars)
            errors.append(f"序列{i}包含无效字符: {', '.join(invalid_chars)}")
        
        # 检查序列长度是否合理
        if len(seq) > 100000:  # 防止过大的序列
            errors.append(f"序列{i}长度({len(seq)})过大，可能导致性能问题")
    
    return {"valid": len(errors) == 0, "errors": errors}


def validate_parameters(params: Dict[str, Any], 
                        required_params: Optional[List[str]] = None,
                        param_constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    验证模型参数是否符合要求
    
    Args:
        params: 要验证的参数字典
        required_params: 必需的参数列表（可选）
        param_constraints: 参数约束条件字典（可选），格式为:
            {
                "param_name": {
                    "type": expected_type,  # 期望的类型
                    "min": min_value,       # 最小值（如果适用）
                    "max": max_value,       # 最大值（如果适用）
                    "allowed_values": []    # 允许的值列表（如果适用）
                }
            }
    
    Returns:
        包含验证结果的字典，格式为 {"valid": bool, "errors": List[str]}
    """
    errors = []
    required_params = required_params or []
    param_constraints = param_constraints or {}
    
    # 验证必需参数是否存在
    for required in required_params:
        if required not in params:
            errors.append(f"缺少必需参数: {required}")
    
    # 验证参数类型和约束
    for param_name, value in params.items():
        if param_name in param_constraints:
            constraints = param_constraints[param_name]
            
            # 验证类型
            if "type" in constraints and not isinstance(value, constraints["type"]):
                errors.append(f"参数 {param_name} 应为 {constraints['type'].__name__} 类型，得到 {type(value).__name__}")
            
            # 验证最小值
            if "min" in constraints and value < constraints["min"]:
                errors.append(f"参数 {param_name} 不能小于 {constraints['min']}")
            
            # 验证最大值
            if "max" in constraints and value > constraints["max"]:
                errors.append(f"参数 {param_name} 不能大于 {constraints['max']}")
            
            # 验证允许的值
            if "allowed_values" in constraints and value not in constraints["allowed_values"]:
                errors.append(f"参数 {param_name} 的值必须是以下之一: {', '.join(map(str, constraints['allowed_values']))}")
    
    # 默认验证一些常见参数
    if "max_sequence_length" in params:
        if params["max_sequence_length"] <= 0:
            errors.append("max_sequence_length必须大于0")
        elif params["max_sequence_length"] > 10000:
            errors.append("max_sequence_length过大，建议不超过10000")
    
    if "batch_size" in params:
        if params["batch_size"] <= 0:
            errors.append("batch_size必须大于0")
        elif params["batch_size"] > 1024:
            errors.append("batch_size过大，建议不超过1024")
    
    if "temperature" in params:
        if params["temperature"] < 0:
            errors.append("temperature不能小于0")
    
    return {"valid": len(errors) == 0, "errors": errors}


def normalize_sequences(sequences: List[str], sequence_type: str = "DNA") -> List[str]:
    """
    规范化序列，转换为大写并去除空格
    
    Args:
        sequences: 要规范化的序列列表
        sequence_type: 序列类型
    
    Returns:
        规范化后的序列列表
    """
    normalized = []
    for seq in sequences:
        if isinstance(seq, str):
            # 去除空格和换行符
            seq = seq.strip().replace(' ', '').replace('\n', '')
            # 转换为大写
            seq = seq.upper()
            # 对于RNA序列，将T替换为U
            if sequence_type.upper() == "RNA":
                seq = seq.replace('T', 'U')
            # 对于DNA序列，将U替换为T
            elif sequence_type.upper() == "DNA":
                seq = seq.replace('U', 'T')
            normalized.append(seq)
    return normalized


def validate_model_inputs(sequences: List[str], 
                           task_type: str, 
                           sequence_type: str = "DNA",
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    综合验证模型输入（序列、任务类型、参数）
    
    Args:
        sequences: 输入序列列表
        task_type: 任务类型
        sequence_type: 序列类型
        params: 模型参数
    
    Returns:
        包含验证结果的字典，格式为 {"valid": bool, "errors": List[str]}
    """
    errors = []
    
    # 验证序列
    sequence_result = validate_sequences(sequences, sequence_type)
    if not sequence_result["valid"]:
        errors.extend(sequence_result["errors"])
    
    # 验证任务类型
    supported_tasks = ["CLASSIFICATION", "EMBEDDING", "PREDICTION", "GENERATION", "ANNOTATION"]
    if task_type.upper() not in supported_tasks:
        errors.append(f"不支持的任务类型: {task_type}，支持的任务类型: {', '.join(supported_tasks)}")
    
    # 验证参数
    if params:
        # 根据任务类型设置参数约束
        task_constraints = {
            "EMBEDDING": {
                "required_params": ["max_sequence_length"],
                "param_constraints": {
                    "max_sequence_length": {"min": 1, "max": 10000}
                }
            },
            "CLASSIFICATION": {
                "required_params": ["max_sequence_length"],
                "param_constraints": {
                    "max_sequence_length": {"min": 1, "max": 10000},
                    "batch_size": {"min": 1, "max": 1024}
                }
            }
        }
        
        constraints = task_constraints.get(task_type.upper(), {})
        param_result = validate_parameters(
            params, 
            constraints.get("required_params"),
            constraints.get("param_constraints")
        )
        
        if not param_result["valid"]:
            errors.extend(param_result["errors"])
    
    return {"valid": len(errors) == 0, "errors": errors}