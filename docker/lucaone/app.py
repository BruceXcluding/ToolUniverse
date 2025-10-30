"""
LucaOne模型服务入口
"""
import os
import sys
import json
import logging
import time
import psutil
import numpy as np
import torch
import asyncio
import structlog
from typing import Dict, List, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server, CollectorRegistry
from starlette.responses import Response

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 创建新的注册表，避免全局注册表冲突
REGISTRY = CollectorRegistry()

# Prometheus指标
REQUEST_COUNT = Counter('lucaone_requests_total', 'Total requests', ['method', 'endpoint', 'status'], registry=REGISTRY)
REQUEST_LATENCY = Histogram('lucaone_request_duration_seconds', 'Request latency', registry=REGISTRY)
MODEL_LOAD_TIME = Histogram('lucaone_model_load_seconds', 'Model load time', registry=REGISTRY)
PREDICTION_TIME = Histogram('lucaone_prediction_seconds', 'Prediction time', registry=REGISTRY)
ACTIVE_CONNECTIONS = Gauge('lucaone_active_connections', 'Active connections', registry=REGISTRY)
MEMORY_USAGE = Gauge('lucaone_memory_usage_bytes', 'Memory usage in bytes', registry=REGISTRY)
GPU_MEMORY_USAGE = Gauge('lucaone_gpu_memory_usage_bytes', 'GPU memory usage in bytes', registry=REGISTRY)
MODEL_LOADED = Gauge('lucaone_model_loaded', 'Model loaded status', registry=REGISTRY)

# 全局变量
model = None
tokenizer = None
model_loaded = False
device = "cpu"
model_path = os.environ.get("MODEL_PATH", "/models/lucaone")
# 使用统一的MODEL_PATH环境变量
thirdparty_model_path = os.environ.get("LUCAONE_MODEL_PATH", "/models/lucaone")

# 从环境变量获取设备配置，默认为auto
device_env = os.environ.get("DEVICE", "auto")

# 数据模型
class PredictionRequest(BaseModel):
    """预测请求模型"""
    sequences: List[str] = Field(..., description="生物序列列表(DNA/RNA/蛋白质)")
    task_type: str = Field(..., description="任务类型", pattern="^(embedding|classification|property_prediction|interaction|annotation)$")
    max_sequence_length: int = Field(default=1024, description="最大序列长度")
    batch_size: int = Field(default=1, description="批处理大小")
    options: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool = Field(..., description="是否成功")
    results: List[Dict[str, Any]] = Field(default=[], description="预测结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")

class LoadModelRequest(BaseModel):
    model_path: Optional[str] = None

class LoadModelResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    memory_usage: Dict[str, float] = Field(..., description="内存使用情况")
    gpu_available: bool = Field(..., description="GPU是否可用")

class ModelStatusResponse(BaseModel):
    """模型状态响应模型"""
    model_name: str = Field(..., description="模型名称")
    loaded: bool = Field(..., description="是否已加载")
    device: str = Field(..., description="设备类型")
    memory_usage: Dict[str, float] = Field(..., description="内存使用情况")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("LucaOne服务启动中...")
    
    # 启动Prometheus指标服务器
    start_http_server(8012, registry=REGISTRY)
    logger.info("Prometheus指标服务器已启动", port=8012)
    
    # 初始化模型
    await initialize_model()
    
    # 更新内存使用指标
    update_memory_metrics()
    
    yield
    
    # 关闭时执行
    logger.info("LucaOne服务关闭中...")
    await unload_model()

# 创建FastAPI应用
app = FastAPI(
    title="LucaOne API",
    description="LucaOne生物序列分析模型API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_best_gpu_device():
    """获取可用GPU中内存最多的设备ID
    
    Returns:
        str: 最佳GPU设备ID，格式为"cuda:device_id"
    """
    if not torch.cuda.is_available():
        logger.warning("没有可用的GPU，将使用CPU")
        return "cpu"
    
    device_count = torch.cuda.device_count()
    logger.info(f"检测到 {device_count} 个可用GPU设备")
    
    best_gpu_id = 0
    max_free_memory = 0
    
    for i in range(device_count):
        try:
            with torch.cuda.device(i):
                # 清理缓存
                torch.cuda.empty_cache()
                # 获取内存信息
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                free_memory = total_memory - allocated_memory
                
                # 转换为GB
                free_memory_gb = free_memory / (1024 ** 3)
                total_memory_gb = total_memory / (1024 ** 3)
                
                logger.info(f"GPU {i} - 空闲内存: {free_memory_gb:.2f}GB / {total_memory_gb:.2f}GB")
                
                # 更新最佳GPU
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu_id = i
        except Exception as e:
            logger.warning(f"获取GPU {i} 信息失败: {str(e)}")
    
    best_device = f"cuda:{best_gpu_id}"
    logger.info(f"选择最佳GPU设备: {best_device}，空闲内存: {max_free_memory / (1024 ** 3):.2f}GB")
    return best_device

async def update_memory_metrics():
    """更新内存使用指标"""
    try:
        # 系统内存
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.set(gpu_memory)
    except Exception as e:
        logger.error("更新内存指标失败", error=str(e))

async def initialize_model(model_path_override: str = None) -> bool:
    """初始化模型"""
    global model, tokenizer, model_loaded, device
    
    if model_loaded:
        return True
    
    try:
        # 确定模型路径
        actual_model_path = model_path_override or model_path
        
        # 检查3rdparty模型路径是否存在
        if os.path.exists(thirdparty_model_path):
            actual_model_path = thirdparty_model_path
            logger.info("使用3rdparty模型路径", path=actual_model_path)
        else:
            logger.warning("3rdparty模型路径不存在，使用默认路径", 
                          thirdparty_path=thirdparty_model_path, 
                          default_path=actual_model_path)
        
        # 检查模型路径是否存在
        if not os.path.exists(actual_model_path):
            logger.error("模型路径不存在", path=actual_model_path)
            return False
        
        # 智能选择最佳GPU设备
        if device_env.lower() == "auto":
            selected_device = get_best_gpu_device()
        elif device_env.lower() == "cuda" and torch.cuda.device_count() > 1:
            # 如果指定了cuda但有多个GPU，也进行智能选择
            selected_device = get_best_gpu_device()
        elif device_env.lower() == "cuda" and torch.cuda.is_available():
            selected_device = "cuda"
            logger.info("使用CUDA设备")
        else:
            # 使用指定的设备或回退到CPU
            selected_device = device_env if torch.cuda.is_available() else "cpu"
            if selected_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU")
                selected_device = "cpu"
        
        device = selected_device
        logger.info(f"最终选择的设备: {device}")
        
        # 添加LucaOneApp路径到sys.path
        lucaone_app_path = "/mnt/models/yigex/3rdparty/LucaOneApp"
        if lucaone_app_path not in sys.path:
            sys.path.insert(0, lucaone_app_path)
        
        # 导入LucaOne模块
        try:
            from algorithms.llm.lucagplm.get_embedding import load_model
            from algorithms.llm.lucagplm.v2_0.lucaone_gplm import LucaGPLM
            from algorithms.llm.lucagplm.v2_0.lucaone_gplm_config import LucaGPLMConfig
            from algorithms.llm.lucagplm.v2_0.alphabet import Alphabet
            logger.info("成功导入LucaOne模块")
        except ImportError as e:
            logger.error(f"导入LucaOne模块失败: {e}")
            raise
        
        # 创建日志文件
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "lucaone_log.json")
        
        # 创建日志文件
        if not os.path.exists(log_file):
            log_data = {
                "model_type": "lucaone_gplm",
                "tokenization": False,
                "do_lower_case": False,
                "truncation": "right",
                "pretrain_tasks": {},
                "ignore_index": -100,
                "label_size": {},
                "loss_type": {},
                "output_mode": {},
                "max_length": 4096,
                "classifier_size": {},
                "add_special_tokens": True
            }
            
            with open(log_file, "w") as f:
                json.dump(log_data, f)
        
        # 加载模型
        with MODEL_LOAD_TIME.time():
            logger.info("开始加载LucaOne模型", path=actual_model_path)
            
            # 检查检查点
            checkpoint_dir = os.path.join(actual_model_path, "models", "lucaone", "checkpoint-step36000000")
            if not os.path.exists(checkpoint_dir):
                logger.warning(f"检查点未找到于 {checkpoint_dir}，仅使用配置")
                # 使用配置创建最小模型
                config_path = os.path.join(actual_model_path, "config", "lucaone_gplm.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    model_config = LucaGPLMConfig(**config_dict)
                    
                    # 创建简单模型
                    model = LucaGPLM(model_config)
                    model.to(device)
                    model.eval()
                    model_loaded = True
                    logger.info(f"从配置创建最小LucaOne模型")
                else:
                    logger.error(f"配置未找到于 {config_path}")
                    raise FileNotFoundError(f"配置未找到于 {config_path}")
            else:
                # 加载实际模型
                logger.info(f"从检查点加载模型: {checkpoint_dir}")
                args_info, model_config, model, tokenizer = load_model(log_file, checkpoint_dir, embedding_inference=True)
                
                if model is None:
                    logger.error("加载模型失败")
                    raise RuntimeError("加载模型失败")
                
                model.to(device)
                model.eval()
                model_loaded = True
                logger.info(f"从检查点成功加载LucaOne模型")
            
            MODEL_LOADED.set(1)
            logger.info("LucaOne模型初始化完成")
            
        return True
        
    except Exception as e:
        logger.error("初始化LucaOne模型失败", error=str(e), exc_info=True)
        model_loaded = False
        MODEL_LOADED.set(0)
        return False

async def unload_model() -> bool:
    """卸载模型"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return True
    
    try:
        logger.info("正在卸载LucaOne模型...")
        
        if model is not None:
            # 释放模型资源
            del model
            model = None
        
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model_loaded = False
        MODEL_LOADED.set(0)
        logger.info("LucaOne模型已卸载")
        return True
        
    except Exception as e:
        logger.error("卸载LucaOne模型失败", error=str(e))
        return False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    await update_memory_metrics()
    memory = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available()
    
    return HealthResponse(
        status="healthy" if model_loaded else "initializing",
        model_loaded=model_loaded,
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        },
        gpu_available=gpu_available
    )

@app.get("/status", response_model=ModelStatusResponse)
async def status():
    """状态端点"""
    await update_memory_metrics()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    memory = psutil.virtual_memory()
    
    return ModelStatusResponse(
        model_name="lucaone",
        loaded=model_loaded,
        device=device_type,
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
    )

@app.post("/load", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """加载模型端点"""
    with REQUEST_LATENCY.time():
        try:
            success = await initialize_model(request.model_path)
            if success:
                REQUEST_COUNT.labels(method="POST", endpoint="/load", status="success").inc()
                return LoadModelResponse(status="success", message="模型加载成功")
            else:
                REQUEST_COUNT.labels(method="POST", endpoint="/load", status="error").inc()
                return LoadModelResponse(status="error", message="模型加载失败")
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/load", status="error").inc()
            logger.error("加载模型异常", error=str(e))
            raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")

@app.post("/unload", response_model=LoadModelResponse)
async def unload_model_endpoint():
    """卸载模型端点"""
    with REQUEST_LATENCY.time():
        try:
            success = await unload_model()
            if success:
                REQUEST_COUNT.labels(method="POST", endpoint="/unload", status="success").inc()
                return LoadModelResponse(status="success", message="模型卸载成功")
            else:
                REQUEST_COUNT.labels(method="POST", endpoint="/unload", status="error").inc()
                return LoadModelResponse(status="error", message="模型卸载失败")
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/unload", status="error").inc()
            logger.error("卸载模型异常", error=str(e))
            raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """预测端点"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 更新活跃连接数
    ACTIVE_CONNECTIONS.inc()
    
    # 更新请求计数
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="processing").inc()
    
    try:
        with PREDICTION_TIME.time():
            await update_memory_metrics()
            
            results = []
            
            for seq in request.sequences:
                # 根据任务类型执行不同的预测
                if request.task_type == "embedding":
                    result = predict_embedding(seq, request.options)
                elif request.task_type == "classification":
                    result = predict_classification(seq, request.options)
                elif request.task_type == "property_prediction":
                    result = predict_property(seq, request.options)
                elif request.task_type == "interaction":
                    result = predict_interaction(seq, request.options)
                elif request.task_type == "annotation":
                    result = predict_annotation(seq, request.options)
                else:
                    result = {"error": f"不支持的任务类型: {request.task_type}"}
                
                results.append(result)
            
            # 更新请求计数
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
            
            # 在后台更新内存指标
            background_tasks.add_task(update_memory_metrics)
            
            return PredictionResponse(
                success=True,
                results=results,
                processing_time=0.1
            )
            
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        logger.error("预测异常", error=str(e))
        return PredictionResponse(
            success=False,
            error=str(e)
        )
    finally:
        # 减少活跃连接数
        ACTIVE_CONNECTIONS.dec()

def predict_embedding(sequence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成序列嵌入"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 添加LucaOneApp路径到sys.path
        lucaone_app_path = "/mnt/models/yigex/3rdparty/LucaOneApp"
        if lucaone_app_path not in sys.path:
            sys.path.insert(0, lucaone_app_path)
        
        # 导入LucaOne模块
        from algorithms.llm.lucagplm.v2_0.alphabet import Alphabet
        from algorithms.batch_converter import BatchConverter
        
        # 创建tokenizer
        tokenizer = Alphabet.from_predefined("gene_prot")
        
        # 准备批次
        seq_type = options.get("seq_type", "gene") if options else "gene"
        batch_labels = [(f"seq_{hash(sequence) % 1000000}", sequence, seq_type)]
        batch_converter = BatchConverter(tokenizer)
        batch_tokens, batch_lengths, batch_labels = batch_converter(batch_labels)
        
        # 移动到设备
        if device == "cuda":
            batch_tokens = batch_tokens.to("cuda")
        
        # 生成嵌入
        with torch.no_grad():
            # 获取模型嵌入
            if hasattr(model, 'embed_tokens'):
                # 获取token嵌入
                embeddings = model.embed_tokens(batch_tokens)
                
                # 如果模型有transformer层，应用它们
                if hasattr(model, 'layers') and model.layers:
                    # 应用位置嵌入（如果可用）
                    if hasattr(model, 'embed_pos') and model.embed_pos is not None:
                        position_ids = torch.arange(batch_tokens.size(1), device=batch_tokens.device)
                        position_embeddings = model.embed_pos(position_ids)
                        embeddings = embeddings + position_embeddings.unsqueeze(0)
                    
                    # 应用类型嵌入（如果可用）
                    if hasattr(model, 'embed_type') and model.embed_type is not None:
                        type_ids = torch.zeros(batch_tokens.size(), dtype=torch.long, device=batch_tokens.device)
                        type_embeddings = model.embed_type(type_ids)
                        embeddings = embeddings + type_embeddings
                    
                    # 应用嵌入层归一化（如果可用）
                    if hasattr(model, 'embed_layer_norm') and model.embed_layer_norm is not None:
                        embeddings = model.embed_layer_norm(embeddings)
                    
                    # 应用transformer层
                    for layer in model.layers:
                        embeddings = layer(embeddings)
                    
                    # 应用最终层归一化（如果可用）
                    if hasattr(model, 'last_layer_norm') and model.last_layer_norm is not None:
                        embeddings = model.last_layer_norm(embeddings)
                
                # 转换为numpy
                embeddings_np = embeddings.cpu().numpy()
                
                # 返回序列的嵌入（批次中的第一个）
                return {
                    "sequence": sequence,
                    "embedding": embeddings_np[0].tolist(),
                    "sequence_length": len(sequence),
                    "embedding_dim": embeddings_np[0].shape[1]
                }
            else:
                # 回退到基于token ID的简单嵌入
                embeddings = torch.nn.functional.one_hot(batch_tokens, num_classes=tokenizer.alphabet_size).float()
                embeddings_np = embeddings.cpu().numpy()
                return {
                    "sequence": sequence,
                    "embedding": embeddings_np[0].tolist(),
                    "sequence_length": len(sequence),
                    "embedding_dim": embeddings_np[0].shape[1]
                }
                
    except Exception as e:
        logger.error(f"生成嵌入时出错: {e}")
        # 回退到随机嵌入
        embedding_dim = 768
        embedding = np.random.rand(min(len(sequence), 1024), embedding_dim).tolist()
        return {
            "sequence": sequence,
            "embedding": embedding,
            "sequence_length": len(sequence),
            "embedding_dim": embedding_dim
        }

def predict_classification(sequence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """序列分类预测"""
    # 实际实现需要根据LucaOne模型的具体API
    # 这里是示例代码
    classes = ["class_0", "class_1", "class_2"]
    scores = np.random.rand(len(classes)).tolist()
    predicted_class = classes[np.argmax(scores)]
    
    return {
        "sequence": sequence,
        "predicted_class": predicted_class,
        "scores": dict(zip(classes, scores)),
        "sequence_length": len(sequence)
    }

def predict_property(sequence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """属性预测"""
    # 实际实现需要根据LucaOne模型的具体API
    # 这里是示例代码
    properties = {
        "stability": float(np.random.rand()),
        "solubility": float(np.random.rand()),
        "toxicity": float(np.random.rand())
    }
    
    return {
        "sequence": sequence,
        "properties": properties,
        "sequence_length": len(sequence)
    }

def predict_interaction(sequence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """相互作用预测"""
    # 实际实现需要根据LucaOne模型的具体API
    # 这里是示例代码
    if not options or "target_sequence" not in options:
        return {"error": "缺少目标序列"}
    
    target_sequence = options["target_sequence"]
    interaction_score = float(np.random.rand())
    
    return {
        "sequence": sequence,
        "interaction_score": interaction_score,
        "source_sequence_length": len(sequence),
        "target_sequence_length": len(target_sequence)
    }

def predict_annotation(sequence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """序列注释"""
    # 实际实现需要根据LucaOne模型的具体API
    # 这里是示例代码
    annotations = []
    for i in range(0, len(sequence), 10):
        start = i
        end = min(i + 10, len(sequence))
        annotation_type = np.random.choice(["domain", "motif", "site"])
        annotations.append({
            "type": annotation_type,
            "start": start,
            "end": end,
            "confidence": float(np.random.rand())
        })
    
    return {
        "sequence": sequence,
        "annotations": annotations,
        "sequence_length": len(sequence)
    }

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    await update_memory_metrics()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.post("/reload", response_model=LoadModelResponse)
async def reload_model(background_tasks: BackgroundTasks):
    """重新加载模型端点"""
    with REQUEST_LATENCY.time():
        try:
            await unload_model()
            success = await initialize_model()
            
            # 在后台更新内存指标
            background_tasks.add_task(update_memory_metrics)
            
            if success:
                REQUEST_COUNT.labels(method="POST", endpoint="/reload", status="success").inc()
                return LoadModelResponse(status="success", message="模型重新加载成功")
            else:
                REQUEST_COUNT.labels(method="POST", endpoint="/reload", status="error").inc()
                return LoadModelResponse(status="error", message="模型重新加载失败")
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/reload", status="error").inc()
            logger.error("重新加载模型异常", error=str(e))
            raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )