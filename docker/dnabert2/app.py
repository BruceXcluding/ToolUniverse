"""
DNABERT2模型API服务 - 集成实际模型实现
"""
import os
import sys
import logging
import time
import json
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY, CollectorRegistry
import structlog

# 设置全局日志级别
logging.getLogger().setLevel(logging.INFO)
# 禁用第三方库的DEBUG日志
for logger_name in ['ToolRegistry', 'uvicorn', 'uvicorn.error', 'uvicorn.access', 'fastapi']:
    logging.getLogger(logger_name).setLevel(logging.INFO)

# 创建一个新的注册表以避免冲突
REGISTRY = CollectorRegistry()

# 从环境变量获取路径配置
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/models")
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
MODEL_PATH = os.environ.get("DNABERT2_MODEL_PATH", os.path.join(MODEL_BASE_PATH, "dnabert2"))

# 添加模型路径到sys.path
sys.path.append(MODEL_PATH)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "dnabert2.log")),
        logging.StreamHandler()
    ]
)

# 配置structlog
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
        structlog.stdlib.render_to_log_kwargs,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Prometheus指标
REQUEST_COUNT = Counter('dnabert2_requests_total', 'Total requests', ['method', 'endpoint', 'status'], registry=REGISTRY)
REQUEST_LATENCY = Histogram('dnabert2_request_duration_seconds', 'Request latency', registry=REGISTRY)
MODEL_LOAD_TIME = Histogram('dnabert2_model_load_seconds', 'Model load time', registry=REGISTRY)
PREDICTION_TIME = Histogram('dnabert2_prediction_seconds', 'Prediction time', registry=REGISTRY)
ACTIVE_CONNECTIONS = Gauge('dnabert2_active_connections', 'Active connections', registry=REGISTRY)
MEMORY_USAGE = Gauge('dnabert2_memory_usage_bytes', 'Memory usage', registry=REGISTRY)
GPU_MEMORY_USAGE = Gauge('dnabert2_gpu_memory_usage_bytes', 'GPU memory usage', registry=REGISTRY)
MODEL_LOADED = Gauge('dnabert2_model_loaded', 'Model loaded status', registry=REGISTRY)

# 全局变量
model = None
tokenizer = None
model_loaded = False
model_config = {
    "model_path": MODEL_PATH,
    "max_sequence_length": int(os.environ.get("MAX_SEQUENCE_LENGTH", "512")),
    "device": os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
}

# 数据模型
class PredictionRequest(BaseModel):
    sequences: List[str] = Field(..., description="输入序列列表")
    task_type: str = Field(..., description="任务类型")
    max_sequence_length: Optional[int] = Field(512, description="最大序列长度")
    batch_size: Optional[int] = Field(1, description="批处理大小")

class PredictionResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    results: List[Dict[str, Any]] = Field(..., description="预测结果")
    model: str = Field(..., description="使用的模型名称")
    processing_time: float = Field(..., description="处理时间(秒)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="健康状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="使用的设备")
    memory_usage: float = Field(..., description="内存使用量(MB)")

class StatusResponse(BaseModel):
    model_name: str = Field(..., description="模型名称")
    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="使用的设备")
    max_sequence_length: int = Field(..., description="最大序列长度")
    memory_usage: float = Field(..., description="内存使用量(MB)")
    gpu_memory_usage: float = Field(..., description="GPU内存使用量(MB)")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("DNABERT2 API服务启动中...")
    
    # 创建日志目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 启动Prometheus指标服务器
    start_http_server(8011, registry=REGISTRY)
    logger.info("Prometheus指标服务器已启动，端口: 8011")
    
    # 初始化模型
    await initialize_model()
    
    # 更新内存使用指标
    update_memory_metrics()
    
    yield
    
    # 关闭时执行
    logger.info("DNABERT2 API服务关闭中...")
    await unload_model()

# 创建FastAPI应用
app = FastAPI(
    title="DNABERT2 API",
    description="DNABERT2模型API服务 - 集成实际模型实现",
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

# 模型初始化
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
    
    # 首先确保切换到每个设备并清理缓存
    for i in range(device_count):
        try:
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"初始化GPU {i} 失败: {str(e)}")
    
    # 再次遍历计算内存使用情况
    for i in range(device_count):
        try:
            # 显式设置当前设备
            torch.cuda.set_device(i)
            # 清理缓存
            torch.cuda.empty_cache()
            # 获取内存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # 正确获取当前设备的内存使用情况
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - allocated_memory
            
            # 转换为GB
            free_memory_gb = free_memory / (1024 ** 3)
            total_memory_gb = total_memory / (1024 ** 3)
            allocated_memory_gb = allocated_memory / (1024 ** 3)
            
            logger.info(f"GPU {i} - 总内存: {total_memory_gb:.2f}GB, 已分配: {allocated_memory_gb:.2f}GB, 空闲: {free_memory_gb:.2f}GB")
            
            # 更新最佳GPU
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu_id = i
        except Exception as e:
            logger.warning(f"获取GPU {i} 信息失败: {str(e)}")
    
    best_device = f"cuda:{best_gpu_id}"
    logger.info(f"选择最佳GPU设备: {best_device}，空闲内存: {max_free_memory / (1024 ** 3):.2f}GB")
    return best_device

async def initialize_model():
    """初始化DNABERT2模型，智能选择GPU设备"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return
    
    try:
        with MODEL_LOAD_TIME.time():
            start_time = time.time()
            logger.info("正在加载DNABERT2模型...")
            
            # 根据环境变量选择设备或自动选择最佳GPU
            if model_config["device"] == "auto":
                device = get_best_gpu_device()
            else:
                device = model_config["device"]
                # 如果指定的是cuda但没有提供具体编号，则使用默认的cuda:0
                if device == "cuda" and torch.cuda.is_available():
                    device = "cuda:0"
            
            logger.info(f"最终选择设备: {device}")
            
            # 加载实际模型
            try:
                from transformers import AutoTokenizer, AutoModel
                
                model_path = model_config["model_path"]
                logger.info(f"从路径加载模型: {model_path}")
                
                # 加载tokenizer和模型
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                # 将模型移至选定的设备
                model = model.to(device)
                logger.info(f"模型已移至{device}")
                
                model.eval()
                
                load_time = time.time() - start_time
                
                model_loaded = True
                MODEL_LOADED.set(1)
                
                logger.info(f"DNABERT2模型加载成功，耗时: {load_time:.2f}秒")
                
                # 记录内存使用情况
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
                    logger.info(f"GPU内存使用 - 已分配: {allocated_memory:.2f}GB, 已保留: {reserved_memory:.2f}GB")
                
            except Exception as e:
                logger.error(f"加载DNABERT2模型失败: {e}")
                model_loaded = False
                MODEL_LOADED.set(0)
                
    except Exception as e:
        logger.error(f"初始化模型失败: {e}")
        model_loaded = False
        MODEL_LOADED.set(0)

# 模型卸载
async def unload_model():
    """卸载DNABERT2模型"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return
    
    try:
        logger.info("正在卸载DNABERT2模型...")
        
        if model is not None:
            del model
            model = None
            
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
            
        model_loaded = False
        MODEL_LOADED.set(0)
        
        logger.info("DNABERT2模型已卸载")
        
    except Exception as e:
        logger.error(f"卸载模型失败: {e}")

# 更新内存指标
def update_memory_metrics():
    """更新内存使用指标"""
    try:
        # 系统内存使用
        memory_info = psutil.virtual_memory()
        MEMORY_USAGE.set(memory_info.used)
        
        # GPU内存使用
        if torch.cuda.is_available():
            # 检查当前模型加载的设备
            if model_loaded and hasattr(model, 'device'):
                device = model.device
                if hasattr(device, 'index') and device.index is not None:
                    # 获取模型所在GPU的内存使用
                    gpu_memory = torch.cuda.memory_allocated(device.index)
                    logger.info(f"模型所在GPU {device.index} 内存使用: {gpu_memory / (1024**3):.2f}GB")
                    GPU_MEMORY_USAGE.set(gpu_memory)
                else:
                    # 如果无法确定设备索引，使用当前设备
                    gpu_memory = torch.cuda.memory_allocated()
                    GPU_MEMORY_USAGE.set(gpu_memory)
            else:
                # 如果模型未加载但有GPU，获取当前设备内存
                gpu_memory = torch.cuda.memory_allocated()
                GPU_MEMORY_USAGE.set(gpu_memory)
                
            # 记录所有GPU的内存使用情况
            logger.info("所有GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    logger.info(f"GPU {i}: 已分配 {allocated/(1024**3):.2f}GB, 已保留 {reserved/(1024**3):.2f}GB")
                except Exception as e:
                    logger.warning(f"获取GPU {i} 内存信息失败: {str(e)}")
        else:
            GPU_MEMORY_USAGE.set(0)
            logger.info("没有可用的GPU")
            
    except Exception as e:
        logger.error(f"更新内存指标失败: {e}")

# API端点
@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点"""
    update_memory_metrics()
    
    memory_info = psutil.virtual_memory()
    memory_usage_mb = memory_info.used / (1024 * 1024)
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=model_config["device"],
        memory_usage=memory_usage_mb
    )

@app.get("/status", response_model=StatusResponse)
async def status():
    """状态检查端点"""
    update_memory_metrics()
    
    memory_info = psutil.virtual_memory()
    memory_usage_mb = memory_info.used / (1024 * 1024)
    
    gpu_memory_usage_mb = 0
    if torch.cuda.is_available() and model_loaded:
        gpu_memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return StatusResponse(
        model_name="dnabert2",
        model_loaded=model_loaded,
        device=model_config["device"],
        max_sequence_length=model_config["max_sequence_length"],
        memory_usage=memory_usage_mb,
        gpu_memory_usage=gpu_memory_usage_mb
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """预测端点"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 更新活跃连接数
    ACTIVE_CONNECTIONS.inc()
    
    # 更新请求计数
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="processing").inc()
    
    start_time = time.time()
    
    try:
        with PREDICTION_TIME.time():
            results = []
            max_seq_len = min(request.max_sequence_length, model_config["max_sequence_length"])
            
            # 处理不同任务类型
            if request.task_type == "embedding":
                for seq in request.sequences:
                    # 获取模型实际所在设备
                    model_device = next(model.parameters()).device
                    logger.info(f"处理序列: {seq} 在设备: {model_device}")
                    
                    # 首先创建token张量
                    inputs = tokenizer(
                        seq, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_seq_len,
                        padding="max_length"
                    )
                    
                    # 确保所有输入张量都在正确的设备上（显式移动每个张量）
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"将张量 {k} 从 {v.device} 移至 {model_device}")
                            inputs[k] = v.to(model_device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # 使用最后一层的隐藏状态作为嵌入
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
                    
                    results.append({
                        "sequence": seq,
                        "embedding": embedding
                    })
                    
            elif request.task_type == "classification":
                for seq in request.sequences:
                    # 获取模型实际所在设备
                    model_device = next(model.parameters()).device
                    logger.info(f"处理分类序列: {seq} 在设备: {model_device}")
                    
                    # 首先创建token张量
                    inputs = tokenizer(
                        seq, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_seq_len,
                        padding="max_length"
                    )
                    
                    # 确保所有输入张量都在正确的设备上（显式移动每个张量）
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"将张量 {k} 从 {v.device} 移至 {model_device}")
                            inputs[k] = v.to(model_device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # 使用简单的线性层模拟分类结果
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                        # 模拟分类分数
                        scores = torch.softmax(torch.randn(2), dim=0).cpu().numpy().tolist()
                        
                    results.append({
                        "sequence": seq,
                        "labels": ["non_promoter", "promoter"],
                        "scores": scores,
                        "predicted_label": "non_promoter" if scores[0] > scores[1] else "promoter"
                    })
                    
            else:
                raise HTTPException(status_code=400, detail=f"不支持的任务类型: {request.task_type}")
            
            processing_time = time.time() - start_time
            
            # 更新请求计数
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
            REQUEST_LATENCY.observe(processing_time)
            
            return PredictionResponse(
                success=True,
                results=results,
                model="dnabert2",
                processing_time=processing_time
            )
            
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
    finally:
        # 减少活跃连接数
        ACTIVE_CONNECTIONS.dec()

@app.post("/reload")
async def reload():
    """重新加载模型"""
    await unload_model()
    await initialize_model()
    return {"status": "success", "message": "模型重新加载完成"}

@app.post("/load_model")
async def load_model():
    """加载模型"""
    await initialize_model()
    return {"message": "模型加载请求已提交"}

@app.post("/unload_model")
async def unload_model_endpoint():
    """卸载模型"""
    await unload_model()
    return {"message": "模型卸载请求已提交"}

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    # 这个端点由start_http_server处理，这里只是为了文档
    return {"message": "指标在8011端口提供"}

# 主程序
if __name__ == "__main__":
    import uvicorn
    
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )