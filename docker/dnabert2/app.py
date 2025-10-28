"""
DNABERT2模型服务入口
"""
import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import structlog
from prometheus_client import start_http_server, Counter, Histogram, Gauge, REGISTRY
import psutil

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

# 清理已注册的指标以避免重复注册
for collector in list(REGISTRY._collector_to_names):
    REGISTRY.unregister(collector)

# Prometheus指标
REQUEST_COUNT = Counter('dnabert2_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('dnabert2_request_duration_seconds', 'Request latency')
MODEL_LOAD_TIME = Histogram('dnabert2_model_load_seconds', 'Model load time')
PREDICTION_TIME = Histogram('dnabert2_prediction_seconds', 'Prediction time')
ACTIVE_CONNECTIONS = Gauge('dnabert2_active_connections', 'Active connections')
MEMORY_USAGE = Gauge('dnabert2_memory_usage_bytes', 'Memory usage in bytes')
GPU_MEMORY_USAGE = Gauge('dnabert2_gpu_memory_usage_bytes', 'GPU memory usage in bytes')

# 全局变量
model = None
tokenizer = None
model_loaded = False


class PredictionRequest(BaseModel):
    """预测请求模型"""
    sequences: List[str] = Field(..., description="DNA序列列表")
    task_type: str = Field(default="promoter_prediction", description="任务类型")
    truncation_seq_length: int = Field(default=1024, description="截断序列长度")
    batch_size: int = Field(default=1, description="批处理大小")


class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool = Field(..., description="是否成功")
    results: List[Dict[str, Any]] = Field(default=[], description="预测结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("DNABERT2服务启动中...")
    
    # 启动Prometheus指标服务器
    start_http_server(8011)
    logger.info("Prometheus指标服务器已启动", port=8011)
    
    # 初始化模型
    await initialize_model()
    
    # 更新内存使用指标
    update_memory_metrics()
    
    yield
    
    # 关闭时执行
    logger.info("DNABERT2服务关闭中...")
    await unload_model()


# 创建FastAPI应用
app = FastAPI(
    title="DNABERT2模型服务",
    description="DNABERT2模型的REST API服务",
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


async def initialize_model():
    """初始化模型"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return
    
    try:
        with MODEL_LOAD_TIME.time():
            logger.info("正在加载DNABERT2模型...")
            
            # 这里应该是实际的模型加载代码
            # 由于这是一个示例，我们只是模拟加载过程
            # 在实际实现中，这里会加载真实的DNABERT2模型
            
            # 模拟加载过程
            await asyncio.sleep(2)  # 模拟加载时间
            
            # 模拟模型和tokenizer
            # model = load_dnabert2_model(...)
            # tokenizer = load_dnabert2_tokenizer(...)
            
            model_loaded = True
            logger.info("DNABERT2模型加载成功")
    except Exception as e:
        logger.error("DNABERT2模型加载失败", error=str(e), exc_info=True)
        raise


async def unload_model():
    """卸载模型"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return
    
    try:
        logger.info("正在卸载DNABERT2模型...")
        
        # 释放模型资源
        # if model is not None:
        #     del model
        #     model = None
        # if tokenizer is not None:
        #     del tokenizer
        #     tokenizer = None
        
        model_loaded = False
        logger.info("DNABERT2模型卸载成功")
    except Exception as e:
        logger.error("DNABERT2模型卸载失败", error=str(e), exc_info=True)


def update_memory_metrics():
    """更新内存使用指标"""
    # 系统内存
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.memory_allocated(i)
            GPU_MEMORY_USAGE.set(gpu_memory)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
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
async def model_status():
    """获取模型状态"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = psutil.virtual_memory()
    
    return ModelStatusResponse(
        model_name="dnabert2",
        loaded=model_loaded,
        device=device,
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """执行预测"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 更新活跃连接数
    ACTIVE_CONNECTIONS.inc()
    
    # 更新请求计数
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="processing").inc()
    
    try:
        with PREDICTION_TIME.time():
            # 这里应该是实际的预测代码
            # 由于这是一个示例，我们只是模拟预测过程
            
            results = []
            
            for dna_seq in request.sequences:
                # 模拟启动子活性预测
                import numpy as np
                promoter_activity = np.random.rand().astype(np.float32)  # 0-1之间的值
                confidence = np.random.rand().astype(np.float32)  # 0-1之间的置信度
                
                # 计算GC含量
                gc_count = dna_seq.count('G') + dna_seq.count('C')
                gc_content = gc_count / len(dna_seq) * 100 if len(dna_seq) > 0 else 0
                
                # 检测常见的启动子基序
                tata_box = "TATAAA" in dna_seq
                caat_box = "CCAAT" in dna_seq
                gc_box = "GGGCGG" in dna_seq
                
                result = {
                    "dna_sequence": dna_seq[:50] + "..." if len(dna_seq) > 50 else dna_seq,
                    "promoter_activity": float(promoter_activity),
                    "confidence": float(confidence),
                    "gc_content": float(gc_content),
                    "tata_box": tata_box,
                    "caat_box": caat_box,
                    "gc_box": gc_box,
                    "prediction": "strong" if promoter_activity > 0.7 else "moderate" if promoter_activity > 0.3 else "weak"
                }
                
                results.append(result)
            
            # 更新请求计数
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
            
            # 在后台更新内存指标
            background_tasks.add_task(update_memory_metrics)
            
            return PredictionResponse(
                success=True,
                results=results,
                processing_time=PREDICTION_TIME.observe()
            )
    
    except Exception as e:
        logger.error("预测失败", error=str(e), exc_info=True)
        
        # 更新请求计数
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        
        return PredictionResponse(
            success=False,
            error=str(e)
        )
    finally:
        # 减少活跃连接数
        ACTIVE_CONNECTIONS.dec()


@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """重新加载模型"""
    try:
        await unload_model()
        await initialize_model()
        
        # 在后台更新内存指标
        background_tasks.add_task(update_memory_metrics)
        
        return {"status": "success", "message": "模型重新加载成功"}
    except Exception as e:
        logger.error("模型重新加载失败", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型重新加载失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )