# 使用Docker Hub的官方Python镜像
FROM python:3.10-slim

# 1. 系统依赖
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl libxrender1 libxext6 libfontconfig1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 先拷依赖清单，利用缓存
# 构建上下文是ToolUniverse目录，所以可以直接访问这些文件
COPY . .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv \
 && uv pip install --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e . \
 && uv pip install --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple docker

# 4. 环境变量 & 端口
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "src/tooluniverse/bio_models/agent_api.py"]