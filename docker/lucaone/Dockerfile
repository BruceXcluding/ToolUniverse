# LucaOne模型容器
FROM python:3.10-slim

WORKDIR /app

# 1. 换源 & 装系统依赖 一条 RUN 搞定
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's@deb.debian.org@mirrors.aliyun.com@g; s@security.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list; \
    elif [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's@deb.debian.org@mirrors.aliyun.com@g; s@security.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# 2. 拷依赖 → 一次性用国内源安装
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3. 拷代码
COPY app.py .

# 4. 模型挂载点（运行时用 -v 把真实权重挂进来）
ENV MODEL_PATH=/models/lucaone
RUN mkdir -p /models/lucaone

ENV PYTHONUNBUFFERED=1
EXPOSE 8002 8012

# 5. 健康检查（允许通过 PORT 变量改监听口）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8002}/health || exit 1

CMD ["python", "app.py"]