# DNA-Bert2 服务
FROM python:3.10-slim

WORKDIR /app

# 1. 换源 & 装系统依赖一条完成
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's@deb.debian.org@mirrors.aliyun.com@g; s@security.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list; \
    elif [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's@deb.debian.org@mirrors.aliyun.com@g; s@security.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# 2. 拷依赖 → 国内源安装
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3. 拷代码
COPY app.py .

# 4. 模型挂载点（运行时使用 -v /host/dnabert2:/models/dnabert2）
ENV MODEL_PATH=/models/dnabert2
RUN mkdir -p /models/dnabert2

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8001 8011

# 5. 健康检查（支持 PORT 变量）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

CMD ["python", "app.py"]