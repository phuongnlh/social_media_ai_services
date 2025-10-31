# 1. Chọn base image nhẹ, có python + slim
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Biến môi trường cho config
ARG BUILD_ENVIRONMENT=prod
ENV YOURVIBES_AI_CONFIG_FILE=$BUILD_ENVIRONMENT
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 4. Cài gcc và các thư viện hệ thống cần thiết trong 1 layer duy nhất
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    ca-certificates \
    git \
    ffmpeg \
    libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy chỉ requirements trước (build cache hiệu quả)
COPY requirements.txt .

# 6. Cài Python dependencies, loại bỏ cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --no-deps


# 7. Copy toàn bộ source code và config
COPY src/ /app/src/
COPY config/ /app/config/

# 8. Set lệnh chạy container
CMD ["python", "src/main.py"]
