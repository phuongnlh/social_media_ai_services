# Chọn base image nhẹ
FROM python:3.10-slim

# Thư mục làm việc
WORKDIR /app

# Biến môi trường cho config
ARG BUILD_ENVIRONMENT=prod
ENV YOURVIBES_AI_CONFIG_FILE=$BUILD_ENVIRONMENT

# Cài gcc + ca-certificates trong 1 RUN để giảm layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Nâng cấp pip
RUN pip install --upgrade pip

# Cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

# Copy code & config
COPY src/ /app/src/
COPY config/ /app/config/

# Lệnh chạy container
CMD ["python", "src/main.py"]
