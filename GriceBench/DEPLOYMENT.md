# GriceBench Deployment Guide

## Overview

This guide covers deploying GriceBench models for production use.

## Deployment Options

### 1. Docker Deployment (Recommended)

**Build Image:**
```bash
docker build -t gricebench:latest .
```

**Run Container:**
```bash
docker run --gpus all -p 8000:8000 \\
  -v $(pwd)/models:/app/models \\
  gricebench:latest
```

**With Docker Compose:**
```bash
docker-compose up -d
```

### 2. Kubernetes Deployment

**Create deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gricebench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gricebench
  template:
    metadata:
      labels:
        app: gricebench
    spec:
      containers:
      - name: gricebench
        image: gricebench:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 3. Cloud Deployment

#### AWS SageMaker

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://bucket/gricebench/model.tar.gz',
    role=role,
    framework_version='2.1.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)
```

#### Google Cloud Run

```bash
gcloud run deploy gricebench \\
  --image gcr.io/project/gricebench \\
  --platform managed \\
  --region us-central1 \\
  --memory 16Gi \\
  --cpu 4
```

## Monitoring

### Metrics to Track

- **Latency**: p50, p95, p99
- **Throughput**: Requests per second
- **GPU Utilization**: % usage
- **Error Rate**: 4xx, 5xx responses

### Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gricebench'
    static_configs:
      - targets: ['localhost:8000']
```

## Scaling

### Horizontal Scaling

```bash
# Kubernetes
kubectl scale deployment gricebench --replicas=5

# Docker Compose
docker-compose up --scale api=3
```

### Vertical Scaling

- **CPU**: 4-8 cores recommended
- **RAM**: 16-32GB
- **GPU**: T4 (16GB) or better

## Security

1. **API Authentication**: Add JWT tokens
2. **Rate Limiting**: 100 req/min per user
3. **Input Validation**: Sanitize all inputs
4. **HTTPS**: Use TLS 1.3

---

**Last Updated:** 2026-01-23
