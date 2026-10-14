# Deployment Guide

Backend analysis runs on a **LangGraph** agent network (`backend/orchestrator/graph/`). Ensure `langgraph` and `langchain-core` are installed via `backend/requirements.txt`.

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 16+
- API Keys for external services (`MAPPLS_API_KEY`, `SERP_API_KEY`, `WEATHER_API_KEY`, `GROQ_API_KEY`)

### 1. Clone and Setup

```bash
git clone https://github.com/iareARiES/LAMDAAnalytics.git
cd LAMDAAnalytics
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup

```bash
cd project2
npm install
```

### 4. Start Services

```bash
# Terminal 1: Backend
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8007

# Terminal 2: Frontend
cd project2
npm run dev
```

## Production Deployment

### Docker Deployment

#### Backend Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Create data directory
RUN mkdir -p data

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007"]
```

#### Frontend Dockerfile

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY project2/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY project2/ .

# Build application
RUN npm run build

# Install serve for production
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Start application
CMD ["serve", "-s", "dist", "-l", "3000"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8007:8007"
    environment:
      - GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}
      - SERP_API_KEY=${SERP_API_KEY}
      - WEATHER_API_KEY=${WEATHER_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./backend/data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8007/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: project2/Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    restart: unless-stopped
```

### Environment Configuration

#### Production Environment Variables

```bash
# API Keys
GOOGLE_MAPS_API_KEY=your_production_key
SERP_API_KEY=your_production_key
WEATHER_API_KEY=your_production_key
GEMINI_API_KEY=your_production_key

# Application Settings
WEATHER_PROVIDER=openweather
HTTP_TIMEOUT_SECONDS=30
AGENT_TIMEOUT_SECONDS=40
SCORING_STATE_PATH=/app/data/scoring_state.json
LOG_LEVEL=INFO
ENABLE_GDELT=false
ENABLE_COMTRADE=false
ALLOW_GEMINI_FALLBACK=true

# Production Settings
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
```

### Nginx Configuration

```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8007;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name yourdomain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # API routes
        location /api/ {
            proxy_pass http://backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Frontend routes
        location / {
            proxy_pass http://frontend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Cloud Deployment

### AWS Deployment

#### EC2 Setup

```bash
# Launch EC2 instance (Ubuntu 22.04)
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose

# Clone repository
git clone https://github.com/iareARiES/LAMDAAnalytics.git
cd LAMDAAnalytics

# Set environment variables
export GOOGLE_MAPS_API_KEY=your_key
export SERP_API_KEY=your_key
export WEATHER_API_KEY=your_key
export GEMINI_API_KEY=your_key

# Deploy with Docker Compose
docker-compose up -d
```

#### ECS Deployment

```yaml
# task-definition.json
{
  "family": "lamda-analytics",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/lamda-backend:latest",
      "portMappings": [
        {
          "containerPort": 8007,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GOOGLE_MAPS_API_KEY",
          "value": "your_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lamda-analytics",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/lamda-backend ./backend
gcloud builds submit --tag gcr.io/PROJECT-ID/lamda-frontend ./project2

# Deploy to Cloud Run
gcloud run deploy lamda-backend \
  --image gcr.io/PROJECT-ID/lamda-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_MAPS_API_KEY=your_key

gcloud run deploy lamda-frontend \
  --image gcr.io/PROJECT-ID/lamda-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name lamda-rg --location eastus

# Deploy backend
az container create \
  --resource-group lamda-rg \
  --name lamda-backend \
  --image your-registry/lamda-backend:latest \
  --dns-name-label lamda-backend \
  --ports 8007 \
  --environment-variables \
    GOOGLE_MAPS_API_KEY=your_key \
    SERP_API_KEY=your_key \
    WEATHER_API_KEY=your_key \
    GEMINI_API_KEY=your_key

# Deploy frontend
az container create \
  --resource-group lamda-rg \
  --name lamda-frontend \
  --image your-registry/lamda-frontend:latest \
  --dns-name-label lamda-frontend \
  --ports 3000
```

## Monitoring and Logging

### Application Monitoring

#### Prometheus Metrics

```python
# Add to backend/main.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_DURATION.observe(time.time() - start_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "services": {
            "database": "healthy",
            "external_apis": "healthy",
            "tgn_model": "healthy"
        }
    }
```

### Logging Configuration

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)
```

## Security

### API Key Management

```python
# Use AWS Secrets Manager or Azure Key Vault
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Load API keys from secrets manager
secrets = get_secret('lamda-api-keys')
GOOGLE_MAPS_API_KEY = secrets['google_maps_api_key']
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
import redis

# Redis caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def cached_geocoding(address: str):
    # Implementation
    pass
```

### Database Optimization

```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0
)
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

# Backup model files
tar -czf model_backup_$(date +%Y%m%d).tar.gz backend/tgn_model.pth

# Backup configuration
cp backend/.env config_backup_$(date +%Y%m%d).env

# Upload to S3
aws s3 cp model_backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
aws s3 cp config_backup_$(date +%Y%m%d).env s3://your-backup-bucket/
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

# Download from S3
aws s3 cp s3://your-backup-bucket/model_backup_latest.tar.gz .
aws s3 cp s3://your-backup-bucket/config_backup_latest.env .

# Restore files
tar -xzf model_backup_latest.tar.gz
cp config_backup_latest.env backend/.env

# Restart services
docker-compose restart
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  backend:
    deploy:
      replicas: 3
    environment:
      - WORKER_ID=${HOSTNAME}
  
  nginx:
    volumes:
      - ./nginx-load-balancer.conf:/etc/nginx/nginx.conf
```

### Load Balancer Configuration

```nginx
upstream backend {
    least_conn;
    server backend_1:8007;
    server backend_2:8007;
    server backend_3:8007;
}
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Check environment variables
   docker exec -it container_name env | grep API_KEY
   ```

2. **Model Loading Issues**
   ```bash
   # Check model file
   ls -la backend/tgn_model.pth
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8007
   ```

### Debug Mode

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug endpoints
@app.get("/debug/agents")
async def debug_agents():
    return {
        "framework": "langgraph",
        "graph": "analysis_graph",
        "nodes": [
            "geocode", "trade", "news", "weather",
            "political", "gscpi", "normalize", "tgn", "report"
        ],
        "status": "healthy"
    }
```

---

This deployment guide provides comprehensive instructions for deploying the LAMDA Analytics system in various environments. Choose the deployment method that best fits your infrastructure requirements.
