# Advanced Configuration

This section covers advanced configuration options for customizing Hayhooks behavior, performance tuning, and deployment optimization.

## Environment Configuration

### Production Settings

```ini
# .env file for production deployment
HAYHOOKS_HOST=0.0.0.0
HAYHOOKS_PORT=1416
HAYHOOKS_MCP_HOST=0.0.0.0
HAYHOOKS_MCP_PORT=1417
HAYHOOKS_PIPELINES_DIR=./pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=./custom_modules
HAYHOOKS_ROOT_PATH=/
HAYHOOKS_DISABLE_SSL=false
HAYHOOKS_USE_HTTPS=false
HAYHOOKS_SHOW_TRACEBACKS=false
LOG=INFO

# Security settings
HAYHOOKS_CORS_ALLOW_ORIGINS=["https://yourdomain.com"]
HAYHOOKS_CORS_ALLOW_METHODS=["*"]
HAYHOOKS_CORS_ALLOW_HEADERS=["*"]
HAYHOOKS_CORS_ALLOW_CREDENTIALS=false
HAYHOOKS_CORS_EXPOSE_HEADERS=[]
HAYHOOKS_CORS_MAX_AGE=600
# Note: to change the number of workers use the CLI flag,
# for example: `hayhooks run --workers 4`
```

### Performance Tuning

- Run multiple workers (CPU or high concurrency):

```bash
hayhooks run --workers 4
```

- Prefer async pipelines and methods for I/O-bound workloads:

```python
from haystack import AsyncPipeline

class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = AsyncPipeline.loads((Path(__file__).parent / "pipeline.yml").read_text())

    async def run_api_async(self, query: str) -> str:
        result = await self.pipeline.run_async({"prompt": {"query": query}})
        return result["llm"]["replies"][0]
```

- Use streaming helpers for chat endpoints to reduce latency:

```python
from hayhooks import async_streaming_generator, get_last_user_message

async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict):
    question = get_last_user_message(messages)
    return async_streaming_generator(
        pipeline=self.pipeline,
        pipeline_run_args={"prompt": {"query": question}},
    )
```

- Scale horizontally behind a load balancer for CPU-bound pipelines or heavy traffic.

## Custom Pipeline Directory Structure

### Organized Pipeline Layout

```
pipelines/
├── production/
│   ├── chat_pipeline/
│   │   ├── pipeline_wrapper.py
│   │   ├── requirements.txt
│   │   └── config.yaml
│   └── rag_pipeline/
│       ├── pipeline_wrapper.py
│       ├── requirements.txt
│       └── config.yaml
├── development/
│   └── experimental/
│       └── new_pipeline.py
└── shared/
    └── common_components/
        └── utils.py
```

### Custom Directory Configuration

```python
# Custom directory configuration
HAYHOOKS_PIPELINES_DIR=./custom_pipelines
HAYHOOKS_ADDITIONAL_PYTHON_PATH=./shared_modules
```

## Custom Routes and Middleware

### Adding Custom Middleware

```python
# custom_middleware.py
from fastapi import Request, Response
from fastapi.middleware import Middleware
from fastapi.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Pre-processing
        start_time = time.time()

        # Custom headers
        request.state.custom_data = "processed"

        # Process request
        response = await call_next(request)

        # Post-processing
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response

# Add to FastAPI app
app.add_middleware(CustomMiddleware)
```

### Custom Endpoints

```python
# custom_routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

custom_router = APIRouter()

@custom_router.get("/custom/health")
async def custom_health():
    return {"status": "healthy", "custom": True}

@custom_router.post("/custom/batch")
async def batch_process(requests: List[Dict[str, Any]]):
    results = []
    for req in requests:
        # Process each request
        result = await process_single_request(req)
        results.append(result)
    return {"results": results}
```

## Security Configuration

### CORS Configuration

```python
# Advanced CORS settings
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trusted-domain.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)
```

### Rate Limiting

```python
# Rate limiting middleware
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/pipeline/run")
@limiter.limit("100/minute")
async def run_pipeline_limited(request: Request):
    return await run_pipeline(request)
```

## Logging and Monitoring

### Advanced Logging Configuration

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging():
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        'hayhooks.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('hayhooks_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('hayhooks_request_duration_seconds', 'Request duration')
ACTIVE_PIPELINES = Gauge('hayhooks_active_pipelines', 'Number of active pipelines')

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    # Record request count
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    # Process request
    response = await call_next(request)

    # Record duration
    REQUEST_DURATION.observe(time.time() - start_time)

    return response
```

## Cache Configuration

### Redis Cache Setup

```python
# cache_config.py
import redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

# Redis configuration
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Initialize cache
@app.on_event("startup")
async def startup():
    FastAPICache.init(RedisBackend(redis_client), prefix="hayhooks-cache")

# Usage in endpoints
@app.get("/pipelines/{pipeline_name}")
@cache(expire=60)
async def get_pipeline(pipeline_name: str):
    return get_pipeline_details(pipeline_name)
```

## Database Configuration

### Pipeline State Management

```python
# database.py
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
DATABASE_URL = "sqlite:///hayhooks.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PipelineState(Base):
    __tablename__ = "pipeline_states"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Initialize database
Base.metadata.create_all(bind=engine)
```

## Custom Authentication

### JWT Authentication

```python
# auth.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
```

## Configuration Validation

### Pydantic Models for Configuration

```python
# config_validation.py
from pydantic import BaseSettings, validator
from typing import List, Optional

class HayhooksConfig(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 1416
    workers: int = 1
    pipelines_dir: str = "./pipelines"
    additional_python_path: Optional[str] = None

    # Security settings
    cors_origins: List[str] = ["*"]
    max_request_size: int = 10 * 1024 * 1024  # 10MB

    # Performance settings
    keepalive_timeout: int = 30
    graceful_shutdown_timeout: int = 10

    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

    @validator('workers')
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError('Workers must be at least 1')
        return v

# Usage
config = HayhooksConfig()
```

## Deployment Templates

### Docker Compose Advanced

```yaml
# docker-compose.advanced.yml
version: '3.8'

services:
  hayhooks:
    image: deepset/hayhooks:latest
    environment:
      - HAYHOOKS_HOST=0.0.0.0
      - HAYHOOKS_PORT=1416
      - HAYHOOKS_WORKERS=4
      - REDIS_URL=redis://redis:6379
    ports:
      - "1416:1416"
    volumes:
      - ./pipelines:/app/pipelines
      - ./custom_modules:/app/custom_modules
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
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
      - hayhooks
    restart: unless-stopped

volumes:
  redis_data:
```

### Kubernetes Advanced

```yaml
# k8s-advanced.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hayhooks-advanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hayhooks
  template:
    metadata:
      labels:
        app: hayhooks
    spec:
      containers:
      - name: hayhooks
        image: deepset/hayhooks:latest
        ports:
        - containerPort: 1416
        env:
        - name: HAYHOOKS_HOST
          value: "0.0.0.0"
        - name: HAYHOOKS_PORT
          value: "1416"
        - name: HAYHOOKS_WORKERS
          value: "4"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 1416
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 1416
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hayhooks-service
spec:
  selector:
    app: hayhooks
  ports:
  - port: 80
    targetPort: 1416
  type: LoadBalancer
```

## Best Practices

### 1. Configuration Management
- Use environment variables for all configuration
- Implement configuration validation
- Separate development and production configurations
- Use configuration management tools for large deployments

### 2. Security
- Implement proper authentication and authorization
- Use HTTPS in production
- Validate all inputs
- Implement rate limiting and request throttling

### 3. Performance
- Use appropriate worker counts for your hardware
- Implement caching for frequently accessed data
- Monitor resource usage and scale accordingly
- Use connection pooling for database connections

### 4. Monitoring
- Implement comprehensive logging
- Set up metrics collection
- Configure health checks
- Set up alerting for critical issues

## Next Steps

- [Deployment Guidelines](../deployment/deployment-guidelines.md) - Production deployment
- [Code Sharing](code-sharing.md) - Reusable components
- [Custom Routes](custom-routes.md) - Custom endpoint development
