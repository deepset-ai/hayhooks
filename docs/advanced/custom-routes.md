# Custom Routes

This section covers how to add custom routes and endpoints to extend Hayhooks functionality beyond the standard pipeline operations.

## Overview

Custom routes allow you to:

- Add specialized endpoints for your application
- Implement custom authentication and authorization
- Create admin interfaces
- Add health checks and monitoring endpoints
- Integrate with external systems

## Adding Custom Routes

### Basic Custom Route

```python
# custom_routes.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import asyncio
import logging

# Create a custom router
custom_router = APIRouter(prefix="/custom", tags=["custom"])

@custom_router.get("/health")
async def custom_health_check():
    """Custom health check endpoint"""
    return {
        "status": "healthy",
        "service": "hayhooks-custom",
        "version": "1.0.0"
    }

@custom_router.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return {
        "active_pipelines": 5,
        "total_requests": 1000,
        "uptime": "2d 5h 30m"
    }

@custom_router.post("/batch-process")
async def batch_process(requests: List[Dict[str, Any]]):
    """Process multiple requests in batch"""
    results = []
    for req in requests:
        try:
            # Process each request
            result = await process_single_request(req)
            results.append({"success": True, "result": result})
        except Exception as e:
            results.append({"success": False, "error": str(e)})

    return {"results": results}
```

### Integrating with Hayhooks

```python
# main.py (extension to Hayhooks main application)
from fastapi import FastAPI
from hayhooks import create_app
from custom_routes import custom_router

# Create the base Hayhooks app
app = create_app()

# Include custom routes
app.include_router(custom_router)

# You can also add middleware
@app.middleware("http")
async def custom_middleware(request: Request, call_next):
    # Custom processing before request
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Custom processing after request
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response
```

## Authentication and Authorization

### JWT Authentication

```python
# auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

security = HTTPBearer()

# JWT Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

@auth_router.post("/login")
async def login(username: str, password: str):
    """Login endpoint"""
    # In a real application, you would validate against a database
    if username == "admin" and password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@auth_router.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    """Protected route requiring authentication"""
    return {"message": f"Hello {current_user}, this is a protected route!"}
```

### API Key Authentication

```python
# api_key_auth.py
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional

api_key_router = APIRouter(prefix="/api-key", tags=["api-key"])

# In a real application, you would store this securely
API_KEYS = {
    "key1": {"user": "user1", "permissions": ["read", "write"]},
    "key2": {"user": "user2", "permissions": ["read"]}
}

async def get_api_user(api_key: str = Header(..., alias="X-API-Key")):
    """Get user from API key"""
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return API_KEYS[api_key]

@api_key_router.get("/data")
async def get_data(user: dict = Depends(get_api_user)):
    """Get data with API key authentication"""
    return {"message": f"Hello {user['user']}, here's your data"}

@api_key_router.post("/data")
async def create_data(data: dict, user: dict = Depends(get_api_user)):
    """Create data with API key authentication"""
    if "write" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    return {"message": "Data created successfully", "data": data}
```

## Admin Interface

### Admin Routes

```python
# admin_routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import asyncio
import logging

admin_router = APIRouter(prefix="/admin", tags=["admin"])

@admin_router.get("/pipelines")
async def list_all_pipelines():
    """List all deployed pipelines"""
    # In a real application, this would query your pipeline store
    return {
        "pipelines": [
            {"name": "chat_pipeline", "status": "active", "version": "1.0.0"},
            {"name": "rag_pipeline", "status": "active", "version": "2.1.0"},
            {"name": "translation_pipeline", "status": "inactive", "version": "1.5.0"}
        ]
    }

@admin_router.post("/pipelines/{pipeline_name}/restart")
async def restart_pipeline(pipeline_name: str, background_tasks: BackgroundTasks):
    """Restart a specific pipeline"""
    background_tasks.add_task(restart_pipeline_task, pipeline_name)
    return {"message": f"Pipeline {pipeline_name} restart initiated"}

@admin_router.get("/logs")
async def get_logs(lines: int = 100):
    """Get application logs"""
    # In a real application, this would read from log files
    return {
        "logs": [
            {"timestamp": "2024-01-01T10:00:00", "level": "INFO", "message": "Pipeline started"},
            {"timestamp": "2024-01-01T10:01:00", "level": "ERROR", "message": "Pipeline failed"},
            {"timestamp": "2024-01-01T10:02:00", "level": "INFO", "message": "Pipeline restarted"}
        ]
    }

@admin_router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "cpu_usage": "45%",
        "memory_usage": "60%",
        "disk_usage": "75%",
        "active_connections": 25,
        "total_requests": 1000
    }

async def restart_pipeline_task(pipeline_name: str):
    """Background task to restart pipeline"""
    try:
        # Simulate pipeline restart
        await asyncio.sleep(5)
        logging.info(f"Pipeline {pipeline_name} restarted successfully")
    except Exception as e:
        logging.error(f"Failed to restart pipeline {pipeline_name}: {str(e)}")
```

## Monitoring and Health Checks

### Health Check Routes

```python
# health_routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import psutil
import time

health_router = APIRouter(prefix="/health", tags=["health"])

@health_router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": time.time()}

@health_router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available": memory.available,
                "disk_free": disk.free
            },
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "pipelines": "healthy"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@health_router.get("/ready")
async def readiness_check():
    """Readiness check - is the service ready to accept requests?"""
    # Check if all required services are ready
    # In a real application, you would check database connections, etc.
    return {"status": "ready", "timestamp": time.time()}

@health_router.get("/live")
async def liveness_check():
    """Liveness check - is the service running?"""
    return {"status": "alive", "timestamp": time.time()}
```

## External Integration Routes

### Webhook Handlers

```python
# webhook_routes.py
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import json
import logging

webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@webhook_router.post("/github")
async def github_webhook(request: Request):
    """Handle GitHub webhooks"""
    try:
        # Get the webhook payload
        payload = await request.json()

        # Verify GitHub signature if needed
        # signature = request.headers.get("X-Hub-Signature")

        # Handle different webhook events
        event_type = request.headers.get("X-GitHub-Event")

        if event_type == "push":
            await handle_push_event(payload)
        elif event_type == "pull_request":
            await handle_pull_request_event(payload)

        return {"status": "success"}

    except Exception as e:
        logging.error(f"Error handling GitHub webhook: {str(e)}")
        raise HTTPException(status_code=400, detail="Webhook processing failed")

async def handle_push_event(payload: Dict[str, Any]):
    """Handle push event from GitHub"""
    repo_name = payload.get("repository", {}).get("full_name")
    commit_id = payload.get("after")

    logging.info(f"Received push event for {repo_name}, commit {commit_id}")

    # Trigger pipeline deployment or other actions
    # This is where you would integrate with your CI/CD pipeline

async def handle_pull_request_event(payload: Dict[str, Any]):
    """Handle pull request event from GitHub"""
    pr_number = payload.get("pull_request", {}).get("number")
    action = payload.get("action")

    logging.info(f"Received pull request {action} event for PR #{pr_number}")

    # Handle pull request actions (opened, closed, etc.)
```

### Slack Integration

```python
# slack_routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import requests

slack_router = APIRouter(prefix="/slack", tags=["slack"])

@slack_router.post("/commands")
async def slack_command(command: str, text: str, response_url: str):
    """Handle Slack slash commands"""
    if command == "/pipeline":
        # Handle pipeline-related commands
        return await handle_pipeline_command(text, response_url)
    elif command == "/status":
        # Handle status commands
        return await handle_status_command(text, response_url)

    return {"text": "Unknown command"}

async def handle_pipeline_command(text: str, response_url: str) -> Dict[str, Any]:
    """Handle pipeline commands"""
    parts = text.split()
    action = parts[0] if parts else "list"

    if action == "list":
        pipelines = ["chat_pipeline", "rag_pipeline", "translation_pipeline"]
        return {"text": f"Available pipelines: {', '.join(pipelines)}"}
    elif action == "run":
        pipeline_name = parts[1] if len(parts) > 1 else None
        if pipeline_name:
            # Trigger pipeline run
            return {"text": f"Running pipeline: {pipeline_name}"}
        else:
            return {"text": "Please specify a pipeline name"}

    return {"text": "Unknown pipeline command"}

async def handle_status_command(text: str, response_url: str) -> Dict[str, Any]:
    """Handle status commands"""
    # Get system status
    return {"text": "System is running normally"}
```

## File Management Routes

### File Upload and Management

```python
# file_routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
from pathlib import Path

file_router = APIRouter(prefix="/files", tags=["files"])

# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@file_router.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a single file"""
    try:
        # Create upload directory if it doesn't exist
        file_path = UPLOAD_DIR / file.filename

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "size": file_path.stat().st_size,
            "path": str(file_path)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")

@file_router.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    results = []

    for file in files:
        try:
            file_path = UPLOAD_DIR / file.filename

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            results.append({
                "filename": file.filename,
                "size": file_path.stat().st_size,
                "status": "success"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return {"results": results}

@file_router.get("/list")
async def list_files():
    """List all uploaded files"""
    files = []

    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created_at": file_path.stat().st_ctime
            })

    return {"files": files}

@file_router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@file_router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete a file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File deletion failed: {str(e)}")
```

## Configuration Management

### Configuration Routes

```python
# config_routes.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
from pathlib import Path

config_router = APIRouter(prefix="/config", tags=["config"])

# Configuration file path
CONFIG_FILE = Path("./config/app.json")

@config_router.get("/")
async def get_config():
    """Get current configuration"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")

@config_router.post("/")
async def update_config(config_data: Dict[str, Any]):
    """Update configuration"""
    try:
        # Validate configuration
        if not validate_config(config_data):
            raise HTTPException(status_code=400, detail="Invalid configuration")

        # Save configuration
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)

        return {"message": "Configuration updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")

@config_router.get("/schema")
async def get_config_schema():
    """Get configuration schema"""
    return {
        "type": "object",
        "properties": {
            "debug": {"type": "boolean", "default": False},
            "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
            "max_file_size": {"type": "integer", "minimum": 1},
            "allowed_extensions": {"type": "array", "items": {"type": "string"}}
        }
    }

def validate_config(config_data: Dict[str, Any]) -> bool:
    """Validate configuration data"""
    # Basic validation
    required_keys = ["debug", "log_level"]

    for key in required_keys:
        if key not in config_data:
            return False

    return True
```

## Integrating Custom Routes with Hayhooks

### Complete Integration Example

```python
# hayhooks_custom.py
from fastapi import FastAPI
from hayhooks.server import create_app
from auth_routes import auth_router
from admin_routes import admin_router
from health_routes import health_router
from webhook_routes import webhook_router
from file_routes import file_router
from config_routes import config_router

def create_custom_app():
    """Create Hayhooks app with custom routes"""
    # Create base Hayhooks app
    app = create_app()

    # Include custom routers
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    app.include_router(webhook_router)
    app.include_router(file_router)
    app.include_router(config_router)

    # Add custom middleware
    @app.middleware("http")
    async def logging_middleware(request, call_next):
        start_time = time.time()

        # Log request
        print(f"Request: {request.method} {request.url}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        print(f"Response: {response.status_code} ({process_time:.3f}s)")

        return response

    return app

# Create the custom app
app = create_custom_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1416)
```

## Best Practices

### 1. Route Design

- Use clear, descriptive route names
- Organize routes by functionality
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Implement proper error handling

### 2. Security

- Implement proper authentication and authorization
- Validate all input data
- Use HTTPS in production
- Implement rate limiting

### 3. Performance

- Use asynchronous code for I/O operations
- Implement caching where appropriate
- Monitor performance metrics
- Optimize database queries

### 4. Documentation

- Document all routes with OpenAPI/Swagger
- Provide clear error messages
- Include usage examples
- Keep API documentation up to date

## Next Steps

- [Advanced Configuration](advanced-configuration.md) - Configuration options
- [Code Sharing](code-sharing.md) - Reusable components
- [Examples](../examples/overview.md) - Working examples
