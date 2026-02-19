# NoFishing Deployment Guide

## Prerequisites

### System Requirements
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### External Services
- Redis (included in Docker Compose)

## Docker Deployment

### 1. Build and Start Services

```bash
# Clone repository
git clone https://github.com/yourusername/nofishing.git
cd nofishing

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Verify Deployment

```bash
# Check health status
curl http://localhost:8080/api/v1/health
curl http://localhost:5000/api/v1/health

# Run test detection
curl -X POST http://localhost:8080/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "http://test-phishing-site.com"}'
```

### 3. Load Chrome Extension

1. Open Chrome: `chrome://extensions/`
2. Enable Developer Mode
3. Click "Load unpacked"
4. Select `nofishing-extension/` directory

## Production Deployment

### Environment Variables

```bash
# Backend
export SERVER_PORT=8080
export REDIS_HOST=redis
export REDIS_PORT=6379
export ML_SERVICE_BASE_URL=http://ml-api:5000/api/v1

# ML API
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export MODEL_TYPE=onnx  # Use ONNX for faster inference
export USE_GPU=false
```

### Scaling

```bash
# Scale backend
docker-compose up -d --scale backend=3

# Scale ML API
docker-compose up -d --scale ml-api=2
```

### Monitoring

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f ml-api

# Restart services
docker-compose restart backend
```

## Troubleshooting

### Backend Issues

```bash
# Check backend logs
docker-compose logs backend

# Restart backend
docker-compose restart backend

# Rebuild backend
docker-compose up -d --build backend
```

### ML API Issues

```bash
# Check ML API logs
docker-compose logs ml-api

# Verify model file exists
docker-compose exec ml-api ls -la /app/models/

# Rebuild ML API
docker-compose up -d --build ml-api
```

### Redis Issues

```bash
# Check Redis logs
docker-compose logs redis

# Connect to Redis CLI
docker-compose exec redis redis-cli

# Test connection
docker-compose exec redis redis-cli ping
```

## Security Considerations

1. **API Keys**: Store in environment variables, never commit to git
2. **CORS**: Configure proper origin whitelist in production
3. **Rate Limiting**: Implement rate limiting for public deployments
4. **HTTPS**: Always use HTTPS in production
5. **Input Validation**: All inputs are validated at multiple layers

## Backup and Recovery

### Backup Redis Data

```bash
# Create backup
docker-compose exec redis redis-cli BGSAVE

# Copy RDB file
docker cp nofishing-redis:/data/dump.rdb ./backup/
```

### Restore Redis Data

```bash
# Stop Redis
docker-compose stop redis

# Copy backup file
docker cp ./backup/dump.rdb nofishing-redis:/data/dump.rdb

# Start Redis
docker-compose start redis
```
