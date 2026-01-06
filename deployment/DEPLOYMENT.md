# Deployment Guide

## 🚀 Production Deployment Options

This guide covers deploying the Churn Prediction Platform to production.

## Prerequisites

- Docker and Docker Compose installed
- Cloud provider account (AWS, GCP, Azure, or container platform)
- Domain name (optional, for custom URLs)
- SSL certificate (recommended for production)

## Option 1: Docker Compose (Simple)

### Quick Start

```bash
# Build images
docker-compose -f deployment/docker-compose.yml build

# Start API service
docker-compose -f deployment/docker-compose.yml up -d churn-api

# Check health
curl http://localhost:8000/

# View logs
docker-compose logs -f churn-api
```

### Run Batch Inference

```bash
docker-compose --profile batch up churn-batch
```

### Run Training

```bash
docker-compose --profile training up churn-training
```

## Option 2: Cloud Platform (Recommended)

### Deploy to Render.com

1. **Create `render.yaml`** (already in deployment folder)

2. **Push to GitHub**

3. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repo
   - Render will auto-detect the Dockerfile

4. **Configure Environment Variables**:
   ```
   LOG_LEVEL=INFO
   MODEL_PATH=/app/models
   ```

5. **Deploy** - Render handles the rest!

### Deploy to Fly.io

```bash
# Install flyctl
brew install flyctl  # macOS
# or curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Initialize
cd deployment
fly launch --dockerfile Dockerfile

# Deploy
fly deploy

# Scale if needed
fly scale count 2

# View logs
fly logs
```

### Deploy to AWS ECS

1. **Build and Push to ECR**:

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name churn-prediction

# Build and tag
docker build -t churn-prediction -f deployment/Dockerfile .
docker tag churn-prediction:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction:latest
```

2. **Create ECS Task Definition** (see `deployment/aws/task-definition.json`)

3. **Create ECS Service**:

```bash
aws ecs create-service \
  --cluster churn-cluster \
  --service-name churn-api \
  --task-definition churn-prediction \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

## Option 3: Kubernetes

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Check status
kubectl get pods -n churn-prediction

# Expose service
kubectl apply -f deployment/kubernetes/ingress.yaml

# Scale
kubectl scale deployment churn-api --replicas=3 -n churn-prediction
```

### Using Helm (Advanced)

```bash
# Create Helm chart
helm create churn-prediction-chart

# Install
helm install churn-prediction ./churn-prediction-chart

# Upgrade
helm upgrade churn-prediction ./churn-prediction-chart
```

## Environment Variables

Configure these environment variables in your deployment:

```bash
# Paths
MODEL_PATH=/app/models
DATA_PATH=/app/data
FEATURE_STORE_PATH=/app/feature_store

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/api.log

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Settings
MODEL_VERSION=production  # or specific version
ENABLE_EXPLANATIONS=true

# Monitoring
ENABLE_MONITORING=true
ALERT_EMAIL=ml-team@company.com
```

## Scheduled Jobs

### Using Cron (Simple)

```bash
# Add to crontab
# Run batch inference daily at 2 AM
0 2 * * * docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs churn-prediction python -m src.inference.batch

# Run monitoring weekly on Monday at 9 AM
0 9 * * 1 docker run --rm -v $(pwd)/models:/app/models churn-prediction python -m src.monitoring.drift_detector --reference-date 2024-01-01 --current-date $(date +%Y-%m-%d)
```

### Using Airflow (Production)

See `deployment/airflow/dags/churn_pipeline_dag.py` for example DAG.

### Using Cloud Scheduler

**AWS EventBridge:**
```bash
aws events put-rule --schedule-expression "cron(0 2 * * ? *)" --name churn-batch-daily
aws events put-targets --rule churn-batch-daily --targets "Id"="1","Arn"="<ecs-task-arn>"
```

**GCP Cloud Scheduler:**
```bash
gcloud scheduler jobs create http churn-batch-daily \
  --schedule="0 2 * * *" \
  --uri="https://your-api.com/batch/run" \
  --http-method=POST
```

## Monitoring & Logging

### Application Logs

```bash
# Docker logs
docker logs -f churn-prediction-api

# Kubernetes logs
kubectl logs -f deployment/churn-api -n churn-prediction

# Write to external logging service
# Configure in configs/logging.yaml
```

### Health Checks

API includes health endpoints:
- `GET /` - Basic health check
- `GET /model-info` - Model status and metrics

### Metrics

Integrate with monitoring tools:
- **Prometheus**: Expose metrics endpoint
- **DataDog**: Use DataDog agent
- **CloudWatch**: For AWS deployments

## Security Best Practices

1. **API Authentication**: Add API key middleware
```python
# In src/inference/api.py
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")
```

2. **HTTPS**: Use reverse proxy (Nginx/Traefik) with SSL
```nginx
server {
    listen 443 ssl;
    server_name churn-api.yourcompany.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

3. **Network Security**: Use VPC, security groups, firewalls

4. **Secrets Management**: Use vault or cloud secrets manager
```bash
# AWS Secrets Manager
aws secretsmanager create-secret --name churn-api-key
```

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up --scale churn-api=3

# Kubernetes
kubectl scale deployment churn-api --replicas=5

# ECS
aws ecs update-service --service churn-api --desired-count 3
```

### Auto-Scaling

**Kubernetes HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Rollback Strategy

```bash
# Docker - keep previous image tagged
docker tag churn-prediction:latest churn-prediction:v1.0

# Kubernetes - rollback deployment
kubectl rollout undo deployment/churn-api

# ECS - update to previous task definition
aws ecs update-service --service churn-api --task-definition churn-prediction:1
```

## Cost Optimization

1. **Right-size resources**: Start small, scale as needed
2. **Use spot instances**: For batch jobs (AWS Spot, GCP Preemptible)
3. **Schedule resources**: Turn off dev/staging environments overnight
4. **Optimize images**: Multi-stage builds, smaller base images

## Troubleshooting

### API not responding
```bash
# Check container status
docker ps
kubectl get pods

# Check logs
docker logs churn-prediction-api
kubectl logs deployment/churn-api

# Check health endpoint
curl http://localhost:8000/
```

### Model not found
```bash
# Verify model volume mounted
docker inspect churn-prediction-api | grep Mounts

# Check model registry
python -c "from src.models.registry import ModelRegistry; print(ModelRegistry().list_models())"
```

### Performance issues
```bash
# Check resource usage
docker stats
kubectl top pods

# Profile API
pip install py-spy
py-spy record -o profile.svg -- python -m uvicorn src.inference.api:app
```

## Production Checklist

- [ ] SSL/TLS configured
- [ ] Authentication enabled
- [ ] Monitoring and alerting set up
- [ ] Logging aggregated
- [ ] Backup strategy for models
- [ ] Auto-scaling configured
- [ ] Health checks enabled
- [ ] Rate limiting implemented
- [ ] API documentation available
- [ ] Incident response plan documented
- [ ] Regular model retraining scheduled
- [ ] Drift monitoring active

## Support

For production issues:
- **Documentation**: [Link to internal docs]
- **On-call**: [PagerDuty/Slack channel]
- **Escalation**: [Team lead contact]

---

**Last Updated**: January 2026
