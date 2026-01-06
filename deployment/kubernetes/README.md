# Kubernetes Deployment for Churn Prediction Platform

This directory contains Kubernetes manifests for deploying the churn prediction platform.

## Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local minikube)
- kubectl configured
- Docker image pushed to registry

## Quick Deploy

```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -n churn-prediction
kubectl get svc -n churn-prediction

# Access API
kubectl port-forward -n churn-prediction svc/churn-api 8000:8000
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Ingress (NGINX)                    │
│              churn-api.example.com                  │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
    ┌─────▼─────┐      ┌───────▼────────┐
    │  API Pods │      │  Batch Jobs    │
    │  (3 replicas)    │  (CronJob)     │
    └───────────┘      └────────────────┘
          │
    ┌─────▼─────────────────────┐
    │  Feature Store (PVC)      │
    │  Model Registry (PVC)     │
    └───────────────────────────┘
```

## Manifests

1. **namespace.yaml**: Namespace and resource quotas
2. **configmap.yaml**: Configuration files
3. **pvc.yaml**: Persistent volume claims for data/models
4. **api-deployment.yaml**: FastAPI service deployment
5. **api-service.yaml**: Service for API
6. **batch-cronjob.yaml**: Scheduled batch inference
7. **training-job.yaml**: Model training job
8. **ingress.yaml**: Ingress for external access

## Configuration

### 1. Update Image
Edit deployments to use your Docker image:
```yaml
image: gcr.io/your-project/churn-prediction:latest
```

### 2. Resource Limits
Adjust based on your workload:
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### 3. Scaling
```bash
# Scale API pods
kubectl scale deployment churn-api -n churn-prediction --replicas=5

# Autoscaling
kubectl autoscale deployment churn-api -n churn-prediction \
  --cpu-percent=70 --min=2 --max=10
```

## Monitoring

### Logs
```bash
# API logs
kubectl logs -n churn-prediction -l app=churn-api -f

# Batch job logs
kubectl logs -n churn-prediction -l app=churn-batch
```

### Metrics
```bash
# Pod metrics
kubectl top pods -n churn-prediction

# Node metrics
kubectl top nodes
```

## Secrets Management

For production, use Kubernetes secrets:

```bash
# Create secret for API keys
kubectl create secret generic churn-secrets \
  --from-literal=api-key=your-key \
  -n churn-prediction

# Mount in deployment
# See api-deployment.yaml for example
```

## Persistent Storage

The platform uses PVCs for:
- Feature store data
- Model artifacts
- Logs

Storage class depends on your cloud provider:
- GKE: `pd-standard` or `pd-ssd`
- EKS: `gp2` or `gp3`
- AKS: `managed-premium`

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
- name: Deploy to GKE
  run: |
    gcloud container clusters get-credentials $CLUSTER_NAME
    kubectl apply -f deployment/kubernetes/
    kubectl rollout status deployment/churn-api -n churn-prediction
```

### ArgoCD
```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: churn-prediction
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/churn-prediction
    targetRevision: main
    path: deployment/kubernetes
```

## Production Checklist

- [ ] Image in production registry
- [ ] Secrets configured
- [ ] Resource limits tuned
- [ ] Monitoring/alerting setup
- [ ] Backup strategy for PVCs
- [ ] Ingress SSL/TLS configured
- [ ] Network policies applied
- [ ] Pod security policies
- [ ] RBAC configured

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod <pod-name> -n churn-prediction
kubectl logs <pod-name> -n churn-prediction
```

### Storage issues
```bash
kubectl get pvc -n churn-prediction
kubectl describe pvc <pvc-name> -n churn-prediction
```

### Network issues
```bash
kubectl get svc -n churn-prediction
kubectl get ingress -n churn-prediction
```
