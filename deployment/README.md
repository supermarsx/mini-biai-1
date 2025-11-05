# Deployment Configurations

This directory contains all deployment configurations, container setups, and cloud deployment templates for the Brain-Inspired Modular AI Framework.

## Structure

```
deployment/
├── README.md                  # This file
├── Makefile                   # Build and deployment commands
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── configs/                   # Configuration files
│   └── production.yaml       # Production configuration
├── docker/                    # Docker configurations
│   ├── Dockerfile            # Main application container
│   ├── docker-compose.yml    # Multi-container setup
│   └── nginx/                # Nginx proxy configuration
├── kubernetes/                # Kubernetes deployments
│   └── deployment.yaml       # K8s deployment manifest
├── monitoring/                # Monitoring stack
│   ├── grafana/             # Grafana dashboards
│   └── prometheus/          # Prometheus configuration
├── scripts/                  # Deployment scripts
│   ├── deploy.sh            # Main deployment script
│   ├── health-check.sh      # Health check utility
│   ├── autoscaling.sh       # Auto-scaling controller
│   ├── test-api.sh          # API testing suite
│   └── nginx/               # Nginx management scripts
└── src/                      # Application code
    └── main.py              # FastAPI application
```

## Quick Deployment

### Development Environment

```bash
# Start development environment
make dev

# Check service health
make health

# View logs
make logs-api
```

### Production Deployment

```bash
# Production with auto-scaling
make deploy-production

# Kubernetes deployment
make deploy-k8s

# Verify deployment
make health && make info
```

## Deployment Types

### Docker Compose

Multi-container deployment with:
- FastAPI application (vLLM backend)
- Nginx load balancer
- Prometheus metrics
- Grafana dashboards
- Redis caching (optional)

### Kubernetes

Production-ready K8s manifests featuring:
- Deployment with auto-scaling (HPA)
- ConfigMaps and Secrets management
- Service and Ingress configurations
- Resource limits and requests
- Health checks and probes

### Cloud Providers

Support for major cloud platforms:
- **AWS**: ECS, EKS deployment configurations
- **GCP**: GKE, Cloud Run templates
- **Azure**: AKS deployment manifests
- **DigitalOcean**: Kubernetes templates

## Configuration Management

### Environment Variables

Core configuration via environment variables:

```bash
# Application settings
ENV=production
LOG_LEVEL=INFO
MODEL_CACHE_DIR=/app/data/models

# Performance tuning
CUDA_VISIBLE_DEVICES=0
MAX_BATCH_SIZE=32
WORKER_PROCESSES=4

# Monitoring
PROMETHEUS_METRICS=true
OTEL_SERVICE_NAME=brain-ai-serving
```

### Configuration Files

Environment-specific configurations:

- `configs/production.yaml`: Production settings
- `configs/staging.yaml`: Staging environment
- `configs/development.yaml`: Development configuration

### Secrets Management

Secure credential handling:

- Kubernetes Secrets
- Docker secrets
- Environment variables
- Vault integration (optional)

## Monitoring & Observability

### Prometheus Metrics

Comprehensive metrics collection:
- Request latency and throughput
- GPU utilization and memory
- Error rates and patterns
- System resource usage

### Grafana Dashboards

Real-time monitoring:
- System overview dashboard
- Performance metrics
- Error analysis
- Resource utilization

### Health Checks

Multi-level health monitoring:
- Application health endpoints
- Dependency checks
- Database connectivity
- External service availability

## Auto-scaling

### Horizontal Pod Autoscaling (HPA)

CPU and memory-based auto-scaling:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Custom Metrics

Application-specific scaling:
- Request rate scaling
- Queue size thresholds
- Response time metrics
- Custom business metrics

## Security

### Container Security

- Multi-stage Docker builds
- Non-root user execution
- Minimal base images
- Security scanning integration

### Network Security

- TLS/SSL termination at load balancer
- Internal service mesh (optional)
- Network policies for Kubernetes
- API rate limiting and throttling

### Secrets Security

- Encrypted secrets storage
- Secret rotation policies
- RBAC for Kubernetes
- Vault integration

## Performance Optimization

### Resource Allocation

Optimal resource configurations:

```yaml
resources:
  requests:
    cpu: 2
    memory: 8Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
```

### Caching Strategy

Multi-level caching:
- Redis for session data
- Model cache for inference
- CDN for static assets
- Database query caching

### Load Balancing

Efficient request distribution:
- Nginx round-robin
- Least connections
- Health-based routing
- Sticky sessions (when needed)

## Deployment Commands

### Make Commands

```bash
make help              # Show all available commands
make install           # Install dependencies
make build             # Build containers
make dev               # Start development environment
make deploy-production # Deploy to production
make deploy-k8s        # Deploy to Kubernetes
make health            # Check service health
make logs              # View application logs
make scale-up          # Manual scaling
make scale-down        # Scale down
make clean             # Clean up resources
make test              # Run integration tests
```

### Script Commands

```bash
# Deployment
./scripts/deploy.sh --env production --scale 3

# Health checks
./scripts/health-check.sh --comprehensive

# Auto-scaling configuration
./scripts/autoscaling.sh --cpu-threshold 70 --memory-threshold 80

# API testing
./scripts/test-api.sh --duration 300 --concurrency 10
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs <container-id>

# Inspect container state
docker inspect <container-id>

# Check resource availability
docker system df
```

#### High Memory Usage
```bash
# Monitor container resources
docker stats

# Check application memory
curl http://localhost:8080/metrics | grep memory

# Scale resources
kubectl scale deployment brain-ai-serving --replicas=5
```

#### API Errors
```bash
# Check application logs
make logs-api | grep ERROR

# Test API endpoints
curl http://localhost:8080/health

# Run API tests
./scripts/test-api.sh
```

### Debug Commands

```bash
# Emergency stop
make emergency-stop

# Clean restart
make clean-all && make build && make deploy-production

# Debug mode
make dev-debug

# Resource analysis
kubectl top pods
docker system df
```

## CI/CD Integration

### GitHub Actions

Automated deployment workflows:
- Build and test on PR
- Deploy to staging on merge
- Production deployment on release
- Automated rollback on failure

### Deployment Pipeline

```yaml
# Example pipeline stages
stages:
  - build: Build containers
  - test: Run integration tests
  - scan: Security scanning
  - deploy-staging: Deploy to staging
  - test-staging: Validate staging
  - deploy-prod: Deploy to production
  - verify: Production verification
```

## Cost Optimization

### Resource Management

- Right-sizing containers based on usage
- Spot instances for non-critical workloads
- Auto-scaling to minimize idle resources
- Resource quotas and limits

### Monitoring Costs

- Cloud provider cost tracking
- Resource utilization monitoring
- Alert on unusual spending patterns
- Regular cost optimization reviews

## Best Practices

### Deployment
- Use immutable deployments
- Implement blue-green or canary deployments
- Maintain rollback procedures
- Monitor deployment metrics

### Configuration
- Separate configuration from code
- Use environment-specific configs
- Implement configuration validation
- Document all configuration options

### Monitoring
- Set up comprehensive monitoring
- Configure appropriate alerts
- Monitor both infrastructure and application
- Regular alert testing

### Security
- Regular security updates
- Implement proper RBAC
- Use network segmentation
- Regular security scanning

## Contributing

When adding new deployment configurations:

1. **Follow naming conventions**: Use descriptive, environment-specific names
2. **Include documentation**: Document new configuration options
3. **Add validation**: Include configuration validation where possible
4. **Test thoroughly**: Test in all relevant environments
5. **Security review**: Ensure new configurations don't introduce security issues
6. **Update this README**: Document new features and changes

## Support

For deployment-related issues:
- Check the troubleshooting section above
- Review application logs: `make logs`
- Run health checks: `make health`
- Consult monitoring dashboards
- Create detailed issue reports with logs and configuration