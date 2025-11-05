# Makefile for Brain-AI Serving Infrastructure

.PHONY: help install build deploy clean test lint format health monitor scale

# Configuration
DOCKER_IMAGE := brain-ai-serving:latest
NAMESPACE := brain-ai-serving
COMPOSE_FILE := docker/docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
help: ## Show this help message
	@echo "$(BLUE)Brain-AI Serving Infrastructure$(NC)"
	@echo "$(BLUE)Available commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Usage examples:$(NC)"
	@echo "  make install        # Install dependencies"
	@echo "  make build         # Build Docker image"
	@echo "  make deploy-dev    # Deploy for development"
	@echo "  make test          # Run API tests"
	@echo "  make health        # Check service health"

# Installation targets
install: ## Install development dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

install-prod: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Production dependencies installed$(NC)"

# Build targets
build: ## Build Docker image
	@echo "$(BLUE)Building Docker image: $(DOCKER_IMAGE)$(NC)"
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Docker image built successfully$(NC)"

build-no-cache: ## Build Docker image without cache
	@echo "$(BLUE)Building Docker image (no cache): $(DOCKER_IMAGE)$(NC)"
	docker build --no-cache -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Docker image built successfully$(NC)"

# Development targets
dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file configs/.env.development up -d
	@echo "$(GREEN)Development environment started$(NC)"
	@echo "$(YELLOW)Access points:$(NC)"
	@echo "  API: http://localhost:8080"
	@echo "  Docs: http://localhost:8080/docs"
	@echo "  Grafana: http://localhost:3001 (admin/admin123)"

dev-logs: ## Show development logs
	docker-compose -f $(COMPOSE_FILE) logs -f

dev-stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)Development environment stopped$(NC)"

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	chmod +x scripts/deploy.sh
	./scripts/deploy.sh --environment staging --docker-compose
	@echo "$(GREEN)Staging deployment completed$(NC)"

deploy-production: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	chmod +x scripts/deploy.sh
	./scripts/deploy.sh --environment production --docker-compose --scale
	@echo "$(GREEN)Production deployment completed$(NC)"

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	chmod +x scripts/deploy.sh
	./scripts/deploy.sh --environment production --kubernetes --scale
	@echo "$(GREEN)Kubernetes deployment completed$(NC)"

# Monitoring targets
monitor: ## Start monitoring stack only
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d prometheus grafana jaeger
	@echo "$(GREEN)Monitoring stack started$(NC)"
	@echo "$(YELLOW)Access points:$(NC)"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Grafana: http://localhost:3001 (admin/admin123)"
	@echo "  Jaeger: http://localhost:16686"

# Health check targets
health: ## Run comprehensive health check
	@echo "$(BLUE)Running health checks...$(NC)"
	chmod +x scripts/health-check.sh
	./scripts/health-check.sh --component all --verbose

health-quick: ## Quick health check
	@echo "$(BLUE)Running quick health check...$(NC)"
	curl -s http://localhost:8080/health | jq '.' || echo "Health endpoint not available"

health-continuous: ## Continuous health monitoring
	@echo "$(BLUE)Starting continuous health monitoring...$(NC)"
	chmod +x scripts/health-check.sh
	./scripts/health-check.sh --max-retries -1 --check-interval 30

# Auto-scaling targets
scale-up: ## Scale up to 5 replicas
	@echo "$(BLUE)Scaling up service...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d --scale brain-ai-serving=5
	@echo "$(GREEN)Service scaled up to 5 replicas$(NC)"

scale-down: ## Scale down to 1 replica
	@echo "$(BLUE)Scaling down service...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d --scale brain-ai-serving=1
	@echo "$(GREEN)Service scaled down to 1 replica$(NC)"

scale-status: ## Show current scaling status
	@echo "$(BLUE)Current scaling status...$(NC)"
	docker-compose -f $(COMPOSE_FILE) ps brain-ai-serving

autoscaling: ## Start auto-scaling controller
	@echo "$(BLUE)Starting auto-scaling controller...$(NC)"
	chmod +x scripts/autoscaling.sh
	./scripts/autoscaling.sh --verbose

# Testing targets
test: ## Run API tests
	@echo "$(BLUE)Running API tests...$(NC)"
	chmod +x scripts/test-api.sh
	./scripts/test-api.sh --verbose

test-health: ## Run health check tests only
	@echo "$(BLUE)Running health check tests...$(NC)"
	chmod +x scripts/test-api.sh
	./scripts/test-api.sh --health-only

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	chmod +x scripts/test-api.sh
	./scripts/test-api.sh --performance-only

test-load: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	chmod +x scripts/test-api.sh
	./scripts/test-api.sh --load-test

# Code quality targets
lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(NC)"
	flake8 src/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)Linting completed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ scripts/ --line-length=88
	isort src/ scripts/ --profile black
	@echo "$(GREEN)Code formatted$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src/ scripts/ --line-length=88
	isort --check-only src/ scripts/ --profile black

security: ## Run security scanning
	@echo "$(BLUE)Running security scans...$(NC)"
	safety check
	bandit -r src/ -ll
	@echo "$(GREEN)Security scanning completed$(NC)"

# Cleanup targets
clean: ## Clean up containers and images
	@echo "$(BLUE)Cleaning up...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-all: ## Clean up everything including volumes
	@echo "$(BLUE)Performing deep cleanup...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans --rmi all
	docker system prune -af
	docker volume prune -f
	@echo "$(GREEN)Deep cleanup completed$(NC)"

# Logs and debugging
logs: ## Show all service logs
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-api: ## Show API service logs
	docker-compose -f $(COMPOSE_FILE) logs -f brain-ai-serving

logs-monitor: ## Show monitoring service logs
	docker-compose -f $(COMPOSE_FILE) logs -f prometheus grafana jaeger

# Database targets
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U prometheus -d monitoring -f /docker-entrypoint-initdb.d/init.sql
	@echo "$(GREEN)Database migrations completed$(NC)"

db-backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec postgres pg_dump -U prometheus monitoring > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backup created$(NC)"

# Model management
model-cache: ## Pre-download and cache model
	@echo "$(BLUE)Caching model...$(NC)"
	docker run --rm -v serving_data:/app/serving/data $(DOCKER_IMAGE) python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; import os; os.environ['MODEL_CACHE_DIR'] = '/app/serving/data/models'; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium'); AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')"
	@echo "$(GREEN)Model cached successfully$(NC)"

model-clean: ## Clean model cache
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	rm -rf data/models/*
	@echo "$(GREEN)Model cache cleaned$(NC)"

# Performance monitoring
perf-monitor: ## Monitor performance metrics
	@echo "$(BLUE)Starting performance monitoring...$(NC)"
	docker stats --no-stream

gpu-monitor: ## Monitor GPU usage
	@echo "$(BLUE)Monitoring GPU usage...$(NC)"
	watch -n 1 nvidia-smi

# Documentation targets
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build --clean --strict
	@echo "$(GREEN)Documentation built in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve --dev-addr=localhost:8000

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	mkdocs gh-deploy --clean
	@echo "$(GREEN)Documentation deployed$(NC)"

# Environment setup
env-setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	mkdir -p configs data logs data/models data/batch data/cache
	cp configs/production.yaml configs/.env.development.template
	@echo "$(GREEN)Development environment setup completed$(NC)"

env-config: ## Create environment configuration
	@echo "$(BLUE)Creating environment configuration...$(NC)"
	@echo "ENV=development" > configs/.env.development
	@echo "LOG_LEVEL=DEBUG" >> configs/.env.development
	@echo "MODEL_CACHE_DIR=/app/serving/data/models" >> configs/.env.development
	@echo "$(GREEN)Environment configuration created$(NC)"

# Status and information
status: ## Show deployment status
	@echo "$(BLUE)Deployment status:$(NC)"
	@echo "Docker containers:"
	docker-compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "K8s pods:"
	-kubectl get pods -n $(NAMESPACE) 2>/dev/null || echo "Kubernetes not configured"

info: ## Show service information
	@echo "$(BLUE)Brain-AI Serving Infrastructure Information$(NC)"
	@echo ""
	@echo "$(YELLOW)Service URLs:$(NC)"
	@echo "  API: http://localhost:8080"
	@echo "  API Docs: http://localhost:8080/docs"
	@echo "  Health: http://localhost:8080/health"
	@echo "  Metrics: http://localhost:8080/metrics"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Grafana: http://localhost:3001 (admin/admin123)"
	@echo "  Jaeger: http://localhost:16686"
	@echo ""
	@echo "$(YELLOW)Image:$(NC) $(DOCKER_IMAGE)"
	@echo "$(YELLOW)Namespace:$(NC) $(NAMESPACE)"

# Emergency targets
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP - Stopping all services immediately$(NC)"
	docker-compose -f $(COMPOSE_FILE) kill
	docker-compose -f $(COMPOSE_FILE) down -t 0
	@echo "$(GREEN)All services stopped$(NC)"

emergency-restart: ## Emergency restart all services
	@echo "$(RED)EMERGENCY RESTART - Restarting all services$(NC)"
	docker-compose -f $(COMPOSE_FILE) restart
	@echo "$(GREEN)All services restarted$(NC)"