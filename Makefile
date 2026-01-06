.PHONY: help setup install test lint format clean run-pipeline train infer api docker-build docker-up docker-down

# Variables
PYTHON := python3
PIP := pip
VENV := venv
SRC := src
TESTS := tests

help:  ## Show this help message
	@echo "Customer Churn Prediction Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Create virtual environment and install dependencies
	@echo "🔧 Setting up environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created"
	@echo ""
	@echo "Activate with: source $(VENV)/bin/activate"

install:  ## Install dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✓ Dependencies installed"

install-dev:  ## Install development dependencies
	@echo "📦 Installing dev dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest black isort flake8 mypy
	@echo "✓ Dev dependencies installed"

test:  ## Run test suite
	@echo "🧪 Running tests..."
	pytest $(TESTS) -v --tb=short

test-coverage:  ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=html --cov-report=term

lint:  ## Run linting checks
	@echo "🔍 Running linters..."
	flake8 $(SRC) --max-line-length=100 --exclude=__pycache__
	mypy $(SRC) --ignore-missing-imports

format:  ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black $(SRC) $(TESTS) --line-length=100
	isort $(SRC) $(TESTS) --profile=black

format-check:  ## Check code formatting without modifying
	@echo "🎨 Checking code format..."
	black $(SRC) $(TESTS) --line-length=100 --check
	isort $(SRC) $(TESTS) --profile=black --check

clean:  ## Clean generated files
	@echo "🧹 Cleaning..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
	@echo "✓ Cleaned"

run-pipeline:  ## Run full end-to-end pipeline
	@echo "🚀 Running full pipeline..."
	$(PYTHON) run_pipeline.py --mode full

run-data:  ## Run data ingestion only
	@echo "📊 Running data ingestion..."
	$(PYTHON) -m $(SRC).data_ingestion.run

train:  ## Train model
	@echo "🤖 Training model..."
	$(PYTHON) run_pipeline.py --mode train_only

infer:  ## Run batch inference
	@echo "🔮 Running batch inference..."
	$(PYTHON) run_pipeline.py --mode inference_only

api:  ## Start FastAPI server
	@echo "🌐 Starting API server..."
	uvicorn $(SRC).inference.api:app --reload --host 0.0.0.0 --port 8000

api-prod:  ## Start API server in production mode
	@echo "🌐 Starting API server (production)..."
	uvicorn $(SRC).inference.api:app --host 0.0.0.0 --port 8000 --workers 4

monitor:  ## Run drift monitoring
	@echo "📈 Running drift monitoring..."
	$(PYTHON) -m $(SRC).monitoring.drift_detector

docker-build:  ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t churn-prediction:latest .

docker-up:  ## Start Docker services
	@echo "🐳 Starting Docker services..."
	docker-compose up -d

docker-down:  ## Stop Docker services
	@echo "🐳 Stopping Docker services..."
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

validate-config:  ## Validate configuration files
	@echo "✅ Validating configs..."
	@for file in configs/*.yaml; do \
		echo "Checking $$file..."; \
		$(PYTHON) -c "import yaml; yaml.safe_load(open('$$file'))"; \
	done
	@echo "✓ All configs valid"

init-dirs:  ## Create necessary directories
	@echo "📁 Creating directories..."
	mkdir -p data/raw data/processed
	mkdir -p models/registry models/artifacts
	mkdir -p feature_store/offline feature_store/online
	mkdir -p outputs logs
	@echo "✓ Directories created"

check-data:  ## Check if dataset exists
	@if [ -f "data/raw/telco.csv" ]; then \
		echo "✓ Dataset found: data/raw/telco.csv"; \
	else \
		echo "❌ Dataset not found!"; \
		echo "Please download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"; \
		exit 1; \
	fi

all: install init-dirs check-data validate-config  ## Setup everything

ci: format-check lint test  ## Run CI checks

# Quick start workflow
quickstart: setup install init-dirs  ## Quick start setup
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source $(VENV)/bin/activate"
	@echo "  2. Download dataset to data/raw/telco.csv"
	@echo "  3. Run pipeline: make run-pipeline"
	@echo "  4. Start API: make api"
	@echo ""
