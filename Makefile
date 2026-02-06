# YOLOv8 CSV-to-YOLO Pipeline — Makefile
# ========================================

.PHONY: help install convert train predict export clean lint format

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"

convert: ## Convert CSV annotations to YOLO format
	python scripts/convert_csv_to_yolo.py --config configs/default.yaml

train: ## Train YOLOv8 model
	python scripts/train.py --config configs/default.yaml

predict: ## Run inference on images
	python scripts/predict.py --weights artifacts/best.pt --source data/images

export: ## Export model to ONNX
	python -c "from ultralytics import YOLO; YOLO('artifacts/best.pt').export(format='onnx')"

clean: ## Clean generated files
	rm -rf runs/ dataset/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint: ## Run linter
	ruff check src/ scripts/

format: ## Format code with Black
	black src/ scripts/ --line-length 100
