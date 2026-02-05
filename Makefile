.PHONY: install install-dev install-ops prepare train evaluate pipeline predict \
       test lint dvc-repro dvc-status dvc-metrics dvc-plots mlflow-ui

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml]"

install-ops:
	pip install -e ".[dev,ml,ops]"

prepare:
	churn-prepare --config config/default.yaml

train:
	churn-train --config config/default.yaml

evaluate:
	churn-evaluate --config config/default.yaml

pipeline:
	make prepare
	make train
	make evaluate

predict:
	churn-predict --config config/default.yaml --input data/new_customers.csv --output data/predictions.csv

# DVC
dvc-repro:
	dvc repro

dvc-status:
	dvc status

dvc-metrics:
	dvc metrics show

dvc-plots:
	dvc plots show models/threshold_analysis_val.csv

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri mlruns

test:
	pytest

lint:
	ruff check src tests
