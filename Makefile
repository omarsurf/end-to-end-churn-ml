.PHONY: install install-dev prepare train evaluate pipeline test lint

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml]"

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

test:
	pytest

lint:
	ruff check src tests
