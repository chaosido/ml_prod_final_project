# ASR QE Pipeline - Makefile
.PHONY: help install test lint up down e2e traffic clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	uv sync --all-extras

test:  ## Run all tests
	uv run --extra dev pytest tests/ -v

lint:  ## Run linter and fix issues
	uv run --extra dev ruff check . --fix

up:  ## Start all Docker services
	docker compose up -d

down:  ## Stop all Docker services
	docker compose down

e2e:  ## Run full E2E pipeline test
	./scripts/test_pipeline_e2e.sh

traffic:  ## Generate API traffic for Grafana
	./scripts/generate_traffic.sh

clean:  ## Full cleanup (Docker + data)
	docker compose down -v
	docker run --rm -v "$(PWD):/app" busybox sh -c "rm -rf /app/data/features /app/data/incoming/* /app/data/archive/* /app/models/staging/* /app/models/production/*"
