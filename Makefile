DEV_COMPOSE := docker compose -f ./docker/compose/docker-compose.dev.yml
PROD_COMPOSE := docker compose -f ./docker/compose/docker-compose.prod.yml


.PHONY: \
	dev-config \
	dev-build \
	dev-up \
	dev-down \
	dev-restart \
	dev-logs \
	dev-ps \
	prod-config \
	prod-build \
	prod-up \
	prod-down \
	prod-logs \
	prod-ps \
	ollama-list \
	ollama-pull \
	model-health \
	docker-health \
	docker-test \
	docker-mypy \
	docker-ruff \
	semantic-smoke \
	concurrency-smoke \
	export-orchestration-graph \
	export-runtime-workflow \
	export-graphs


# ---------------------------------------------------
# Development Docker
# ---------------------------------------------------

dev-config:
	$(DEV_COMPOSE) config


dev-build:
	$(DEV_COMPOSE) build


dev-up:
	$(DEV_COMPOSE) up -d --build


dev-down:
	$(DEV_COMPOSE) down


dev-restart:
	$(DEV_COMPOSE) down
	$(DEV_COMPOSE) up -d --build


dev-logs:
	$(DEV_COMPOSE) logs -f


dev-ps:
	$(DEV_COMPOSE) ps


# ---------------------------------------------------
# Production Docker
# ---------------------------------------------------

prod-config:
	$(PROD_COMPOSE) config


prod-build:
	$(PROD_COMPOSE) build


prod-up:
	$(PROD_COMPOSE) up -d --build


prod-down:
	$(PROD_COMPOSE) down


prod-logs:
	$(PROD_COMPOSE) logs -f


prod-ps:
	$(PROD_COMPOSE) ps


# ---------------------------------------------------
# Ollama
# ---------------------------------------------------

ollama-list:
	$(DEV_COMPOSE) exec ollama ollama list


ollama-pull:
	$(DEV_COMPOSE) exec ollama ollama pull qwen3:4b
	$(DEV_COMPOSE) exec ollama ollama pull nomic-embed-text-v2-moe


# ---------------------------------------------------
# Health
# ---------------------------------------------------

model-health:
	curl --fail http://localhost:8000/model/health


docker-health:
	$(DEV_COMPOSE) ps
	$(DEV_COMPOSE) exec ollama ollama list
	curl --fail http://localhost:8000/model/health
