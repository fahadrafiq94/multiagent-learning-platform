.PHONY: dev-up dev-down ollama-list ollama-pull test-model-health

dev-up:
	docker compose -f ./docker/compose/docker-compose.dev.yml up --build

dev-down:
	docker compose -f ./docker/compose/docker-compose.dev.yml down

ollama-list:
	docker exec -it fredi-ollama-dev ollama list

ollama-pull:
	docker exec -it fredi-ollama-dev ollama pull qwen3:4b
	docker exec -it fredi-ollama-dev ollama pull nomic-embed-text-v2-moe

test-model-health:
	curl http://localhost:8000/model/health


# Server deployment with NVIDIA GPU acceleration
prod-up:
	docker compose -f ./docker/compose/docker-compose.dev.yml -f ./docker/compose/docker-compose.prod.yml up -d --build

prod-down:
	docker compose -f ./docker/compose/docker-compose.dev.yml -f ./docker/compose/docker-compose.prod.yml down
