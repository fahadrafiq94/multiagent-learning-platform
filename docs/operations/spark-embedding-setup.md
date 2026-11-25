# Spark embedding service: initial setup

Prepared: 2026-11-16. The user executed setup on Spark; independent laptop checks
confirmed version 0.32.1, the expected model digest and two unit-norm 768-dimensional
vectors. See [Checkpoint 0](../checkpoints/sprint-6-checkpoint-0.md).
These commands are for **Bash on DGX Spark**, not PowerShell on the laptop.
Do not recreate the existing container when revisiting these instructions.
Leave the existing vLLM service and its
`vllm-start` / `vllm-stop` scripts unchanged.

Use a separate CPU-only Ollama container initially. This avoids reserving GPU
resources from the running chat service while establishing embedding compatibility.
GPU acceleration can be considered later with measured resource availability.
The Docker pattern follows [Ollama's official Docker instructions](https://docs.ollama.com/docker).
Version 0.32.1 is an explicit initial pin, not a claim that it is the newest release.

## 1. Check existing services

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
ss -ltn '( sport = :11434 )'
```

If port 11434 is already listening, inspect the existing service before proceeding.
Do not stop or replace it. The block below also refuses an occupied port or existing
container name. It preserves existing containers and volumes.

## 2. Create the separate embedding service

Run the following block once. The subshell contains its fail-fast behavior.

```bash
(
  set -euo pipefail

  if [[ -n "$(ss -H -ltn '( sport = :11434 )')" ]]; then
    echo 'Port 11434 is occupied. Stop here and report the existing service.' >&2
    exit 1
  fi
  if docker container inspect fredi-embeddings >/dev/null 2>&1; then
    echo 'fredi-embeddings already exists. Inspect it instead of replacing it.' >&2
    exit 1
  fi

  docker pull ollama/ollama:0.32.1
  docker run -d \
    --name fredi-embeddings \
    --restart unless-stopped \
    --cpus 4 \
    --memory 6g \
    -e OLLAMA_VULKAN=0 \
    -e OLLAMA_NUM_PARALLEL=1 \
    -e OLLAMA_MAX_LOADED_MODELS=1 \
    -p 192.168.40.19:11434:11434 \
    -v fredi-embedding-models:/root/.ollama \
    ollama/ollama:0.32.1

  for attempt in {1..30}; do
    if curl -fsS http://192.168.40.19:11434/api/version >/dev/null; then
      break
    fi
    sleep 2
  done
  curl -fsS http://192.168.40.19:11434/api/version

  docker exec fredi-embeddings ollama pull nomic-embed-text-v2-moe:latest
  docker exec fredi-embeddings ollama list
  docker image inspect ollama/ollama:0.32.1 --format '{{json .RepoDigests}}'
)
```

The published address is the supplied Spark LAN address; change the bind IP if the
device's address changes. No `--gpus` or GPU device mapping is used. The model volume
persists across container stops. Do not use `vllm-stop` to manage this container.
If this block fails after creation, keep the container/logs for diagnosis rather
than deleting volumes or rerunning replacement commands.

The model's official page documents the pull command and its 768-dimensional
representation. A floating tag can change, so the next step verifies its digest
before using it with the existing index.
[Nomic model documentation](https://ollama.com/library/nomic-embed-text-v2-moe)

## 3. Verify model identity and a synthetic embedding

Run this on Spark. It sends no student data and writes nothing to PostgreSQL.
It stops if the downloaded model differs from the repository's current embedding
profile. Do not edit the expected digest to make a mismatch disappear.

```bash
python3 - <<'PY'
import json
import math
import urllib.request

base = 'http://192.168.40.19:11434'
model = 'nomic-embed-text-v2-moe:latest'
expected = 'ff9c2f10ef5e3722623a1b396e1e04efc27a93112c83e9b7b7b9ca1d05620965'

with urllib.request.urlopen(base + '/api/tags', timeout=15) as response:
    models = json.load(response)['models']
entry = next((item for item in models if item.get('name') == model), None)
if entry is None:
    raise SystemExit('Expected embedding model is missing.')
digest = entry['digest'].removeprefix('sha256:')
print('Model:', model)
print('Digest:', digest)
if digest != expected:
    raise SystemExit('STOP: digest differs from the existing index profile. Report it before integration.')

payload = json.dumps({
    'model': model,
    'input': [
        'search_query: Where is the client address assigned?',
        'search_document: The client has an assigned address.',
    ],
    'truncate': False,
}).encode()
request = urllib.request.Request(
    base + '/api/embed', data=payload,
    headers={'Content-Type': 'application/json'}, method='POST',
)
with urllib.request.urlopen(request, timeout=120) as response:
    vectors = json.load(response)['embeddings']
if len(vectors) != 2 or any(
    len(vector) != 768 or not all(math.isfinite(value) for value in vector)
    or not any(value != 0 for value in vector)
    for vector in vectors
):
    raise SystemExit('Invalid embedding count, dimensions or values.')
print('Vectors:', len(vectors))
print('Dimensions:', [len(vector) for vector in vectors])
print('Norms:', [round(math.sqrt(sum(value * value for value in vector)), 6) for vector in vectors])
print('Preflight passed; backend profile checks are still required before using the index.')
PY
```

The API paths are documented in [model listing](https://docs.ollama.com/api/tags)
and [embedding generation](https://docs.ollama.com/api/embed). `truncate=False`
preserves the application's explicit input-limit behavior.

Send back the version, model digest, image digest and validation output. If there
is an error, also capture `docker logs --tail 60 fredi-embeddings`. Do not send
credential files or unrelated student request logs.

## 4. Backend configuration (Checkpoint 2)

Configured endpoint roles:

- Chat: `http://192.168.40.19:8000/v1`, vLLM, Qwen3.8-27B-NVFP4.
- Embeddings: `http://192.168.40.19:11434`, Ollama, the verified existing profile.

Independent `.env` routing is implemented in Sprint 6 Checkpoint 2; see the
[model-profile runbook](vllm-provider.md) for the actual settings and precedence.
These setup commands do not themselves enable backend retrieval or intake.
Do not rebuild/publish the index merely because the service moved to another host.

## 5. Running container but refused connections

During Checkpoint 2, the container was running and `docker exec fredi-embeddings
ollama list` showed the correct model, but the host endpoint refused connections.
The configured port binding existed while active networks and published ports were
empty. Inspect only the relevant networking and volume metadata:

```bash
docker inspect --format 'NetworkMode={{.HostConfig.NetworkMode}} Bindings={{json .HostConfig.PortBindings}} Ports={{json .NetworkSettings.Ports}}' fredi-embeddings
docker inspect --format '{{json .NetworkSettings.Networks}}' fredi-embeddings
docker inspect --format '{{json .Mounts}}' fredi-embeddings
```

If bridge mode is configured, `Networks={}` and the expected host binding is still
present, reconnect the existing container and verify the endpoint:

```bash
docker network connect bridge fredi-embeddings
docker port fredi-embeddings
curl -fsS http://192.168.40.19:11434/api/version
```

If it still fails, restart only `fredi-embeddings` and repeat the version check.
The model lives in the `fredi-embedding-models` named volume. No model download,
container recreation or change to the vLLM service is required for this recovery.
The user confirmed version 0.32.1, and the subsequent full synthetic backend probe
passed with the original embedding digest.
