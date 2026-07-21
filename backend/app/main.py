from fastapi import FastAPI

app = FastAPI(
    title="Multiagent AI Learning Platform",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}