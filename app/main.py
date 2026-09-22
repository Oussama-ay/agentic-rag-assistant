from fastapi import FastAPI

app = FastAPI(title="RAGent", description="Agentic Document Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
