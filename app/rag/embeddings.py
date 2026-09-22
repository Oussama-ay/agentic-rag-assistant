from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"


class LocalEmbeddings(Embeddings):
    """Adapt FastEmbed's local model to LangChain's embedding interface."""

    def __init__(self) -> None:
        self.model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_CACHE),
            threads=2,
            cuda=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()
