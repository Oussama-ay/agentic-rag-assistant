from pathlib import Path

from langchain_core.documents import Document


def load_text(path: str | Path) -> Document:
    """Read one UTF-8 text file and retain its filename for future citations."""
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported at this milestone.")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("The document must contain non-whitespace text.")

    return Document(page_content=text, metadata={"source": path.name})
