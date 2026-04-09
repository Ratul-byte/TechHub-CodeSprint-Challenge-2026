import os
import sqlite3
import time
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    category: str
    year: int
    pub_status: str
    first_author: str


class RAGPipeline:
    def __init__(
        self,
        db_path: str = "data/arxiv.db",
        persist_dir: str = "vector_store",
        collection_name: str = "papers_chunks",
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        embedding_backend: str = "local",
        local_model_name: str = "sentence-transformers/all-minilm-l6-v2",
        openrouter_model: str = "text-embedding-3-small",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_api_key: str | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.db_path = Path(db_path)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_backend = embedding_backend.lower()

        self.local_model_name = local_model_name
        self.openrouter_model = openrouter_model
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        self._local_embedder = None

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _connect_db(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def _fetch_papers(self) -> list[PaperRecord]:
        sql = """
        SELECT
            arxiv_id,
            title,
            abstract,
            primary_category,
            submitted_year,
            pub_status,
            first_author
        FROM papers
        """
        with self._connect_db() as conn:
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    "Failed to read papers table. Run clean.sql first so papers exists."
                ) from exc

        results: list[PaperRecord] = []
        for row in rows:
            arxiv_id = str(row[0]).strip()
            title = str(row[1] or "").strip()
            abstract = str(row[2] or "").strip()
            category = str(row[3] or "").strip()
            try:
                year = int(row[4])
            except (TypeError, ValueError):
                continue
            raw_status = str(row[5] or "").strip().lower()
            pub_status = "Published" if raw_status == "published" else "Preprint"
            first_author = str(row[6] or "").strip()

            if not arxiv_id or not title or not abstract or not category or not first_author:
                continue

            results.append(
                PaperRecord(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    category=category,
                    year=year,
                    pub_status=pub_status,
                    first_author=first_author,
                )
            )
        return results

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start += step
        return chunks

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._local_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_embedder = SentenceTransformer(self.local_model_name)
            except Exception:
                # Torch / model import can fail on some Windows+Python setups.
                # Fall back to a deterministic hashed bag-of-words embedding.
                self._local_embedder = "hashed_fallback"

        if self._local_embedder == "hashed_fallback":
            dim = 384
            out: list[list[float]] = []
            for text in texts:
                vec = [0.0] * dim
                for token in text.lower().split():
                    token = token.strip()
                    if not token:
                        continue
                    digest = hashlib.md5(token.encode("utf-8")).digest()
                    idx = int.from_bytes(digest[:4], "little") % dim
                    vec[idx] += 1.0

                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0.0:
                    vec = [v / norm for v in vec]
                out.append(vec)
            return out

        vectors = self._local_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def _embed_openrouter(self, texts: list[str]) -> list[list[float]]:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Set it as an environment variable or pass it in."
            )
        url = f"{self.openrouter_base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        vectors: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.openrouter_model, "input": batch}
            last_error: Exception | None = None
            data: list[dict[str, Any]] = []
            for attempt in range(4):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep(1.5 * (attempt + 1))
            if last_error is not None:
                raise RuntimeError(
                    f"OpenRouter embedding request failed after retries at batch starting index {i}."
                ) from last_error
            data = sorted(data, key=lambda item: item.get("index", 0))
            vectors.extend([item["embedding"] for item in data])
        return vectors

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_backend == "local":
            return self._embed_local(texts)
        if self.embedding_backend == "openrouter":
            return self._embed_openrouter(texts)
        raise ValueError("embedding_backend must be 'local' or 'openrouter'")

    def clear_index(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, reset: bool = False) -> dict[str, int]:
        if reset:
            self.clear_index()

        papers = self._fetch_papers()
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for paper in papers:
            chunks = self._chunk_text(paper.abstract)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{paper.arxiv_id}:{idx}"
                ids.append(chunk_id)
                docs.append(chunk_text)
                metas.append(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "category": paper.category,
                        "year": int(paper.year),
                        "pub_status": paper.pub_status,
                        "first_author": paper.first_author,
                    }
                )

        if not ids:
            return {"papers": 0, "chunks": 0}

        batch_size = 256
        for i in range(0, len(ids), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_embeddings = self._embed_texts(batch_docs)
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=batch_docs,
                metadatas=metas[i : i + batch_size],
                embeddings=batch_embeddings,
            )

        return {"papers": len(papers), "chunks": len(ids)}

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        if top_k <= 0:
            top_k = 5

        query_embedding = self._embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        output: list[dict[str, Any]] = []
        for chunk_text, meta, dist in zip(docs, metas, dists):
            if not meta:
                continue
            output.append(
                {
                    "chunk_text": chunk_text,
                    "arxiv_id": str(meta.get("arxiv_id", "")),
                    "title": str(meta.get("title", "")),
                    "category": str(meta.get("category", "")),
                    "year": int(meta.get("year", 0)),
                    "pub_status": str(meta.get("pub_status", "Preprint")),
                    "first_author": str(meta.get("first_author", "")),
                    "distance": float(dist),
                }
            )
        return output

    def retrive(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.retrieve(query=query, top_k=top_k)

    def index_size(self) -> int:
        return self.collection.count()
