import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


DB_PATH = Path("data/arxiv.db")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


def _read_papers(limit: int, offset: int) -> list[dict]:
    sql = """
    SELECT
        arxiv_id,
        title,
        primary_category AS category,
        submitted_year AS year,
        CASE WHEN lower(pub_status) = 'published' THEN 'Published' ELSE 'Preprint' END AS pub_status,
        first_author
    FROM papers
    ORDER BY submitted_year DESC, arxiv_id
    LIMIT ? OFFSET ?
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        try:
            rows = conn.execute(sql, (limit, offset)).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError("papers table missing. Run clean.sql first.") from exc

    return [
        {
            "arxiv_id": str(row[0]),
            "title": str(row[1]),
            "category": str(row[2]),
            "year": int(row[3]),
            "pub_status": str(row[4]),
            "first_author": str(row[5]),
        }
        for row in rows
    ]


app = FastAPI(title="ArXiv RAG Server", version="1.0.0")

pipeline = RAGPipeline(
    db_path=str(DB_PATH),
    persist_dir="vector_store",
    collection_name="papers_chunks",
    chunk_size=200,
    chunk_overlap=40,
    embedding_backend=os.getenv("EMBEDDING_BACKEND", "openrouter"),
    local_model_name=os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-minilm-l6-v2"),
    openrouter_model=os.getenv("OPENROUTER_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
)


def _safe_index_size() -> int:
    try:
        return int(pipeline.index_size())
    except Exception:
        return 0


@app.on_event("startup")
def startup_event() -> None:
    # Keep startup fast by default; can be enabled with AUTO_BUILD_INDEX=true.
    auto_build = os.getenv("AUTO_BUILD_INDEX", "false").lower() in {"1", "true", "yes"}
    if auto_build and _safe_index_size() == 0:
        try:
            summary = pipeline.build_index(reset=False)
            print(f"Startup index build complete: {summary}")
        except Exception as exc:
            print(f"Startup index build skipped due to error: {exc}")
    else:
        print("API startup complete. Index will be built on first query.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "db_path": str(DB_PATH),
        "index_chunks": _safe_index_size(),
        "embedding_backend": pipeline.embedding_backend,
        "embedding_model": pipeline.openrouter_model if pipeline.embedding_backend == "openrouter" else pipeline.local_model_name,
        "persist_dir": str(pipeline.persist_dir),
    }


@app.get("/papers")
def papers(limit: int = Query(default=20, ge=1, le=200), offset: int = Query(default=0, ge=0)) -> dict:
    try:
        items = _read_papers(limit=limit, offset=offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(items), "items": items}


@app.post("/query")
def query(req: QueryRequest) -> list[dict]:
    try:
        if _safe_index_size() == 0:
            pipeline.build_index(reset=False)
        results = pipeline.retrieve(query=req.query, top_k=req.top_k)
        return results
    except Exception as exc:
        print(f"Error during query: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(exc)}") from exc
