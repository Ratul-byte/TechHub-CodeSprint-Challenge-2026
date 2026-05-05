import json
import os
from pathlib import Path
from typing import Any

from rag_pipeline import RAGPipeline

INPUT_FILE = Path("questions.json")
OUTPUT_FILE = Path("answer.json")
OPENROUTER_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")
TOP_K = 8


def _normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _apply_filters(
    items: list[dict[str, Any]], category_filter: str | None, year_filter: Any
) -> list[dict[str, Any]]:
    filtered = items

    if category_filter:
        c = str(category_filter).strip()
        filtered = [x for x in filtered if str(x.get("category", "")).strip() == c]

    y = _normalize_year(year_filter)
    if y is not None:
        filtered = [x for x in filtered if _normalize_year(x.get("year")) == y]

    return filtered


def _format_sources(items: list[dict[str, Any]], max_sources: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in items:
        arxiv_id = str(row.get("arxiv_id", "")).strip()
        if not arxiv_id or arxiv_id in seen:
            continue

        seen.add(arxiv_id)
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": str(row.get("title", "")).strip(),
                "category": str(row.get("category", "")).strip(),
                "year": _normalize_year(row.get("year")),
                "pub_status": str(row.get("pub_status", "Preprint")).strip(),
                "distance": float(row.get("distance", 0.0)),
            }
        )
        if len(out) >= max_sources:
            break

    return out


def _format_answer(items: list[dict[str, Any]], max_points: int = 3) -> str:
    if not items:
        return "No relevant evidence found for this query."

    parts: list[str] = []
    for row in items[:max_points]:
        title = str(row.get("title", "Unknown Title")).strip()
        chunk = str(row.get("chunk_text", "")).strip().replace("\n", " ")
        if len(chunk) > 280:
            chunk = chunk[:277].rstrip() + "..."
        parts.append(f"{title}: {chunk}")

    return " ".join(parts)


def _render_progress(prefix: str, done: int, total: int, width: int = 30) -> None:
    total = max(total, 1)
    done = max(0, min(done, total))
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * done / total)
    print(f"\r{prefix} [{bar}] {done}/{total} ({pct}%)", end="", flush=True)
    if done == total:
        print()


def run() -> None:
    questions = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and Path("temp.txt").exists():
        api_key = Path("temp.txt").read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found. Set env var or place key in temp.txt")

    pipeline = RAGPipeline(
        db_path="data/arxiv.db",
        persist_dir="vector_store",
        collection_name="papers_chunks_openrouter",
        chunk_size=200,
        chunk_overlap=40,
        embedding_backend="openrouter",
        openrouter_model=OPENROUTER_MODEL,
        openrouter_api_key=api_key,
    )

    if pipeline.index_size() == 0:
        print("Building vector index...")
        info = pipeline.build_index(
            reset=False,
            progress_callback=lambda done, total: _render_progress("Indexing", done, total),
        )
        print(f"Built index: {info}")
    else:
        print(f"Using existing index chunks: {pipeline.index_size()} (model={OPENROUTER_MODEL})")

    answers: list[dict[str, Any]] = []

    total_questions = len(questions)
    _render_progress("Questions", 0, total_questions)

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id")
        question = str(q.get("question", "")).strip()
        category_filter = q.get("category_filter")
        year_filter = q.get("year_filter")

        query_text = question
        if category_filter:
            query_text += f" category {category_filter}"
        if year_filter:
            query_text += f" year {year_filter}"

        try:
            raw = pipeline.retrieve(query=query_text, top_k=TOP_K)
            filtered = _apply_filters(raw, category_filter, year_filter)
            if not filtered:
                filtered = raw

            answer_text = _format_answer(filtered)
            sources = _format_sources(filtered)
        except Exception as exc:
            answer_text = f"Query failed: {exc}"
            sources = []

        answers.append(
            {
                "question_id": qid,
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "model_used": OPENROUTER_MODEL,
                "category_filter": category_filter,
                "year_filter": year_filter,
            }
        )

        _render_progress("Questions", idx, total_questions)
        print(f"Processed question {qid}")

    OUTPUT_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(answers)} answers to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
