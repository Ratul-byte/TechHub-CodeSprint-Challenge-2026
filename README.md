# TechHub CodeSprint Challenge 2026 — ArXiv RAG Pipeline

![CodeSprint Certificate](CodeSprint%20Certificate.jpeg)

## Overview

This project is a **Retrieval-Augmented Generation (RAG) pipeline** built over a sampled subset of the [arXiv](https://arxiv.org/) academic paper metadata. It was developed as a submission for the **TechHub CodeSprint Challenge 2026**.

The system ingests arXiv paper metadata from a CSV snapshot, stores it in SQLite, builds a vector index for semantic search, and exposes a REST API to query papers by natural language questions. It also includes visualization utilities and a batch question-answering runner.

---

## Project Structure

```
.
├── ingest.py           # Ingest arXiv CSV into SQLite + JSON
├── clean.sql           # SQL script to clean/normalize raw data into analytical tables
├── rag_pipeline.py     # Core RAG pipeline (ChromaDB vector store + embedding support)
├── server.py           # FastAPI REST API server
├── query_runner.py     # Batch question answering runner
├── visualize.py        # Generate plots from the SQLite database
├── questions.json      # Sample questions with grading criteria
├── answer.json         # Pre-generated answers from the pipeline
└── requirements.txt    # Python dependencies
```

---

## Features

- **Data ingestion**: Load arXiv paper metadata from CSV, filter by category, and persist to SQLite and JSON.
- **Data cleaning**: Deduplicate papers, extract submission year, count authors, compute abstract word counts, and classify publication status via a single SQL script.
- **RAG pipeline**: Chunk paper abstracts and embed them into a ChromaDB vector store. Supports two embedding backends:
  - `local` — [sentence-transformers](https://www.sbert.net/) (runs fully offline)
  - `openrouter` — [OpenRouter](https://openrouter.ai/) API (e.g., `nomic-ai/nomic-embed-text-v1.5`)
- **REST API**: FastAPI server with endpoints to list papers, run semantic queries, and check system health.
- **Batch Q&A**: Run a list of natural-language questions against the pipeline and save structured answers to `answer.json`.
- **Visualizations**: Generate four matplotlib charts summarising the dataset.

---

## Setup

### Prerequisites

- Python 3.10+
- `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

For **local embeddings**, no additional setup is needed — the model is downloaded automatically on first use.

For **OpenRouter embeddings**, set your API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

---

## Usage

### 1 — Ingest data

Place the arXiv CSV snapshot (`sampled-arxiv-metadata-oai-snapshot.csv`) in the project root, then run:

```bash
python ingest.py --input sampled-arxiv-metadata-oai-snapshot.csv --output-dir data
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | `sampled-arxiv-metadata-oai-snapshot.csv` | Path to input CSV |
| `--output-dir` | `data` | Directory for outputs (`papers_raw.json`, `arxiv.db`) |
| `--categories` | `cs.AI cs.LG cs.CL stat.ML cs.CV` | Space-separated list of arXiv categories to keep |
| `--sample-size` | — | Fixed number of rows to sample |
| `--sample-frac` | — | Fraction of rows to sample (0, 1] |
| `--random-state` | `42` | Random seed for reproducibility |

### 2 — Clean the database

```bash
sqlite3 data/arxiv.db < clean.sql
```

This creates five tables: `papers`, `category_stats`, `yearly_trends`, `publication_status`, and `author_stats`.

### 3 — Start the API server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Set `AUTO_BUILD_INDEX=true` to build the vector index automatically on startup:

```bash
AUTO_BUILD_INDEX=true uvicorn server:app --host 0.0.0.0 --port 8000
```

Override the embedding backend or model with environment variables:

```bash
EMBEDDING_BACKEND=local LOCAL_EMBED_MODEL=sentence-transformers/all-minilm-l6-v2 uvicorn server:app
```

#### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health, index size, and config |
| `GET` | `/papers?limit=20&offset=0` | Paginated list of papers |
| `POST` | `/query` | Semantic search over abstracts |

**Query example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer models for text classification", "top_k": 5}'
```

### 4 — Run batch question answering

```bash
python query_runner.py
```

This reads `questions.json`, queries the pipeline for each question, applies optional category and year filters, and writes structured answers to `answer.json`.

Set `OPENROUTER_API_KEY` or place the key in a `temp.txt` file in the project root.

### 5 — Generate visualizations

```bash
python visualize.py
```

Plots are saved to `data/plots/`:

| File | Description |
|---|---|
| `01_papers_per_category.png` | Paper counts by category and publication status |
| `02_submission_trend_over_time.png` | Submission volume over time by category |
| `03_publication_status_breakdown.png` | Published vs preprint pie chart |
| `04_abstract_length_distribution.png` | Abstract word count distribution by category |

Use `--rebuild-clean` to re-run `clean.sql` automatically if the cleaned tables are missing.

---

## Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | API key for OpenRouter embedding requests |
| `EMBEDDING_BACKEND` | `openrouter` | `local` or `openrouter` |
| `LOCAL_EMBED_MODEL` | `sentence-transformers/all-minilm-l6-v2` | HuggingFace model name for local embeddings |
| `OPENROUTER_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | OpenRouter embedding model |
| `AUTO_BUILD_INDEX` | `false` | Build the vector index automatically at API startup |

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `chromadb` | Vector store for embeddings |
| `sentence-transformers` | Local embedding model |
| `requests` | HTTP client for OpenRouter API |

---

## License

This project was created as a competition submission for the **TechHub CodeSprint Challenge 2026**.
