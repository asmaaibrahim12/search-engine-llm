# Semantic search app

- Semantic search over the outdoors Q&A dataset (~5,331 titles)
- `sentence-transformers` (`roberta-base-nli-stsb-mean-tokens`, 768-dim) for
  embedding
- Postgres + `pgvector` for storage and similarity search
- FastAPI + HTMX frontend
- Claude Opus 4.7 streams a cited summary per query

## Local development

```bash
# 1. Postgres with pgvector
docker run --rm -d --name pgv -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=outdoors \
  pgvector/pgvector:pg16

# 2. Python deps
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Config
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY and DATABASE_URL

# 4. Index the data (~2-3 min on CPU)
python scripts/index_outdoors.py data/outdoors/posts.csv

# 5. Run the app
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Tests

```bash
# Fast tier (no DB, no model)
pytest tests/test_rag.py tests/test_embeddings.py

# Full tier (requires Postgres + pgvector + the model)
export TEST_DATABASE_URL=postgres://postgres:postgres@localhost:5432/outdoors
export RUN_MODEL_TESTS=1
pytest
```

CI runs both tiers — see `.github/workflows/ci.yml`.

## Deploy to Railway

1. Push this repo to GitHub and connect it to a new Railway project.
2. Railway detects the `Dockerfile` and builds the web service.
3. Add the **Postgres** plugin to the project — this injects `DATABASE_URL`
   into the web service env.
4. In the Railway dashboard, set `ANTHROPIC_API_KEY`.
5. Run the indexer once (pick one):
   - Locally against the Railway Postgres URL:
     `DATABASE_URL=... python scripts/index_outdoors.py`
   - Or as a Railway one-off job.
6. Hit the public URL.

**Memory note:** torch + the roberta model sit around ~1.5 GB resident. Pick a
Railway plan with at least 2 GB of RAM.

## Repo layout

```
app/
├── main.py            # FastAPI routes, startup hook
├── embeddings.py      # SentenceTransformer wrapper
├── search.py          # pgvector similarity query
├── rag.py             # prompt builder + Claude streaming
├── db.py              # asyncpg pool + schema
└── templates/         # index.html + results.html (HTMX)
scripts/
└── index_outdoors.py  # one-time indexing
tests/
rag_semantic_search.ipynb   # reference notebook
Dockerfile
railway.toml
```
