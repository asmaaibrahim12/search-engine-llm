"""One-off indexing job: load the outdoors CSV, embed titles, upsert into pgvector.

Run locally once, or as a Railway one-off job, before the app can serve queries.

    python scripts/index_outdoors.py path/to/posts.csv

If no path is given, reads OUTDOORS_CSV_PATH or falls back to data/outdoors/posts.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import create_pool, ensure_schema  # noqa: E402
from app.embeddings import embed_batch_passages  # noqa: E402

# Upper bound on body length we feed into the embedder. Bodies longer than
# this tend to drag the embedding away from the core topic and waste tokens.
BODY_TRUNCATE = 1000

UPSERT_SQL = """
INSERT INTO outdoors (id, title, body, content_embedding)
VALUES ($1, $2, $3, $4)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    content_embedding = EXCLUDED.content_embedding
"""


def load_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    if "body" not in df.columns:
        df["body"] = ""
    df["body"] = df["body"].fillna("")
    return df[["id", "title", "body"]].reset_index(drop=True)


def embedding_text(title: str, body: str) -> str:
    """Build the text we actually feed the encoder.

    Title is repeated once to bias the embedding toward title words (helps on
    short-question datasets like this one), then a truncated body is
    appended for topical coverage."""
    body_trimmed = (body or "").strip()[:BODY_TRUNCATE]
    return f"{title}\n\n{title}\n\n{body_trimmed}".strip()


async def index(csv_path: Path) -> int:
    df = load_rows(csv_path)
    texts = [embedding_text(row.title, row.body) for row in df.itertuples(index=False)]
    print(f"Encoding {len(texts)} documents (title + body)...", flush=True)
    embeddings = embed_batch_passages(texts)

    pool = await create_pool()
    await ensure_schema(pool)
    try:
        records = [
            (
                int(row.id),
                row.title,
                row.body,
                np.array(embeddings[i], dtype=np.float32),
            )
            for i, row in enumerate(df.itertuples(index=False))
        ]
        async with pool.acquire() as conn:
            await conn.executemany(UPSERT_SQL, records)
    finally:
        await pool.close()

    return len(df)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.environ.get("OUTDOORS_CSV_PATH", "data/outdoors/posts.csv"),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    count = asyncio.run(index(csv_path))
    print(f"Indexed {count} rows.")


if __name__ == "__main__":
    main()
