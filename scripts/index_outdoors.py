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
from app.embeddings import embed_batch  # noqa: E402

UPSERT_SQL = """
INSERT INTO outdoors (id, title, body, title_embedding)
VALUES ($1, $2, $3, $4)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    title_embedding = EXCLUDED.title_embedding
"""


def load_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    if "body" not in df.columns:
        df["body"] = ""
    df["body"] = df["body"].fillna("")
    return df[["id", "title", "body"]].reset_index(drop=True)


async def index(csv_path: Path) -> int:
    df = load_rows(csv_path)
    titles = df["title"].tolist()
    print(f"Encoding {len(titles)} titles...", flush=True)
    embeddings = embed_batch(titles)

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
