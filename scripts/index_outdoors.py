"""One-off indexing job: load the outdoors CSV, embed questions + answers,
upsert into pgvector.

Run locally once (or as a Railway one-off job) before the app can serve queries.

    python scripts/index_outdoors.py path/to/posts.csv

If no path is given, reads OUTDOORS_CSV_PATH or falls back to
data/outdoors/posts.csv.

The script handles the full Stack Exchange Posts schema when available
(PostTypeId, ParentId, Score, AcceptedAnswerId, Tags) and falls back to
"title present -> question, title missing -> answer" when those columns
aren't in the CSV.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import create_pool, ensure_schema  # noqa: E402
from app.embeddings import embed_batch_passages  # noqa: E402

# Upper bound on body length we feed into the embedder. Bodies longer than
# this drag the embedding away from the core topic and waste tokens.
BODY_TRUNCATE = 1000

UPSERT_SQL = """
INSERT INTO outdoors
    (id, parent_id, item_type, title, body, score, is_accepted, tags,
     content_embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (id) DO UPDATE SET
    parent_id = EXCLUDED.parent_id,
    item_type = EXCLUDED.item_type,
    title = EXCLUDED.title,
    body = EXCLUDED.body,
    score = EXCLUDED.score,
    is_accepted = EXCLUDED.is_accepted,
    tags = EXCLUDED.tags,
    content_embedding = EXCLUDED.content_embedding
"""


# Stack Exchange stores tags as "<tag1><tag2><tag3>" in the Tags column.
_TAG_RE = re.compile(r"<([^<>]+)>")


def parse_tags(raw) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    return _TAG_RE.findall(raw)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept CSVs that use either PascalCase (SE dump) or lowercase names,
    and backfill missing columns with sensible defaults.

    Returns a DataFrame with canonical lowercase columns:
        id, post_type_id, parent_id, title, body, score, accepted_answer_id, tags
    """
    lower = {c.lower(): c for c in df.columns}

    def pick(*aliases, default=None):
        for a in aliases:
            if a in lower:
                return df[lower[a]]
        return default

    id_col = pick("id")
    if id_col is None:
        raise ValueError(f"CSV has no 'id' column. Columns: {list(df.columns)}")

    out = pd.DataFrame({"id": id_col})
    out["post_type_id"] = pick("posttypeid", "post_type_id", default=pd.Series([None] * len(df)))
    out["parent_id"]    = pick("parentid",    "parent_id",    default=pd.Series([None] * len(df)))
    out["title"]        = pick("title",       default=pd.Series([None] * len(df)))
    out["body"]         = pick("body",        default=pd.Series([""] * len(df)))
    out["score"]        = pick("score",       default=pd.Series([0] * len(df)))
    out["accepted_answer_id"] = pick(
        "acceptedanswerid", "accepted_answer_id", default=pd.Series([None] * len(df))
    )
    out["tags_raw"]     = pick("tags",        default=pd.Series([""] * len(df)))
    return out


def classify_item_type(row: pd.Series) -> str:
    """Question if PostTypeId == 1 (or null + has title), else answer."""
    pt = row.get("post_type_id")
    if pd.notna(pt):
        return "question" if int(pt) == 1 else "answer"
    # Fallback when the column isn't present: title → question, no title → answer.
    return "question" if pd.notna(row.get("title")) and str(row["title"]).strip() else "answer"


def load_rows(csv_path: Path) -> tuple[pd.DataFrame, set[int]]:
    """Return (items_df, accepted_answer_ids).

    items_df has both questions and answers, with body coerced to str and tags
    parsed into a list. accepted_answer_ids is the set of answer IDs marked
    as accepted by some question; used downstream to set is_accepted per row.
    """
    raw = pd.read_csv(csv_path)
    df = normalize_columns(raw)
    df["body"] = df["body"].fillna("").astype(str)
    df["item_type"] = df.apply(classify_item_type, axis=1)
    df["tags"] = df["tags_raw"].apply(parse_tags)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

    # Drop rows with no usable content.
    has_text = df["body"].str.strip().ne("") | df["title"].fillna("").astype(str).str.strip().ne("")
    df = df[has_text].reset_index(drop=True)

    accepted = set()
    q_mask = df["item_type"] == "question"
    for aid in df.loc[q_mask, "accepted_answer_id"].dropna():
        try:
            accepted.add(int(aid))
        except (TypeError, ValueError):
            pass

    # Answers inherit their parent question's tags (nice for tag-filtering
    # answers directly), then propagate down so every row has a tag list.
    q_tags = dict(zip(df.loc[q_mask, "id"].astype(int), df.loc[q_mask, "tags"]))
    def effective_tags(row):
        if row["item_type"] == "answer" and pd.notna(row["parent_id"]):
            return q_tags.get(int(row["parent_id"]), row["tags"])
        return row["tags"]
    df["tags"] = df.apply(effective_tags, axis=1)

    return df, accepted


def embedding_text(row: pd.Series, question_titles: dict[int, str]) -> str:
    """Build the text we feed the encoder for one row.

    Question: `title\n\ntitle\n\nbody[:1000]` (title repeated to emphasize).
    Answer:   `<parent question title>\n\nbody[:1000]` (answer embedded with
              the question context — makes retrieval match the answer to
              queries that would otherwise only match the question).
    """
    body = (row["body"] or "").strip()[:BODY_TRUNCATE]
    if row["item_type"] == "question":
        title = (row["title"] or "").strip()
        return f"{title}\n\n{title}\n\n{body}".strip()
    parent_title = ""
    if pd.notna(row.get("parent_id")):
        parent_title = question_titles.get(int(row["parent_id"]), "") or ""
    return f"{parent_title}\n\n{body}".strip()


async def index(csv_path: Path) -> int:
    df, accepted_ids = load_rows(csv_path)
    n_questions = int((df["item_type"] == "question").sum())
    n_answers = int((df["item_type"] == "answer").sum())
    print(
        f"Loaded {len(df)} rows ({n_questions} questions, {n_answers} answers)",
        flush=True,
    )

    question_titles = dict(
        zip(
            df.loc[df["item_type"] == "question", "id"].astype(int),
            df.loc[df["item_type"] == "question", "title"].fillna(""),
        )
    )

    texts = [embedding_text(row, question_titles) for _, row in df.iterrows()]
    print(f"Encoding {len(texts)} documents...", flush=True)
    embeddings = embed_batch_passages(texts)

    records = []
    for i, row in df.iterrows():
        rid = int(row["id"])
        pid = int(row["parent_id"]) if pd.notna(row.get("parent_id")) else None
        title = row["title"] if pd.notna(row["title"]) and str(row["title"]).strip() else None
        is_accepted = rid in accepted_ids
        records.append((
            rid,
            pid,
            row["item_type"],
            title,
            row["body"] or "",
            int(row["score"]),
            is_accepted,
            list(row["tags"]),
            np.array(embeddings[i], dtype=np.float32),
        ))

    pool = await create_pool()
    await ensure_schema(pool)
    try:
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
