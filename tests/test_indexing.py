from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from app.db import create_pool, ensure_schema
from tests.conftest import TEST_DATABASE_URL, needs_db, needs_model

pytestmark_int = [pytest.mark.asyncio, needs_db, needs_model]


# -----------------------------------------------------------------------------
# Unit tests for pure helpers (no DB / no model)
# -----------------------------------------------------------------------------


def test_parse_tags_standard_format():
    from scripts.index_outdoors import parse_tags

    assert parse_tags("<hiking><water><safety>") == ["hiking", "water", "safety"]


def test_parse_tags_handles_empty_or_null():
    from scripts.index_outdoors import parse_tags

    assert parse_tags("") == []
    assert parse_tags(None) == []
    assert parse_tags(float("nan")) == []


def test_parse_tags_single_tag():
    from scripts.index_outdoors import parse_tags

    assert parse_tags("<hiking>") == ["hiking"]


def test_classify_item_type_from_post_type_id():
    from scripts.index_outdoors import classify_item_type

    assert classify_item_type(pd.Series({"post_type_id": 1, "title": "Q"})) == "question"
    assert classify_item_type(pd.Series({"post_type_id": 2, "title": None})) == "answer"


def test_classify_item_type_falls_back_on_title():
    """When PostTypeId is missing, presence of a non-empty title means question."""
    from scripts.index_outdoors import classify_item_type

    assert classify_item_type(pd.Series({"post_type_id": None, "title": "Some Q"})) == "question"
    assert classify_item_type(pd.Series({"post_type_id": None, "title": None})) == "answer"
    assert classify_item_type(pd.Series({"post_type_id": None, "title": "   "})) == "answer"


def test_normalize_columns_accepts_pascal_case():
    from scripts.index_outdoors import normalize_columns

    df = pd.DataFrame({
        "Id": [1, 2],
        "PostTypeId": [1, 2],
        "ParentId": [None, 1],
        "Title": ["Q", None],
        "Body": ["qbody", "abody"],
        "Score": [5, 3],
        "Tags": ["<hiking>", None],
    })
    out = normalize_columns(df)
    assert list(out.columns) >= ["id", "post_type_id", "parent_id", "title", "body", "score", "tags_raw"]
    assert out.loc[0, "post_type_id"] == 1
    assert out.loc[1, "tags_raw"] is None or pd.isna(out.loc[1, "tags_raw"])


def test_normalize_columns_backfills_missing_columns():
    from scripts.index_outdoors import normalize_columns

    df = pd.DataFrame({"id": [1, 2], "title": ["a", "b"], "body": ["x", "y"]})
    out = normalize_columns(df)
    # missing PostTypeId, Score, ParentId, Tags, AcceptedAnswerId all backfilled
    assert "post_type_id" in out.columns
    assert "score" in out.columns
    assert "tags_raw" in out.columns


def test_normalize_columns_errors_without_id():
    from scripts.index_outdoors import normalize_columns

    df = pd.DataFrame({"title": ["a"]})
    with pytest.raises(ValueError, match="id"):
        normalize_columns(df)


# -----------------------------------------------------------------------------
# Integration tests (need DB + model)
# -----------------------------------------------------------------------------


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(textwrap.dedent(rows).lstrip())
    return path


async def _truncate():
    pool = await create_pool(TEST_DATABASE_URL)
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
    await pool.close()


@pytest.mark.asyncio
@needs_db
@needs_model
async def test_index_populates_both_questions_and_answers(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        Id,PostTypeId,ParentId,Title,Body,Score,AcceptedAnswerId,Tags
        1,1,,What are minimalist shoes?,Thin soles...,10,3,<hiking><footwear>
        2,2,1,,You want <5mm drop...,4,,
        3,2,1,,Minimalist means barefoot-style...,15,,
        4,1,,How to purify water?,Boil or filter...,6,,<water><safety>
        5,2,4,,Iodine works...,2,,
        """,
    )

    from scripts.index_outdoors import index

    count = await index(csv)
    assert count == 5

    pool = await create_pool(TEST_DATABASE_URL)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, item_type, parent_id, score, is_accepted, tags "
            "FROM outdoors ORDER BY id"
        )
    await pool.close()

    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["item_type"] == "question"
    assert by_id[2]["item_type"] == "answer"
    assert by_id[2]["parent_id"] == 1
    # Answer 3 was marked accepted by question 1's AcceptedAnswerId=3
    assert by_id[3]["is_accepted"] is True
    assert by_id[2]["is_accepted"] is False
    assert by_id[1]["score"] == 10
    # Answers inherit parent question tags
    assert "hiking" in list(by_id[2]["tags"])


@pytest.mark.asyncio
@needs_db
@needs_model
async def test_index_idempotent_reindex_keeps_same_count(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,post_type_id,title,body
        1,1,Boots,ankle support
        2,2,,answer text
        """,
    )

    from scripts.index_outdoors import index

    await index(csv)
    await index(csv)  # second run upserts, not duplicates

    pool = await create_pool(TEST_DATABASE_URL)
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM outdoors")
    await pool.close()
    assert count == 2


@pytest.mark.asyncio
@needs_db
@needs_model
async def test_index_skips_rows_with_no_content(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,post_type_id,title,body
        1,1,valid title,valid body
        2,2,,""
        3,2,,
        4,1,another,valid
        """,
    )

    from scripts.index_outdoors import index

    count = await index(csv)
    assert count == 2  # rows 2 and 3 skipped (no title AND no body)


@pytest.mark.asyncio
@needs_db
@needs_model
async def test_index_embeddings_are_normalized(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,post_type_id,title,body
        1,1,hiking boots,ankle support
        """,
    )

    from scripts.index_outdoors import index

    await index(csv)

    pool = await create_pool(TEST_DATABASE_URL)
    async with pool.acquire() as conn:
        vec = await conn.fetchval("SELECT content_embedding FROM outdoors WHERE id = 1")
    await pool.close()

    values = list(vec)
    assert len(values) == 768
    norm = math.sqrt(sum(x * x for x in values))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)
