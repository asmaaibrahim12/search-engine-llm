from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

from app.db import create_pool, ensure_schema
from tests.conftest import TEST_DATABASE_URL, needs_db, needs_model

pytestmark = [pytest.mark.asyncio, needs_db, needs_model]


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(textwrap.dedent(rows).lstrip())
    return path


async def _truncate():
    pool = await create_pool(TEST_DATABASE_URL)
    await ensure_schema(pool)
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE outdoors")
    await pool.close()


async def test_index_outdoors_populates_table(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,title,body
        1,minimalist shoes,thin soles
        2,winter tent,snow-rated
        3,climbing ropes,dynamic rope
        """,
    )

    from scripts.index_outdoors import index

    count = await index(csv)
    assert count == 3

    pool = await create_pool(TEST_DATABASE_URL)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, content_embedding FROM outdoors ORDER BY id"
        )
    await pool.close()

    assert [r["id"] for r in rows] == [1, 2, 3]
    for row in rows:
        vec = list(row["content_embedding"])
        assert len(vec) == 768
        assert math.isclose(
            math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-4
        )


async def test_index_outdoors_idempotent(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,title,body
        1,hiking boots,ankle support
        2,trail shoes,lightweight
        """,
    )

    from scripts.index_outdoors import index

    await index(csv)
    await index(csv)  # second run should upsert, not duplicate

    pool = await create_pool(TEST_DATABASE_URL)
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM outdoors")
    await pool.close()
    assert count == 2


async def test_index_outdoors_skips_null_titles(tmp_path, monkeypatch):
    await _truncate()
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)

    csv = _write_csv(
        tmp_path / "posts.csv",
        """\
        id,title,body
        1,valid title,body 1
        2,,body 2
        3,   ,body 3
        4,another valid,body 4
        """,
    )

    from scripts.index_outdoors import index

    count = await index(csv)
    assert count == 2
