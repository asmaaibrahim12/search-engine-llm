"""Unit tests for filter-input cleaners in app/main.py.

These run without DB or model — they just exercise the pure-Python
input-normalization helpers that gate the /search and /summary endpoints.
"""
from __future__ import annotations

from app.main import VALID_ITEM_TYPES, _clean_item_types, _clean_tags


def test_clean_tags_strips_and_lowercases():
    assert _clean_tags(["  Hiking  ", "BOOTS"]) == ["hiking", "boots"]


def test_clean_tags_drops_blanks_and_empties():
    assert _clean_tags(["", "  ", "hiking", None]) == ["hiking"]


def test_clean_tags_none_and_empty_list_return_none():
    assert _clean_tags(None) is None
    assert _clean_tags([]) is None
    # All blanks also collapses to None so the SQL sees no filter
    assert _clean_tags(["", "  "]) is None


def test_clean_tags_caps_at_ten():
    out = _clean_tags([f"t{i}" for i in range(50)])
    assert out is not None
    assert len(out) == 10


def test_clean_item_types_rejects_invalid_values():
    assert _clean_item_types(["question", "answer"]) == ["question", "answer"]
    assert _clean_item_types(["question", "garbage"]) == ["question"]
    assert _clean_item_types(["nonsense"]) is None
    assert _clean_item_types([]) is None
    assert _clean_item_types(None) is None


def test_valid_item_types_constant_is_exactly_those_two():
    # Regression guard: if a third type is added without thought, e.g.
    # "comment", the filter UI needs a matching checkbox.
    assert VALID_ITEM_TYPES == {"question", "answer"}
