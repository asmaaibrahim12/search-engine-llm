"""Tests for the JSON logging setup.

Pure-Python — doesn't touch the DB or any model. Verifies the log
records round-trip through json.loads and that extra= kwargs make it
into the emitted payload.
"""
from __future__ import annotations

import io
import json
import logging

import pytest

from app.logging_setup import LOGGER_NAME, _JsonFormatter, configure


@pytest.fixture
def captured_log():
    """A fresh logger instance + a StringIO stream attached."""
    log = logging.getLogger(LOGGER_NAME)
    # Save + restore handlers to keep test isolation; configure() is
    # idempotent so we can't rely on it rebuilding state.
    saved_handlers = log.handlers[:]
    saved_level = log.level
    log.handlers.clear()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    log.addHandler(handler)
    log.setLevel("DEBUG")
    try:
        yield log, buf
    finally:
        log.handlers = saved_handlers
        log.setLevel(saved_level)


def test_json_formatter_emits_single_line_json(captured_log):
    log, buf = captured_log
    log.info("hello world")
    out = buf.getvalue().strip().splitlines()
    assert len(out) == 1
    rec = json.loads(out[0])
    assert rec["msg"] == "hello world"
    assert rec["level"] == "info"
    assert rec["logger"] == LOGGER_NAME
    assert "ts" in rec and rec["ts"].endswith("Z")


def test_extra_fields_land_in_payload(captured_log):
    log, buf = captured_log
    log.warning("refresh failed", extra={"err": "boom", "attempt": 3})
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["msg"] == "refresh failed"
    assert rec["err"] == "boom"
    assert rec["attempt"] == 3


def test_configure_is_idempotent():
    log_a = configure()
    log_b = configure()
    # Same instance, handler not duplicated on second call.
    assert log_a is log_b
    assert len(log_a.handlers) == 1


def test_formatter_tolerates_exc_info(captured_log):
    log, buf = captured_log
    try:
        raise ValueError("nope")
    except ValueError:
        log.exception("caught it")
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["msg"] == "caught it"
    assert "exc" in rec
    # Single line — our formatter flattens newlines in the traceback.
    assert "\n" not in rec["exc"]
