"""One-line JSON logger.

Configured once from main.lifespan so every module can just do
`log = logging.getLogger("search")` and emit structured records that
Railway / any log aggregator will index. No deps beyond stdlib.

Kept intentionally tiny: one record per line, ISO-8601 timestamp, no
multiline tracebacks (we include exc_info as a single-line string so a
grep-based workflow doesn't fall apart).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

LOGGER_NAME = "search"

_STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                  .isoformat(timespec="milliseconds")
                  .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Any custom kwargs passed as extra= land on the record as
        # attributes; surface them alongside the message.
        for k, v in record.__dict__.items():
            if k in _STANDARD_ATTRS or k.startswith("_"):
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info).replace("\n", " \\n ")
        return json.dumps(payload, default=str)


def configure() -> logging.Logger:
    """Idempotent. Call this once from startup; subsequent calls are no-ops.

    Log level via LOG_LEVEL env (default INFO). Writes to stderr so uvicorn
    interleaving stays sensible on Railway.
    """
    log = logging.getLogger(LOGGER_NAME)
    # Idempotency: if we've already added a handler, bail. Simpler and
    # more obvious than stashing a sentinel attribute on the Logger.
    if log.handlers:
        return log
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    log.addHandler(handler)
    log.propagate = False
    return log


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
