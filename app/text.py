"""Tiny HTML-to-text helper.

Stack Exchange stores post bodies as HTML. We need plain text for two
reasons:
1. The LLM prompt shouldn't spend tokens on markup — it waste and
   distracts from signal.
2. The UI currently displays raw `&lt;p&gt;...` because Jinja auto-escapes
   the HTML source when it rehydrates.

Deliberately simple: `html.unescape` + a regex tag-strip + whitespace
collapse. No beautifulsoup dep.
"""
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    """Return plain text from a possibly-HTML string.

    Handles nested tags, HTML entities, and collapses whitespace. Safe on
    None / empty input.
    """
    if not text:
        return ""
    # Decode entities first (&amp; -> &, &lt; -> <) so the tag regex
    # matches real tags, not their escaped text form.
    unescaped = html.unescape(text)
    stripped = _TAG_RE.sub(" ", unescaped)
    return _WS_RE.sub(" ", stripped).strip()
