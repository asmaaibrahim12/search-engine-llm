from __future__ import annotations

from app.text import strip_html


def test_strip_html_removes_tags():
    assert strip_html("<p>hello <b>world</b></p>") == "hello world"


def test_strip_html_decodes_entities():
    """Inputs that are already HTML-escaped (e.g. pulled from a DB that
    double-escaped on insert) should round-trip cleanly."""
    assert strip_html("&lt;p&gt;hello&lt;/p&gt;") == "hello"
    assert strip_html("A &amp; B") == "A & B"
    assert strip_html("&#xA;trim&#xA;") == "trim"


def test_strip_html_collapses_whitespace():
    """Block-level tags leave gaps; we replace them with a single space."""
    assert strip_html("<p>a</p><p>b</p>") == "a b"
    assert strip_html("x\n\n\n   y") == "x y"


def test_strip_html_handles_none_and_empty():
    assert strip_html(None) == ""
    assert strip_html("") == ""
    assert strip_html("   ") == ""


def test_strip_html_preserves_plain_text():
    assert strip_html("just plain text") == "just plain text"


def test_strip_html_removes_nested_tags():
    html = "<div><p>hi <span class='x'><em>there</em></span></p></div>"
    assert strip_html(html) == "hi there"


def test_strip_html_leaves_punctuation_alone():
    assert strip_html("<p>It's &ldquo;nice&rdquo;.</p>") == "It's “nice”."


def test_strip_html_handles_incomplete_markup_gracefully():
    """Malformed HTML shouldn't blow up — best effort."""
    # Unclosed tag — regex strips from '<' to nearest '>' which may eat
    # more than intended, but at minimum it must not raise.
    out = strip_html("<p>hello <unclosed")
    assert "hello" in out


def test_strip_html_realistic_stackexchange_body():
    """Simulates what's actually in the DB."""
    body = (
        "<p>I've always used any old shoes for hiking. "
        "Are there any real benefits to using specially made boots? "
        "</p>&#xA;"
    )
    out = strip_html(body)
    assert "I've always used any old shoes for hiking." in out
    assert "<" not in out
    assert "&" not in out
