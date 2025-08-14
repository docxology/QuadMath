from __future__ import annotations

import os

from glossary_gen import build_api_index, generate_markdown_table, inject_between_markers


def test_build_api_index_includes_public_symbols():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    src_dir = os.path.abspath(src_dir)
    entries = build_api_index(src_dir)
    # Basic sanity: expect at least a few known entries
    names = {(e.module, e.name) for e in entries}
    assert ("quadray", "Quadray") in names
    assert ("information", "free_energy") in names
    assert ("linalg_utils", "bareiss_determinant_int") in names


def test_generate_markdown_table_formats_rows():
    from dataclasses import dataclass

    @dataclass
    class E:
        module: str
        name: str
        kind: str
        signature: str | None
        summary: str

    rows = [
        E("m", "f", "function", "(x)", "do f"),
        E("m", "C", "class", None, "class C"),
    ]
    table = generate_markdown_table(rows)  # type: ignore[arg-type]
    assert "| Module | Symbol | Kind | Signature | Summary |" in table
    assert "`m` | `f` | function | `(x)` | do f" in table
    assert "`m` | `C` | class | `` | class C" in table


def test_inject_between_markers_inserts_payload_when_missing_markers():
    original = "alpha"
    begin, end = "<!-- BEGIN X -->", "<!-- END X -->"
    payload = "content"
    updated = inject_between_markers(original, begin, end, payload)
    assert begin in updated and end in updated and payload in updated


def test_inject_between_markers_replaces_between_existing_markers():
    begin, end = "<!-- BEGIN X -->", "<!-- END X -->"
    text = f"pre\n{begin}\nold\n{end}\npost\n"
    updated = inject_between_markers(text, begin, end, "new")
    assert "old" not in updated and "new" in updated


