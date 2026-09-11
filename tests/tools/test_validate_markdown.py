"""Tests for the manuscript-validation contract (quadmath/scripts/validate_markdown.py)."""
from __future__ import annotations

from validate_markdown import (
    collect_symbols,
    strip_code_fences,
    validate_images,
    validate_math,
    validate_refs,
)


def test_strip_code_fences_removes_fenced_content():
    text = (
        "before\n"
        "```latex\n"
        "![img](../output/figures/x.png)\n"
        "```\n"
        "after\n"
        "~~~\n"
        "also hidden\n"
        "~~~\n"
        "end"
    )
    stripped = strip_code_fences(text)
    assert "![img]" not in stripped
    assert "also hidden" not in stripped
    assert "before" in stripped and "after" in stripped and "end" in stripped


def test_validate_images_missing_and_present(tmp_path):
    figs = tmp_path / "figures"
    figs.mkdir()
    (figs / "real.png").write_bytes(b"png")
    good = tmp_path / "good.md"
    good.write_text("![Figure](figures/real.png)\n", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("![Figure](figures/missing.png)\n", encoding="utf-8")
    assert validate_images([str(good)], str(tmp_path)) == []
    problems = validate_images([str(bad)], str(tmp_path))
    assert len(problems) == 1
    assert "missing.png" in problems[0] and "Missing image" in problems[0]


def test_validate_images_ignores_code_fences(tmp_path):
    hidden = tmp_path / "hidden.md"
    hidden.write_text(
        "```\n![Fence example](figures/none.png)\n```\n", encoding="utf-8"
    )
    assert validate_images([str(hidden)], str(tmp_path)) == []


def test_validate_refs_flags_bare_and_uninformative_links(tmp_path):
    md = tmp_path / "refs.md"
    md.write_text(
        "See \\eqref{eq:known} and [anchor](#sec-anchor).\n"
        "Refer to https://example.com plainly.\n"
        "[https://example.com/doc](https://example.com/doc) is uninformative.\n",
        encoding="utf-8",
    )
    problems = validate_refs([str(md)], {"eq:known"}, {"sec-anchor"}, str(tmp_path))
    assert any("Bare URL" in p for p in problems)
    assert any("Non-informative link text" in p for p in problems)
    # Known label and anchor must not be reported missing
    assert not any("Missing equation label" in p for p in problems)
    assert not any("Missing anchor/label" in p for p in problems)


def test_validate_refs_missing_label(tmp_path):
    md = tmp_path / "bad_ref.md"
    md.write_text("See \\eqref{eq:missing}.\n", encoding="utf-8")
    problems = validate_refs([str(md)], set(), set(), str(tmp_path))
    assert any("Missing equation label" in p and "eq:missing" in p for p in problems)


def test_validate_math_rules(tmp_path):
    ok = tmp_path / "ok.md"
    ok.write_text(
        "\\begin{equation}\n\\label{eq:a}\nE = mc^2\n\\end{equation}\n",
        encoding="utf-8",
    )
    assert validate_math([str(ok)], str(tmp_path)) == []

    dollar = tmp_path / "dollar.md"
    dollar.write_text("$$E = mc^2$$\n", encoding="utf-8")
    assert any("$$" in p for p in validate_math([str(dollar)], str(tmp_path)))

    no_label = tmp_path / "nolabel.md"
    no_label.write_text(
        "\\begin{equation}\nE = mc^2\n\\end{equation}\n", encoding="utf-8"
    )
    assert any("missing" in p.lower() and "label" in p.lower()
               for p in validate_math([str(no_label)], str(tmp_path)))

    a = tmp_path / "a.md"
    a.write_text(
        "\\begin{equation}\n\\label{eq:dup}\nx\n\\end{equation}\n", encoding="utf-8"
    )
    b = tmp_path / "b.md"
    b.write_text(
        "\\begin{equation}\n\\label{eq:dup}\ny\n\\end{equation}\n", encoding="utf-8"
    )
    assert any("Duplicate" in p for p in validate_math([str(a), str(b)], str(tmp_path)))


def test_collect_symbols(tmp_path):
    md = tmp_path / "sym.md"
    md.write_text("\\label{eq:one}\n{#sec-two}\n", encoding="utf-8")
    labels, anchors = collect_symbols([str(md)])
    assert labels == {"eq:one"}
    assert anchors == {"sec-two"}
