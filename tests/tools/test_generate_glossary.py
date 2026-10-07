"""Tests for the glossary regeneration contract (quadmath/scripts/generate_glossary.py)."""
from __future__ import annotations

import generate_glossary

BEGIN = "<!-- BEGIN: AUTO-API-GLOSSARY -->"
END = "<!-- END: AUTO-API-GLOSSARY -->"


def _make_repo(tmp_path, glossary_text: str):
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "mod.py").write_text('def f(x):\n    """Doc for f."""\n', encoding="utf-8")
    (repo / "quadmath" / "markdown").mkdir(parents=True)
    glossary = repo / "quadmath" / "markdown" / "10_symbols_glossary.md"
    glossary.write_text(glossary_text, encoding="utf-8")
    return repo, glossary


def test_default_run_writes_regenerated_table(tmp_path, monkeypatch):
    repo, glossary = _make_repo(tmp_path, f"# Glossary\n{BEGIN}\nold\n{END}\n")
    monkeypatch.setattr(generate_glossary, "_repo_root", lambda: str(repo))
    assert generate_glossary.main([]) == 0
    assert "`pkg.mod`" in glossary.read_text(encoding="utf-8")


def test_check_exits_1_and_does_not_write_when_stale(tmp_path, monkeypatch):
    stale = f"# Glossary\n{BEGIN}\nold\n{END}\n"
    repo, glossary = _make_repo(tmp_path, stale)
    monkeypatch.setattr(generate_glossary, "_repo_root", lambda: str(repo))
    assert generate_glossary.main(["--check"]) == 1
    assert glossary.read_text(encoding="utf-8") == stale


def test_check_exits_0_when_up_to_date(tmp_path, monkeypatch):
    repo, glossary = _make_repo(tmp_path, f"# Glossary\n{BEGIN}\nold\n{END}\n")
    monkeypatch.setattr(generate_glossary, "_repo_root", lambda: str(repo))
    generate_glossary.main([])
    regenerated = glossary.read_text(encoding="utf-8")
    assert generate_glossary.main(["--check"]) == 0
    assert glossary.read_text(encoding="utf-8") == regenerated
