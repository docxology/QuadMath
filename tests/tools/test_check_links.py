"""Consumer-visible failures of the shared Markdown link gate."""
from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

CHECKER_PATH = Path(__file__).resolve().parents[2] / "docs/development/check_links.py"
check_docs = runpy.run_path(str(CHECKER_PATH))["check_docs"]


def test_same_page_and_cross_file_anchors_reject_renamed_headings(tmp_path):
    index = tmp_path / "index.md"
    target = tmp_path / "target.md"
    index.write_text("# Index\n\n[Here](#index)\n[There](target.md#target)\n", encoding="utf-8")
    target.write_text("# Target\n", encoding="utf-8")
    assert check_docs(tmp_path) == []
    index.write_text(index.read_text().replace("# Index", "# Renamed index"), encoding="utf-8")
    target.write_text("# Renamed target\n", encoding="utf-8")
    problems = check_docs(tmp_path)
    assert len(problems) == 2
    assert any("#index" in problem for problem in problems)
    assert any("#target" in problem for problem in problems)


def test_encoded_reference_and_external_root_targets_resolve(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "API reference.md").write_text(
        "# Usage\n\n# Usage\n\n## Loss — gradient units\n\n<a id=\"stable-api\"></a>\n",
        encoding="utf-8",
    )
    index = docs / "index.md"
    index.write_text(
        "# Index\n\n[Encoded](../API%20reference.md#usage-1)\n"
        "[Spaced](<../API reference.md#stable-api>)\n"
        "[Spaced punctuation](../API%20reference.md#loss--gradient-units)\n"
        "[API][api]\n\n[api]: <../API reference.md#usage>\n"
        "`[Not a link](missing.md)`\n\n```md\n[Example](missing.md)\n```\n",
        encoding="utf-8",
    )
    assert check_docs(docs) == []
    (tmp_path / "API reference.md").write_text("# Usage\n\n## Loss — gradient units\n", encoding="utf-8")
    problems = check_docs(docs)
    assert len(problems) == 2
    assert any("#usage-1" in problem for problem in problems)
    assert any("#stable-api" in problem for problem in problems)


def test_reference_definition_target_is_not_silently_skipped(tmp_path):
    (tmp_path / "index.md").write_text("# Index\n\n[API][api]\n\n[api]: missing.md\n", encoding="utf-8")
    problems = check_docs(tmp_path)
    assert len(problems) == 1
    assert "missing.md" in problems[0]


def test_cli_checks_every_root_and_rejects_missing_or_empty_roots(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "index.md").write_text("# First\n", encoding="utf-8")
    page = second / "index.md"
    page.write_text("# Second\n\n[Broken](#absent)\n", encoding="utf-8")

    def run(*roots):
        return subprocess.run([sys.executable, "-I", str(CHECKER_PATH), *map(str, roots)], capture_output=True)

    assert run(first, second).returncode == 1
    page.write_text("# Second\n\n[Local](#second)\n", encoding="utf-8")
    assert run(first, second).returncode == 0
    assert run(first, tmp_path / "absent").returncode == 1
    page.unlink()
    assert run(first, second).returncode == 1


def test_generated_markdown_does_not_make_an_empty_authored_root_pass(tmp_path):
    generated = tmp_path / "output"
    generated.mkdir()
    (generated / "README.md").write_text("# Artifact data\n", encoding="utf-8")
    assert subprocess.run([sys.executable, "-I", str(CHECKER_PATH), str(tmp_path)], capture_output=True).returncode == 1
    authored = tmp_path / "README.md"
    authored.write_text("# Project\n\n[Artifact](output/README.md#absent)\n", encoding="utf-8")
    problems = check_docs(tmp_path)
    assert len(problems) == 1
    assert "#absent" in problems[0]
    authored.write_text("# Project\n\n[Artifact](output/README.md#artifact-data)\n", encoding="utf-8")
    assert check_docs(tmp_path) == []


def test_long_fence_does_not_close_at_short_literal_delimiters(tmp_path):
    (tmp_path / "index.md").write_text(
        "# Real heading\n\n[Real](#real-heading)\n\n"
        "````markdown\n```\n[Literal example](absent.md)\n````\n",
        encoding="utf-8",
    )
    assert check_docs(tmp_path) == []
