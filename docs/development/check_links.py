#!/usr/bin/env python3
"""Check that every relative markdown link/image under docs/ resolves on disk.

Scans ``docs/**/*.md`` for link/image targets using target-side matching
(``](target)`` — every ``](...)`` occurrence outside fenced code blocks and
inline code spans), so targets whose ``[alt]`` text itself contains brackets
(e.g. math like ``$w = [1.0, 2.0]$`` in a caption) are still checked.
Absolute URLs (``http(s)``/``mailto``) and pure-fragment links are skipped;
each remaining relative target is verified:

- the target file exists (fragment identifiers stripped), and
- when a target carries a ``#fragment``, the fragment resolves in that file
  against explicit ``{#id}`` anchors or GitHub-style heading slugs.

Exit code 0 when every relative link resolves, 1 otherwise.

Usage::

    uv run python docs/development/check_links.py
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent

# Target-side matching: every `](target)` occurrence. Alt text may itself
# contain `]` (math brackets in captions), so `[...](...)`-shaped patterns
# with a bracket-free alt would silently skip those embeds.
TARGET_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Explicit Pandoc-style anchor: ## Heading {#some-id}
EXPLICIT_ANCHOR_RE = re.compile(r"\{#([A-Za-z0-9_:.-]+)\}")

# Inline-code span, stripped before link scanning so example syntax in
# backticks never counts as a link.
CODE_SPAN_RE = re.compile(r"`[^`\n]+`")


def strip_fenced_code(text: str) -> str:
    """Remove fenced code blocks so example markdown is not checked.

    Parameters
    ----------
    text : str
        Full markdown file content.

    Returns
    -------
    str
        The markdown with fenced blocks replaced by blank lines.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if fence is None:
            match = FENCE_RE.match(line)
            if match:
                fence = match.group(1)[:3]
                out.append("")
                continue
            out.append(line)
        else:
            if line.strip().startswith(fence):
                fence = None
            out.append("")
    return "\n".join(out)


def heading_slug(heading_text: str) -> str:
    """Return the GitHub-style slug for a heading's visible text.

    Parameters
    ----------
    heading_text : str
        Heading text with markdown markers removed.

    Returns
    -------
    str
        Lowercase, space-collapsed, punctuation-stripped slug joined by
        hyphens.
    """
    text = heading_text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s]+", "-", text).strip("-")


def collect_anchors(text: str) -> set[str]:
    """Collect every anchor id a markdown file defines.

    Parameters
    ----------
    text : str
        Full markdown file content (fenced blocks included; they are skipped
        per line here).

    Returns
    -------
    set of str
        Explicit ``{#id}`` anchors plus heading slugs.
    """
    anchors: set[str] = set()
    fence: str | None = None
    for line in text.splitlines():
        if fence is None:
            match = FENCE_RE.match(line)
            if match:
                fence = match.group(1)[:3]
                continue
        else:
            if line.strip().startswith(fence):
                fence = None
            continue
        anchors.update(EXPLICIT_ANCHOR_RE.findall(line))
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s*\{#[^}]+\})?\s*#*\s*$", line)
        if heading:
            anchors.add(heading_slug(EXPLICIT_ANCHOR_RE.sub("", heading.group(1))))
    return anchors


def iter_targets_from_line(line: str) -> list[str]:
    """Extract link/image targets requiring on-disk resolution from one line.

    Parameters
    ----------
    line : str
        A single markdown line with fenced blocks already stripped and inline
        code spans removed.

    Returns
    -------
    list of str
        Targets that must resolve on disk relative to the file.
    """
    targets: list[str] = []
    for target in TARGET_RE.findall(line):
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if "://" in target or target.startswith(("mailto:", "#")):
            continue
        targets.append(target)
    return targets


def check_docs(docs_root: Path) -> list[str]:
    """Check every relative link/image and fragment under ``docs_root``.

    Parameters
    ----------
    docs_root : Path
        Directory scanned recursively for ``*.md`` files.

    Returns
    -------
    list of str
        Human-readable problems; empty when every relative link resolves.
    """
    anchors_by_file: dict[Path, set[str]] = {}
    for md in sorted(docs_root.rglob("*.md")):
        anchors_by_file[md.resolve()] = collect_anchors(md.read_text(encoding="utf-8"))

    problems: list[str] = []
    for md in sorted(docs_root.rglob("*.md")):
        raw = md.read_text(encoding="utf-8")
        text = strip_fenced_code(raw)
        for lineno, line in enumerate(text.splitlines(), 1):
            line = CODE_SPAN_RE.sub("", line)
            for target in iter_targets_from_line(line):
                if target.startswith("/"):
                    problems.append(
                        f"{md.relative_to(docs_root)}:{lineno}: absolute path "
                        f"link is not repo-relative: {target}"
                    )
                    continue
                path_part, _, fragment = target.partition("#")
                if not path_part:
                    continue
                resolved = (md.parent / path_part).resolve()
                if not resolved.exists():
                    problems.append(
                        f"{md.relative_to(docs_root)}:{lineno}: unresolved "
                        f"link target: {target}"
                    )
                    continue
                if fragment and resolved.suffix == ".md":
                    anchors = anchors_by_file.get(resolved, set())
                    if fragment not in anchors:
                        problems.append(
                            f"{md.relative_to(docs_root)}:{lineno}: dangling "
                            f"fragment #{fragment} in {path_part}"
                        )
    return problems


def main() -> int:
    """Run the checker over ``docs/`` and report.

    Returns
    -------
    int
        0 when every relative link and fragment resolves, 1 otherwise.
    """
    problems = check_docs(DOCS_ROOT)
    scanned = sum(1 for _ in DOCS_ROOT.rglob("*.md"))
    if problems:
        print(f"FAIL: {len(problems)} unresolved link(s) in {scanned} markdown file(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"PASS: every relative link/image (and fragment) in {scanned} markdown file(s) under docs/ resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
