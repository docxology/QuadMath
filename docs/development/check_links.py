#!/usr/bin/env python3
"""Validate relative Markdown links, images and heading fragments.

Usage: ``python docs/development/check_links.py [ROOT ...]``.
Without roots, scan this repository's docs directory. Missing or empty roots
fail. Fenced/inline code and external URLs are excluded; fragment-only links,
reference definitions, URL-encoded paths, explicit anchors and duplicate heading
slugs are checked. Linked Markdown outside a scanned root is read on demand.
Repository-root scans prune dependency/build/generated directories rather than
mistaking their Markdown for authored documentation. Linked targets inside
those directories are still resolved and checked on demand.
"""

from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

DOCS_ROOT = Path(__file__).resolve().parent.parent
DISCOVERY_EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".tox", "node_modules", "output", "build", "dist", "venv"}

# Target-side matching: every `](target)` occurrence. Alt text may itself
# contain `]` (math brackets in captions), so `[...](...)`-shaped patterns
# with a bracket-free alt would silently skip those embeds.
TARGET_RE = re.compile(r'\]\((<[^>\n]+>|[^)\s]+)(?:\s+["\'][^"\']*["\'])?\)')
REFERENCE_RE = re.compile(r'^\s{0,3}\[[^\]]+\]:\s*(<[^>\n]+>|\S+)')
EXTERNAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
HTML_ANCHOR_RE = re.compile(r'<[a-z][^>]*\s(?:id|name)=["\']([^"\']+)["\']', re.IGNORECASE)

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

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
                fence = match.group(1)
                out.append("")
                continue
            out.append(line)
        else:
            if re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
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
        Lowercase, punctuation-stripped slug, retaining each space as a hyphen.
        Punctuation between spaces can therefore produce consecutive hyphens.
    """
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading_text)
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


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
    heading_slugs: set[str] = set()
    previous = ""
    for line in strip_fenced_code(text).splitlines():
        anchors.update(EXPLICIT_ANCHOR_RE.findall(line))
        anchors.update(HTML_ANCHOR_RE.findall(line))
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        heading_text = heading.group(1) if heading else None
        if heading_text is None and previous.strip() and re.fullmatch(r"\s{0,3}(?:=+|-+)\s*", line):
            heading_text = previous
        if heading_text is not None:
            slug = heading_slug(EXPLICIT_ANCHOR_RE.sub("", heading_text))
            candidate, suffix = slug, 0
            while candidate in heading_slugs:
                suffix += 1
                candidate = f"{slug}-{suffix}"
            heading_slugs.add(candidate)
            anchors.add(candidate)
        previous = line
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
    matches = TARGET_RE.findall(line)
    reference = REFERENCE_RE.match(line)
    if reference:
        matches.append(reference.group(1))
    for target in matches:
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if EXTERNAL_RE.match(target) or target.startswith("//"):
            continue
        targets.append(target)
    return targets


def _markdown_files(root: Path) -> list[Path]:
    """Discover authored Markdown without walking dependency/generated trees."""
    files: list[Path] = []
    for directory, children, names in os.walk(root):
        children[:] = [
            name for name in children
            if name not in DISCOVERY_EXCLUDES and not name.startswith(".venv")
        ]
        files.extend(Path(directory) / name for name in names if name.lower().endswith(".md"))
    return sorted(files)


def check_docs(docs_root: Path) -> list[str]:
    """Check every relative link/image and fragment under ``docs_root``.

    Parameters
    ----------
    docs_root : Path
        Directory scanned for authored Markdown, pruning generated/dependency trees.

    Returns
    -------
    list of str
        Human-readable problems; empty when every relative link resolves.
    """
    docs_root = docs_root.resolve()
    if not docs_root.is_dir():
        return [f"documentation root is not a directory: {docs_root}"]
    markdown = _markdown_files(docs_root)
    if not markdown:
        return [f"documentation root contains no Markdown files: {docs_root}"]
    contents = {md: md.read_text(encoding="utf-8") for md in markdown}
    anchors_by_file = {md: collect_anchors(raw) for md, raw in contents.items()}
    problems: list[str] = []
    for md, raw in contents.items():
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
                parsed = urlsplit(target)
                path_part = unquote(parsed.path)
                fragment = unquote(parsed.fragment)
                if path_part.startswith("/"):
                    problems.append(
                        f"{md.relative_to(docs_root)}:{lineno}: absolute path "
                        f"link is not repo-relative: {target}"
                    )
                    continue
                resolved = (md.parent / path_part).resolve() if path_part else md
                if not resolved.exists():
                    problems.append(
                        f"{md.relative_to(docs_root)}:{lineno}: unresolved "
                        f"link target: {target}"
                    )
                    continue
                if fragment and resolved.suffix.lower() == ".md":
                    if resolved not in anchors_by_file:
                        anchors_by_file[resolved] = collect_anchors(resolved.read_text(encoding="utf-8"))
                    anchors = anchors_by_file[resolved]
                    if fragment not in anchors:
                        problems.append(
                            f"{md.relative_to(docs_root)}:{lineno}: dangling "
                            f"fragment #{fragment} in {path_part}"
                        )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check every requested root; return one if any root or link is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="*", help="Markdown directories (default: docs/)")
    roots = parser.parse_args(argv).roots or [DOCS_ROOT]
    problems = [
        f"{root}: {problem}"
        for root in roots
        for problem in check_docs(root)
    ]
    scanned = sum(len(_markdown_files(root)) for root in roots if root.is_dir())
    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {scanned} Markdown file(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"PASS: relative links and fragments resolve in {scanned} Markdown file(s) across {len(roots)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
