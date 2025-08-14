#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    repo = _repo_root()
    src_dir = os.path.join(repo, "src")
    glossary_md = os.path.join(repo, "quadmath", "markdown", "10_symbols_glossary.md")

    sys.path.insert(0, src_dir)
    from glossary_gen import build_api_index, generate_markdown_table, inject_between_markers  # type: ignore

    with open(glossary_md, "r", encoding="utf-8") as fh:
        text = fh.read()

    entries = build_api_index(src_dir)
    table = generate_markdown_table(entries)
    begin = "<!-- BEGIN: AUTO-API-GLOSSARY -->"
    end = "<!-- END: AUTO-API-GLOSSARY -->"
    new_text = inject_between_markers(text, begin, end, table)

    if new_text != text:
        with open(glossary_md, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        print(f"Updated glossary: {glossary_md}")
    else:
        print("Glossary up-to-date")


if __name__ == "__main__":
    main()


