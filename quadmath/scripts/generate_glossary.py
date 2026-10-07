#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_glossary(repo: str) -> Tuple[str, str, str]:
    """Return (glossary_path, current_text, regenerated_text) for the repo."""
    src_dir = os.path.join(repo, "src")
    glossary_md = os.path.join(repo, "quadmath", "markdown", "10_symbols_glossary.md")

    sys.path.insert(0, src_dir)
    from quadmath.tools.glossary_gen import build_api_index, generate_markdown_table, inject_between_markers  # type: ignore

    with open(glossary_md, "r", encoding="utf-8") as fh:
        text = fh.read()

    entries = build_api_index(src_dir)
    table = generate_markdown_table(entries)
    begin = "<!-- BEGIN: AUTO-API-GLOSSARY -->"
    end = "<!-- END: AUTO-API-GLOSSARY -->"
    return glossary_md, text, inject_between_markers(text, begin, end, table)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the API glossary table.")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the glossary is out of date; never write")
    args = parser.parse_args(argv)

    glossary_md, text, new_text = build_glossary(_repo_root())
    if args.check:
        if new_text != text:
            print(f"Glossary out of date: {glossary_md} (run generate_glossary.py to regenerate)")
            return 1
        print("Glossary up-to-date")
        return 0

    if new_text != text:
        with open(glossary_md, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        print(f"Updated glossary: {glossary_md}")
    else:
        print("Glossary up-to-date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
