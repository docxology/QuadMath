"""Tooling layer: auto-documentation utilities (API glossary generation) and atomic data writes."""

from .atomic_write import atomic_open, atomic_savez, atomic_write_text
from .glossary_gen import ApiEntry, build_api_index, generate_markdown_table, inject_between_markers

__all__ = [
    "ApiEntry",
    "atomic_open",
    "atomic_savez",
    "atomic_write_text",
    "build_api_index",
    "generate_markdown_table",
    "inject_between_markers",
]
