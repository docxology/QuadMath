from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ApiEntry:
    module: str
    name: str
    kind: str  # 'function' | 'class' | 'constant'
    signature: Optional[str]
    summary: str


def _is_constant(name: str, node: ast.AST) -> bool:
    return (
        name.isupper()
        and isinstance(node, (ast.Assign, ast.AnnAssign))
    )


def _first_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _format_function_signature(node: ast.FunctionDef) -> str:
    # Minimal signature (names only) to avoid heavy introspection
    params: List[str] = []
    for arg in getattr(node.args, "args", []) or []:
        params.append(arg.arg)
    if getattr(node.args, "vararg", None):
        params.append(f"*{node.args.vararg.arg}")
    for kw in getattr(node.args, "kwonlyargs", []) or []:
        params.append(f"{kw.arg}=")
    if getattr(node.args, "kwarg", None):
        params.append(f"**{node.args.kwarg.arg}")
    return f"({', '.join(params)})"


def build_api_index(src_dir: str) -> List[ApiEntry]:
    entries: List[ApiEntry] = []
    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            module = os.path.splitext(os.path.relpath(path, src_dir))[0].replace(os.sep, ".")
            with open(path, "r", encoding="utf-8") as fh:
                code = fh.read()
            try:
                tree = ast.parse(code, filename=path)
            except SyntaxError:
                continue

            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    entries.append(
                        ApiEntry(
                            module=module,
                            name=node.name,
                            kind="function",
                            signature=_format_function_signature(node),
                            summary=_first_line(ast.get_docstring(node)),
                        )
                    )
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    entries.append(
                        ApiEntry(
                            module=module,
                            name=node.name,
                            kind="class",
                            signature=None,
                            summary=_first_line(ast.get_docstring(node)),
                        )
                    )
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    # Simple top-level constants in ALL_CAPS
                    targets: Iterable[ast.expr]
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    else:
                        targets = [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name) and _is_constant(t.id, node):
                            entries.append(
                                ApiEntry(
                                    module=module,
                                    name=t.id,
                                    kind="constant",
                                    signature=None,
                                    summary="",
                                )
                            )
    return entries


def generate_markdown_table(entries: List[ApiEntry]) -> str:
    header = "| Module | Symbol | Kind | Signature | Summary |\n| --- | --- | --- | --- | --- |"
    rows: List[str] = []
    # Stable sort by module then name
    for e in sorted(entries, key=lambda x: (x.module, x.name)):
        sig = e.signature or ""
        summary = e.summary.replace("|", "\\|")
        rows.append(f"| `{e.module}` | `{e.name}` | {e.kind} | `{sig}` | {summary} |")
    return "\n".join([header, *rows])


def inject_between_markers(markdown_text: str, begin: str, end: str, payload: str) -> str:
    if begin not in markdown_text or end not in markdown_text:
        # Append at end with markers if not present
        return markdown_text.rstrip() + f"\n\n{begin}\n{payload}\n{end}\n"
    pre, rest = markdown_text.split(begin, 1)
    _old, post = rest.split(end, 1)
    return pre + begin + "\n" + payload + "\n" + end + post


__all__ = [
    "ApiEntry",
    "build_api_index",
    "generate_markdown_table",
    "inject_between_markers",
]


