"""Link equation references in QuadMath's template-generated HTML reader."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

NAME = r"[A-Za-z0-9_.:-]+"
LABEL = rf"eq:{NAME}"
AUX_EQUATION = re.compile(
    rf"^\\newlabel\{{(?P<label>{LABEL})\}}\{{\{{"
    r"(?P<number>[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)\}.*\{equation\.[^{}]+\}",
    re.MULTILINE,
)
DISPLAY = re.compile(r'<span\s+class="math display">(?P<tex>.*?)</span>', re.DOTALL)
DECLARATION = re.compile(rf"\\label\{{({LABEL})\}}")
REFERENCE = re.compile(
    rf'<span\s+class="math inline">\s*\\\(\s*\\eqref\{{({LABEL})\}}\s*\\\)\s*</span>'
)
RAW_REFERENCE = re.compile(
    rf'<span\s+class="math (?:inline|display)">(?:(?!</span>).)*?'
    rf'\\eqref\{{{LABEL}\}}',
    re.DOTALL,
)
ANCHOR_PREFIX = "quadmath-equation-"
LINKED_TARGET = re.compile(
    rf'<span id="{ANCHOR_PREFIX}{NAME}" class="quadmath-equation">'
    r'(?P<display><span\s+class="math display">.*?</span>) '
    r'<span class="quadmath-equation-number">\([^<>]+\)</span></span>',
    re.DOTALL,
)
LINKED_REFERENCE = re.compile(
    rf'<a class="quadmath-equation-reference" href="(?:index\.html)?'
    rf'#{ANCHOR_PREFIX}(?P<name>{NAME})" aria-label="Equation [^"<>]+">'
    r'\([^<>]+\)</a>'
)


def display_labels(document: str) -> list[str]:
    """Read labels in this writer's display math, preserving document order."""
    labels = []
    for display in DISPLAY.finditer(document):
        found = DECLARATION.findall(display["tex"])
        if len(found) > 1:
            raise ValueError("Multiple equation labels in one HTML math span.")
        labels.extend(found)
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate equation labels in the HTML page.")
    return labels


def link_equations(document: str, numbers: dict[str, str]) -> str:
    """Add distinct static targets and replace the writer's inline references."""
    original = document
    document = LINKED_TARGET.sub(r"\g<display>", document)
    document = LINKED_REFERENCE.sub(
        lambda match: rf'<span class="math inline">\(\eqref{{eq:{match["name"]}}}\)</span>',
        document,
    )
    if f'id="{ANCHOR_PREFIX}' in document:
        raise ValueError("Unrecognized pre-existing equation target.")
    local = set(display_labels(document))
    unknown = local - numbers.keys()
    if unknown:
        raise ValueError(f"No PDF numbers for HTML equation labels: {sorted(unknown)}")

    def insert_target(match: re.Match[str]) -> str:
        label = DECLARATION.search(match["tex"])
        if label is None:
            return match[0]
        name = label[1]
        anchor = ANCHOR_PREFIX + name.removeprefix("eq:")
        number = numbers[name]
        return (
            f'<span id="{anchor}" class="quadmath-equation">'
            f'{match[0]} <span class="quadmath-equation-number">({number})</span></span>'
        )

    def replace_reference(match: re.Match[str]) -> str:
        name = match[1]
        if name not in numbers:
            raise ValueError(f"No PDF equation number for {name}.")
        anchor = ANCHOR_PREFIX + name.removeprefix("eq:")
        target = "" if name in local else "index.html"
        number = numbers[name]
        return (
            f'<a class="quadmath-equation-reference" href="{target}#{anchor}" '
            f'aria-label="Equation {number}">({number})</a>'
        )

    result = REFERENCE.sub(replace_reference, DISPLAY.sub(insert_target, document))
    if RAW_REFERENCE.search(result):
        raise ValueError("Unsupported raw equation reference in generated HTML.")
    if original != document and original != result:
        raise ValueError("Existing equation links do not match the current PDF.")
    return result


def replace_page(page: Path, text: str) -> None:
    """Replace a complete page, leaving the old page intact on write failure."""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=page.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.chmod(page.stat().st_mode & 0o777)
        os.replace(temporary, page)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    """Post-process a single completed Template Stage 03 output tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Project output directory")
    output = parser.parse_args().output
    web = output / "web"
    index = web / "index.html"
    aux = output / "pdf" / "_combined_manuscript.aux"
    try:
        pairs = AUX_EQUATION.findall(aux.read_text(encoding="utf-8"))
        numbers = dict(pairs)
        if not numbers or len(numbers) != len(pairs):
            raise ValueError("Missing or duplicate PDF equation labels.")
        pages = [index, *sorted(web.glob("manuscript__*.html"))]
        documents = {page: page.read_text(encoding="utf-8") for page in pages}
        if display_labels(documents[index]) != list(numbers):
            raise ValueError("The combined HTML and PDF equation label order differs.")
        updated = {page: link_equations(text, numbers) for page, text in documents.items()}
        for page, text in updated.items():
            if text != documents[page]:
                replace_page(page, text)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
