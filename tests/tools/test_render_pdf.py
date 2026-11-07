"""Contract tests for quadmath/scripts/render_pdf.sh.

The script is sourced (its main() is guarded), so MODULES, EXCLUDED_MODULES and
the build helpers run against stub pandoc/xelatex binaries on PATH. No real
TeX or pandoc process is started.
"""
from __future__ import annotations

import os
import re
import subprocess
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN = ROOT / "quadmath" / "markdown"
RENDER = ROOT / "quadmath" / "scripts" / "render_pdf.sh"
README = MARKDOWN / "README.md"
SECTION_GLOB = "[0-9][0-9]_*.md"

# Records every argument (one per line) into the file named by -o.
PANDOC_STUB = r"""#!/bin/bash
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-o" ]; then out="$a"; fi
  prev="$a"
done
if [ -n "${PANDOC_FAIL:-}" ]; then
  printf 'partial tex\n' > "$out"
  echo "pandoc fixture cannot convert input" >&2
  exit 1
fi
printf '%s\n' "$@" > "$out"
"""

# Logs one line per pass; writes a stub PDF; optionally writes an aux file
# with an unresolved-reference marker so the script takes the extra pass.
XELATEX_STUB = r"""#!/bin/bash
outdir=""
tex=""
for a in "$@"; do
  case "$a" in
    -output-directory=*) outdir="${a#-output-directory=}" ;;
    *.tex) tex="$a" ;;
  esac
done
base="$(basename "$tex" .tex)"
echo "$base" >> "$XELATEX_CALLS"
if [ -n "${XELATEX_AUX:-}" ]; then printf '\\@ref{x}\n' > "$outdir/$base.aux"; fi
printf 'stub pdf for %s\n' "$base" > "$outdir/$base.pdf"
if [ -n "${XELATEX_FAIL:-}" ]; then echo "fake failure after PDF write" >&2; exit 1; fi
"""


def _bash(body: str, markdown: Path = MARKDOWN) -> subprocess.CompletedProcess:
    script = f'source "{RENDER}"\nMARKDOWN_DIR="{markdown}"\n{body}'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)


def _pairs(var: str) -> list[tuple[str, str]]:
    result = _bash(f'printf "%s\\n" "${{{var}[@]}}"')
    assert result.returncode == 0, result.stderr
    pairs = []
    for line in result.stdout.splitlines():
        if line:
            name, _, reason = line.partition("|")
            pairs.append((name, reason))
    return pairs


def _disk_sections(markdown: Path = MARKDOWN) -> list[str]:
    return sorted(p.name for p in markdown.glob(SECTION_GLOB))


@pytest.fixture
def sandbox(tmp_path) -> SimpleNamespace:
    markdown = tmp_path / "markdown"
    markdown.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("pandoc", PANDOC_STUB), ("xelatex", XELATEX_STUB)):
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    preamble = tmp_path / "preamble.tex"
    preamble.write_text("% stub preamble\n", encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["XELATEX_CALLS"] = str(tmp_path / "xelatex_calls.txt")
    env.pop("XELATEX_AUX", None)
    env.pop("XELATEX_FAIL", None)
    env.pop("PANDOC_FAIL", None)
    return SimpleNamespace(
        markdown=markdown,
        output=tmp_path / "output",
        preamble=preamble,
        env=env,
        calls=tmp_path / "xelatex_calls.txt",
    )


def _render(sb: SimpleNamespace, body: str, **extra_env: str) -> subprocess.CompletedProcess:
    prefix = (
        f'source "{RENDER}"\n'
        f'MARKDOWN_DIR="{sb.markdown}"\n'
        f'OUTPUT_DIR="{sb.output}"\n'
        f'PDF_DIR="{sb.output}/pdf"\n'
        f'TEX_DIR="{sb.output}/tex"\n'
        f'LATEX_TEMP_DIR="{sb.output}/latex_temp"\n'
        'mkdir -p "$PDF_DIR" "$TEX_DIR" "$LATEX_TEMP_DIR"\n'
    )
    env = dict(sb.env, **extra_env)
    return subprocess.run(["bash", "-c", prefix + body], capture_output=True, text=True, env=env, check=False)


def _calls(sb: SimpleNamespace) -> list[str]:
    return sb.calls.read_text(encoding="utf-8").split() if sb.calls.exists() else []


def _margin_lines(tex_text: str) -> list[str]:
    return [line for line in tex_text.splitlines() if line.startswith("geometry:")]


# --- section coverage contract -------------------------------------------------


def test_modules_and_exclusions_cover_every_section_file_exactly_once():
    listed = [name for name, _ in _pairs("MODULES")] + [name for name, _ in _pairs("EXCLUDED_MODULES")]
    assert len(listed) == len(set(listed)), "a section file is listed twice"
    assert sorted(listed) == _disk_sections()


def test_modules_are_built_in_numeric_order_with_titles():
    modules = _pairs("MODULES")
    names = [name for name, _ in modules]
    assert names == sorted(names)
    assert all(title.strip() for _, title in modules)


def test_module_titles_contain_no_latex_special_characters():
    # pandoc copies -V title into \title{} unescaped, so & % $ # _ would break xelatex
    special = re.compile(r"[&%$#_{}~^\\]")
    offenders = [(name, title) for name, title in _pairs("MODULES") if special.search(title)]
    assert offenders == []


def test_excluded_sections_carry_a_reason():
    excluded = _pairs("EXCLUDED_MODULES")
    assert excluded, "expected at least the preamble exclusion"
    assert all(reason.strip() for _, reason in excluded)


def test_readme_section_table_lists_the_same_files_as_disk():
    text = README.read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(\d\d_[^`]+\.md)` \|", text, flags=re.MULTILINE)
    assert sorted(rows) == _disk_sections()


def test_check_module_coverage_passes_for_repository():
    result = _bash("check_module_coverage")
    assert result.returncode == 0, result.stderr


def test_check_module_coverage_rejects_unlisted_section(tmp_path):
    (tmp_path / "01_a.md").write_text("a\n", encoding="utf-8")
    (tmp_path / "02_b.md").write_text("b\n", encoding="utf-8")
    result = _bash('MODULES=("01_a.md|A"); EXCLUDED_MODULES=(); check_module_coverage', markdown=tmp_path)
    assert result.returncode != 0
    assert "02_b.md" in result.stderr


def test_check_module_coverage_rejects_excluded_section_without_reason(tmp_path):
    (tmp_path / "01_a.md").write_text("a\n", encoding="utf-8")
    (tmp_path / "00_pre.md").write_text("p\n", encoding="utf-8")
    result = _bash(
        'MODULES=("01_a.md|A"); EXCLUDED_MODULES=("00_pre.md|"); check_module_coverage', markdown=tmp_path
    )
    assert result.returncode != 0
    assert "00_pre.md" in result.stderr


def test_check_module_coverage_rejects_module_file_that_is_missing(tmp_path):
    (tmp_path / "01_a.md").write_text("a\n", encoding="utf-8")
    result = _bash('MODULES=("01_a.md|A" "03_gone.md|G"); EXCLUDED_MODULES=(); check_module_coverage', markdown=tmp_path)
    assert result.returncode != 0
    assert "03_gone.md" in result.stderr


# --- build helpers (stubbed pandoc/xelatex) ------------------------------------


def test_build_one_writes_tex_and_runs_three_xelatex_passes_without_aux(sandbox):
    (sandbox.markdown / "11_demo.md").write_text("# Demo\n", encoding="utf-8")
    result = _render(sandbox, f'build_one "11_demo.md" "Demo Title" "{sandbox.preamble}"')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (sandbox.output / "pdf" / "11_demo.pdf").exists()
    tex_args = (sandbox.output / "tex" / "11_demo.tex").read_text(encoding="utf-8").splitlines()
    assert "title=Demo Title" in tex_args
    assert _calls(sandbox) == ["11_demo"] * 3


def test_build_one_runs_extra_pass_when_references_are_unresolved(sandbox):
    (sandbox.markdown / "12_refs.md").write_text("See \\ref{x}.\n", encoding="utf-8")
    result = _render(
        sandbox, f'build_one "12_refs.md" "Refs" "{sandbox.preamble}"', XELATEX_AUX="1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _calls(sandbox) == ["12_refs"] * 3


def test_build_one_reports_failure_and_leaves_no_pdf(sandbox):
    (sandbox.markdown / "13_broken.md").write_text("# Broken\n", encoding="utf-8")
    body = (
        f'if build_one "13_broken.md" "Broken" "{sandbox.preamble}"; then echo BUILD_OK; '
        'else echo BUILD_FAILED; fi'
    )
    result = _render(sandbox, body, XELATEX_FAIL="1")
    assert "BUILD_FAILED" in result.stdout, result.stdout + result.stderr
    assert not (sandbox.output / "pdf" / "13_broken.pdf").exists()
    assert "fake failure after PDF write" in result.stderr
    assert list((sandbox.output / "pdf").iterdir()) == []


@pytest.mark.parametrize("combined", [False, True])
def test_pandoc_failure_invalidates_stale_pdf_without_running_xelatex(sandbox, combined):
    (sandbox.markdown / "01_a.md").write_text("Alpha\n", encoding="utf-8")
    pdf_dir = sandbox.output / "pdf"
    pdf_dir.mkdir(parents=True)
    base = "quadmath_review" if combined else "01_a"
    previous = pdf_dir / f"{base}.pdf"
    previous.write_bytes(b"stale prior pdf")
    command = (
        f'MODULES=("01_a.md|Alpha"); build_combined "{sandbox.preamble}"'
        if combined else f'build_one "01_a.md" "Alpha" "{sandbox.preamble}"'
    )
    result = _render(sandbox, f"if {command}; then echo BUILD_OK; else echo BUILD_FAILED; fi", PANDOC_FAIL="1")
    assert "BUILD_FAILED" in result.stdout, result.stdout + result.stderr
    assert "pandoc fixture cannot convert input" in result.stderr
    assert not previous.exists()
    assert _calls(sandbox) == []


def test_combined_input_failure_keeps_prior_markdown_but_no_current_pdf(sandbox):
    (sandbox.markdown / "01_a.md").write_text("Alpha\n", encoding="utf-8")
    sandbox.output.mkdir()
    previous_md = sandbox.output / "quadmath_review.md"
    previous_md.write_text("previous complete manuscript\n", encoding="utf-8")
    pdf_dir = sandbox.output / "pdf"
    pdf_dir.mkdir()
    previous_pdf = pdf_dir / "quadmath_review.pdf"
    previous_pdf.write_bytes(b"stale")
    result = _render(
        sandbox,
        'MODULES=("01_a.md|Alpha" "02_missing.md|Missing"); '
        f'if build_combined "{sandbox.preamble}"; then echo BUILD_OK; else echo BUILD_FAILED; fi',
    )
    assert "BUILD_FAILED" in result.stdout
    assert "02_missing.md" in result.stderr
    assert previous_md.read_text(encoding="utf-8") == "previous complete manuscript\n"
    assert not previous_pdf.exists()
    assert list(sandbox.output.glob("quadmath_review.md.*")) == []
    assert _calls(sandbox) == []


def test_generation_failure_preserves_child_stdout_and_stderr(tmp_path):
    scripts = tmp_path / "repo space" / "quadmath" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "make_all_figures.py").write_text(
        "import sys\n"
        "print('generator reached data import')\n"
        "print('required data file is missing', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "uv"
    launcher.write_text(f'#!/bin/bash\nshift 2\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    result = subprocess.run(
        ["bash", "-c", f'source "{RENDER}"\nREPO_ROOT="{scripts.parents[1]}"\nrun_generation_scripts'],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "generator reached data import" in result.stdout
    assert "required data file is missing" in result.stderr


def test_renderer_entrypoint_uses_repository_cwd_from_spaced_nonroot_path(tmp_path):
    root = tmp_path / "Quad Math"
    scripts = root / "quadmath" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(RENDER, scripts / "render_pdf.sh")
    markdown = root / "quadmath" / "markdown"
    markdown.mkdir()
    for name, _ in _pairs("MODULES"):
        (markdown / name).write_text("# Section\n", encoding="utf-8")
    (markdown / "00_preamble.md").write_text("```latex\n% preamble\n```\n", encoding="utf-8")
    (scripts / "make_all_figures.py").write_text(
        "raise SystemExit('unexpected figure generation')\n", encoding="utf-8",
    )
    (scripts / "generate_glossary.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "root = Path(__file__).resolve().parents[2]\n"
        "if Path.cwd() != root:\n"
        "    raise SystemExit('wrong project cwd')\n"
        "(root / 'generation-started').write_text('correct project')\n"
        "raise SystemExit('stop before document build')\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fixtures = {
        "pandoc": "#!/bin/bash\nexit 99\n",
        "xelatex": "#!/bin/bash\nexit 99\n",
        "fc-list": "#!/bin/bash\nprintf 'DejaVu Serif\\nDejaVu Sans Mono\\n'\n",
        "uv": f'#!/bin/bash\nshift 2\nexec "{sys.executable}" "$@"\n',
    }
    for name, body in fixtures.items():
        executable = bin_dir / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
    elsewhere = tmp_path / "other project"
    elsewhere.mkdir()
    result = subprocess.run(
        ["bash", str(scripts / "render_pdf.sh"), "--skip-figures"],
        cwd=elsewhere, env=dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert (root / "generation-started").read_text() == "correct project"
    assert "stop before document build" in result.stderr
    assert not (elsewhere / "generation-started").exists()


def test_build_combined_joins_module_pairs_and_uses_one_margin(sandbox):
    (sandbox.markdown / "01_a.md").write_text("Alpha body\n", encoding="utf-8")
    (sandbox.markdown / "02_b.md").write_text("Beta body\n", encoding="utf-8")
    body = (
        'MODULES=("01_a.md|Alpha" "02_b.md|Beta"); '
        f'build_combined "{sandbox.preamble}"'
    )
    result = _render(sandbox, body)
    assert result.returncode == 0, result.stdout + result.stderr

    combined_md = (sandbox.output / "quadmath_review.md").read_text(encoding="utf-8")
    assert "Alpha body" in combined_md and "Beta body" in combined_md
    assert "\\newpage" in combined_md

    combined_tex = (sandbox.output / "tex" / "quadmath_review.tex").read_text(encoding="utf-8")
    assert "title=QuadMath: An Analytical Review of 4D and Quadray Coordinates" in combined_tex
    assert (sandbox.output / "pdf" / "quadmath_review.pdf").exists()

    # One margin for every PDF: module and combined builds share the same geometry.
    module_result = _render(sandbox, f'build_one "01_a.md" "Alpha" "{sandbox.preamble}"')
    assert module_result.returncode == 0, module_result.stdout + module_result.stderr
    module_tex = (sandbox.output / "tex" / "01_a.tex").read_text(encoding="utf-8")
    assert _margin_lines(module_tex) == _margin_lines(combined_tex)
    assert "geometry:left=1cm" in _margin_lines(combined_tex)
    assert "geometry:right=1cm" in _margin_lines(combined_tex)
