import pytest
import os

from quadmath.paths import get_repo_root, get_output_dir, get_data_dir, get_figure_dir


@pytest.fixture(autouse=True)
def isolated_output_configuration(monkeypatch):
    monkeypatch.delenv("QUADMATH_OUTPUT_DIR", raising=False)


def test_paths_helpers_basic():
    root = get_repo_root()
    assert isinstance(root, str) and root
    # Root must be the true repository root, not a subpackage like src/
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    out = get_output_dir()
    assert out == os.path.join(root, "quadmath", "output")


def test_get_output_dir_creates_dir():
    out = get_output_dir()
    assert os.path.isdir(out)


def test_get_repo_root_is_parent():
    here = os.path.abspath(os.path.dirname(__file__))
    root = get_repo_root(start=here)
    # tests/ has a bare README.md; traversal must continue to the real root
    assert root != here
    assert os.path.exists(os.path.join(root, "pyproject.toml"))


def test_get_repo_root_from_src():
    """Regression: src/ contains a README.md; root detection must skip it."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))
    root = get_repo_root(start=src_dir)
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    assert not root.rstrip(os.sep).endswith("src")


def test_get_repo_root_terminal_branch():
    # Starting from the filesystem root with no marker above it must fail, not
    # return '/' (which would direct generated outputs to /quadmath/output).
    start = os.path.abspath(os.sep)
    with pytest.raises(RuntimeError, match="no repository root"):
        get_repo_root(start=start)


def test_get_data_and_figure_dirs():
    # Direct coverage of the real helpers (visualize tests monkeypatch them)
    out = get_output_dir()
    data_dir = get_data_dir()
    figure_dir = get_figure_dir()
    assert data_dir == os.path.join(out, "data") and os.path.isdir(data_dir)
    assert figure_dir == os.path.join(out, "figures") and os.path.isdir(figure_dir)


def test_explicit_output_root_works_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "generated"
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(target))
    assert get_output_dir() == str(target)
    assert get_data_dir() == str(target / "data")
    assert get_figure_dir() == str(target / "figures")
    assert (target / "data").is_dir()
    assert (target / "figures").is_dir()


def test_relative_output_root_is_resolved_from_the_calling_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", "results")
    assert get_output_dir() == str(tmp_path / "results")
    assert (tmp_path / "results").is_dir()
