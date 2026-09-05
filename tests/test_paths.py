import os

from paths import get_repo_root, get_output_dir, get_data_dir, get_figure_dir


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
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
    root = get_repo_root(start=src_dir)
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    assert not root.rstrip(os.sep).endswith("src")


def test_get_repo_root_terminal_branch():
    # On Unix, starting from '/' will immediately hit terminal branch
    start = os.path.abspath(os.sep)
    root = get_repo_root(start=start)
    assert isinstance(root, str)


def test_get_data_and_figure_dirs():
    # Direct coverage of the real helpers (visualize tests monkeypatch them)
    out = get_output_dir()
    data_dir = get_data_dir()
    figure_dir = get_figure_dir()
    assert data_dir == os.path.join(out, "data") and os.path.isdir(data_dir)
    assert figure_dir == os.path.join(out, "figures") and os.path.isdir(figure_dir)
