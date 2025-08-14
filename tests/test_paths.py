from paths import get_repo_root, get_output_dir


def test_paths_helpers_basic():
    root = get_repo_root()
    assert isinstance(root, str) and root
    out = get_output_dir()
    assert out.endswith("quadmath/output")
import os

from paths import get_repo_root, get_output_dir


def test_get_output_dir_creates_dir():
    out = get_output_dir()
    assert os.path.isdir(out)


def test_get_repo_root_is_parent():
    here = os.path.abspath(os.path.dirname(__file__))
    root = get_repo_root(start=here)
    assert os.path.exists(os.path.join(root, "README.md")) or os.path.isdir(os.path.join(root, ".git"))


def test_get_repo_root_terminal_branch():
    # On Unix, starting from '/' will immediately hit terminal branch
    start = os.path.abspath(os.sep)
    root = get_repo_root(start=start)
    assert isinstance(root, str)
