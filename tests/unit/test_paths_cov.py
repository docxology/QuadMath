import pytest

from quadmath.paths import get_repo_root




def test_paths_traversal_multiple_levels(tmp_path):
    """Test that get_repo_root traverses multiple parent directories (covers line 20)."""
    
    # Create nested directories without .git or README.md
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    
    # Add README.md plus pyproject.toml at the top level to stop traversal
    (tmp_path / "README.md").write_text("# Test")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    
    # Start from deeply nested directory
    root = get_repo_root(start=str(nested))
    
    # Should traverse up and find tmp_path (which has both markers)
    assert root == str(tmp_path)


def test_bare_readme_is_not_repo_root(tmp_path):
    """Regression: a directory with only a README.md must not stop traversal.

    Mirrors the src/ layout (README.md, no pyproject.toml) that sent generated
    outputs to src/quadmath/output/ instead of quadmath/output/.
    """
    nested = tmp_path / "subpkg" / "deeper"
    nested.mkdir(parents=True)
    (tmp_path / "README.md").write_text("# Test")
    
    with pytest.raises(RuntimeError, match="no repository root"):
        get_repo_root(start=str(nested))
