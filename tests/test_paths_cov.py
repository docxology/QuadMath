from paths import get_repo_root, get_output_dir


def test_paths_cover_helpers(tmp_path, monkeypatch):
    # Force repo root discovery from a nested location
    root = get_repo_root()
    assert isinstance(root, str) and len(root) > 0
    out = get_output_dir()
    assert out.endswith("quadmath/output")


def test_paths_traversal_multiple_levels(tmp_path):
    """Test that get_repo_root traverses multiple parent directories (covers line 20)."""
    import os
    
    # Create nested directories without .git or README.md
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    
    # Add README.md at the top level to stop traversal
    (tmp_path / "README.md").write_text("# Test")
    
    # Start from deeply nested directory
    root = get_repo_root(start=str(nested))
    
    # Should traverse up and find tmp_path (which has README.md)
    assert root == str(tmp_path)
