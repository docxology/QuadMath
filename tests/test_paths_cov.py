from paths import get_repo_root, get_output_dir


def test_paths_cover_helpers(tmp_path, monkeypatch):
    # Force repo root discovery from a nested location
    root = get_repo_root()
    assert isinstance(root, str) and len(root) > 0
    out = get_output_dir()
    assert out.endswith("quadmath/output")

