from pathlib import Path


def test_expected_project_directories_exist():
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "app").exists()
    assert (repo_root / "config").exists()
    assert (repo_root / "data").exists()
    assert (repo_root / "tests").exists()
