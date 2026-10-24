import tomllib
from pathlib import Path

import fitstream as fts


def test_public_version_matches_project_version() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())
    assert fts.__version__ == project["project"]["version"]
