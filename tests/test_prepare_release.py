import runpy
from pathlib import Path
from unittest.mock import MagicMock, Mock
from urllib.error import HTTPError

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.py"
HISTORY = "# Changelog\n\nRelease history.\n\n## v0.3.1\n\nPrevious release.\n"


@pytest.fixture
def release_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAW_VERSION", "v0.4.0")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    (tmp_path / "docs/releases").mkdir(parents=True)
    (tmp_path / "docs/releases/0.4.0.md").write_text("- New feature.\n")
    (tmp_path / "changelog.md").write_text(HISTORY)
    monkeypatch.setattr("subprocess.check_output", Mock(return_value=""))
    update = Mock()
    monkeypatch.setattr("subprocess.run", update)
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=HTTPError("", 404, "", None, None)))
    return update


@pytest.mark.parametrize("raw_version,version", [("0.4.0", "0.4.0"), ("v0.4.0", "0.4.0"), ("0.4.0rc1", "0.4.0rc1")])
def test_prepare_release_updates_notes_and_version(release_env, tmp_path, monkeypatch, raw_version, version):
    monkeypatch.setenv("RAW_VERSION", raw_version)
    (tmp_path / f"docs/releases/{version}.md").write_text("- New feature.\n")

    runpy.run_path(str(SCRIPT), run_name="__main__")

    release_env.assert_called_once_with(["uv", "version", "--no-sync", version], check=True)
    changelog = (tmp_path / "changelog.md").read_text()
    assert changelog.startswith(f"# Changelog\n\nRelease history.\n\n## v{version} (")
    assert changelog.endswith("- New feature.\n\n## v0.3.1\n\nPrevious release.\n")
    assert (tmp_path / "release-notes.md").read_text() == "- New feature.\n"
    prerelease = "true" if version.endswith("rc1") else "false"
    assert (tmp_path / "outputs").read_text() == f"version={version}\ntag=v{version}\nprerelease={prerelease}\n"


@pytest.mark.parametrize("notes", [None, " \n"])
def test_prepare_release_requires_notes(release_env, tmp_path, notes):
    path = tmp_path / "docs/releases/0.4.0.md"
    if notes is None:
        path.unlink()
    else:
        path.write_text(notes)
    with pytest.raises(SystemExit, match="Add release notes"):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    release_env.assert_not_called()
    assert (tmp_path / "changelog.md").read_text() == HISTORY


@pytest.mark.parametrize("raw_version", ["invalid", "0.4.0\ninjected=true", "0.4.0+local"])
def test_prepare_release_rejects_invalid_versions(release_env, monkeypatch, raw_version):
    monkeypatch.setenv("RAW_VERSION", raw_version)
    with pytest.raises((ValueError, SystemExit)):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    release_env.assert_not_called()


@pytest.mark.parametrize("existing", ["tag", "pypi", "changelog"])
def test_prepare_release_rejects_existing_release(release_env, tmp_path, monkeypatch, existing):
    if existing == "tag":
        monkeypatch.setattr("subprocess.check_output", Mock(return_value="v0.4.0\n"))
    elif existing == "pypi":
        monkeypatch.setattr("urllib.request.urlopen", MagicMock())
    else:
        (tmp_path / "changelog.md").write_text(HISTORY + "\n## v0.4.0\n\nExisting notes.\n")
    with pytest.raises(SystemExit, match="already"):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    release_env.assert_not_called()


def test_prepare_release_stops_on_pypi_failure(release_env, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", Mock(side_effect=HTTPError("", 503, "", None, None)))
    with pytest.raises(HTTPError):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    release_env.assert_not_called()
