"""Prepare release metadata from the repository root using RAW_VERSION and GITHUB_OUTPUT."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from packaging.version import Version


def main() -> None:
    version = Version(os.environ["RAW_VERSION"])
    if version.local is not None:
        raise SystemExit("PyPI releases cannot use local version identifiers.")
    tag = f"v{version}"
    if subprocess.check_output(["git", "tag", "--list", tag], text=True).strip():
        raise SystemExit(f"Git tag {tag} already exists.")
    try:
        with urlopen(f"https://pypi.org/pypi/fitstream/{version}/json", timeout=30):
            raise SystemExit(f"fitstream {version} already exists on PyPI.")
    except HTTPError as error:
        if error.code != 404:
            raise

    notes_path = Path(f"docs/releases/{version}.md")
    notes = notes_path.read_text().strip() if notes_path.is_file() else ""
    if not notes:
        raise SystemExit(f"Add release notes to {notes_path} first.")
    changelog = Path("changelog.md")
    text = changelog.read_text()
    if any(line.split()[:2] == ["##", tag] for line in text.splitlines()):
        raise SystemExit(f"Changelog already contains {tag}.")
    header, separator, history = text.partition("\n## v")
    date = datetime.now(timezone.utc).date()
    entry = f"## {tag} ({date})\n\n{notes}\n"
    changelog.write_text(header.rstrip() + "\n\n" + entry + separator + history)
    Path("release-notes.md").write_text(notes + "\n")
    subprocess.run(["uv", "version", "--no-sync", str(version)], check=True)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"version={version}\ntag={tag}\n")
        output.write(f"prerelease={str(version.is_prerelease).lower()}\n")


if __name__ == "__main__":
    main()
