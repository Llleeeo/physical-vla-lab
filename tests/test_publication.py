"""Publication checks use isolated temporary repositories, never live credentials."""
import importlib.util
from pathlib import Path
import subprocess
import pytest

spec = importlib.util.spec_from_file_location("publication", Path(__file__).resolve().parents[1] / "scripts/check_publication.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def stage(repo, path, content):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", path], check=True)


def test_allowed_text(repo):
    stage(repo, "README.md", "research")
    assert publication.audit(repo)["failures"] == []


def test_forced_raw_data_rejected(repo):
    stage(repo, "data/episode.json", "{}")
    assert "allowlist" in publication.audit(repo)["failures"][0][1]


def test_secret_rejected_without_echoing_value(repo):
    secret = "hf_" + "X" * 30
    stage(repo, "README.md", secret)
    result = publication.audit(repo)
    assert result["failures"] and secret not in str(result)


def test_audits_staged_blob_not_working_copy(repo):
    stage(repo, "README.md", "hf_" + "X" * 30)
    (repo / "README.md").write_text("safe working copy")
    assert publication.audit(repo)["failures"]


def test_symlink_rejected(repo):
    (repo / "README.md").symlink_to("missing-private-file")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    assert "symlink" in publication.audit(repo)["failures"][0][1]


def test_private_ip_rejected(repo):
    stage(repo, "README.md", ".".join(["192", "168", "20", "30"]))
    assert "private network" in publication.audit(repo)["failures"][0][1]
