#!/usr/bin/env python3
"""Inspect the staged Git index; never reads ignored credentials or raw data."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT_FILES = {".gitignore", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "THIRD_PARTY.md", "requirements-cpu.txt"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yml", ".yaml"}
SECRET_PATTERNS = [
    re.compile(rb"(?<![A-Za-z0-9_])(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}(?![A-Za-z0-9_])"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
]
PRIVATE_IP = re.compile(rb"(?<![0-9.])(?:10\.[0-9]{1,3}|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9.])")


def permitted(path):
    if path in ROOT_FILES or path == "configs/dynamicvla_data_constraints.txt":
        return True
    if Path(path).suffix not in TEXT_SUFFIXES:
        return False
    return path.startswith(("src/capstone_vla/", "scripts/", "tests/", "configs/examples/", "docs/project/", "results/summaries/", ".github/"))


def audit(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    failures, count, total = [], 0, 0
    for entry in entries:
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t", 1)
        mode, sha, stage = meta.decode().split()
        path = raw_path.decode()
        count += 1
        if stage != "0" or mode not in ("100644", "100755"):
            failures.append((path, "unmerged entry, symlink or nested repository"))
            continue
        if not permitted(path):
            failures.append((path, "outside publication allowlist"))
            continue
        size = int(git("cat-file", "-s", sha))
        total += size
        if size > 2 * 1024 * 1024:
            failures.append((path, "file exceeds 2 MiB publication budget"))
            continue
        content = git("cat-file", "blob", sha)
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            failures.append((path, "non-UTF-8 content"))
            continue
        if b"\0" in content:
            failures.append((path, "binary content"))
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            failures.append((path, "credential-like content; inspect locally"))
        if PRIVATE_IP.search(content):
            failures.append((path, "private network address; use a local configuration"))
    if not count:
        failures.append(("INDEX", "no staged/tracked files to inspect"))
    return {"files": count, "bytes": total, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    result = audit(parser.parse_args().root)
    for path, reason in result["failures"]:
        print(f"REJECT {path}: {reason}")  # Never print a matching secret.
    if result["failures"]:
        raise SystemExit(1)
    print(f"PUBLICATION_CHECK_PASS files={result['files']} bytes={result['bytes']}")


if __name__ == "__main__":
    main()
