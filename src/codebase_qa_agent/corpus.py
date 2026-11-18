"""Clone and enumerate files in the pinned corpus."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Any


def load_pin(project_root: Path) -> dict[str, Any]:
    with (project_root / "data" / "pin.json").open(encoding="utf-8") as f:
        return json.load(f)


def resolve_default_commit(repo_url: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repo_url, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    line = result.stdout.strip().split()[0]
    return line


def ensure_corpus(project_root: Path, pin: dict[str, Any], *, write_pin: bool = False) -> Path:
    corpus_dir = project_root / "data" / "corpus"
    sha = pin.get("commit_sha") or ""
    if sha in ("", "REPLACE_AFTER_FIRST_PIN"):
        sha = resolve_default_commit(pin["repo_url"])
        pin = {**pin, "commit_sha": sha}
        if write_pin:
            pin_path = project_root / "data" / "pin.json"
            with pin_path.open("w", encoding="utf-8") as f:
                json.dump(pin, f, indent=2)
                f.write("\n")

    if not (corpus_dir / ".git").is_dir():
        corpus_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", pin["repo_url"], str(corpus_dir)], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", sha], cwd=corpus_dir, check=True)
    subprocess.run(["git", "checkout", sha], cwd=corpus_dir, check=True)
    return corpus_dir


def _excluded(relative: str, exclude_globs: list[str]) -> bool:
    parts = relative.split("/")
    for pattern in exclude_globs:
        pat = pattern.replace("**/", "").replace("/**", "")
        if pat in relative or pat in parts:
            return True
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(relative, pattern.lstrip("/")):
            return True
    if relative.startswith("test_") and relative.endswith(".py"):
        return True
    return False


def iter_corpus_files(
    corpus_dir: Path,
    scope_paths: list[str],
    exclude_globs: list[str],
    max_file_bytes: int,
) -> list[Path]:
    files: list[Path] = []
    for scope in scope_paths:
        root = corpus_dir / scope
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(corpus_dir).as_posix()
            if _excluded(relative, exclude_globs):
                continue
            if path.stat().st_size > max_file_bytes:
                continue
            if not relative.endswith((".py", ".md", ".mdx")):
                continue
            files.append(path)
    return sorted(files, key=lambda p: p.as_posix())