"""Carry the 1.6.0 Windows notices forward to 1.6.1, checked.

Usage: python notices161.py WORKTREE [--install]

1.6.1 adds no package to the Windows executable: this refuses to run unless
the locked default Windows x86_64 normal/build graph is exactly the package
inventory THIRD_PARTY_NOTICES.txt lists, every package Cargo.lock gained since
v1.6.0 lies outside that graph (the offscreen UI tests' dev-dependencies),
none was removed, and the toolchain is the one the notices name. It then
changes the two version lines and prints what it changed. Nothing is built or
run besides `cargo tree`, `git show` and `rustc -V`.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

OWN_CODEC = "moosik-codec-opus@0.1.0"
RUSTC = "rustc 1.94.0 (4a4ef493e 2026-03-02)"


def run(worktree: Path, *args: str) -> str:
    return subprocess.run(args, cwd=worktree, check=True, capture_output=True, text=True).stdout


def graph(worktree: Path) -> set[str]:
    raw = run(worktree, "cargo", "tree", "--locked", "--offline", "--target", "x86_64-pc-windows-msvc",
              "--edges", "normal,build", "--prefix", "none", "--format", "{p}")
    out = set()
    for line in raw.splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+) v(\S+)(?: |$)", line)
        if not m:
            raise RuntimeError(f"Unexpected cargo tree line: {line!r}")
        if m[1] != "moosik":
            out.add(f"{m[1]}@{m[2]}")
    return out - {OWN_CODEC}


def lock_packages(text: str) -> dict[str, dict]:
    return {f"{p['name']}@{p['version']}": p for p in tomllib.loads(text)["package"] if p["name"] != "moosik"}


def main() -> None:
    worktree = Path(sys.argv[1]).resolve()
    install = "--install" in sys.argv[2:]
    notices_path = worktree / "THIRD_PARTY_NOTICES.txt"
    text = notices_path.read_text(encoding="utf-8")

    listed = set(re.findall(r"^([A-Za-z0-9_+.-]+@\S+)  \[", text, flags=re.MULTILINE))
    current = graph(worktree)
    if current != listed:
        raise RuntimeError(f"Graph changed since the notices: +{sorted(current - listed)} -{sorted(listed - current)}")
    before = lock_packages(run(worktree, "git", "show", "v1.6.0:Cargo.lock"))
    after = lock_packages((worktree / "Cargo.lock").read_text(encoding="utf-8"))
    added, removed = set(after) - set(before), set(before) - set(after)
    if removed:
        raise RuntimeError(f"Cargo.lock lost packages since v1.6.0: {sorted(removed)}")
    if added & current:
        raise RuntimeError(f"New packages reach the executable: {sorted(added & current)}")
    if run(worktree, "rustc", "-V").strip() != RUSTC:
        raise RuntimeError("Toolchain differs from the one the notices name")

    changes = []

    def replace(old: str, new: str) -> None:
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError(f"Anchor not found exactly once: {old[:80]!r}")
        text = text.replace(old, new)
        changes.append(old.splitlines()[0][:70])

    replace("MOOSIK 1.6.0: THIRD-PARTY NOTICES", "MOOSIK 1.6.1: THIRD-PARTY NOTICES")
    replace("third-party code and data in the Moosik 1.6.0 Windows x86_64",
            "third-party code and data in the Moosik 1.6.1 Windows x86_64")
    for c in changes:
        print("changed:", c)
    print(f"graph: {len(current)} third-party packages, unchanged")
    print(f"Cargo.lock: {len(added)} dev-only packages added, none in the graph: {', '.join(sorted(added))}")
    if install:
        notices_path.write_text(text, encoding="utf-8", newline="\n")
        print("installed", notices_path)


if __name__ == "__main__":
    main()
