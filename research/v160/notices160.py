"""Carry the 1.5.5 Windows notices forward to 1.6.0, checked.

Usage: python notices160.py WORKTREE [--install]

1.6.0 adds no package: this refuses to run unless the locked default Windows
x86_64 normal/build graph is exactly the package inventory the current
THIRD_PARTY_NOTICES.txt lists, and Cargo.lock differs from v1.5.5 only in
Moosik's own version. It then makes single-match checked substitutions (the
version lines, and the one new item: code ported from lv-chordia 1.1.0, whose
upstream LICENSE must be the pinned bytes in third_party/) and prints what it
changed. Nothing is built or run besides `cargo tree` and `git show`.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

OWN_CODEC = "moosik-codec-opus@0.1.0"
LV_LICENSE = "third_party/lv-chordia-1.1.0-LICENSE"
LV_LICENSE_SHA = "24fe07e4202d179f2235a1192db8e027dc3e1cd6c321ac59085a78eb2e8e0512"
LV_WHEEL_SHA = "292241f03e1029e6739ee7a2a91fbe348c25c97b8b3641db76cbef9f9dee0cb0"
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
    if lock_packages(run(worktree, "git", "show", "v1.5.5:Cargo.lock")) != lock_packages((worktree / "Cargo.lock").read_text(encoding="utf-8")):
        raise RuntimeError("Cargo.lock differs from v1.5.5 beyond Moosik's own version")
    if run(worktree, "rustc", "-V").strip() != RUSTC:
        raise RuntimeError("Toolchain differs from the one the notices name")
    lv = (worktree / LV_LICENSE).read_bytes()
    if hashlib.sha256(lv).hexdigest() != LV_LICENSE_SHA:
        raise RuntimeError(f"{LV_LICENSE} is not the pinned upstream text")

    changes = []

    def replace(before: str, after: str) -> None:
        nonlocal text
        if text.count(before) != 1:
            raise RuntimeError(f"Anchor not found exactly once: {before[:80]!r}")
        text = text.replace(before, after)
        changes.append(before.splitlines()[0][:70])

    replace("MOOSIK 1.5.5: THIRD-PARTY NOTICES", "MOOSIK 1.6.0: THIRD-PARTY NOTICES")
    replace("third-party code and data in the Moosik 1.5.5 Windows x86_64",
            "third-party code and data in the Moosik 1.6.0 Windows x86_64")
    replace("  third_party/opusic-sys-0.7.5-LICENSE.txt          (libopus BSD notice)\n",
            "  third_party/opusic-sys-0.7.5-LICENSE.txt          (libopus BSD notice)\n"
            "  third_party/lv-chordia-1.1.0-LICENSE              (lv-chordia MIT notice)\n")
    anchor = ("third_party/rust-std-COPYRIGHT-library.html, copied unmodified from the toolchain's\n"
              "share/doc/rust/COPYRIGHT-library.html (SHA-256 af70aaabed1b73e872f14f9130db37e09f3f4d73a5f7c598b9173697a5d2729f).\n")
    lv_text = lv.decode("utf-8").rstrip("\n")
    replace(anchor, anchor + f"""
1.5 Code ported from lv-chordia 1.1.0 (chord recognition)

Moosik's chord recognition (src/chords/) is a Rust port of the network, constant-Q
front end and chord vocabulary of lv-chordia 1.1.0 by Junyan Jiang, Ke Chen, Wei Li and
Gus Xia (ISMIR 2019), MIT License. It is not a Cargo package. The upstream LICENSE,
taken from the lv_chordia-1.1.0 wheel (SHA-256 {LV_WHEEL_SHA}),
is reproduced byte for byte in {LV_LICENSE} (SHA-256 {LV_LICENSE_SHA[:16]}...) and here:

{lv_text}

The model weights are not in this executable. They are downloaded from that wheel only
when the user asks, each file checked against a pinned SHA-256.
""")
    for c in changes:
        print("changed:", c)
    print(f"graph: {len(current)} third-party packages, unchanged; lv-chordia LICENSE verified")
    if install:
        notices_path.write_text(text, encoding="utf-8", newline="\n")
        print("installed", notices_path)


if __name__ == "__main__":
    main()
