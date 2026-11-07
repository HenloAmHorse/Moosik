"""Re-enumerate and assemble 1.5.5 Windows prereview notices without builds.

Usage: python notices155.py WORKTREE EVIDENCE_DIR [--install]

Reuses the immutable 1.5.4 assembler and its retained, SHA-checked upstream
evidence. All changes to that assembler are single-match checked substitutions
recorded in the output. No old evidence is edited. Cargo tree is locked/offline;
every selected registry archive is checked against the current Cargo.lock and
every vendored upstream archive against recorded provenance. Moosik's original
local codec leaf is accounted for separately, not counted as third-party code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib


LEGACY = Path(r"C:\Users\PC\Programming\moosik-v154-release\scripts\notices154.py")
LEGACY_SHA = "e7b4673671b90d2cc2edb447de0eaa345ee3341e5f32fd7521444ff05b374c03"
OLD_HOME = Path(r"C:\Users\PC\Programming\moosik-153-home")
OLD_MAP = OLD_HOME / "raw/a12/e/notice_bundle/NOTICE_MAP.json"
OLD_TEXTS = OLD_MAP.parent / "texts"
OLD_FETCH = OLD_HOME / "raw/close/notices/fetch/results.json"
OWN_CODEC = "moosik-codec-opus@0.1.0"
NEW_REGISTRY = {
    "cmake@0.1.58",
    "opusic-sys@0.7.5",
    "symphonia-codec-alac@0.5.5",
    "symphonia-format-caf@0.5.5",
}
VENDOR = {
    "eframe@0.31.1": "vendor/eframe-0.31.1",
    "egui@0.31.1": "vendor/egui-0.31.1",
    "rustfft@6.4.1": "vendor/rustfft-6.4.1",
    "rodio@0.20.1": "vendor/rodio-0.20.1",
    "symphonia@0.5.5": "vendor/symphonia-0.5.5",
    "symphonia-format-ogg@0.5.5": "vendor/symphonia-format-ogg-0.5.5",
    "symphonia-format-mkv@0.5.5": "vendor/symphonia-format-mkv-0.5.5",
}


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    body = path.read_bytes()
    if digest(body) != expected:
        raise RuntimeError(f"Hash mismatch: {path}")
    return body


def record_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command(worktree: Path, *args: str) -> str:
    result = subprocess.run(args, cwd=worktree, check=True, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.stdout


def cargo_packages(worktree: Path, evidence: Path, edges: str, filename: str) -> set[str]:
    raw = command(worktree, "cargo", "tree", "--locked", "--offline", "--target",
                  "x86_64-pc-windows-msvc", "--edges", edges, "--prefix", "none", "--format", "{p}")
    (evidence / (filename + ".raw.txt")).write_text(raw, encoding="utf-8")
    packages = set()
    for line in raw.splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+) v(\S+)(?: |$)", line)
        if not match:
            raise RuntimeError(f"Unexpected cargo tree line: {line!r}")
        if match[1] != "moosik":
            packages.add(f"{match[1]}@{match[2]}")
    # Keep the complete graph in raw output, and make the assembler's input
    # explicitly a third-party-only set rather than mislabel original app code.
    third_party = packages - {OWN_CODEC}
    (evidence / filename).write_text("\n".join(sorted(third_party)) + "\n", encoding="utf-8")
    return packages


def archive_texts(archive: Path, name: str, version: str, paths: list[str]) -> tuple[dict, list[dict], dict[str, bytes]]:
    prefix = f"{name}-{version}/"
    with tarfile.open(archive) as tar:
        metadata = tomllib.loads(tar.extractfile(prefix + "Cargo.toml").read().decode("utf-8"))
        package = metadata["package"]
        if (package["name"], package["version"]) != (name, version):
            raise RuntimeError(f"Wrong archive identity: {archive}")
        bodies = {path: tar.extractfile(prefix + path).read() for path in paths}
        vcs_member = tar.getmember(prefix + ".cargo_vcs_info.json")
        vcs = json.loads(tar.extractfile(vcs_member).read())
    texts = [{"source": f"package archive ({name}-{version}.crate): {path}",
              "sha256": digest(body), "kind": "license-or-copyright"}
             for path, body in bodies.items()]
    return package, texts, {"vcs": vcs, "bodies": bodies}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--install", action="store_true")
    options = parser.parse_args()
    worktree, evidence = options.worktree.resolve(), options.evidence.resolve()
    if evidence.exists():
        raise RuntimeError(f"Use a new evidence directory, refusing overwrite: {evidence}")
    evidence.mkdir(parents=True)
    (evidence / "texts").mkdir()
    (evidence / "inputs").mkdir()
    original = checked(LEGACY, LEGACY_SHA).decode("utf-8").replace("\r\n", "\n")
    (evidence / "inputs/notices154.py").write_bytes(LEGACY.read_bytes())
    (evidence / "inputs/THIRD_PARTY_NOTICES.before.txt").write_bytes((worktree / "THIRD_PARTY_NOTICES.txt").read_bytes())
    all_packages = cargo_packages(worktree, evidence, "normal,build", "tree_all.txt")
    normal_packages = cargo_packages(worktree, evidence, "normal", "tree_normal.txt")
    all_third_party = all_packages - {OWN_CODEC}
    if OWN_CODEC not in all_packages:
        raise RuntimeError("The expected original Moosik codec is absent from the graph")
    notice_map = json.loads(OLD_MAP.read_text(encoding="utf-8"))
    old_rows = {row["package"]: row for row in notice_map["packages"] if row["in_selected_windows_tree"]}
    if all_third_party - set(old_rows) != NEW_REGISTRY or set(old_rows) - all_third_party:
        raise RuntimeError(f"Unreviewed graph delta: +{sorted(all_third_party - set(old_rows))}, "
                           f"-{sorted(set(old_rows) - all_third_party)}")
    rows = copy.deepcopy(old_rows)
    lock_bytes = (worktree / "Cargo.lock").read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    locked = {f"{p['name']}@{p['version']}": p for p in lock["package"]}
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    caches = list((cargo_home / "registry/cache").glob("*"))
    archives = {}
    archive_records = []
    for key in sorted(all_third_party):
        name, version = key.split("@", 1)
        package = locked[key]
        expected = package.get("checksum")
        if key in VENDOR:
            if package.get("source"):
                raise RuntimeError(f"Expected a local vendored package: {key}")
            expected = (old_rows[key]["lock_checksum_archive_sha256"] or
                        old_rows[key]["vendor"]["upstream_archive_sha256"])
            vendor = worktree / VENDOR[key]
            manifest = tomllib.loads((vendor / "Cargo.toml").read_text(encoding="utf-8"))["package"]
            if (manifest["name"], manifest["version"]) != (name, version):
                raise RuntimeError(f"Vendored manifest identity mismatch: {key}")
            if not (vendor / "PATCH.md").is_file():
                raise RuntimeError(f"Missing vendored patch provenance: {key}")
            rows[key]["vendor"] = {**(rows[key].get("vendor") or {}), "dir": VENDOR[key]}
        elif package.get("source") != "registry+https://github.com/rust-lang/crates.io-index" or not expected:
            raise RuntimeError(f"Unreviewed package source: {key}")
        matches = [cache / f"{name}-{version}.crate" for cache in caches
                   if (cache / f"{name}-{version}.crate").is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"Need one retained published archive for {key}: {matches}")
        archive = matches[0]
        checked(archive, expected)
        archives[key] = archive
        archive_records.append({"package": key, "archive": str(archive), "sha256": expected,
                                "binding": "retained upstream archive provenance" if key in VENDOR else "current Cargo.lock",
                                "current_source": VENDOR[key] if key in VENDOR else package["source"]})
        if (key in old_rows and old_rows[key]["lock_checksum_archive_sha256"] is not None
                and old_rows[key]["lock_checksum_archive_sha256"] != expected):
            raise RuntimeError(f"Old and current archive identity disagree: {key}")
    # New MPL crates are at exactly the same packaged workspace commit as the
    # already-verified Symphonia LICENSE, not guessed from a generic SPDX label.
    for key in sorted(NEW_REGISTRY):
        name, version = key.split("@", 1)
        paths = (["LICENSE-APACHE", "LICENSE-MIT"] if name == "cmake" else
                 ["LICENSE", "opus/COPYING", "opus/AUTHORS"] if name == "opusic-sys" else [])
        package, texts, provenance = archive_texts(archives[key], name, version, paths)
        if name.startswith("symphonia-"):
            if provenance["vcs"]["git"]["sha1"] != "6d533f26150953a882a6a111ebd13f0abf7129d5":
                raise RuntimeError(f"New Symphonia crate commit not covered by retained LICENSE: {key}")
            texts = copy.deepcopy(old_rows["symphonia@0.5.5"]["texts"])
        for body in provenance["bodies"].values():
            (evidence / "texts" / (digest(body) + ".txt")).write_bytes(body)
        rows[key] = {"package": key, "license_expression": package["license"],
                     "vendor": None, "texts": texts, "in_selected_windows_tree": True}
        archive_records_by_key = next(r for r in archive_records if r["package"] == key)
        archive_records_by_key["packaged_vcs"] = provenance["vcs"]
    for key in VENDOR:
        for text in rows[key]["texts"]:
            if text["source"].startswith("package archive"):
                filename = text["source"].split(": ", 1)[1].split(" ", 1)[0]
                checked(worktree / VENDOR[key] / filename, text["sha256"])
    opus_license = (evidence / "texts" / (rows["opusic-sys@0.7.5"]["texts"][0]["sha256"] + ".txt")).read_bytes()
    if opus_license != (worktree / "third_party/opusic-sys-0.7.5-LICENSE.txt").read_bytes():
        raise RuntimeError("Existing standalone Opus notice does not match the verified archive")
    own_manifest = tomllib.loads((worktree / "vendor/moosik-codec-opus/Cargo.toml").read_text(encoding="utf-8"))["package"]
    if own_manifest["license"] != "AGPL-3.0-only" or own_manifest["name"] != "moosik-codec-opus":
        raise RuntimeError("Original local codec identity/license changed")

    # Adapt only known pieces of the retained generator. All upstream text
    # resolution, exception policies and embedded-data handling stay in it.
    source = original
    replacements = []
    def replace(before: str, after: str) -> None:
        nonlocal source
        if source.count(before) != 1:
            raise RuntimeError(f"Legacy assembler anchor changed: {before[:90]!r}")
        source = source.replace(before, after)
        replacements.append({"before": before, "after": after})
    replace("DEP = r'C:\\Users\\PC\\Programming\\moosik-v154-release\\deps'", "DEP = CURRENT_DEP")
    replace("nm = json.load(open(os.path.join(B, 'NOTICE_MAP.json'), encoding='utf-8'))\nrows = {r['package']: r for r in nm['packages'] if r['in_selected_windows_tree']}",
            "rows = CURRENT_ROWS")
    replace("tp = os.path.join(B, 'texts', t['sha256'] + '.txt')",
            "tp = os.path.join(B, 'texts', t['sha256'] + '.txt')\n        if not os.path.isfile(tp):\n            candidate = os.path.join(CURRENT_TEXTS, t['sha256'] + '.txt')\n            if os.path.isfile(candidate): tp = candidate")
    replace("bad = [p for p, (ids, note) in pk.items() if not ids and not note]",
            """for v in CURRENT_VENDORS:
    if v not in ('eframe@0.31.1', 'egui@0.31.1', 'rustfft@6.4.1'):
        d = rows[v]['vendor']['dir']
        option = 'Distributed by Moosik under the MIT option of MIT OR Apache-2.0.' if v.startswith('rodio@') else 'Modified Source Code Form is retained under that directory under MPL-2.0, including PATCH.md; it is not the unmodified crates.io source.'
        pk[v][1] = f'Modified copy vendored at {d}; see {d}/PATCH.md. ' + option
pk['opusic-sys@0.7.5'][1] = 'Statically linked native binding and bundled libopus 1.6.1 (BSD-3-Clause). LICENSE and opus/COPYING are byte-identical; both source labels are retained with the upstream AUTHORS list. No standalone Opus DLL is required.'
bad = [p for p, (ids, note) in pk.items() if not ids and not note]""")
    replace("w('MOOSIK 1.5.4: THIRD-PARTY NOTICES')", "w('MOOSIK 1.5.5 PREREVIEW: THIRD-PARTY NOTICES')")
    replace("w('This file covers the third-party code and data in the Moosik 1.5.4 Windows x86_64')\nw('release executable')",
            "w('This file covers the third-party code and data in the Moosik 1.5.5 Windows x86_64')\nw('prereview executable (not a published release)')")
    replace("w('packages are marked; their notices are included anyway. Every package archive was')\nw('checked against the checksum in Cargo.lock.')",
            "w('packages are marked; their notices are included anyway. Registry archives were checked')\nw('against the current Cargo.lock; vendored upstream archives against retained provenance.')\nw('Original moosik-codec-opus 0.1.0 is Moosik AGPL-3.0-only code, accounted for separately')\nw('and not counted as a third-party package.')")
    replace("w('  third_party/rust-std-COPYRIGHT-library.html       (Rust standard library notices)')",
            "w('  third_party/rust-std-COPYRIGHT-library.html       (Rust standard library notices)')\nw('  third_party/MPL-2.0.txt                        (MPL-2.0 license text)')\nw('  third_party/opusic-sys-0.7.5-LICENSE.txt          (libopus BSD notice)')")
    replace("mpl = sorted(p for p in rows if 'MPL' in rows[p]['license_expression'])\nw('MPL-2.0 packages, used unmodified: ' + ', '.join(mpl) + '.')",
            "mpl = sorted(p for p in rows if 'MPL' in rows[p]['license_expression'] and p not in CURRENT_VENDORS)\nw('MPL-2.0 packages, used unmodified: ' + ', '.join(mpl) + '.')")
    replace("w('https://github.com/pdeljanov/Symphonia. The MPL-2.0 text is in Part 4.')",
            "w('https://github.com/pdeljanov/Symphonia. The MPL-2.0 text is in Part 4.')\nw('The modified symphonia, symphonia-format-ogg and symphonia-format-mkv sources are')\nw('available in Moosik vendor/ at the corresponding exact version directories, with')\nw('their original MPL source headers and PATCH.md change records. They are not described')\nw('as unmodified. The corresponding Moosik source snapshot must accompany this binary.')")
    replace("for v in ('eframe@0.31.1', 'egui@0.31.1', 'rustfft@6.4.1'):\n    w(f'{v}: {pk[v][1]}')",
            "for v in CURRENT_VENDORS:\n    w(f'{v}: {pk[v][1]}')")
    out = evidence / "generated"
    sys_argv = sys.argv
    try:
        sys.argv = [str(LEGACY), str(worktree), str(out)]
        namespace = {"CURRENT_DEP": str(evidence), "CURRENT_ROWS": rows,
                     "CURRENT_TEXTS": str(evidence / "texts"), "CURRENT_VENDORS": tuple(VENDOR)}
        exec(compile(source, "retained-notices154-with-checked-155-adaptation", "exec"), namespace)
    finally:
        sys.argv = sys_argv
    # The generated file references these two already-verified companion files.
    for path in ("third_party/MPL-2.0.txt", "third_party/opusic-sys-0.7.5-LICENSE.txt"):
        shutil.copyfile(worktree / path, out / path)
    notice = (out / "THIRD_PARTY_NOTICES.txt").read_bytes()
    package_lines = re.findall(rb"^([A-Za-z0-9_+.-]+@[\S]+)  \[", notice, flags=re.MULTILINE)
    if {line.decode() for line in package_lines} != all_third_party or len(package_lines) != len(all_third_party):
        raise RuntimeError("Emitted package inventory is not the exact current graph")
    modified = sorted(key for key in VENDOR if key.startswith("symphonia"))
    if any(key in notice.decode().split("MPL-2.0 packages, used unmodified: ")[1].split("\n", 1)[0] for key in modified):
        raise RuntimeError("Modified MPL source mislabeled unmodified")
    provenance = {
        "scope": "locked default Windows x86_64 normal/build graph; no build or execution",
        "head": command(worktree, "git", "rev-parse", "HEAD").strip(),
        "cargo_lock_sha256": digest(lock_bytes), "legacy_generator": str(LEGACY),
        "legacy_generator_sha256": LEGACY_SHA,
        "retained_notice_map_sha256": digest(OLD_MAP.read_bytes()),
        "retained_fetch_map_sha256": digest(OLD_FETCH.read_bytes()),
        "all_dependency_nodes_excluding_app": len(all_packages),
        "third_party_packages": len(all_third_party),
        "normal_third_party_packages": len(normal_packages - {OWN_CODEC}),
        "original_local_code": [OWN_CODEC],
        "added_third_party_packages": sorted(NEW_REGISTRY), "removed_third_party_packages": [],
        "archive_checks": archive_records,
        "notices_sha256": digest(notice), "notices_bytes": len(notice),
        "text_count": len(namespace["texts"]), "unresolved": [],
        "source_snapshot_required_for_modified_mpl": modified,
        "graph_output_exactly_matches_emitted_package_inventory": True,
    }
    record_json(evidence / "PROVENANCE.json", provenance)
    record_json(evidence / "ASSEMBLER_ADAPTATION.json", replacements)
    record_json(evidence / "CURRENT_NOTICE_MAP.json", {"packages": [rows[key] for key in sorted(rows)]})
    (evidence / "adapted_notices155.py").write_text(source, encoding="utf-8")
    if options.install:
        for path in ("THIRD_PARTY_NOTICES.txt", "third_party/MPL-2.0.txt", "third_party/opusic-sys-0.7.5-LICENSE.txt",
                     "third_party/mecab-ipadic-2.7.0-20250920-COPYING", "third_party/rust-std-COPYRIGHT-library.html"):
            shutil.copyfile(out / path, worktree / path)
    print(json.dumps({key: provenance[key] for key in ("third_party_packages", "text_count", "notices_sha256", "unresolved")}, indent=2))


if __name__ == "__main__":
    main()
