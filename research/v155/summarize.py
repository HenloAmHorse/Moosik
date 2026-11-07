"""Read isolated Rust probe CSV; no owner data, no package dependencies."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("csv", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = list(csv.DictReader(a.csv.open(encoding="utf-8-sig", newline="")))
if len(rows) != 135:
    raise SystemExit(f"expected 135 screening rows, found {len(rows)}")
groups = {}
for r in rows:
    key = (int(r["sr"]), float(r["bpo"]), float(r["freq"]))
    groups.setdefault(key, []).append(r)
out = []
for (sr, bpo, f), rs in sorted(groups.items()):
    if sorted(int(r["rep"]) for r in rs) != [0, 1, 2]:
        raise SystemExit("repetition identity mismatch")
    d = statistics.median(float(r["direct_ms"]) for r in rs)
    s = statistics.median(float(r["sliding_ms"]) for r in rs)
    err = max(float(r["max_complex_error"]) for r in rs)
    changed = max(int(r["changed_research_codes"]) for r in rs)
    if d <= 0 or s <= 0 or err > 2e-9:
        raise SystemExit(f"invalid time / error at {(sr,bpo,f)}")
    out.append(dict(sr=sr,bpo=bpo,freq=f,direct_median_ms=d,
                    sliding_median_ms=s,speedup=d/s,max_complex_error=err,
                    max_changed_research_codes=changed))
a.output.write_text(json.dumps({"scope":"single-bin f64 CPU, 2s chirp+noise, 360 frames, 3 reps; not app or bank throughput", "results":out}, indent=2)+"\n",encoding="utf-8")
for sr in [44100,48000,96000]:
    subset=[r for r in out if r["sr"]==sr]
    print(f"{sr} Hz: speedup range {min(r['speedup'] for r in subset):.2f}..{max(r['speedup'] for r in subset):.2f}; max complex error {max(r['max_complex_error'] for r in subset):.3g}; max changed codes {max(r['max_changed_research_codes'] for r in subset)}")
