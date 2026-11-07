"""Validate the grouped-FFT screen; no product/pixel equivalence claim."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

folder=Path(sys.argv[1])
rows=list(csv.DictReader((folder/"grouped.csv").open(encoding="utf-8-sig",newline="")))
if len(rows)!=18:
    raise SystemExit(f"expected18 grouped rows, found{len(rows)}")
groups={}
for r in rows:
    if not all(math.isfinite(float(v)) for v in r.values()):
        raise SystemExit("nonfinite field")
    if int(r["peak_explicit_bytes"])>int(r["limit_bytes"]) or int(r["kernel_ffts"])!=int(r["bins"]):
        raise SystemExit("workspace bound or kernel count failed")
    if min(float(r["per_bin_ms"]),float(r["grouped_ms"]))<=0 or float(r["max_complex_error"])>2e-11:
        raise SystemExit("timing / complex error failed")
    groups.setdefault((r["sr"],r["bpo"],r["limit_bytes"]),[]).append(r)
out=[]
for (sr,bpo,limit),rs in groups.items():
    if sorted(int(r["rep"]) for r in rs)!=[0,1,2]:
        raise SystemExit("repetition identities failed")
    a=statistics.median(float(r["per_bin_ms"]) for r in rs)
    b=statistics.median(float(r["grouped_ms"]) for r in rs)
    out.append(dict(sr=int(sr),bpo=float(bpo),limit_bytes=int(limit),
                    per_bin_ms=a,grouped_ms=b,speedup=a/b,
                    max_complex_error=max(float(r["max_complex_error"]) for r in rs),
                    research_changed_codes=max(int(r["changed_research_codes"]) for r in rs),
                    input_ffts=int(rs[0]["input_ffts"]),inverse_ffts=int(rs[0]["inverse_ffts"]),
                    peak_explicit_bytes=max(int(r["peak_explicit_bytes"]) for r in rs)))
(folder/"grouped-summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,indent=2))
