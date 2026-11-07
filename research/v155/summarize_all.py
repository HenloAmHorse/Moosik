"""Validate and summarize all first-batch lanes; no third-party Python modules."""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

root=Path(sys.argv[1])

def rows(name,expected):
    out=list(csv.DictReader((root/name).open(encoding="utf-8-sig",newline="")))
    if len(out)!=expected:
        raise SystemExit(f"{name}: expected {expected} rows, got {len(out)}")
    for row in out:
        for k,v in row.items():
            if k not in ("kind","equal_complex_and_norm_bits"):
                if not math.isfinite(float(v)):
                    raise SystemExit(f"{name}: nonfinite {k}")
    return out

def grouped(rs,keys,reps):
    out={}
    for r in rs:
        out.setdefault(tuple(r[k] for k in keys),[]).append(r)
    for key,group in out.items():
        if sorted(int(r["rep"]) for r in group)!=list(range(reps)):
            raise SystemExit(f"bad repetitions: {key}")
    return out

screen=rows("screen.csv",135)
if len(grouped(screen,["sr","bpo","freq"],3))!=45:
    raise SystemExit("wrong screen identities")
bank=rows("bank.csv",9)
fft=rows("fft.csv",36)
rt=rows("rt-fft.csv",15)
summary={"source_commit":"6b34309", "bank":[],"single_bin_fft":[],"isolated_live_fft":[],
         "note":"All CPU synthetic; CQT is f64 and research quantizer, live FFT f32. Not product throughput/pixel gates."}
for (sr,bpo,seconds),rs in grouped(bank,["sr","bpo","seconds"],3).items():
    plan=st.median(float(r["plan_ms"]) for r in rs)
    direct=st.median(float(r["direct_ms"]) for r in rs)
    hybrid=st.median(float(r["hybrid_ms"]) for r in rs)
    if min(direct,hybrid)<=0:
        raise SystemExit("nonpositive bank timing")
    summary["bank"].append(dict(sr=sr,bpo=bpo,seconds=seconds,bins=int(rs[0]["bins"]),plan_ms=plan,
        direct_ms=direct,hybrid_ms=hybrid,process_speedup=direct/hybrid,
        including_plan_speedup=(plan+direct)/(plan+hybrid)))
for (sr,bpo,f),rs in grouped(fft,["sr","bpo","freq"],3).items():
    d=st.median(float(r["direct_ms"]) for r in rs)
    v=st.median(float(r["fft_ms"]) for r in rs)
    if min(d,v)<=0:
        raise SystemExit("nonpositive FFT timing")
    summary["single_bin_fft"].append(dict(sr=sr,bpo=bpo,freq=f,direct_ms=d,fft_ms=v,speedup=d/v))
for (size,),rs in grouped(rt,["size"],5).items():
    if not all(r["equal_complex_and_norm_bits"]=="true" and int(r["reused_alloc_calls"])==0 for r in rs):
        raise SystemExit("failed live FFT exactness/allocation assertion")
    b=st.median(float(r["baseline_us"]) for r in rs)
    v=st.median(float(r["reused_us"]) for r in rs)
    summary["isolated_live_fft"].append(dict(size=int(size),baseline_us=b,reused_us=v,
        isolated_savings_percent=(b-v)/b*100,baseline_alloc_calls=int(rs[0]["baseline_alloc_calls"]),
        allocated_bytes_per_baseline_call=int(rs[0]["baseline_alloc_bytes"]),
        timing_status="trend: not a robust speed claim" if size=="4096" else "screen only, no app claim"))
summary["max_cqt_complex_error"]=max(float(r["max_complex_error"]) for r in screen+bank+fft)
summary["changed_research_codes"]=sum(int(r["changed_research_codes"]) for r in screen+bank+fft)
if summary["max_cqt_complex_error"]>2e-9:
    raise SystemExit("CQT numerical screen failed")
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
