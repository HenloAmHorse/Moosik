"""Generate temporary research mutants only; never edit the source in place."""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--evidence",type=Path,required=True)
a=p.parse_args()
a.evidence.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name("cqt_probe.rs").read_text(encoding="utf-8")
cases=[
    ("reversed_phase","C::cis(-TAU * f * (i as f64 - h as f64) / sr as f64)","C::cis(TAU * f * (i as f64 - h as f64) / sr as f64)","tests::direct_matches_independent_reference"),
    ("wrong_incoming","incoming: C::cis(-w * (h + 1) as f64)","incoming: C::cis(-w * h as f64)","tests::sliding_matches_reference_across_rebases_edges_and_noninteger_hops"),
    ("missing_hann_sideband","self.rect[2].sum.scale(0.25)","self.rect[2].sum.scale(0.0)","tests::sliding_matches_reference_across_rebases_edges_and_noninteger_hops"),
    ("wrong_gain","let gain = 2.0 / weights.iter().sum::<f64>();","let gain = 1.0 / weights.iter().sum::<f64>();","tests::tone_gain_phase_and_impulse_have_independent_analytic_checks"),
    ("rounded_hop_drift","((i as u64 * sr as u64) / fps as u64) as usize","i * (sr as f64 / fps as f64).round() as usize","tests::frame_centres_do_not_drift_and_cap_is_explicit"),
]
results=[]
for name,old,new,test in cases:
    if source.count(old)!=1:
        raise SystemExit(f"ambiguous / absent mutation: {name}")
    folder=a.evidence/name
    folder.mkdir(exist_ok=False)
    src=folder/"mutant.rs"
    src.write_text(source.replace(old,new),encoding="utf-8")
    exe=folder/"test.exe"
    build=subprocess.run(["rustc","--edition","2024","--test","-O",str(src),"-o",str(exe)],capture_output=True,text=True,timeout=90)
    (folder/"build.log").write_text(build.stdout+build.stderr,encoding="utf-8")
    if build.returncode:
        raise SystemExit(f"mutant {name} did not compile; not a caught mutant")
    run=subprocess.run([str(exe),"--exact",test,"--nocapture"],capture_output=True,text=True,timeout=30)
    (folder/"test.log").write_text(run.stdout+run.stderr,encoding="utf-8")
    caught=run.returncode!=0 and "1 failed" in run.stdout
    results.append(dict(name=name,test=test,exit=run.returncode,caught=caught))
    if not caught:
        raise SystemExit(f"mutant survived: {name}")
(a.evidence/"results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
print(f"{len(results)}/{len(cases)} compiled mutants rejected by the named tests")
