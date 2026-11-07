"""Analytical real-tone CQT coefficient mixtures, not perceptual audio quality.

Importing window_response validates/prints its table, so this script reimplements
the closed-form response locally and independently checks selected cases by a
slow finite-window sum. No actual music, UI or ASLT comparison is claimed.
"""
import csv
import math
import sys

WINDOWS={"Hann":[0.5,0.5],"Hamming":[0.54,0.46],
         "Blackman":[0.42,0.5,0.08],"Blackman-Harris-4":[0.35875,0.48829,0.14128,0.01168]}
SR=48000
BPO=24
Q=1/(2**(1/BPO)-1)

def rectangle(omega,h):
    omega=math.remainder(omega,2*math.pi)
    if abs(omega)<1e-14:
        return 2*h+1
    return math.sin((h+0.5)*omega)/math.sin(omega/2)

def response(coeff,omega,h):
    return sum(a*0.5*(rectangle(omega-m*math.pi/h,h)+rectangle(omega+m*math.pi/h,h))
               for m,a in enumerate(coeff))

def coefficient(coeff,centre,tone,amplitude,phase,h):
    gain=response(coeff,0,h)
    diff=response(coeff,2*math.pi*(tone-centre)/SR,h)/gain
    negative=response(coeff,-2*math.pi*(tone+centre)/SR,h)/gain
    return amplitude*complex(math.cos(phase)*(diff+negative),math.sin(phase)*(diff-negative))

for coeff in WINDOWS.values():
    for phase in [0.0,0.41,math.pi/2]:
        h=83;centre=450;f=440
        gain=sum(sum(a*math.cos(m*math.pi*j/h) for m,a in enumerate(coeff)) for j in range(-h,h+1))
        direct=sum(math.cos(2*math.pi*f*j/SR+phase)
                   *sum(a*math.cos(m*math.pi*j/h) for m,a in enumerate(coeff))
                   *complex(math.cos(-2*math.pi*centre*j/SR),math.sin(-2*math.pi*centre*j/SR))
                   for j in range(-h,h+1))*2/gain
        assert abs(direct-coefficient(coeff,centre,f,1,phase,h))<1e-11

w=csv.writer(sys.stdout,lineterminator="\n")
w.writerow(["window","support_policy","bpo","strong_hz","weak_hz","spacing_cqt_bins","half_samples",
            "weak_amplitude","strong_leakage_into_weak_bin_db","leakage_relative_to_weak_db",
            "min_combined_magnitude_over_4_phases","max_combined_magnitude_over_4_phases",
            "full_impulse_half_amplitude_width_s"])
def half_power_delta(coeff,h):
    gain=response(coeff,0,h)
    lo,hi=0.0,2.0
    for _ in range(60):
        mid=(lo+hi)/2
        magnitude=abs(response(coeff,math.pi*mid/h,h)/gain)
        if magnitude>2**-0.5:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2

hann_delta=half_power_delta(WINDOWS["Hann"],4096)
for name,coeff in WINDOWS.items():
  for policy in ["same_support_Q","approximately_matched_Hann_bandwidth"]:
    factor=1.0 if policy=="same_support_Q" else half_power_delta(coeff,4096)/hann_delta
    for spacing in [1,2,3,4,6]:
        strong=440.0;weak=strong*2**(spacing/BPO);h=math.ceil(factor*Q*SR/(2*weak))
        leakage=coefficient(coeff,weak,strong,1.0,0.0,h)
        weak_only=coefficient(coeff,weak,weak,0.01,0.0,h)
        mags=[abs(leakage+coefficient(coeff,weak,weak,0.01,p,h)) for p in [0,math.pi/2,math.pi,3*math.pi/2]]
        # Impulse response envelope half-amplitude crossing, not -3 dB width.
        def env(j):
            return sum(a*math.cos(m*math.pi*j/h) for m,a in enumerate(coeff))
        j=next(j for j in range(h+1) if env(j)<=env(0)*0.5)
        w.writerow([name,policy,BPO,strong,f"{weak:.6f}",spacing,h,0.01,
                    f"{20*math.log10(max(abs(leakage),1e-30)):.6f}",
                    f"{20*math.log10(max(abs(leakage)/abs(weak_only),1e-30)):.6f}",
                    f"{min(mags):.10g}",f"{max(mags):.10g}",f"{2*j/SR:.9f}"])
