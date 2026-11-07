"""Analytical DTFT of finite sum-of-cosines windows (no audio files).

Reports measured-in-model shape, not musical perceptual quality. Frequency units
are full-window DFT bins (sample_rate / (2*h)); h=4096. Independent slow direct
checks protect the analytical implementation before writing a table.
"""
import csv
import math
import sys

H = 4096
WINDOWS = {
    "Hann": [0.5, 0.5],
    "Hamming": [0.54, 0.46],
    "Blackman": [0.42, 0.50, 0.08],
    "Blackman-Harris-4": [0.35875, 0.48829, 0.14128, 0.01168],
}

def rect(a, h):
    a = math.remainder(a, 2*math.pi)
    return 2*h+1 if abs(a) < 1e-14 else math.sin((2*h+1)*a/2)/math.sin(a/2)

def response(coeff, delta, h):
    omega = math.pi*delta/h
    return sum(a*0.5*(rect(omega-m*math.pi/h,h)+rect(omega+m*math.pi/h,h))
               for m,a in enumerate(coeff))

for coeff in WINDOWS.values():
    for d in [0.0,0.17,0.5,1.25,2.99,4.7]:
        direct = sum(sum(a*math.cos(m*math.pi*j/47) for m,a in enumerate(coeff))
                     * math.cos(math.pi*d*j/47) for j in range(-47,48))
        assert abs(direct-response(coeff,d,47)) < 1e-10

w = csv.writer(sys.stdout,lineterminator="\n")
w.writerow(["window","half_power_full_width_bins","first_minimum_offset_bins",
            "peak_sidelobe_db_after_first_minimum","rectangle_components"])
for name,coeff in WINDOWS.items():
    gain = response(coeff,0,H)
    ys = [abs(response(coeff,i/1000,H)/gain) for i in range(20001)]
    half = next(i/1000 for i,v in enumerate(ys) if v <= 2**-0.5)
    first = next(i for i in range(1,len(ys)-1) if ys[i]<ys[i-1] and ys[i]<ys[i+1])
    side = max(ys[first+1:])
    w.writerow([name,f"{2*half:.3f}",f"{first/1000:.3f}",f"{20*math.log10(side):.2f}",2*len(coeff)-1])
