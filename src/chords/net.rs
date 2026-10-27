//! ChordNet (Jiang et al., ISMIR 2019; MIT): a CNN over the CQT, a
//! bidirectional LSTM and six softmax heads (triad, bass, 7th, 9th, 11th,
//! 13th). Five trained copies are averaged. Written out by hand so it runs
//! inside Moosik with nothing to install but the weights.

use super::pickle::Tensor;
use rayon::prelude::*;
use std::collections::HashMap;

/// CQT bins the network looks at: 252 of the 288, from bin 18.
pub const FIRST_BIN: usize = 18;
pub const SPEC: usize = 252;
/// Head sizes, in output order.
pub const HEADS: [usize; 6] = [73, 13, 4, 4, 3, 3];
const HIDDEN: usize = 96;

struct Conv {
    w: Vec<f32>, // [out][in][3][3]
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    pad_f: usize,
}

pub struct Net {
    convs: Vec<Conv>,
    lstm: [(Vec<f32>, Vec<f32>, Vec<f32>); 2], // (w_ih [4H][240], w_hh [4H][H], b_ih + b_hh), forward and reverse
    fc_w: Vec<f32>,                              // [100][192]
    fc_b: Vec<f32>,
}

fn take(t: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let x = t.remove(name).ok_or_else(|| format!("the weights lack {name}"))?;
    if x.shape != shape {
        return Err(format!("{name} is {:?}, expected {shape:?}", x.shape));
    }
    Ok(x.data)
}

impl Net {
    pub fn from_state(mut t: HashMap<String, Tensor>) -> Result<Net, String> {
        let spec = [("1a", 1, 16, 1), ("1b", 16, 16, 1), ("1c", 16, 16, 1), ("2a", 16, 32, 1), ("2b", 32, 32, 1), ("2c", 32, 32, 1),
            ("3a", 32, 64, 1), ("3b", 64, 64, 1), ("4a", 64, 80, 0), ("4b", 80, 80, 0)];
        let mut convs = Vec::new();
        for (n, cin, cout, pad_f) in spec {
            let w = take(&mut t, &format!("audio_feature_block.conv{n}.weight"), &[cout, cin, 3, 3])?;
            let b = take(&mut t, &format!("audio_feature_block.conv{n}.bias"), &[cout])?;
            convs.push(Conv { w, b, cin, cout, pad_f });
        }
        let mut dir = |suffix: &str| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
            let wi = take(&mut t, &format!("lstm1.weight_ih_l0{suffix}"), &[4 * HIDDEN, 240])?;
            let wh = take(&mut t, &format!("lstm1.weight_hh_l0{suffix}"), &[4 * HIDDEN, HIDDEN])?;
            let bi = take(&mut t, &format!("lstm1.bias_ih_l0{suffix}"), &[4 * HIDDEN])?;
            let bh = take(&mut t, &format!("lstm1.bias_hh_l0{suffix}"), &[4 * HIDDEN])?;
            Ok((wi, wh, bi.iter().zip(&bh).map(|(a, b)| a + b).collect()))
        };
        let lstm = [dir("")?, dir("_reverse")?];
        let fc_w = take(&mut t, "final_fc1.weight", &[100, 2 * HIDDEN])?;
        let fc_b = take(&mut t, "final_fc1.bias", &[100])?;
        Ok(Net { convs, lstm, fc_w, fc_b })
    }

    /// Per frame, the six heads' probabilities, concatenated (100 values).
    pub fn forward(&self, cqt: &[[f32; super::cqt::BINS]]) -> Vec<[f32; 100]> {
        let t = cqt.len();
        // [time][freq][channel]
        let mut f = SPEC;
        let mut x: Vec<f32> = cqt.iter().flat_map(|r| r[FIRST_BIN..FIRST_BIN + SPEC].iter().copied()).collect();
        let mut c = 1;
        for (i, conv) in self.convs.iter().enumerate() {
            debug_assert_eq!(conv.cin, c);
            let fo = f + 2 * conv.pad_f - 2;
            x = conv2d(&x, conv, t, f, fo);
            f = fo;
            c = conv.cout;
            instance_norm_selu(&mut x, c);
            // max-pool along frequency after 1c, 2c and 3b
            let pool = match i {
                2 | 5 => 3,
                7 => 4,
                _ => 1,
            };
            if pool > 1 {
                let fp = f / pool;
                let mut y = vec![f32::MIN; t * fp * c];
                for (tj, out) in y.chunks_mut(c).enumerate() {
                    let (ti, j) = (tj / fp, tj % fp);
                    for k in 0..pool {
                        let src = &x[(ti * f + j * pool + k) * c..][..c];
                        out.iter_mut().zip(src).for_each(|(o, s)| *o = o.max(*s));
                    }
                }
                x = y;
                f = fp;
            }
        }
        // Features per frame: channel-major (c * f + freq), 80 x 3 = 240.
        let feat: Vec<f32> = (0..t).flat_map(|ti| (0..c).flat_map(move |ci| (0..f).map(move |fi| (ci, ti, fi)))).map(|(ci, ti, fi)| x[(ti * f + fi) * c + ci]).collect();
        let n_in = c * f;
        let outs: Vec<Vec<f32>> = self.lstm.par_iter().enumerate().map(|(d, w)| lstm_dir(&feat, n_in, t, w, d == 1)).collect();
        (0..t)
            .into_par_iter()
            .map(|ti| {
                let h: Vec<f32> = outs[0][ti * HIDDEN..(ti + 1) * HIDDEN].iter().chain(&outs[1][ti * HIDDEN..(ti + 1) * HIDDEN]).copied().collect();
                let mut z = [0f32; 100];
                for (o, zo) in z.iter_mut().enumerate() {
                    *zo = self.fc_b[o] + self.fc_w[o * 2 * HIDDEN..(o + 1) * 2 * HIDDEN].iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
                }
                let mut at = 0;
                for n in HEADS {
                    let s = &mut z[at..at + n];
                    let m = s.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0.0;
                    for v in s.iter_mut() {
                        *v = (*v - m).exp();
                        sum += *v;
                    }
                    s.iter_mut().for_each(|v| *v /= sum);
                    at += n;
                }
                z
            })
            .collect()
    }
}

/// 3x3 convolution, time padded by one, frequency by `pad_f`, channels
/// last. The innermost loop runs over output channels with the weights laid
/// out contiguously for it, so it vectorises however narrow the frequency
/// axis has become.
fn conv2d(x: &[f32], k: &Conv, t: usize, f: usize, fo: usize) -> Vec<f32> {
    let (cin, cout) = (k.cin, k.cout);
    // [dt][df][in][out]
    let mut w = vec![0f32; 9 * cin * cout];
    for o in 0..cout {
        for i in 0..cin {
            for tap in 0..9 {
                w[(tap * cin + i) * cout + o] = k.w[(o * cin + i) * 9 + tap];
            }
        }
    }
    let mut y = vec![0f32; t * fo * cout];
    y.par_chunks_mut(fo * cout).enumerate().for_each(|(to, row)| conv_row(x, &w, k, t, f, to, row));
    y
}

/// One output time row of `conv2d`. With AVX2 the same arithmetic runs
/// eight lanes wide (no FMA, so results are identical either way).
fn conv_row(x: &[f32], w: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2, checked just above.
        return unsafe { conv_row_avx2(x, w, k, t, f, to, row) };
    }
    conv_row_any(x, w, k, t, f, to, row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv_row_avx2(x: &[f32], w: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    conv_row_any(x, w, k, t, f, to, row)
}

/// The network's channel counts get a fixed-size accumulator that stays in
/// registers; any other count takes the general loop.
#[inline(always)]
fn conv_row_any(x: &[f32], w: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    match k.cout {
        16 => conv_row_n::<16>(x, w, k, t, f, to, row),
        32 => conv_row_n::<32>(x, w, k, t, f, to, row),
        64 => conv_row_n::<64>(x, w, k, t, f, to, row),
        80 => conv_row_n::<80>(x, w, k, t, f, to, row),
        _ => conv_row_dyn(x, w, k, t, f, to, row),
    }
}

#[inline(always)]
fn conv_row_n<const C: usize>(x: &[f32], w: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    let cin = k.cin;
    for (j, out) in row.chunks_exact_mut(C).enumerate() {
        let mut acc: [f32; C] = k.b[..C].try_into().unwrap();
        for dt in 0..3 {
            let ti = to as isize + dt as isize - 1;
            if ti < 0 || ti >= t as isize {
                continue;
            }
            for df in 0..3 {
                // output column j reads input column j + df - pad_f
                let fi = j as isize + df as isize - k.pad_f as isize;
                if fi < 0 || fi >= f as isize {
                    continue;
                }
                let xin = &x[(ti as usize * f + fi as usize) * cin..][..cin];
                let wt = &w[(dt * 3 + df) * cin * C..][..cin * C];
                for (xv, wi) in xin.iter().zip(wt.chunks_exact(C)) {
                    let wi: &[f32; C] = wi.try_into().unwrap();
                    for o in 0..C {
                        acc[o] += xv * wi[o];
                    }
                }
            }
        }
        out.copy_from_slice(&acc);
    }
}

#[inline(always)]
fn conv_row_dyn(x: &[f32], w: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    let (cin, cout) = (k.cin, k.cout);
    for (j, acc) in row.chunks_mut(cout).enumerate() {
        acc.copy_from_slice(&k.b);
        for dt in 0..3 {
            let ti = to as isize + dt as isize - 1;
            if ti < 0 || ti >= t as isize {
                continue;
            }
            for df in 0..3 {
                let fi = j as isize + df as isize - k.pad_f as isize;
                if fi < 0 || fi >= f as isize {
                    continue;
                }
                let xin = &x[(ti as usize * f + fi as usize) * cin..][..cin];
                let wt = &w[(dt * 3 + df) * cin * cout..][..cin * cout];
                for (xv, wi) in xin.iter().zip(wt.chunks_exact(cout)) {
                    acc.iter_mut().zip(wi).for_each(|(a, b)| *a += xv * b);
                }
            }
        }
    }
}

/// InstanceNorm2d (no affine, eps 1e-5) over each channel's whole map, then
/// SELU. `x` is channels last.
fn instance_norm_selu(x: &mut [f32], c: usize) {
    const SCALE: f32 = 1.050_700_987_355_480_5;
    const ALPHA: f32 = 1.673_263_242_354_377_3;
    let n = x.len() / c;
    let (sum, sq) = x
        .par_chunks(c * 256)
        .map(|blk| {
            let (mut s, mut q) = (vec![0f64; c], vec![0f64; c]);
            for px in blk.chunks_exact(c) {
                for (ci, &v) in px.iter().enumerate() {
                    s[ci] += v as f64;
                    q[ci] += v as f64 * v as f64;
                }
            }
            (s, q)
        })
        .reduce(|| (vec![0f64; c], vec![0f64; c]), |(mut s, mut q), (s2, q2)| {
            s.iter_mut().zip(&s2).for_each(|(a, b)| *a += b);
            q.iter_mut().zip(&q2).for_each(|(a, b)| *a += b);
            (s, q)
        });
    let mean: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
    let inv: Vec<f64> = sq.iter().zip(&mean).map(|(q, m)| 1.0 / ((q / n as f64 - m * m).max(0.0) + 1e-5).sqrt()).collect();
    x.par_chunks_mut(c).for_each(|px| {
        for (ci, v) in px.iter_mut().enumerate() {
            let z = ((*v as f64 - mean[ci]) * inv[ci]) as f32;
            *v = SCALE * if z > 0.0 { z } else { ALPHA * (z.exp() - 1.0) };
        }
    });
}

/// One direction of the LSTM over the whole sequence (gates i, f, g, o).
fn lstm_dir(x: &[f32], n_in: usize, t: usize, (wi, wh, b): &(Vec<f32>, Vec<f32>, Vec<f32>), reverse: bool) -> Vec<f32> {
    let g4 = 4 * HIDDEN;
    // Weights transposed ([in][gate]) so each input adds a whole gate row at once.
    let tr = |w: &[f32], n: usize| -> Vec<f32> { (0..n * g4).map(|i| w[(i % g4) * n + i / g4]).collect() };
    let (wi, wh) = (tr(wi, n_in), tr(wh, HIDDEN));
    // Input projections for every frame at once.
    let mut pre = vec![0f32; t * g4];
    pre.par_chunks_mut(g4).enumerate().for_each(|(ti, acc)| {
        acc.copy_from_slice(b);
        axpy_rows(acc, &x[ti * n_in..(ti + 1) * n_in], &wi);
    });
    let mut out = vec![0f32; t * HIDDEN];
    let (mut h, mut cs) = (vec![0f32; HIDDEN], vec![0f32; HIDDEN]);
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let mut gates = vec![0f32; g4];
    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        gates.copy_from_slice(&pre[ti * g4..(ti + 1) * g4]);
        axpy_rows(&mut gates, &h, &wh);
        for j in 0..HIDDEN {
            let (i, f, gg, o) = (sig(gates[j]), sig(gates[HIDDEN + j]), gates[2 * HIDDEN + j].tanh(), sig(gates[3 * HIDDEN + j]));
            cs[j] = f * cs[j] + i * gg;
            h[j] = o * cs[j].tanh();
        }
        out[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&h);
    }
    out
}

/// `acc += xs[i] * w[i]` over the rows of `w` (each `acc.len()` long).
fn axpy_rows(acc: &mut [f32], xs: &[f32], w: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2, checked just above.
        return unsafe { axpy_rows_avx2(acc, xs, w) };
    }
    axpy_rows_any(acc, xs, w)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpy_rows_avx2(acc: &mut [f32], xs: &[f32], w: &[f32]) {
    axpy_rows_any(acc, xs, w)
}

#[inline(always)]
fn axpy_rows_any(acc: &mut [f32], xs: &[f32], w: &[f32]) {
    for (xv, row) in xs.iter().zip(w.chunks_exact(acc.len())) {
        acc.iter_mut().zip(row).for_each(|(a, b)| *a += xv * b);
    }
}
