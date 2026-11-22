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
    w: Vec<f32>, // [dt*3+df][in][out], packed in `from_state` for contiguous out-channel SIMD
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    pad_f: usize,
}

const PRE_TILE: usize = 16;
const REC_TILE: usize = 64;

pub struct Net {
    convs: Vec<Conv>,
    // Packed LSTM weights per direction (forward, reverse):
    // (wi_packed [4H/PRE_TILE][240][PRE_TILE], wh_packed [4H/REC_TILE][H][REC_TILE], b_ih + b_hh)
    lstm: [(Vec<f32>, Vec<f32>, Vec<f32>); 2],
    fc_w_tr: Vec<f32>, // [2*HIDDEN][100]
    fc_b: Vec<f32>,
}

fn pack_conv_weights(raw: &[f32], cin: usize, cout: usize) -> Vec<f32> {
    let mut w = vec![0f32; 9 * cin * cout];
    for o in 0..cout {
        for i in 0..cin {
            for tap in 0..9 {
                w[(tap * cin + i) * cout + o] = raw[(o * cin + i) * 9 + tap];
            }
        }
    }
    w
}

fn pack_lstm_tiled(raw: &[f32], n_in: usize, tile: usize) -> Vec<f32> {
    let g4 = 4 * HIDDEN;
    debug_assert_eq!(g4 % tile, 0);
    let tiles = g4 / tile;
    let mut out = vec![0f32; g4 * n_in];
    for t_idx in 0..tiles {
        for i in 0..n_in {
            for k in 0..tile {
                let g = t_idx * tile + k;
                out[(t_idx * n_in + i) * tile + k] = raw[g * n_in + i];
            }
        }
    }
    out
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
            let w_raw = take(&mut t, &format!("audio_feature_block.conv{n}.weight"), &[cout, cin, 3, 3])?;
            let b = take(&mut t, &format!("audio_feature_block.conv{n}.bias"), &[cout])?;
            let w = pack_conv_weights(&w_raw, cin, cout);
            convs.push(Conv { w, b, cin, cout, pad_f });
        }
        let mut dir = |suffix: &str| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
            let wi = take(&mut t, &format!("lstm1.weight_ih_l0{suffix}"), &[4 * HIDDEN, 240])?;
            let wh = take(&mut t, &format!("lstm1.weight_hh_l0{suffix}"), &[4 * HIDDEN, HIDDEN])?;
            let bi = take(&mut t, &format!("lstm1.bias_ih_l0{suffix}"), &[4 * HIDDEN])?;
            let bh = take(&mut t, &format!("lstm1.bias_hh_l0{suffix}"), &[4 * HIDDEN])?;
            Ok((pack_lstm_tiled(&wi, 240, PRE_TILE), pack_lstm_tiled(&wh, HIDDEN, REC_TILE), bi.iter().zip(&bh).map(|(a, b)| a + b).collect()))
        };
        let lstm = [dir("")?, dir("_reverse")?];
        let fc_w = take(&mut t, "final_fc1.weight", &[100, 2 * HIDDEN])?;
        let fc_b = take(&mut t, "final_fc1.bias", &[100])?;
        let n_fc = 2 * HIDDEN;
        let mut fc_w_tr = vec![0f32; n_fc * 100];
        for o in 0..100 {
            for i in 0..n_fc {
                fc_w_tr[i * 100 + o] = fc_w[o * n_fc + i];
            }
        }
        Ok(Net { convs, lstm, fc_w_tr, fc_b })
    }

    /// Per frame, the six heads' probabilities, concatenated (100 values).
    pub fn forward(&self, cqt: &[[f32; super::cqt::BINS]]) -> Vec<[f32; 100]> {
        let t = cqt.len();
        if t == 0 {
            return Vec::new();
        }
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
                let mut y = vec![0f32; t * fp * c];
                y.par_chunks_mut(fp * c).enumerate().with_min_len(16).for_each(|(ti, row)| {
                    for (j, out) in row.chunks_exact_mut(c).enumerate() {
                        let base = (ti * f + j * pool) * c;
                        out.copy_from_slice(&x[base..base + c]);
                        for k in 1..pool {
                            let src = &x[base + k * c..base + (k + 1) * c];
                            out.iter_mut().zip(src).for_each(|(o, s)| *o = o.max(*s));
                        }
                    }
                });
                x = y;
                f = fp;
            }
        }
        // Features per frame: channel-major (c * f + freq), 80 x 3 = 240.
        let n_in = c * f;
        let mut feat = vec![0f32; t * n_in];
        feat.par_chunks_mut(n_in).enumerate().with_min_len(64).for_each(|(ti, dst)| {
            let src = &x[ti * n_in..];
            for ci in 0..c {
                for fi in 0..f {
                    dst[ci * f + fi] = src[fi * c + ci];
                }
            }
        });
        let outs: Vec<Vec<f32>> = self.lstm.par_iter().enumerate().map(|(d, w)| lstm_dir(&feat, n_in, t, w, d == 1)).collect();
        let mut probs = vec![[0f32; 100]; t];
        probs.par_chunks_mut(16).enumerate().for_each(|(blk_i, chunk)| {
            let base_t = blk_i * 16;
            for (k, out_z) in chunk.iter_mut().enumerate() {
                let ti = base_t + k;
                let mut h = [0f32; 2 * HIDDEN];
                h[..HIDDEN].copy_from_slice(&outs[0][ti * HIDDEN..(ti + 1) * HIDDEN]);
                h[HIDDEN..].copy_from_slice(&outs[1][ti * HIDDEN..(ti + 1) * HIDDEN]);
                *out_z = fc_softmax(&h, &self.fc_w_tr, &self.fc_b);
            }
        });
        probs
    }
}

fn fc_softmax(h: &[f32; 2 * HIDDEN], fc_w_tr: &[f32], fc_b: &[f32]) -> [f32; 100] {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability verified just above.
        return unsafe { fc_softmax_avx2(h, fc_w_tr, fc_b) };
    }
    fc_softmax_any(h, fc_w_tr, fc_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc_softmax_avx2(h: &[f32; 2 * HIDDEN], fc_w_tr: &[f32], fc_b: &[f32]) -> [f32; 100] {
    fc_softmax_any(h, fc_w_tr, fc_b)
}

#[inline(always)]
fn fc_softmax_any(h: &[f32; 2 * HIDDEN], fc_w_tr: &[f32], fc_b: &[f32]) -> [f32; 100] {
    let mut acc = [0f32; 100];
    for (hi, row) in h.iter().zip(fc_w_tr.chunks_exact(100)) {
        for (a, w) in acc.iter_mut().zip(row) {
            *a += *w * *hi;
        }
    }
    let mut z = [0f32; 100];
    for o in 0..100 {
        z[o] = fc_b[o] + acc[o];
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
}

/// 3x3 convolution, time padded by one, frequency by `pad_f`, channels
/// last, using pre-packed `[dt*3+df][in][out]` weights from `Net::from_state`.
fn conv2d(x: &[f32], k: &Conv, t: usize, f: usize, fo: usize) -> Vec<f32> {
    if t == 0 || fo == 0 {
        return Vec::new();
    }
    let mut y = vec![0f32; t * fo * k.cout];
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");
    y.par_chunks_mut(fo * k.cout).enumerate().with_min_len(16).for_each(|(to, row)| {
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            // SAFETY: AVX2 availability verified once outside the parallel loop.
            unsafe { conv_row_avx2(x, k, t, f, fo, to, row) };
            return;
        }
        conv_row_any(x, k, t, f, fo, to, row);
    });
    y
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv_row_avx2(x: &[f32], k: &Conv, t: usize, f: usize, fo: usize, to: usize, row: &mut [f32]) {
    conv_row_any(x, k, t, f, fo, to, row)
}

/// The network's channel counts get a fixed-size accumulator that stays in
/// registers; any other count takes the general loop.
#[inline(always)]
fn conv_row_any(x: &[f32], k: &Conv, t: usize, f: usize, fo: usize, to: usize, row: &mut [f32]) {
    match k.cout {
        16 => conv_row_n::<16>(x, k, t, f, fo, to, row),
        32 => conv_row_n::<32>(x, k, t, f, fo, to, row),
        64 => conv_row_n::<64>(x, k, t, f, fo, to, row),
        80 => conv_row_n::<80>(x, k, t, f, fo, to, row),
        _ => conv_row_dyn(x, k, t, f, to, row),
    }
}

#[inline(always)]
fn conv_tap<const C: usize>(acc: &mut [f32; C], xin: &[f32], wt: &[f32]) {
    for (xv, wi) in xin.iter().zip(wt.chunks_exact(C)) {
        let wi: &[f32; C] = wi.try_into().unwrap();
        for o in 0..C {
            acc[o] += xv * wi[o];
        }
    }
}

#[inline(always)]
fn conv_row_n<const C: usize>(x: &[f32], k: &Conv, t: usize, f: usize, fo: usize, to: usize, row: &mut [f32]) {
    let cin = k.cin;
    let w = &k.w;
    let b: [f32; C] = k.b[..C].try_into().unwrap();
    for out in row.chunks_exact_mut(C) {
        out.copy_from_slice(&b);
    }
    for dt in 0..3 {
        let ti = to as isize + dt as isize - 1;
        if ti < 0 || ti >= t as isize {
            continue;
        }
        let x_row = &x[ti as usize * f * cin..];
        for df in 0..3 {
            let pad = k.pad_f as isize - df as isize;
            let j_lo = pad.max(0).min(fo as isize) as usize;
            let j_hi = (f as isize + pad).max(0).min(fo as isize) as usize;
            if j_lo >= j_hi {
                continue;
            }
            let fi_lo = (j_lo as isize - pad) as usize;
            let wt = &w[(dt * 3 + df) * cin * C..][..cin * C];
            for (out, xin) in row[j_lo * C..j_hi * C].chunks_exact_mut(C).zip(x_row[fi_lo * cin..].chunks_exact(cin)) {
                let mut acc: [f32; C] = out.try_into().unwrap();
                conv_tap(&mut acc, xin, wt);
                out.copy_from_slice(&acc);
            }
        }
    }
}

#[inline(always)]
fn conv_row_dyn(x: &[f32], k: &Conv, t: usize, f: usize, to: usize, row: &mut [f32]) {
    let (cin, cout) = (k.cin, k.cout);
    let fo = row.len() / cout;
    let w = &k.w;
    for out in row.chunks_mut(cout) {
        out.copy_from_slice(&k.b);
    }
    for dt in 0..3 {
        let ti = to as isize + dt as isize - 1;
        if ti < 0 || ti >= t as isize {
            continue;
        }
        let x_row = &x[ti as usize * f * cin..];
        for df in 0..3 {
            let pad = k.pad_f as isize - df as isize;
            let j_lo = pad.max(0).min(fo as isize) as usize;
            let j_hi = (f as isize + pad).max(0).min(fo as isize) as usize;
            if j_lo >= j_hi {
                continue;
            }
            let fi_lo = (j_lo as isize - pad) as usize;
            let wt = &w[(dt * 3 + df) * cin * cout..][..cin * cout];
            for (acc, xin) in row[j_lo * cout..j_hi * cout].chunks_exact_mut(cout).zip(x_row[fi_lo * cin..].chunks_exact(cin)) {
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
    if x.is_empty() || c == 0 {
        return;
    }
    match c {
        16 => instance_norm_selu_n::<16>(x),
        32 => instance_norm_selu_n::<32>(x),
        64 => instance_norm_selu_n::<64>(x),
        80 => instance_norm_selu_n::<80>(x),
        _ => instance_norm_selu_dyn(x, c),
    }
}

#[inline(always)]
fn sum_sq_block_any<const C: usize>(blk: &[f32]) -> ([f64; C], [f64; C]) {
    let (mut s, mut q) = ([0f64; C], [0f64; C]);
    for px in blk.chunks_exact(C) {
        for ci in 0..C {
            let vf = px[ci] as f64;
            s[ci] += vf;
            q[ci] += vf * vf;
        }
    }
    (s, q)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_block_avx2<const C: usize>(blk: &[f32]) -> ([f64; C], [f64; C]) {
    sum_sq_block_any::<C>(blk)
}

#[inline(always)]
fn norm_selu_block_any<const C: usize>(blk: &mut [f32], mean: &[f64; C], inv: &[f64; C]) {
    const SCALE: f32 = 1.050_700_987_355_480_5;
    const ALPHA: f32 = 1.673_263_242_354_377_3;
    for px in blk.chunks_exact_mut(C) {
        for ci in 0..C {
            let z = ((px[ci] as f64 - mean[ci]) * inv[ci]) as f32;
            px[ci] = SCALE * if z > 0.0 { z } else { ALPHA * (z.exp() - 1.0) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn norm_selu_block_avx2<const C: usize>(blk: &mut [f32], mean: &[f64; C], inv: &[f64; C]) {
    norm_selu_block_any::<C>(blk, mean, inv)
}

fn instance_norm_selu_n<const C: usize>(x: &mut [f32]) {
    let n = x.len() / C;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");
    let (sum, sq) = x
        .par_chunks(C * 256)
        .map(|blk| {
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                // SAFETY: AVX2 availability verified once outside the parallel loop.
                return unsafe { sum_sq_block_avx2::<C>(blk) };
            }
            sum_sq_block_any::<C>(blk)
        })
        .reduce(
            || ([0f64; C], [0f64; C]),
            |(mut s, mut q), (s2, q2)| {
                for ci in 0..C {
                    s[ci] += s2[ci];
                    q[ci] += q2[ci];
                }
                (s, q)
            },
        );
    let mut mean = [0f64; C];
    let mut inv = [0f64; C];
    let nf = n as f64;
    for ci in 0..C {
        let m = sum[ci] / nf;
        mean[ci] = m;
        inv[ci] = 1.0 / ((sq[ci] / nf - m * m).max(0.0) + 1e-5).sqrt();
    }
    x.par_chunks_mut(C * 256).with_min_len(2).for_each(|blk| {
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            // SAFETY: AVX2 availability verified once outside the parallel loop.
            unsafe { norm_selu_block_avx2::<C>(blk, &mean, &inv) };
            return;
        }
        norm_selu_block_any::<C>(blk, &mean, &inv);
    });
}

fn instance_norm_selu_dyn(x: &mut [f32], c: usize) {
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
    x.par_chunks_mut(c * 256).with_min_len(2).for_each(|blk| {
        for px in blk.chunks_exact_mut(c) {
            for (ci, v) in px.iter_mut().enumerate() {
                let z = ((*v as f64 - mean[ci]) * inv[ci]) as f32;
                *v = SCALE * if z > 0.0 { z } else { ALPHA * (z.exp() - 1.0) };
            }
        }
    });
}

/// One direction of the LSTM over the whole sequence (gates i, f, g, o).
fn lstm_dir(x: &[f32], n_in: usize, t: usize, (wi, wh, b): &(Vec<f32>, Vec<f32>, Vec<f32>), reverse: bool) -> Vec<f32> {
    let g4 = 4 * HIDDEN;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");
    // Input projections for every frame at once (8 frames per chunk so wi tile + x + output fit in L1D).
    let mut pre = vec![0f32; t * g4];
    pre.par_chunks_mut(8 * g4).enumerate().for_each(|(blk_i, chunk)| {
        let f_start = blk_i * 8;
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            // SAFETY: AVX2 availability checked once outside parallel loop; lengths verified below.
            unsafe { lstm_pre_chunk_avx2(chunk, &x[f_start * n_in..], n_in, wi, b) };
            return;
        }
        lstm_pre_chunk_any(chunk, &x[f_start * n_in..], n_in, wi, b);
    });
    let mut out = vec![0f32; t * HIDDEN];
    let (mut h, mut cs) = ([0f32; HIDDEN], [0f32; HIDDEN]);
    let mut gates = [0f32; 4 * HIDDEN];
    #[inline(always)]
    fn sig(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }
    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        let pre_t = &pre[ti * g4..(ti + 1) * g4];
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            // SAFETY: AVX2 verified once outside loop.
            unsafe { lstm_rec_step_avx2(&mut gates, pre_t, &h, wh) };
        } else {
            lstm_rec_step_any(&mut gates, pre_t, &h, wh);
        }
        #[cfg(not(target_arch = "x86_64"))]
        lstm_rec_step_any(&mut gates, pre_t, &h, wh);
        for j in 0..HIDDEN {
            let (i, f, gg, o) = (sig(gates[j]), sig(gates[HIDDEN + j]), gates[2 * HIDDEN + j].tanh(), sig(gates[3 * HIDDEN + j]));
            cs[j] = f * cs[j] + i * gg;
            h[j] = o * cs[j].tanh();
        }
        out[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&h);
    }
    out
}

fn lstm_pre_chunk_any(chunk: &mut [f32], x_sub: &[f32], n_in: usize, wi: &[f32], b: &[f32]) {
    let g4 = 4 * HIDDEN;
    let m = chunk.len() / g4;
    let tiles = g4 / PRE_TILE;
    for t_idx in 0..tiles {
        let g_off = t_idx * PRE_TILE;
        let b_tile = &b[g_off..g_off + PRE_TILE];
        let w_tile = &wi[t_idx * n_in * PRE_TILE..(t_idx + 1) * n_in * PRE_TILE];
        for f_rel in 0..m {
            let mut acc: [f32; PRE_TILE] = b_tile.try_into().unwrap();
            let xf = &x_sub[f_rel * n_in..(f_rel + 1) * n_in];
            for (i, row) in w_tile.chunks_exact(PRE_TILE).enumerate() {
                let xv = xf[i];
                for k in 0..PRE_TILE {
                    acc[k] += xv * row[k];
                }
            }
            chunk[f_rel * g4 + g_off..][..PRE_TILE].copy_from_slice(&acc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn lstm_pre_chunk_avx2(chunk: &mut [f32], x_sub: &[f32], n_in: usize, wi: &[f32], b: &[f32]) {
    use std::arch::x86_64::*;
    let g4 = 4 * HIDDEN;
    let m = chunk.len() / g4;
    let tiles = g4 / PRE_TILE;
    for t_idx in 0..tiles {
        let g_off = t_idx * PRE_TILE;
        let bp = b.as_ptr().add(g_off);
        let wp = wi.as_ptr().add(t_idx * n_in * PRE_TILE);
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let mut f_rel = 0;
        while f_rel + 4 <= m {
            let mut a00 = b0;
            let mut a01 = b1;
            let mut a10 = b0;
            let mut a11 = b1;
            let mut a20 = b0;
            let mut a21 = b1;
            let mut a30 = b0;
            let mut a31 = b1;
            let xp0 = x_sub.as_ptr().add(f_rel * n_in);
            let xp1 = xp0.add(n_in);
            let xp2 = xp0.add(2 * n_in);
            let xp3 = xp0.add(3 * n_in);
            for i in 0..n_in {
                let wr = wp.add(i * PRE_TILE);
                let w0 = _mm256_loadu_ps(wr);
                let w1 = _mm256_loadu_ps(wr.add(8));
                let x0 = _mm256_broadcast_ss(&*xp0.add(i));
                a00 = _mm256_add_ps(a00, _mm256_mul_ps(x0, w0));
                a01 = _mm256_add_ps(a01, _mm256_mul_ps(x0, w1));
                let x1 = _mm256_broadcast_ss(&*xp1.add(i));
                a10 = _mm256_add_ps(a10, _mm256_mul_ps(x1, w0));
                a11 = _mm256_add_ps(a11, _mm256_mul_ps(x1, w1));
                let x2 = _mm256_broadcast_ss(&*xp2.add(i));
                a20 = _mm256_add_ps(a20, _mm256_mul_ps(x2, w0));
                a21 = _mm256_add_ps(a21, _mm256_mul_ps(x2, w1));
                let x3 = _mm256_broadcast_ss(&*xp3.add(i));
                a30 = _mm256_add_ps(a30, _mm256_mul_ps(x3, w0));
                a31 = _mm256_add_ps(a31, _mm256_mul_ps(x3, w1));
            }
            let dst = chunk.as_mut_ptr().add(f_rel * g4 + g_off);
            _mm256_storeu_ps(dst, a00);
            _mm256_storeu_ps(dst.add(8), a01);
            _mm256_storeu_ps(dst.add(g4), a10);
            _mm256_storeu_ps(dst.add(g4 + 8), a11);
            _mm256_storeu_ps(dst.add(2 * g4), a20);
            _mm256_storeu_ps(dst.add(2 * g4 + 8), a21);
            _mm256_storeu_ps(dst.add(3 * g4), a30);
            _mm256_storeu_ps(dst.add(3 * g4 + 8), a31);
            f_rel += 4;
        }
        while f_rel < m {
            let mut a0 = b0;
            let mut a1 = b1;
            let xp = x_sub.as_ptr().add(f_rel * n_in);
            for i in 0..n_in {
                let wr = wp.add(i * PRE_TILE);
                let w0 = _mm256_loadu_ps(wr);
                let w1 = _mm256_loadu_ps(wr.add(8));
                let xi = _mm256_broadcast_ss(&*xp.add(i));
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(xi, w0));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(xi, w1));
            }
            let dst = chunk.as_mut_ptr().add(f_rel * g4 + g_off);
            _mm256_storeu_ps(dst, a0);
            _mm256_storeu_ps(dst.add(8), a1);
            f_rel += 1;
        }
    }
}

fn lstm_rec_step_any(gates: &mut [f32; 4 * HIDDEN], pre_t: &[f32], h: &[f32; HIDDEN], wh: &[f32]) {
    let tiles = (4 * HIDDEN) / REC_TILE;
    for t_idx in 0..tiles {
        let g_off = t_idx * REC_TILE;
        let mut acc: [f32; REC_TILE] = pre_t[g_off..g_off + REC_TILE].try_into().unwrap();
        let w_tile = &wh[t_idx * HIDDEN * REC_TILE..(t_idx + 1) * HIDDEN * REC_TILE];
        for (i, row) in w_tile.chunks_exact(REC_TILE).enumerate() {
            let hi = h[i];
            for k in 0..REC_TILE {
                acc[k] += hi * row[k];
            }
        }
        gates[g_off..g_off + REC_TILE].copy_from_slice(&acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn lstm_rec_step_avx2(gates: &mut [f32; 4 * HIDDEN], pre_t: &[f32], h: &[f32; HIDDEN], wh: &[f32]) {
    use std::arch::x86_64::*;
    let tiles = (4 * HIDDEN) / REC_TILE;
    for t_idx in 0..tiles {
        let g_off = t_idx * REC_TILE;
        let pp = pre_t.as_ptr().add(g_off);
        let mut a0 = _mm256_loadu_ps(pp);
        let mut a1 = _mm256_loadu_ps(pp.add(8));
        let mut a2 = _mm256_loadu_ps(pp.add(16));
        let mut a3 = _mm256_loadu_ps(pp.add(24));
        let mut a4 = _mm256_loadu_ps(pp.add(32));
        let mut a5 = _mm256_loadu_ps(pp.add(40));
        let mut a6 = _mm256_loadu_ps(pp.add(48));
        let mut a7 = _mm256_loadu_ps(pp.add(56));
        let wp = wh.as_ptr().add(t_idx * HIDDEN * REC_TILE);
        let hp = h.as_ptr();
        for i in 0..HIDDEN {
            let hi = _mm256_broadcast_ss(&*hp.add(i));
            let wr = wp.add(i * REC_TILE);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(hi, _mm256_loadu_ps(wr)));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(8))));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(16))));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(24))));
            a4 = _mm256_add_ps(a4, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(32))));
            a5 = _mm256_add_ps(a5, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(40))));
            a6 = _mm256_add_ps(a6, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(48))));
            a7 = _mm256_add_ps(a7, _mm256_mul_ps(hi, _mm256_loadu_ps(wr.add(56))));
        }
        let gp = gates.as_mut_ptr().add(g_off);
        _mm256_storeu_ps(gp, a0);
        _mm256_storeu_ps(gp.add(8), a1);
        _mm256_storeu_ps(gp.add(16), a2);
        _mm256_storeu_ps(gp.add(24), a3);
        _mm256_storeu_ps(gp.add(32), a4);
        _mm256_storeu_ps(gp.add(40), a5);
        _mm256_storeu_ps(gp.add(48), a6);
        _mm256_storeu_ps(gp.add(56), a7);
    }
}

#[cfg(test)]
pub mod old_de4fcb3 {
    use super::*;

    struct OldConv {
        w: Vec<f32>,
        b: Vec<f32>,
        cin: usize,
        cout: usize,
        pad_f: usize,
    }

    pub struct OldNet {
        convs: Vec<OldConv>,
        lstm: [(Vec<f32>, Vec<f32>, Vec<f32>); 2],
        fc_w: Vec<f32>,
        fc_b: Vec<f32>,
    }

    impl OldNet {
        pub fn from_state(mut t: HashMap<String, Tensor>) -> Result<OldNet, String> {
            let spec = [
                ("1a", 1, 16, 1),
                ("1b", 16, 16, 1),
                ("1c", 16, 16, 1),
                ("2a", 16, 32, 1),
                ("2b", 32, 32, 1),
                ("2c", 32, 32, 1),
                ("3a", 32, 64, 1),
                ("3b", 64, 64, 1),
                ("4a", 64, 80, 0),
                ("4b", 80, 80, 0),
            ];
            let mut convs = Vec::new();
            for (n, cin, cout, pad_f) in spec {
                let w = take(&mut t, &format!("audio_feature_block.conv{n}.weight"), &[cout, cin, 3, 3])?;
                let b = take(&mut t, &format!("audio_feature_block.conv{n}.bias"), &[cout])?;
                convs.push(OldConv { w, b, cin, cout, pad_f });
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
            Ok(OldNet { convs, lstm, fc_w, fc_b })
        }

        pub fn forward(&self, cqt: &[[f32; super::super::cqt::BINS]]) -> Vec<[f32; 100]> {
            let t = cqt.len();
            let mut f = SPEC;
            let mut x: Vec<f32> = cqt.iter().flat_map(|r| r[FIRST_BIN..FIRST_BIN + SPEC].iter().copied()).collect();
            let mut c = 1;
            for (i, conv) in self.convs.iter().enumerate() {
                let fo = f + 2 * conv.pad_f - 2;
                x = old_conv2d(&x, conv, t, f, fo);
                f = fo;
                c = conv.cout;
                old_instance_norm_selu(&mut x, c);
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
            let feat: Vec<f32> = (0..t)
                .flat_map(|ti| (0..c).flat_map(move |ci| (0..f).map(move |fi| (ci, ti, fi))))
                .map(|(ci, ti, fi)| x[(ti * f + fi) * c + ci])
                .collect();
            let n_in = c * f;
            let outs: Vec<Vec<f32>> = self.lstm.par_iter().enumerate().map(|(d, w)| old_lstm_dir(&feat, n_in, t, w, d == 1)).collect();
            (0..t)
                .into_par_iter()
                .map(|ti| {
                    let h: Vec<f32> = outs[0][ti * HIDDEN..(ti + 1) * HIDDEN]
                        .iter()
                        .chain(&outs[1][ti * HIDDEN..(ti + 1) * HIDDEN])
                        .copied()
                        .collect();
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

    fn old_conv2d(x: &[f32], k: &OldConv, t: usize, f: usize, fo: usize) -> Vec<f32> {
        let (cin, cout) = (k.cin, k.cout);
        let mut w = vec![0f32; 9 * cin * cout];
        for o in 0..cout {
            for i in 0..cin {
                for tap in 0..9 {
                    w[(tap * cin + i) * cout + o] = k.w[(o * cin + i) * 9 + tap];
                }
            }
        }
        let mut y = vec![0f32; t * fo * cout];
        y.par_chunks_mut(fo * cout).enumerate().for_each(|(to, row)| old_conv_row(x, &w, k, t, f, to, row));
        y
    }

    fn old_conv_row(x: &[f32], w: &[f32], k: &OldConv, t: usize, f: usize, to: usize, row: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return unsafe { old_conv_row_avx2(x, w, k, t, f, to, row) };
        }
        old_conv_row_any(x, w, k, t, f, to, row)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn old_conv_row_avx2(x: &[f32], w: &[f32], k: &OldConv, t: usize, f: usize, to: usize, row: &mut [f32]) {
        old_conv_row_any(x, w, k, t, f, to, row)
    }

    #[inline(always)]
    fn old_conv_row_any(x: &[f32], w: &[f32], k: &OldConv, t: usize, f: usize, to: usize, row: &mut [f32]) {
        match k.cout {
            16 => old_conv_row_n::<16>(x, w, k, t, f, to, row),
            32 => old_conv_row_n::<32>(x, w, k, t, f, to, row),
            64 => old_conv_row_n::<64>(x, w, k, t, f, to, row),
            80 => old_conv_row_n::<80>(x, w, k, t, f, to, row),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn old_conv_row_n<const C: usize>(x: &[f32], w: &[f32], k: &OldConv, t: usize, f: usize, to: usize, row: &mut [f32]) {
        let cin = k.cin;
        for (j, out) in row.chunks_exact_mut(C).enumerate() {
            let mut acc: [f32; C] = k.b[..C].try_into().unwrap();
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

    fn old_instance_norm_selu(x: &mut [f32], c: usize) {
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

    fn old_lstm_dir(x: &[f32], n_in: usize, t: usize, (wi, wh, b): &(Vec<f32>, Vec<f32>, Vec<f32>), reverse: bool) -> Vec<f32> {
        let g4 = 4 * HIDDEN;
        let tr = |w: &[f32], n: usize| -> Vec<f32> { (0..n * g4).map(|i| w[(i % g4) * n + i / g4]).collect() };
        let (wi, wh) = (tr(wi, n_in), tr(wh, HIDDEN));
        let mut pre = vec![0f32; t * g4];
        pre.par_chunks_mut(g4).enumerate().for_each(|(ti, acc)| {
            acc.copy_from_slice(b);
            old_axpy_rows(acc, &x[ti * n_in..(ti + 1) * n_in], &wi);
        });
        let mut out = vec![0f32; t * HIDDEN];
        let (mut h, mut cs) = (vec![0f32; HIDDEN], vec![0f32; HIDDEN]);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut gates = vec![0f32; g4];
        for step in 0..t {
            let ti = if reverse { t - 1 - step } else { step };
            gates.copy_from_slice(&pre[ti * g4..(ti + 1) * g4]);
            old_axpy_rows(&mut gates, &h, &wh);
            for j in 0..HIDDEN {
                let (i, f, gg, o) = (sig(gates[j]), sig(gates[HIDDEN + j]), gates[2 * HIDDEN + j].tanh(), sig(gates[3 * HIDDEN + j]));
                cs[j] = f * cs[j] + i * gg;
                h[j] = o * cs[j].tanh();
            }
            out[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&h);
        }
        out
    }

    fn old_axpy_rows(acc: &mut [f32], xs: &[f32], w: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return unsafe { old_axpy_rows_avx2(acc, xs, w) };
        }
        old_axpy_rows_any(acc, xs, w)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn old_axpy_rows_avx2(acc: &mut [f32], xs: &[f32], w: &[f32]) {
        old_axpy_rows_any(acc, xs, w)
    }

    #[inline(always)]
    fn old_axpy_rows_any(acc: &mut [f32], xs: &[f32], w: &[f32]) {
        for (xv, row) in xs.iter().zip(w.chunks_exact(acc.len())) {
            acc.iter_mut().zip(row).for_each(|(a, b)| *a += xv * b);
        }
    }
}
