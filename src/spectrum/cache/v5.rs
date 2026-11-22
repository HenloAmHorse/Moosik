//! The v5 plane codec: lossless, about a third smaller than v4.
//!
//! # Model
//!
//! Codes are predicted and only the differences are stored, entropy coded.
//! A row is coded in two halves. The even bars first, each predicted from the
//! four rows before it. Then the odd bars, each predicted from those rows and
//! from the even bars of its own row, which sit on both sides of it. No cell
//! depends on another cell of its own half in the row being coded, so each
//! half is predicted in plain loops over its bars before any of it is read.
//!
//! Per cell, six predictors are blended, each weighted by the inverse square
//! of its recent error around that bar: a per-bar linear predictor fitted to
//! the plane (15 taps and a bias), three extrapolations in time, the mean of
//! the two neighbours, and, with a reference plane, the reference's value
//! moved by this plane's last offset from it. The residual's context is the
//! bar's band and the size of the residuals around it.
//!
//! A reference plane is another plane of the same analysis that the reader
//! already holds (Left and Right are coded against Mix). A blob coded against
//! a reference cannot be read without the same one.
//!
//! # Determinism
//!
//! The linear predictor is wrapping integer arithmetic. The blend is f32,
//! evaluated in a fixed order with IEEE operations only (no fused or
//! reassociated arithmetic), so it gives the same bits on every CPU. Each
//! chunk also carries a checksum of its decoded codes: whatever the cause, a
//! decode that is not the plane that was encoded is refused, never shown.
//!
//! # Layout (the body; the header is `cache.rs`'s)
//!
//! ```text
//!  0  u32  chunk_rows      rows per chunk, 1..=65536
//!  4  u8   pb              rANS precision in bits, 8..=12
//!  5  u8   0
//!  6  u16  0
//!  8  ..   coefficients    per bar, even bars then odd: 15 × i16, then i32
//! ..  ..   frequencies     400 contexts × 52 symbols × u16; each context
//!                          sums to 1 << pb
//! ..  ..   directory       per chunk: u32 rANS words, u32 raw words, u64
//!                          checksum of the chunk's codes
//! ..  ..   chunks          per chunk: rANS words (u16), raw words (u32)
//! ```
//!
//! All integers little-endian. The body must be exactly that long. Chunks
//! start from an empty history, so they decode independently and in parallel.
//!
//! Nothing here trusts the bytes: every length is checked before use, every
//! table index is bounded by construction, a stream that ends early reads
//! zeros, and the checksum has the last word.

/// Taps of each per-bar linear predictor, and its bias.
const TAPS: usize = 15;
const K: usize = TAPS + 1;
/// Predictors blended per cell; the last needs a reference.
const NP: usize = 6;
/// Padding cells on each side of a half row.
const PH: usize = 2;
/// Fixed point of the linear predictor's coefficients.
const QB: u32 = 10;
const COEF_MAX: i64 = i16::MAX as i64;
const BIAS_MAX: i64 = 1 << 22;
const GROUPS: usize = 8;
const QCTX: usize = 25;
/// Contexts: band × activity, for each half.
const NCTX: usize = 2 * GROUPS * QCTX;
const NSYM: usize = 52;
const WCAP: usize = 1024;
/// rANS: 32-bit states, 16-bit words.
const RL: u32 = 1 << 16;
const MAX: i32 = 4095;
/// Rows per chunk, and the rANS precision, as written.
const CHUNK_ROWS: usize = 512;
const PB: u32 = 10;
/// Every FIT_STEP-th row is enough to fit sixteen numbers per bar.
const FIT_STEP: usize = 8;
const HEAD: usize = 8;

/// Where each tap of the even half's linear predictor reads: (plane half,
/// rows back, offset). Half 0 is even, 1 is odd; 2 and 3 are the reference's
/// even and odd cells in the row being coded. Bar 2k's neighbours one bar
/// away are odd cells k-1 and k.
const TAPS_E: [(usize, usize, isize); TAPS] = [
    (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0), (1, 1, -1), (1, 1, 0), (1, 2, -1), (1, 2, 0), (1, 3, -1), (1, 3, 0), (0, 1, -1), (0, 1, 1),
    (2, 0, 0), (3, 0, -1), (3, 0, 0),
];
/// The odd half's: rows back 0 is the row being coded, whose even cells k and
/// k+1 flank odd bar 2k+1.
const TAPS_O: [(usize, usize, isize); TAPS] = [
    (1, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0), (0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
    (3, 0, 0), (2, 0, 0), (2, 0, 1),
];

/// Why a body could not be decoded. Always a reason to recompute.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    /// A length, a table or a parameter is not what this format allows.
    Malformed(&'static str),
    /// The decoded codes are not the ones that were encoded.
    Checksum,
}

// ---------------------------------------------------------------------------
// Small pieces: threads, checksum, tokens, bit and word streams
// ---------------------------------------------------------------------------

/// `f(i)` for every `i < n`, results in order, on the analysis pool: the
/// user's thread budget holds for coding as for analysing, and no thread is
/// started here. A pool that would not build leaves the work to the global
/// pool, which is capped too.
fn par<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    use rayon::prelude::*;
    #[cfg(test)]
    let call = probe::Call::new();
    let job = || {
        (0..n)
            .into_par_iter()
            .map(|i| {
                #[cfg(test)]
                let _at_work = call.enter();
                f(i)
            })
            .collect()
    };
    match crate::spectrum::gpu_calib::pool() {
        Some(pool) => pool.install(job),
        None => job(),
    }
}

/// Checks only: for each `par` call, who asked, how many pieces of its work
/// ran at once, and the size of the pool they ran on.
#[cfg(test)]
pub(crate) mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use std::sync::Mutex;

    /// (calling thread, most pieces at once, the pool's size, pieces off the analysis pool).
    pub static CALLS: Mutex<Vec<(std::thread::ThreadId, usize, usize, usize)>> = Mutex::new(Vec::new());

    #[derive(Default)]
    pub struct Call {
        active: AtomicUsize,
        peak: AtomicUsize,
        pool: AtomicUsize,
        stray: AtomicUsize,
    }

    pub struct AtWork<'a>(&'a Call);

    impl Call {
        pub fn new() -> Call {
            Call::default()
        }

        pub fn enter(&self) -> AtWork<'_> {
            self.peak.fetch_max(self.active.fetch_add(1, Relaxed) + 1, Relaxed);
            self.pool.store(rayon::current_num_threads(), Relaxed);
            if !std::thread::current().name().is_some_and(|n| n.starts_with("moosik-analyse-")) {
                self.stray.fetch_add(1, Relaxed);
            }
            AtWork(self)
        }
    }

    impl Drop for AtWork<'_> {
        fn drop(&mut self) {
            self.0.active.fetch_sub(1, Relaxed);
        }
    }

    impl Drop for Call {
        fn drop(&mut self) {
            let row = (std::thread::current().id(), self.peak.load(Relaxed), self.pool.load(Relaxed), self.stray.load(Relaxed));
            CALLS.lock().unwrap_or_else(|e| e.into_inner()).push(row);
        }
    }
}

/// A checksum of codes: order-sensitive, four codes a step.
fn checksum(codes: &[u16]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ codes.len() as u64;
    let mut it = codes.chunks_exact(4);
    for c in &mut it {
        let w = c[0] as u64 | (c[1] as u64) << 16 | (c[2] as u64) << 32 | (c[3] as u64) << 48;
        h = (h ^ w).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 32;
    }
    for &c in it.remainder() {
        h = (h ^ c as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 32;
    }
    h
}

/// A residual as (token, raw bit count, raw bits): small ones whole, large
/// ones as an exponent and two top bits with the rest raw.
fn tokenize(q: i32) -> (u32, u32, u32) {
    let z = ((q << 1) ^ (q >> 31)) as u32;
    if z < 16 {
        (z, 0, 0)
    } else {
        let n = 31 - z.leading_zeros();
        ((16 + (n - 4) * 4) + ((z >> (n - 2)) & 3), n - 2, z & ((1 << (n - 2)) - 1))
    }
}

/// Per token: the value without its raw bits, and how many raw bits follow.
fn token_table() -> [(u32, u32); NSYM] {
    std::array::from_fn(|tok| {
        let tok = tok as u32;
        if tok < 16 {
            (tok, 0)
        } else {
            let n = (tok - 16) / 4 + 4;
            ((4 | ((tok - 16) & 3)) << (n - 2), n - 2)
        }
    })
}

struct BitW {
    acc: u64,
    n: u32,
    out: Vec<u32>,
}

impl BitW {
    fn put(&mut self, v: u32, k: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += k;
        if self.n >= 32 {
            self.out.push(self.acc as u32);
            self.acc >>= 32;
            self.n -= 32;
        }
    }
    fn finish(mut self) -> Vec<u32> {
        if self.n > 0 {
            self.out.push(self.acc as u32);
        }
        self.out
    }
}

/// Raw bits, read from little-endian u32 words. Past the end it reads zeros.
struct BitR<'a> {
    acc: u64,
    n: u32,
    src: &'a [u8],
    pos: usize,
}

impl BitR<'_> {
    #[inline]
    fn read(&mut self, k: u32) -> u32 {
        let need = (self.n < 32) as u64;
        let word = self.src.get(self.pos..self.pos + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])) as u64;
        self.acc |= (word & need.wrapping_neg()) << (self.n & 31);
        self.pos += 4 * need as usize;
        self.n += 32 * need as u32;
        let v = (self.acc & ((1 << k) - 1)) as u32;
        self.acc >>= k;
        self.n -= k;
        v
    }
}

// ---------------------------------------------------------------------------
// Static per-context frequency tables
// ---------------------------------------------------------------------------

struct Tables {
    pb: u32,
    /// (freq, start) per context and token.
    fs: Vec<(u16, u16)>,
    /// Slot -> token, per context.
    s2s: Vec<u8>,
}

impl Tables {
    /// From each context's frequencies, which must sum to `1 << pb`.
    fn from_freqs(freqs: &[u16], pb: u32) -> Result<Tables, Refused> {
        let m = 1usize << pb;
        let mut fs = vec![(0u16, 0u16); NCTX * NSYM];
        let mut s2s = vec![0u8; NCTX * m];
        for c in 0..NCTX {
            let f = &freqs[c * NSYM..(c + 1) * NSYM];
            if f.iter().map(|&v| v as usize).sum::<usize>() != m {
                return Err(Refused::Malformed("a context's frequencies do not sum to the table size"));
            }
            let mut start = 0usize;
            for (s, &fv) in f.iter().enumerate() {
                fs[c * NSYM + s] = (fv, start as u16);
                s2s[c * m + start..c * m + start + fv as usize].fill(s as u8);
                start += fv as usize;
            }
        }
        Ok(Tables { pb, fs, s2s })
    }

    /// Frequencies for counted symbols: every seen symbol at least 1, the
    /// rounding taken up by the most frequent.
    fn freqs_for(hist: &[u64], pb: u32) -> Vec<u16> {
        let m = 1u64 << pb;
        let mut out = vec![0u16; NCTX * NSYM];
        for c in 0..NCTX {
            let h = &hist[c * NSYM..(c + 1) * NSYM];
            let total: u64 = h.iter().sum();
            let f = &mut out[c * NSYM..(c + 1) * NSYM];
            if total == 0 {
                f[0] = m as u16;
                continue;
            }
            let mut v = [0u64; NSYM];
            for s in 0..NSYM {
                if h[s] > 0 {
                    v[s] = ((h[s] * m + total / 2) / total).max(1);
                }
            }
            // Bring the sum to exactly m, taking from (or giving to) the
            // largest entries, never below 1.
            let mut sum: u64 = v.iter().sum();
            while sum != m {
                let big = (0..NSYM).max_by_key(|&s| v[s]).expect("symbols");
                if sum > m {
                    let take = (sum - m).min(v[big] - 1).max(1);
                    v[big] -= take;
                    sum -= take;
                } else {
                    v[big] += m - sum;
                    sum = m;
                }
            }
            for s in 0..NSYM {
                f[s] = v[s] as u16;
            }
        }
        out
    }
}

/// Two rANS states take the symbols alternately and share one word stream.
fn rans_encode(syms: &[u16], t: &Tables) -> Vec<u16> {
    let pb = t.pb;
    let mut xs: [u32; 2] = [RL, RL];
    let mut out = Vec::with_capacity(syms.len() / 2 + 8);
    for (i, &s) in syms.iter().enumerate().rev() {
        let x = &mut xs[i & 1];
        let (freq, start) = t.fs[(s >> 6) as usize * NSYM + (s & 63) as usize];
        let (freq, start) = (freq as u32, start as u32);
        if *x as u64 >= (freq as u64) << (32 - pb) {
            out.push(*x as u16);
            *x >>= 16;
        }
        *x = ((*x / freq) << pb) + (*x % freq) + start;
    }
    for x in [xs[1], xs[0]] {
        out.push(x as u16);
        out.push((x >> 16) as u16);
    }
    out.reverse();
    out
}

// ---------------------------------------------------------------------------
// The row model, shared by encoder and decoder
// ---------------------------------------------------------------------------

/// 1/s² for s below WCAP, in f32: the blend's weights.
fn inv_squares() -> &'static [f32; WCAP] {
    static LUT: std::sync::OnceLock<[f32; WCAP]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|s| {
            let s = s.max(1) as f32;
            1.0 / (s * s)
        })
    })
}

#[inline]
fn qact(act: i32) -> usize {
    let a = (act + 2) as u32;
    let lz = 31 - a.leading_zeros();
    let frac = (a >> (lz - 1)) & 1;
    ((2 * lz + frac) as usize - 2).min(QCTX - 2)
}

/// One half's state: four rows of history, the previous row's per-predictor
/// errors and residual sizes, and the row being coded. Rows are padded by
/// `PH` cells of zero on each side.
struct Half {
    n: usize,
    h: [Vec<i32>; 4],
    err: [Vec<i32>; NP],
    res: Vec<i32>,
    ps: [Vec<i32>; NP],
    pred: Vec<i32>,
    ctx: Vec<u16>,
    cur: Vec<i32>,
    q: Vec<i32>,
    /// The reference's row at the row being coded, and the one before (zero
    /// without a reference).
    rcur: Vec<i32>,
    rprev: Vec<i32>,
    num: Vec<f32>,
    den: Vec<f32>,
    /// Predictors in the blend.
    np: usize,
}

impl Half {
    fn new(n: usize, len: usize, np: usize) -> Half {
        let row = || vec![0i32; len + 2 * PH];
        Half {
            n,
            h: std::array::from_fn(|_| row()),
            err: std::array::from_fn(|_| row()),
            res: row(),
            ps: std::array::from_fn(|_| vec![0; n]),
            pred: vec![0; n],
            ctx: vec![0; n],
            cur: row(),
            q: row(),
            rcur: row(),
            rprev: row(),
            num: vec![0.0; n],
            den: vec![0.0; n],
            np,
        }
    }

    /// The blend: each predictor weighted by the inverse square of its recent
    /// error at this bar and its two neighbours in the half.
    fn blend(&mut self) {
        let n = self.n;
        let lut = inv_squares();
        self.num.fill(0.0);
        self.den.fill(0.0);
        for (e, p) in self.err.iter().zip(self.ps.iter()).take(self.np) {
            let (e0, e1, e2) = (&e[PH - 1..PH - 1 + n], &e[PH..PH + n], &e[PH + 1..PH + 1 + n]);
            for i in 0..n {
                let wt = lut[((2 + e0[i] + 2 * e1[i] + e2[i]) as usize).min(WCAP - 1)];
                self.num[i] += wt * p[i] as f32;
                self.den[i] += wt;
            }
        }
        for i in 0..n {
            self.pred[i] = ((self.num[i] / self.den[i] + 0.5) as i32).clamp(0, MAX);
        }
    }

    /// Move the coded row into history and refresh the error state.
    fn commit(&mut self) {
        let n = self.n;
        self.h.rotate_right(1);
        self.h[0][PH..PH + n].copy_from_slice(&self.cur[PH..PH + n]);
        let x = &self.cur[PH..PH + n];
        for (e, p) in self.err.iter_mut().zip(self.ps.iter()) {
            for ((o, &xv), &pv) in e[PH..PH + n].iter_mut().zip(x).zip(p) {
                *o = (xv - pv).abs();
            }
        }
        for (o, &qv) in self.res[PH..PH + n].iter_mut().zip(&self.q[PH..PH + n]) {
            *o = qv.abs();
        }
    }
}

struct Model<'a> {
    even: Half,
    odd: Half,
    coef_e: &'a [i32],
    coef_o: &'a [i32],
    group_e: Vec<u16>,
    group_o: Vec<u16>,
}

/// One half's linear predictor: the bias, then one pass per tap. Coefficients
/// are tap-major, `coefs[j * n + k]`.
fn linear(out: &mut [i32], coefs: &[i32], taps: &[(usize, usize, isize); TAPS], even: &Half, odd: &Half) {
    let n = out.len();
    out.copy_from_slice(&coefs[TAPS * n..K * n]);
    for (j, &(half, back, off)) in taps.iter().enumerate() {
        let h = if half % 2 == 0 { even } else { odd };
        let row = if half >= 2 { &h.rcur } else if back == 0 { &h.cur } else { &h.h[back - 1] };
        let src = &row[(PH as isize + off) as usize..][..n];
        let cf = &coefs[j * n..(j + 1) * n];
        for ((o, &c), &x) in out.iter_mut().zip(cf).zip(src) {
            *o = o.wrapping_add(c.wrapping_mul(x));
        }
    }
    for o in out.iter_mut() {
        *o = (o.wrapping_add(1 << (QB - 1)) >> QB).clamp(0, MAX);
    }
}

impl<'a> Model<'a> {
    fn new(bars: usize, coef_e: &'a [i32], coef_o: &'a [i32], has_ref: bool) -> Self {
        let (ne, no) = (bars.div_ceil(2), bars / 2);
        let np = if has_ref { NP } else { NP - 1 };
        Model {
            even: Half::new(ne, ne, np),
            odd: Half::new(no, ne, np),
            coef_e,
            coef_o,
            group_e: (0..ne).map(|k| ((2 * k * GROUPS / bars) * QCTX) as u16).collect(),
            group_o: (0..no).map(|k| (((2 * k + 1) * GROUPS / bars + GROUPS) * QCTX) as u16).collect(),
        }
    }

    /// The reference's rows for the row about to be coded.
    fn set_ref(&mut self, cur: &[u16], prev: Option<&[u16]>) {
        for (half, h) in [&mut self.even, &mut self.odd].into_iter().enumerate() {
            for k in 0..h.n {
                h.rcur[PH + k] = cur[2 * k + half] as i32;
                h.rprev[PH + k] = prev.map_or(0, |p| p[2 * k + half] as i32);
            }
        }
    }

    /// The even half of the next row, from history alone.
    fn predict_even(&mut self) {
        let n = self.even.n;
        let mut p0 = std::mem::take(&mut self.even.ps[0]);
        linear(&mut p0, self.coef_e, &TAPS_E, &self.even, &self.odd);
        self.even.ps[0] = p0;
        let (e, o) = (&mut self.even, &self.odd);
        let [_, p1, p2, p3, p4, p5] = &mut e.ps;
        let (a, b, c) = (&e.h[0][PH..PH + n], &e.h[1][PH..PH + n], &e.h[2][PH..PH + n]);
        let (rc, rp) = (&e.rcur[PH..PH + n], &e.rprev[PH..PH + n]);
        let (left, right) = (&o.h[0][PH - 1..PH - 1 + n], &o.h[0][PH..PH + n]);
        for i in 0..n {
            p1[i] = a[i];
            p2[i] = (2 * a[i] - b[i]).clamp(0, MAX);
            p3[i] = (3 * a[i] - 3 * b[i] + c[i]).clamp(0, MAX);
            p4[i] = (left[i] + right[i] + 1) >> 1;
            p5[i] = (rc[i] + a[i] - rp[i]).clamp(0, MAX);
        }
        e.blend();
        let (r0, r1, r2) = (&e.res[PH - 1..PH - 1 + n], &e.res[PH..PH + n], &e.res[PH + 1..PH + 1 + n]);
        let (s0, s1) = (&o.res[PH - 1..PH - 1 + n], &o.res[PH..PH + n]);
        let (a, left) = (&e.h[0][PH..PH + n], &o.h[0][PH - 1..PH - 1 + n]);
        for i in 0..n {
            let act = r0[i] + 2 * r1[i] + r2[i] + s0[i] + s1[i];
            let zero = a[i] == 0 && left[i] == 0;
            e.ctx[i] = self.group_e[i] + if zero { QCTX - 1 } else { qact(act) } as u16;
        }
    }

    /// The odd half, from history and the row's even half (`even.cur` and
    /// `even.q`), which must be in place.
    fn predict_odd(&mut self) {
        let n = self.odd.n;
        let mut p0 = std::mem::take(&mut self.odd.ps[0]);
        linear(&mut p0, self.coef_o, &TAPS_O, &self.even, &self.odd);
        self.odd.ps[0] = p0;
        let (e, o) = (&self.even, &mut self.odd);
        let [_, p1, p2, p3, p4, p5] = &mut o.ps;
        let (a, b) = (&o.h[0][PH..PH + n], &o.h[1][PH..PH + n]);
        let (rc, rp) = (&o.rcur[PH..PH + n], &o.rprev[PH..PH + n]);
        let (cl, cr) = (&e.cur[PH..PH + n], &e.cur[PH + 1..PH + 1 + n]);
        let (hl, hr) = (&e.h[0][PH..PH + n], &e.h[0][PH + 1..PH + 1 + n]);
        for i in 0..n {
            let mid = (cl[i] + cr[i] + 1) >> 1;
            p1[i] = mid;
            p2[i] = (mid + a[i] - ((hl[i] + hr[i] + 1) >> 1)).clamp(0, MAX);
            p3[i] = a[i];
            p4[i] = (2 * a[i] - b[i]).clamp(0, MAX);
            p5[i] = (rc[i] + a[i] - rp[i]).clamp(0, MAX);
        }
        o.blend();
        let (r0, r1, r2) = (&o.res[PH - 1..PH - 1 + n], &o.res[PH..PH + n], &o.res[PH + 1..PH + 1 + n]);
        let (ql, qr) = (&e.q[PH..PH + n], &e.q[PH + 1..PH + 1 + n]);
        for i in 0..n {
            let act = r0[i] + 2 * r1[i] + r2[i] + ql[i].abs() + qr[i].abs();
            let zero = cl[i] == 0 && cr[i] == 0;
            o.ctx[i] = self.group_o[i] + if zero { QCTX - 1 } else { qact(act) } as u16;
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder (with AVX2 fitter and pass1; decoder below stays 100% safe)
// ---------------------------------------------------------------------------

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn commit_avx2(h: &mut Half) {
    use std::arch::x86_64::*;
    let n = h.n;
    let np = h.np;
    h.h.rotate_right(1);
    let x_ptr = h.cur.as_ptr().add(PH);
    let h0_ptr = h.h[0].as_mut_ptr().add(PH);
    let q_ptr = h.q.as_ptr().add(PH);
    let r_ptr = h.res.as_mut_ptr().add(PH);
    let mut i = 0usize;
    while i + 8 <= n {
        let xv = _mm256_loadu_si256(x_ptr.add(i) as *const __m256i);
        _mm256_storeu_si256(h0_ptr.add(i) as *mut __m256i, xv);
        let qv = _mm256_abs_epi32(_mm256_loadu_si256(q_ptr.add(i) as *const __m256i));
        _mm256_storeu_si256(r_ptr.add(i) as *mut __m256i, qv);
        for m in 0..np {
            let pv = _mm256_loadu_si256(h.ps.get_unchecked(m).as_ptr().add(i) as *const __m256i);
            let d = _mm256_abs_epi32(_mm256_sub_epi32(xv, pv));
            _mm256_storeu_si256(h.err.get_unchecked_mut(m).as_mut_ptr().add(PH + i) as *mut __m256i, d);
        }
        i += 8;
    }
    while i < n {
        let xv = *x_ptr.add(i);
        *h0_ptr.add(i) = xv;
        *r_ptr.add(i) = (*q_ptr.add(i)).abs();
        for m in 0..np {
            *h.err.get_unchecked_mut(m).as_mut_ptr().add(PH + i) = (xv - *h.ps.get_unchecked(m).get_unchecked(i)).abs();
        }
        i += 1;
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_row_avx2(src: &[u16], e_out: &mut [i32], o_out: &mut [i32], ne: usize, no: usize) {
    use std::arch::x86_64::*;
    let v_mask = _mm256_set1_epi32(0xFFFF);
    let sp = src.as_ptr();
    let ep = e_out.as_mut_ptr();
    let op = o_out.as_mut_ptr();
    let mut k = 0usize;
    while k + 8 <= no {
        let v = _mm256_loadu_si256(sp.add(2 * k) as *const __m256i);
        let ev = _mm256_and_si256(v, v_mask);
        let od = _mm256_srli_epi32(v, 16);
        _mm256_storeu_si256(ep.add(k) as *mut __m256i, ev);
        _mm256_storeu_si256(op.add(k) as *mut __m256i, od);
        k += 8;
    }
    while k < no {
        *ep.add(k) = *sp.add(2 * k) as i32;
        *op.add(k) = *sp.add(2 * k + 1) as i32;
        k += 1;
    }
    if ne > no {
        *ep.add(no) = *sp.add(2 * no) as i32;
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn predict_even_avx2(even: &mut Half, odd: &Half, coef_e: &[i32], group_e: &[u16]) {
    use std::arch::x86_64::*;
    let n = even.n;
    let np = even.np;
    let n_taps = if np == NP { TAPS } else { TAPS - 3 };
    let mut ptrs = [std::ptr::null::<i32>(); TAPS];
    for (j, &(half, back, off)) in TAPS_E.iter().enumerate().take(n_taps) {
        let h = if half % 2 == 0 { &*even } else { odd };
        let row = if half >= 2 { &h.rcur } else if back == 0 { &h.cur } else { &h.h[back - 1] };
        ptrs[j] = row.as_ptr().offset(PH as isize + off);
    }
    let bias_ptr = coef_e.as_ptr().add(TAPS * n);

    let v_zero = _mm256_setzero_si256();
    let v_max = _mm256_set1_epi32(MAX);
    let v_round = _mm256_set1_epi32(1 << (QB - 1));
    let v_one = _mm256_set1_epi32(1);
    let v_two = _mm256_set1_epi32(2);
    let v_three = _mm256_set1_epi32(3);
    let v_wcap = _mm256_set1_epi32((WCAP - 1) as i32);
    let v_half = _mm256_set1_ps(0.5);
    let lut_ptr = inv_squares().as_ptr();
    let v_256 = _mm256_set1_epi32(256);
    let v_qmax = _mm256_set1_epi32((QCTX - 2) as i32);
    let v_qzero = _mm256_set1_epi32((QCTX - 1) as i32);

    let mut i = 0usize;
    while i + 8 <= n {
        let mut acc = _mm256_loadu_si256(bias_ptr.add(i) as *const __m256i);
        for j in 0..n_taps {
            let cf = _mm256_loadu_si256(coef_e.as_ptr().add(j * n + i) as *const __m256i);
            let x = _mm256_loadu_si256(ptrs[j].add(i) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(cf, x));
        }
        let p0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(acc, v_round), QB as i32), v_zero), v_max);
        _mm256_storeu_si256(even.ps[0].as_mut_ptr().add(i) as *mut __m256i, p0);

        let a = _mm256_loadu_si256(even.h[0].as_ptr().add(PH + i) as *const __m256i);
        let b = _mm256_loadu_si256(even.h[1].as_ptr().add(PH + i) as *const __m256i);
        let c = _mm256_loadu_si256(even.h[2].as_ptr().add(PH + i) as *const __m256i);
        let left = _mm256_loadu_si256(odd.h[0].as_ptr().add(PH - 1 + i) as *const __m256i);
        let right = _mm256_loadu_si256(odd.h[0].as_ptr().add(PH + i) as *const __m256i);

        let p1 = a;
        let a2 = _mm256_slli_epi32(a, 1);
        let p2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(a2, b), v_zero), v_max);
        let b3 = _mm256_mullo_epi32(b, v_three);
        let a3 = _mm256_add_epi32(a2, a);
        let p3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(_mm256_sub_epi32(a3, b3), c), v_zero), v_max);
        let p4 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(left, right), v_one), 1);

        _mm256_storeu_si256(even.ps[1].as_mut_ptr().add(i) as *mut __m256i, p1);
        _mm256_storeu_si256(even.ps[2].as_mut_ptr().add(i) as *mut __m256i, p2);
        _mm256_storeu_si256(even.ps[3].as_mut_ptr().add(i) as *mut __m256i, p3);
        _mm256_storeu_si256(even.ps[4].as_mut_ptr().add(i) as *mut __m256i, p4);

        let p5 = if np == NP {
            let rc = _mm256_loadu_si256(even.rcur.as_ptr().add(PH + i) as *const __m256i);
            let rp = _mm256_loadu_si256(even.rprev.as_ptr().add(PH + i) as *const __m256i);
            let p5v = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_add_epi32(rc, a), rp), v_zero), v_max);
            _mm256_storeu_si256(even.ps[5].as_mut_ptr().add(i) as *mut __m256i, p5v);
            p5v
        } else {
            v_zero
        };

        let ps_arr = [p0, p1, p2, p3, p4, p5];
        let mut num = _mm256_setzero_ps();
        let mut den = _mm256_setzero_ps();
        for m in 0..np {
            let ep = even.err.get_unchecked(m).as_ptr().add(PH + i);
            let e0 = _mm256_loadu_si256(ep.offset(-1) as *const __m256i);
            let e1 = _mm256_loadu_si256(ep as *const __m256i);
            let e2 = _mm256_loadu_si256(ep.offset(1) as *const __m256i);
            let idx = _mm256_min_epi32(_mm256_add_epi32(_mm256_add_epi32(v_two, e0), _mm256_add_epi32(_mm256_add_epi32(e1, e1), e2)), v_wcap);
            let wt = _mm256_i32gather_ps::<4>(lut_ptr, _mm256_max_epi32(idx, v_one));
            let pf = _mm256_cvtepi32_ps(ps_arr[m]);
            // Exact separate IEEE multiply and add (no FMA).
            num = _mm256_add_ps(num, _mm256_mul_ps(wt, pf));
            den = _mm256_add_ps(den, wt);
        }
        let pred = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(num, den), v_half)), v_zero), v_max);
        _mm256_storeu_si256(even.pred.as_mut_ptr().add(i) as *mut __m256i, pred);

        let r0 = _mm256_loadu_si256(even.res.as_ptr().add(PH - 1 + i) as *const __m256i);
        let r1 = _mm256_loadu_si256(even.res.as_ptr().add(PH + i) as *const __m256i);
        let r2 = _mm256_loadu_si256(even.res.as_ptr().add(PH + 1 + i) as *const __m256i);
        let s0 = _mm256_loadu_si256(odd.res.as_ptr().add(PH - 1 + i) as *const __m256i);
        let s1 = _mm256_loadu_si256(odd.res.as_ptr().add(PH + i) as *const __m256i);
        let act = _mm256_add_epi32(_mm256_add_epi32(r0, _mm256_slli_epi32(r1, 1)), _mm256_add_epi32(r2, _mm256_add_epi32(s0, s1)));
        let qa = _mm256_min_epi32(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_add_epi32(act, v_two))), 22), v_256), v_qmax);
        let is_zero = _mm256_and_si256(_mm256_cmpeq_epi32(a, v_zero), _mm256_cmpeq_epi32(left, v_zero));
        let qctx = _mm256_blendv_epi8(qa, v_qzero, is_zero);
        let ge = _mm256_cvtepu16_epi32(_mm_loadu_si128(group_e.as_ptr().add(i) as *const __m128i));
        let c32 = _mm256_add_epi32(ge, qctx);
        let packed = _mm256_packus_epi32(c32, c32);
        let perm = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        _mm_storeu_si128(even.ctx.as_mut_ptr().add(i) as *mut __m128i, _mm256_castsi256_si128(perm));
        i += 8;
    }
    if i < n {
        let lut = inv_squares();
        let [p0, p1, p2, p3, p4, p5] = &mut even.ps;
        let (a, b, c) = (&even.h[0][PH..PH + n], &even.h[1][PH..PH + n], &even.h[2][PH..PH + n]);
        let (rc, rp) = (&even.rcur[PH..PH + n], &even.rprev[PH..PH + n]);
        let (left, right) = (&odd.h[0][PH - 1..PH - 1 + n], &odd.h[0][PH..PH + n]);
        for j in i..n {
            let mut acc = *coef_e.get_unchecked(TAPS * n + j);
            for t in 0..n_taps { acc = acc.wrapping_add((*coef_e.get_unchecked(t * n + j)).wrapping_mul(*ptrs[t].add(j))); }
            p0[j] = (acc.wrapping_add(1 << (QB - 1)) >> QB).clamp(0, MAX);
            p1[j] = a[j];
            p2[j] = (2 * a[j] - b[j]).clamp(0, MAX);
            p3[j] = (3 * a[j] - 3 * b[j] + c[j]).clamp(0, MAX);
            p4[j] = (left[j] + right[j] + 1) >> 1;
            if np == NP { p5[j] = (rc[j] + a[j] - rp[j]).clamp(0, MAX); }
            let mut num = 0.0f32; let mut den = 0.0f32;
            for m in 0..np {
                let e = &even.err[m];
                let wt = lut[((2 + e[PH + j - 1] + 2 * e[PH + j] + e[PH + j + 1]) as usize).min(WCAP - 1)];
                let pv = match m { 0 => p0[j], 1 => p1[j], 2 => p2[j], 3 => p3[j], 4 => p4[j], _ => p5[j] };
                num += wt * pv as f32;
                den += wt;
            }
            even.pred[j] = ((num / den + 0.5) as i32).clamp(0, MAX);
            let act = even.res[PH + j - 1] + 2 * even.res[PH + j] + even.res[PH + j + 1] + odd.res[PH + j - 1] + odd.res[PH + j];
            even.ctx[j] = group_e[j] + if a[j] == 0 && left[j] == 0 { QCTX - 1 } else { qact(act) } as u16;
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn predict_odd_avx2(even: &Half, odd: &mut Half, coef_o: &[i32], group_o: &[u16]) {
    use std::arch::x86_64::*;
    let n = odd.n;
    let np = odd.np;
    let n_taps = if np == NP { TAPS } else { TAPS - 3 };
    let mut ptrs = [std::ptr::null::<i32>(); TAPS];
    for (j, &(half, back, off)) in TAPS_O.iter().enumerate().take(n_taps) {
        let h = if half % 2 == 0 { even } else { &*odd };
        let row = if half >= 2 { &h.rcur } else if back == 0 { &h.cur } else { &h.h[back - 1] };
        ptrs[j] = row.as_ptr().offset(PH as isize + off);
    }
    let bias_ptr = coef_o.as_ptr().add(TAPS * n);

    let v_zero = _mm256_setzero_si256();
    let v_max = _mm256_set1_epi32(MAX);
    let v_round = _mm256_set1_epi32(1 << (QB - 1));
    let v_one = _mm256_set1_epi32(1);
    let v_two = _mm256_set1_epi32(2);
    let v_wcap = _mm256_set1_epi32((WCAP - 1) as i32);
    let v_half = _mm256_set1_ps(0.5);
    let lut_ptr = inv_squares().as_ptr();
    let v_256 = _mm256_set1_epi32(256);
    let v_qmax = _mm256_set1_epi32((QCTX - 2) as i32);
    let v_qzero = _mm256_set1_epi32((QCTX - 1) as i32);

    let mut i = 0usize;
    while i + 8 <= n {
        let mut acc = _mm256_loadu_si256(bias_ptr.add(i) as *const __m256i);
        for j in 0..n_taps {
            let cf = _mm256_loadu_si256(coef_o.as_ptr().add(j * n + i) as *const __m256i);
            let x = _mm256_loadu_si256(ptrs[j].add(i) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(cf, x));
        }
        let p0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(acc, v_round), QB as i32), v_zero), v_max);
        _mm256_storeu_si256(odd.ps[0].as_mut_ptr().add(i) as *mut __m256i, p0);

        let a = _mm256_loadu_si256(odd.h[0].as_ptr().add(PH + i) as *const __m256i);
        let b = _mm256_loadu_si256(odd.h[1].as_ptr().add(PH + i) as *const __m256i);
        let cl = _mm256_loadu_si256(even.cur.as_ptr().add(PH + i) as *const __m256i);
        let cr = _mm256_loadu_si256(even.cur.as_ptr().add(PH + 1 + i) as *const __m256i);
        let hl = _mm256_loadu_si256(even.h[0].as_ptr().add(PH + i) as *const __m256i);
        let hr = _mm256_loadu_si256(even.h[0].as_ptr().add(PH + 1 + i) as *const __m256i);

        let mid = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(cl, cr), v_one), 1);
        let p1 = mid;
        let hmid = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(hl, hr), v_one), 1);
        let p2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_add_epi32(mid, a), hmid), v_zero), v_max);
        let p3 = a;
        let p4 = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_slli_epi32(a, 1), b), v_zero), v_max);

        _mm256_storeu_si256(odd.ps[1].as_mut_ptr().add(i) as *mut __m256i, p1);
        _mm256_storeu_si256(odd.ps[2].as_mut_ptr().add(i) as *mut __m256i, p2);
        _mm256_storeu_si256(odd.ps[3].as_mut_ptr().add(i) as *mut __m256i, p3);
        _mm256_storeu_si256(odd.ps[4].as_mut_ptr().add(i) as *mut __m256i, p4);

        let p5 = if np == NP {
            let rc = _mm256_loadu_si256(odd.rcur.as_ptr().add(PH + i) as *const __m256i);
            let rp = _mm256_loadu_si256(odd.rprev.as_ptr().add(PH + i) as *const __m256i);
            let p5v = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(_mm256_add_epi32(rc, a), rp), v_zero), v_max);
            _mm256_storeu_si256(odd.ps[5].as_mut_ptr().add(i) as *mut __m256i, p5v);
            p5v
        } else {
            v_zero
        };

        let ps_arr = [p0, p1, p2, p3, p4, p5];
        let mut num = _mm256_setzero_ps();
        let mut den = _mm256_setzero_ps();
        for m in 0..np {
            let ep = odd.err.get_unchecked(m).as_ptr().add(PH + i);
            let e0 = _mm256_loadu_si256(ep.offset(-1) as *const __m256i);
            let e1 = _mm256_loadu_si256(ep as *const __m256i);
            let e2 = _mm256_loadu_si256(ep.offset(1) as *const __m256i);
            let idx = _mm256_min_epi32(_mm256_add_epi32(_mm256_add_epi32(v_two, e0), _mm256_add_epi32(_mm256_add_epi32(e1, e1), e2)), v_wcap);
            let wt = _mm256_i32gather_ps::<4>(lut_ptr, _mm256_max_epi32(idx, v_one));
            let pf = _mm256_cvtepi32_ps(ps_arr[m]);
            num = _mm256_add_ps(num, _mm256_mul_ps(wt, pf));
            den = _mm256_add_ps(den, wt);
        }
        let pred = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(num, den), v_half)), v_zero), v_max);
        _mm256_storeu_si256(odd.pred.as_mut_ptr().add(i) as *mut __m256i, pred);

        let r0 = _mm256_loadu_si256(odd.res.as_ptr().add(PH - 1 + i) as *const __m256i);
        let r1 = _mm256_loadu_si256(odd.res.as_ptr().add(PH + i) as *const __m256i);
        let r2 = _mm256_loadu_si256(odd.res.as_ptr().add(PH + 1 + i) as *const __m256i);
        let ql = _mm256_abs_epi32(_mm256_loadu_si256(even.q.as_ptr().add(PH + i) as *const __m256i));
        let qr = _mm256_abs_epi32(_mm256_loadu_si256(even.q.as_ptr().add(PH + 1 + i) as *const __m256i));
        let act = _mm256_add_epi32(_mm256_add_epi32(r0, _mm256_slli_epi32(r1, 1)), _mm256_add_epi32(r2, _mm256_add_epi32(ql, qr)));
        let qa = _mm256_min_epi32(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_add_epi32(act, v_two))), 22), v_256), v_qmax);
        let is_zero = _mm256_and_si256(_mm256_cmpeq_epi32(cl, v_zero), _mm256_cmpeq_epi32(cr, v_zero));
        let qctx = _mm256_blendv_epi8(qa, v_qzero, is_zero);
        let go = _mm256_cvtepu16_epi32(_mm_loadu_si128(group_o.as_ptr().add(i) as *const __m128i));
        let c32 = _mm256_add_epi32(go, qctx);
        let packed = _mm256_packus_epi32(c32, c32);
        let perm = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        _mm_storeu_si128(odd.ctx.as_mut_ptr().add(i) as *mut __m128i, _mm256_castsi256_si128(perm));
        i += 8;
    }
    if i < n {
        let lut = inv_squares();
        let [p0, p1, p2, p3, p4, p5] = &mut odd.ps;
        let (a, b) = (&odd.h[0][PH..PH + n], &odd.h[1][PH..PH + n]);
        let (rc, rp) = (&odd.rcur[PH..PH + n], &odd.rprev[PH..PH + n]);
        let (cl, cr) = (&even.cur[PH..PH + n], &even.cur[PH + 1..PH + 1 + n]);
        let (hl, hr) = (&even.h[0][PH..PH + n], &even.h[0][PH + 1..PH + 1 + n]);
        for j in i..n {
            let mut acc = *coef_o.get_unchecked(TAPS * n + j);
            for t in 0..n_taps { acc = acc.wrapping_add((*coef_o.get_unchecked(t * n + j)).wrapping_mul(*ptrs[t].add(j))); }
            p0[j] = (acc.wrapping_add(1 << (QB - 1)) >> QB).clamp(0, MAX);
            let mid = (cl[j] + cr[j] + 1) >> 1;
            p1[j] = mid;
            p2[j] = (mid + a[j] - ((hl[j] + hr[j] + 1) >> 1)).clamp(0, MAX);
            p3[j] = a[j];
            p4[j] = (2 * a[j] - b[j]).clamp(0, MAX);
            if np == NP { p5[j] = (rc[j] + a[j] - rp[j]).clamp(0, MAX); }
            let mut num = 0.0f32; let mut den = 0.0f32;
            for m in 0..np {
                let e = &odd.err[m];
                let wt = lut[((2 + e[PH + j - 1] + 2 * e[PH + j] + e[PH + j + 1]) as usize).min(WCAP - 1)];
                let pv = match m { 0 => p0[j], 1 => p1[j], 2 => p2[j], 3 => p3[j], 4 => p4[j], _ => p5[j] };
                num += wt * pv as f32;
                den += wt;
            }
            odd.pred[j] = ((num / den + 0.5) as i32).clamp(0, MAX);
            let act = odd.res[PH + j - 1] + 2 * odd.res[PH + j] + odd.res[PH + j + 1] + even.q[PH + j].abs() + even.q[PH + j + 1].abs();
            odd.ctx[j] = group_o[j] + if cl[j] == 0 && cr[j] == 0 { QCTX - 1 } else { qact(act) } as u16;
        }
    }
}

/// Least squares by elimination with partial pivoting, lightly regularised;
/// a degenerate unknown comes out 0.
fn solve(mut a: [f64; K * K], mut b: [f64; K]) -> [f64; K] {
    for i in 0..K {
        a[i * K + i] += 1e-3 * a[i * K + i].abs() + 1e-6;
    }
    for c in 0..K {
        let mut piv = c;
        for r in c + 1..K {
            if a[r * K + c].abs() > a[piv * K + c].abs() {
                piv = r;
            }
        }
        if a[piv * K + c].abs() < 1e-12 {
            continue;
        }
        if piv != c {
            for j in 0..K {
                a.swap(c * K + j, piv * K + j);
            }
            b.swap(c, piv);
        }
        for r in 0..K {
            if r != c {
                let m = a[r * K + c] / a[c * K + c];
                if m != 0.0 {
                    for j in c..K {
                        a[r * K + j] -= m * a[c * K + j];
                    }
                    b[r] -= m * b[c];
                }
            }
        }
    }
    std::array::from_fn(|i| if a[i * K + i].abs() < 1e-12 { 0.0 } else { b[i] / a[i * K + i] })
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fit_8bars_interior_madd16(
    codes: &[u16],
    frames: usize,
    bars: usize,
    reference: Option<&[u16]>,
    f0: usize,
) -> [[i32; K]; 8] {
    use std::arch::x86_64::*;
    let phase = f0 % FIT_STEP;
    let half = phase % 2;
    let taps = if half == 0 { &TAPS_E } else { &TAPS_O };
    let n_active = if reference.is_some() { TAPS } else { TAPS - 3 };
    let k0 = f0 / 2;

    let mut ata_u64 = [[0u64; 8]; K * K];
    let mut aty_u64 = [[0u64; 8]; K];

    let total_t = if frames > phase { (frames - 1 - phase) / FIT_STEP + 1 } else { 0 };
    let mut step_base = 0usize;
    while step_base < total_t {
        let count = (total_t - step_base).min(64);
        let mut ata_i32 = [[0i32; 8]; K * K];
        let mut aty_i32 = [[0i32; 8]; K];
        let mut idx = 0usize;
        while idx + 1 < count {
            let t_a = phase + (step_base + idx) * FIT_STEP;
            let t_b = t_a + FIT_STEP;
            let t0_a = t_a & !(CHUNK_ROWS - 1);
            let t0_b = t_b & !(CHUNK_ROWS - 1);

            let mut y_arr = [0i16; 16];
            let ra = codes.as_ptr().add(t_a * bars + f0);
            let rb = codes.as_ptr().add(t_b * bars + f0);
            for i in 0..8 {
                y_arr[2 * i] = *ra.add(i * 8) as i16;
                y_arr[2 * i + 1] = *rb.add(i * 8) as i16;
            }
            let yv = _mm256_loadu_si256(y_arr.as_ptr() as *const __m256i);

            let mut f_vec = [_mm256_setzero_si256(); K];
            let mut f_arr = [[0i16; 16]; K];
            for j in 0..n_active {
                let (h, back, off) = taps[j];
                let h_off = (h & 1) as isize;
                let base_off = (2 * (k0 as isize + off) + h_off) as usize;
                let src_a: *const u16 = if h >= 2 {
                    reference.unwrap_unchecked().as_ptr().add(t_a * bars + base_off)
                } else if t_a < t0_a + back {
                    std::ptr::null()
                } else {
                    codes.as_ptr().add((t_a - back) * bars + base_off)
                };
                let src_b: *const u16 = if h >= 2 {
                    reference.unwrap_unchecked().as_ptr().add(t_b * bars + base_off)
                } else if t_b < t0_b + back {
                    std::ptr::null()
                } else {
                    codes.as_ptr().add((t_b - back) * bars + base_off)
                };
                if !src_a.is_null() && !src_b.is_null() {
                    for i in 0..8 {
                        f_arr[j][2 * i] = *src_a.add(i * 8) as i16;
                        f_arr[j][2 * i + 1] = *src_b.add(i * 8) as i16;
                    }
                } else {
                    if !src_a.is_null() { for i in 0..8 { f_arr[j][2 * i] = *src_a.add(i * 8) as i16; } }
                    if !src_b.is_null() { for i in 0..8 { f_arr[j][2 * i + 1] = *src_b.add(i * 8) as i16; } }
                }
                f_vec[j] = _mm256_loadu_si256(f_arr[j].as_ptr() as *const __m256i);
            }
            f_vec[K - 1] = _mm256_set1_epi16(1);

            for a in 0..n_active {
                let fa = f_vec[a];
                let yp = _mm256_madd_epi16(fa, yv);
                let ya = _mm256_loadu_si256(aty_i32[a].as_ptr() as *const __m256i);
                _mm256_storeu_si256(aty_i32[a].as_mut_ptr() as *mut __m256i, _mm256_add_epi32(ya, yp));
                let base = a * K;
                for b in a..n_active {
                    let prod = _mm256_madd_epi16(fa, f_vec[b]);
                    let cur = _mm256_loadu_si256(ata_i32[base + b].as_ptr() as *const __m256i);
                    _mm256_storeu_si256(ata_i32[base + b].as_mut_ptr() as *mut __m256i, _mm256_add_epi32(cur, prod));
                }
                let prod_one = _mm256_madd_epi16(fa, f_vec[K - 1]);
                let cur_one = _mm256_loadu_si256(ata_i32[base + K - 1].as_ptr() as *const __m256i);
                _mm256_storeu_si256(ata_i32[base + K - 1].as_mut_ptr() as *mut __m256i, _mm256_add_epi32(cur_one, prod_one));
            }
            let ya = _mm256_loadu_si256(aty_i32[K - 1].as_ptr() as *const __m256i);
            _mm256_storeu_si256(aty_i32[K - 1].as_mut_ptr() as *mut __m256i, _mm256_add_epi32(ya, _mm256_madd_epi16(f_vec[K - 1], yv)));
            ata_i32[(K - 1) * K + (K - 1)][0] += 2;
            idx += 2;
        }
        if idx < count {
            let t = phase + (step_base + idx) * FIT_STEP;
            let t0 = t & !(CHUNK_ROWS - 1);
            let mut y = [0u32; 8];
            let ry = codes.as_ptr().add(t * bars + f0);
            for i in 0..8 { y[i] = *ry.add(i * 8) as u32; }
            let mut feat = [[0u32; 8]; K];
            for j in 0..n_active {
                let (h, back, off) = taps[j];
                let h_off = (h & 1) as isize;
                let base_off = (2 * (k0 as isize + off) + h_off) as usize;
                let src: *const u16 = if h >= 2 {
                    reference.unwrap_unchecked().as_ptr().add(t * bars + base_off)
                } else if t < t0 + back {
                    std::ptr::null()
                } else {
                    codes.as_ptr().add((t - back) * bars + base_off)
                };
                if !src.is_null() { for i in 0..8 { feat[j][i] = *src.add(i * 8) as u32; } }
            }
            feat[K - 1] = [1; 8];
            for i in 0..8 {
                for a in 0..K {
                    let fa = feat[a][i] as u64;
                    aty_u64[a][i] += fa * y[i] as u64;
                    for b in a..K { ata_u64[a * K + b][i] += fa * feat[b][i] as u64; }
                }
            }
        }
        for a in 0..K {
            for i in 0..8 {
                aty_u64[a][i] += aty_i32[a][i] as u64;
                if a < K - 1 {
                    for b in a..n_active { ata_u64[a * K + b][i] += ata_i32[a * K + b][i] as u64; }
                    ata_u64[a * K + K - 1][i] += ata_i32[a * K + K - 1][i] as u64;
                } else {
                    ata_u64[(K - 1) * K + (K - 1)][i] += ata_i32[(K - 1) * K + (K - 1)][0] as u64;
                }
            }
        }
        step_base += 64;
    }

    let scale = (1u32 << QB) as f64;
    let mut out = [[0i32; K]; 8];
    for i in 0..8 {
        let mut mat = [0f64; K * K];
        let mut rhs = [0f64; K];
        for a in 0..K {
            rhs[a] = aty_u64[a][i] as f64;
            for b in a..K {
                let v = ata_u64[a * K + b][i] as f64;
                mat[a * K + b] = v;
                mat[b * K + a] = v;
            }
        }
        let w = solve(mat, rhs);
        for j in 0..TAPS { out[i][j] = ((w[j] * scale).round() as i64).clamp(-COEF_MAX, COEF_MAX) as i32; }
        out[i][TAPS] = ((w[TAPS] * scale).round() as i64).clamp(-BIAS_MAX, BIAS_MAX) as i32;
    }
    out
}

fn fit_scalar_bar(codes: &[u16], frames: usize, bars: usize, reference: Option<&[u16]>, f: usize) -> [i32; K] {
    let (ne, no) = (bars.div_ceil(2), bars / 2);
    let (half, k) = (f % 2, (f / 2) as isize);
    let taps = if half == 0 { &TAPS_E } else { &TAPS_O };
    let mut ata = [0f64; K * K];
    let mut aty = [0f64; K];
    let mut feat = [0f64; K];
    for t in (f % FIT_STEP..frames).step_by(FIT_STEP) {
        let t0 = t / CHUNK_ROWS * CHUNK_ROWS;
        for (j, &(h, back, off)) in taps.iter().enumerate() {
            let n = if h % 2 == 0 { ne } else { no } as isize;
            let kk = k + off;
            feat[j] = if kk < 0 || kk >= n {
                0.0
            } else if h >= 2 {
                reference.map_or(0.0, |r| r[t * bars + 2 * kk as usize + h - 2] as f64)
            } else if t < t0 + back {
                0.0
            } else {
                codes[(t - back) * bars + 2 * kk as usize + h] as f64
            };
        }
        feat[K - 1] = 1.0;
        let y = codes[t * bars + f] as f64;
        for a in 0..K {
            let fa = feat[a];
            if fa == 0.0 {
                continue;
            }
            aty[a] += fa * y;
            for b in a..K {
                ata[a * K + b] += fa * feat[b];
            }
        }
    }
    for a in 0..K {
        for b in 0..a {
            ata[a * K + b] = ata[b * K + a];
        }
    }
    let w = solve(ata, aty);
    let scale = (1u32 << QB) as f64;
    let mut q = [0i32; K];
    for j in 0..TAPS {
        q[j] = ((w[j] * scale).round() as i64).clamp(-COEF_MAX, COEF_MAX) as i32;
    }
    q[TAPS] = ((w[TAPS] * scale).round() as i64).clamp(-BIAS_MAX, BIAS_MAX) as i32;
    q
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_u16_avx2(s: &[u16]) -> u16 {
    use std::arch::x86_64::*;
    let mut vmax = _mm256_setzero_si256();
    let mut i = 0usize;
    let n = s.len();
    let p = s.as_ptr();
    while i + 16 <= n {
        vmax = _mm256_max_epu16(vmax, _mm256_loadu_si256(p.add(i) as *const __m256i));
        i += 16;
    }
    let mut tmp = [0u16; 16];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, vmax);
    let mut m = tmp.into_iter().max().unwrap_or(0);
    while i < n {
        m = m.max(*p.add(i));
        i += 1;
    }
    m
}

fn max_u16(s: &[u16]) -> u16 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support verified above.
        return unsafe { max_u16_avx2(s) };
    }
    s.iter().copied().max().unwrap_or(0)
}

fn fit_verified(codes: &[u16], frames: usize, bars: usize, reference: Option<&[u16]>, codes_checked: bool) -> (Vec<i32>, Vec<i32>) {
    let (ne, no) = (bars.div_ceil(2), bars / 2);
    let use_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            bars >= 128
                && (frames / FIT_STEP + 1) <= 2_097_000
                && is_x86_feature_detected!("avx2")
                && (codes_checked || max_u16(codes) as i32 <= MAX)
                && reference.is_none_or(|r| max_u16(r) as i32 <= MAX)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = codes_checked;
            false
        }
    };
    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        {
            enum Task { Batch8(usize), Single(usize) }
            let mut tasks = Vec::with_capacity(bars / 4);
            for phase in 0..FIT_STEP {
                let mut f = phase;
                if f < 4 {
                    tasks.push(Task::Single(f));
                    f += FIT_STEP;
                }
                while f + 7 * FIT_STEP + 4 < bars {
                    tasks.push(Task::Batch8(f));
                    f += 8 * FIT_STEP;
                }
                while f < bars {
                    tasks.push(Task::Single(f));
                    f += FIT_STEP;
                }
            }
            let done: Vec<(usize, [[i32; K]; 8], bool)> = par(tasks.len(), |i| match tasks[i] {
                Task::Batch8(f0) => {
                    // SAFETY: AVX2 & interior bounds + <= MAX bounds verified above.
                    (f0, unsafe { fit_8bars_interior_madd16(codes, frames, bars, reference, f0) }, true)
                }
                Task::Single(f) => {
                    let mut out = [[0i32; K]; 8];
                    out[0] = fit_scalar_bar(codes, frames, bars, reference, f);
                    (f, out, false)
                }
            });
            let (mut ce, mut co) = (vec![0i32; K * ne], vec![0i32; K * no]);
            for (f0, qs, is_batch) in done {
                let count = if is_batch { 8 } else { 1 };
                for idx in 0..count {
                    let f = f0 + idx * FIT_STEP;
                    let (out, n) = if f % 2 == 0 { (&mut ce, ne) } else { (&mut co, no) };
                    for j in 0..K { out[j * n + f / 2] = qs[idx][j]; }
                }
            }
            return (ce, co);
        }
    }
    let cols = par(bars, |f| fit_scalar_bar(codes, frames, bars, reference, f));
    let (mut ce, mut co) = (vec![0i32; K * ne], vec![0i32; K * no]);
    for (f, q) in cols.iter().enumerate() {
        let (out, n) = if f % 2 == 0 { (&mut ce, ne) } else { (&mut co, no) };
        for j in 0..K {
            out[j * n + f / 2] = q[j];
        }
    }
    (ce, co)
}

struct Pass1 {
    syms: Vec<u16>,
    raw: Vec<u32>,
    hist: Vec<u64>,
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i32_avx2(a: &[i32], b: &[i32], out: &mut [i32], n: usize) {
    use std::arch::x86_64::*;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(bp.add(i) as *const __m256i);
        _mm256_storeu_si256(op.add(i) as *mut __m256i, _mm256_sub_epi32(va, vb));
        i += 8;
    }
    while i < n {
        *op.add(i) = *ap.add(i) - *bp.add(i);
        i += 1;
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn set_ref_avx2(m: &mut Model, r: &[u16], ti: usize, tl: usize, bars: usize) {
    let (ne, no) = (m.even.n, m.odd.n);
    let cur = &r[ti * bars..(ti + 1) * bars];
    if tl > 0 {
        std::mem::swap(&mut m.even.rprev, &mut m.even.rcur);
        std::mem::swap(&mut m.odd.rprev, &mut m.odd.rcur);
        deinterleave_row_avx2(cur, &mut m.even.rcur[PH..PH + ne], &mut m.odd.rcur[PH..PH + no], ne, no);
    } else {
        deinterleave_row_avx2(cur, &mut m.even.rcur[PH..PH + ne], &mut m.odd.rcur[PH..PH + no], ne, no);
        if ti > 0 {
            let prev = &r[(ti - 1) * bars..ti * bars];
            deinterleave_row_avx2(prev, &mut m.even.rprev[PH..PH + ne], &mut m.odd.rprev[PH..PH + no], ne, no);
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pass1_avx2(codes: &[u16], bars: usize, reference: Option<&[u16]>, t0: usize, rows: usize, ce: &[i32], co: &[i32]) -> Pass1 {
    let mut m = Model::new(bars, ce, co, reference.is_some());
    let (ne, no) = (m.even.n, m.odd.n);
    let mut syms = Vec::with_capacity(rows * bars);
    let mut raw_out = Vec::with_capacity(rows * bars / 4 + 16);
    let mut raw_acc = 0u64;
    let mut raw_n = 0u32;
    let mut hist = vec![0u64; NCTX * NSYM];

    for (tl, row) in codes[t0 * bars..(t0 + rows) * bars].chunks_exact(bars).enumerate() {
        if let Some(r) = reference {
            set_ref_avx2(&mut m, r, t0 + tl, tl, bars);
        }
        deinterleave_row_avx2(row, &mut m.even.cur[PH..PH + ne], &mut m.odd.cur[PH..PH + no], ne, no);
        for half in 0..2 {
            if half == 0 {
                predict_even_avx2(&mut m.even, &m.odd, m.coef_e, &m.group_e);
            } else {
                predict_odd_avx2(&m.even, &mut m.odd, m.coef_o, &m.group_o);
            }
            let h = if half == 0 { &mut m.even } else { &mut m.odd };
            let n_half = h.n;
            sub_i32_avx2(&h.cur[PH..PH + n_half], &h.pred[..n_half], &mut h.q[PH..PH + n_half], n_half);
            for k in 0..n_half {
                let q = *h.q.get_unchecked(PH + k);
                let (tok, n, bits) = tokenize(q);
                if n > 0 {
                    raw_acc |= (bits as u64) << raw_n;
                    raw_n += n;
                    if raw_n >= 32 {
                        raw_out.push(raw_acc as u32);
                        raw_acc >>= 32;
                        raw_n -= 32;
                    }
                }
                let ctx = *h.ctx.get_unchecked(k) as usize;
                syms.push((ctx * 64) as u16 | tok as u16);
                hist[ctx * NSYM + tok as usize] += 1;
            }
        }
        commit_avx2(&mut m.even);
        commit_avx2(&mut m.odd);
    }
    if raw_n > 0 {
        raw_out.push(raw_acc as u32);
    }
    Pass1 { syms, raw: raw_out, hist }
}

/// One chunk through the model: its symbols (context and token), raw bits and
/// symbol counts.
fn pass1(codes: &[u16], bars: usize, reference: Option<&[u16]>, t0: usize, rows: usize, ce: &[i32], co: &[i32]) -> Pass1 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability checked once at entry.
        return unsafe { pass1_avx2(codes, bars, reference, t0, rows, ce, co) };
    }
    let mut m = Model::new(bars, ce, co, reference.is_some());
    let mut syms = Vec::with_capacity(rows * bars);
    let mut raw = BitW { acc: 0, n: 0, out: Vec::new() };
    let mut hist = vec![0u64; NCTX * NSYM];
    for t in t0..t0 + rows {
        let row = &codes[t * bars..(t + 1) * bars];
        if let Some(r) = reference {
            m.set_ref(&r[t * bars..(t + 1) * bars], (t > 0).then(|| &r[(t - 1) * bars..t * bars]));
        }
        for half in 0..2 {
            if half == 0 { m.predict_even() } else { m.predict_odd() }
            let h = if half == 0 { &mut m.even } else { &mut m.odd };
            for k in 0..h.n {
                let v = row[2 * k + half] as i32;
                let q = v - h.pred[k];
                h.cur[PH + k] = v;
                h.q[PH + k] = q;
                let (tok, n, bits) = tokenize(q);
                if n > 0 {
                    raw.put(bits, n);
                }
                let ctx = h.ctx[k] as usize;
                syms.push((ctx * 64) as u16 | tok as u16);
                hist[ctx * NSYM + tok as usize] += 1;
            }
        }
        m.even.commit();
        m.odd.commit();
    }
    Pass1 { syms, raw: raw.finish(), hist }
}

/// The body for a plane of 12-bit codes, `frames × bars`, row-major. With a
/// `reference` (same shape) the plane is coded against it and needs the same
/// one to decode.
///
/// # Panics
///
/// If the shape is empty or does not match the slices, or a code is above
/// 4095: the callers own those invariants.
pub fn encode(codes: &[u16], frames: usize, bars: usize, reference: Option<&[u16]>) -> Vec<u8> {
    assert!(frames > 0 && bars > 0 && codes.len() == frames * bars, "v5: shape");
    assert!(reference.is_none_or(|r| r.len() == codes.len()), "v5: reference shape");
    assert!(max_u16(codes) as i32 <= MAX, "v5: a code above 12 bits");
    let (ce, co) = fit_verified(codes, frames, bars, reference, true);
    let n = frames.div_ceil(CHUNK_ROWS);
    let parts = par(n, |c| {
        let t0 = c * CHUNK_ROWS;
        let rows = (frames - t0).min(CHUNK_ROWS);
        (pass1(codes, bars, reference, t0, rows, &ce, &co), checksum(&codes[t0 * bars..(t0 + rows) * bars]))
    });
    let mut hist = vec![0u64; NCTX * NSYM];
    for (r, _) in &parts {
        for (h, v) in hist.iter_mut().zip(&r.hist) {
            *h += v;
        }
    }
    let freqs = Tables::freqs_for(&hist, PB);
    let tables = Tables::from_freqs(&freqs, PB).expect("frequencies built to sum");
    let rans = par(n, |c| rans_encode(&parts[c].0.syms, &tables));

    let payload_bytes: usize = rans.iter().map(|r| r.len() * 2).sum::<usize>()
        + parts.iter().map(|(p, _)| p.raw.len() * 4).sum::<usize>();
    let total_bytes = HEAD + bars * (TAPS * 2 + 4) + NCTX * NSYM * 2 + n * 16 + payload_bytes;
    let mut out = Vec::with_capacity(total_bytes);
    out.extend_from_slice(&(CHUNK_ROWS as u32).to_le_bytes());
    out.extend_from_slice(&[PB as u8, 0, 0, 0]);
    for (coefs, n) in [(&ce, bars.div_ceil(2)), (&co, bars / 2)] {
        for k in 0..n {
            for j in 0..TAPS {
                out.extend_from_slice(&(coefs[j * n + k] as i16).to_le_bytes());
            }
            out.extend_from_slice(&coefs[TAPS * n + k].to_le_bytes());
        }
    }
    for f in &freqs {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for (c, (p, sum)) in parts.iter().enumerate() {
        out.extend_from_slice(&(rans[c].len() as u32).to_le_bytes());
        out.extend_from_slice(&(p.raw.len() as u32).to_le_bytes());
        out.extend_from_slice(&sum.to_le_bytes());
    }
    for (c, (p, _)) in parts.iter().enumerate() {
        for w in &rans[c] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for w in &p.raw {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_chunk(bars: usize, reference: Option<&[u16]>, t0: usize, ce: &[i32], co: &[i32], t: &Tables, rans: &[u8], raw: &[u8], out: &mut [u16]) {
    let mut m = Model::new(bars, ce, co, reference.is_some());
    let mut bits = BitR { acc: 0, n: 0, src: raw, pos: 0 };
    let pb = t.pb;
    let mask = (1u32 << pb) - 1;
    // A stream that ends early reads zeros; the checksum then refuses it.
    let word = |pos: usize| rans.get(2 * pos..2 * pos + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) as u32;
    let mut xs = [word(0) << 16 | word(1), word(2) << 16 | word(3)];
    let mut pos = 4;
    let mut sym = 0usize;
    let tt = token_table();
    for (tl, row) in out.chunks_exact_mut(bars).enumerate() {
        if let Some(r) = reference {
            let t = t0 + tl;
            m.set_ref(&r[t * bars..(t + 1) * bars], (t > 0).then(|| &r[(t - 1) * bars..t * bars]));
        }
        for half in 0..2 {
            if half == 0 { m.predict_even() } else { m.predict_odd() }
            let h = if half == 0 { &mut m.even } else { &mut m.odd };
            for k in 0..h.n {
                let ctx = h.ctx[k] as usize;
                let mut x = xs[sym & 1];
                let slot = x & mask;
                let tok = t.s2s[(ctx << pb) + slot as usize] as usize;
                let (freq, start) = t.fs[ctx * NSYM + tok];
                x = (freq as u32).wrapping_mul(x >> pb).wrapping_add(slot).wrapping_sub(start as u32);
                let need = (x < RL) as u32;
                x = (x << (16 * need)) | (word(pos) & need.wrapping_neg());
                pos += need as usize;
                xs[sym & 1] = x;
                sym += 1;
                let (base, n) = tt[tok];
                let z = base | bits.read(n);
                let q = ((z >> 1) as i32) ^ -((z & 1) as i32);
                let v = (h.pred[k] + q).clamp(0, MAX);
                h.cur[PH + k] = v;
                h.q[PH + k] = q;
                row[2 * k + half] = v as u16;
            }
        }
        m.even.commit();
        m.odd.commit();
    }
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// The plane a body encodes, `frames × bars` codes, row-major. `reference` is
/// the plane it was coded against, if any; the caller knows from the header
/// whether one is needed. The caller has bounded `frames × bars`.
pub fn decode(body: &[u8], frames: usize, bars: usize, reference: Option<&[u16]>) -> Result<Vec<u16>, Refused> {
    let bad = |why| Err(Refused::Malformed(why));
    let cells = frames.checked_mul(bars).filter(|&c| c > 0);
    let Some(cells) = cells else { return bad("an empty plane") };
    if reference.is_some_and(|r| r.len() != cells) {
        return bad("the reference has another shape");
    }
    if body.len() < HEAD {
        return bad("too short for its header");
    }
    let (chunk_rows, pb) = (u32_at(body, 0), body[4] as u32);
    if !(1..=65_536).contains(&chunk_rows) || !(8..=12).contains(&pb) || body[5..8] != [0, 0, 0] {
        return bad("parameters outside the format");
    }
    let (ne, no) = (bars.div_ceil(2), bars / 2);
    let n_chunks = frames.div_ceil(chunk_rows);
    let coef_len = bars * (TAPS * 2 + 4);
    let freq_len = NCTX * NSYM * 2;
    let dir_len = n_chunks.checked_mul(16);
    let fixed = dir_len.and_then(|d| HEAD.checked_add(coef_len)?.checked_add(freq_len)?.checked_add(d));
    let Some(fixed) = fixed.filter(|&f| f <= body.len()) else { return bad("too short for its tables") };

    let mut at = HEAD;
    let mut coefs = |n: usize| {
        let mut c = vec![0i32; K * n];
        for k in 0..n {
            for j in 0..TAPS {
                c[j * n + k] = i16::from_le_bytes([body[at], body[at + 1]]) as i32;
                at += 2;
            }
            c[TAPS * n + k] = i32::from_le_bytes([body[at], body[at + 1], body[at + 2], body[at + 3]]);
            at += 4;
        }
        c
    };
    let (ce, co) = (coefs(ne), coefs(no));
    let freqs: Vec<u16> = body[at..at + freq_len].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let tables = Tables::from_freqs(&freqs, pb)?;
    at += freq_len;

    // The directory: every chunk's byte ranges, summing to exactly the body.
    let mut spans = Vec::with_capacity(n_chunks);
    let mut data = fixed;
    for c in 0..n_chunks {
        let d = at + c * 16;
        let (rans_len, raw_len) = (u32_at(body, d).checked_mul(2), u32_at(body, d + 4).checked_mul(4));
        let sum = u64::from_le_bytes(body[d + 8..d + 16].try_into().expect("eight bytes"));
        let end = rans_len.zip(raw_len).and_then(|(a, b)| data.checked_add(a)?.checked_add(b)).filter(|&e| e <= body.len());
        let (Some(rans_len), Some(end)) = (rans_len, end) else { return bad("a chunk runs past the end") };
        spans.push((data, data + rans_len, end, sum));
        data = end;
    }
    if data != body.len() {
        return bad("bytes after the last chunk");
    }

    let mut out = Vec::new();
    if out.try_reserve_exact(cells).is_err() {
        return bad("no memory for the plane");
    }
    out.resize(cells, 0u16);
    let work: Vec<std::sync::Mutex<&mut [u16]>> = out.chunks_mut(chunk_rows * bars).map(std::sync::Mutex::new).collect();
    let good = par(n_chunks, |c| {
        let (a, b, e, sum) = spans[c];
        let mut o = work[c].lock().unwrap_or_else(|e| e.into_inner());
        decode_chunk(bars, reference, c * chunk_rows, &ce, &co, &tables, &body[a..b], &body[b..e], &mut o);
        checksum(&o) == sum
    });
    drop(work);
    if good.iter().all(|&g| g) { Ok(out) } else { Err(Refused::Checksum) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane with structure in time and across bars, noise, silence and
    /// full-scale cells.
    fn plane(frames: usize, bars: usize, seed: u64) -> Vec<u16> {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as u32
        };
        let mut v = Vec::with_capacity(frames * bars);
        for t in 0..frames {
            for f in 0..bars {
                let tone = 1800.0 + 1500.0 * ((t as f32 * 0.03).sin() * (f as f32 * 0.11).cos());
                let noise = (rnd() % 400) as f32 * (f as f32 / bars.max(1) as f32);
                let quiet = if (t / 37) % 5 == 4 { 0.0 } else { 1.0 };
                let spike = if rnd() % 997 == 0 { 4095.0 } else { 0.0 };
                v.push((quiet * (tone + noise) + spike).clamp(0.0, 4095.0) as u16);
            }
        }
        v
    }

    #[test]
    fn a_plane_comes_back_exactly_with_and_without_a_reference() {
        for (frames, bars) in [(1, 1), (1, 2), (3, 5), (40, 16), (700, 64), (1100, 257), (CHUNK_ROWS, 32), (CHUNK_ROWS + 1, 33)] {
            let a = plane(frames, bars, 7);
            let body = encode(&a, frames, bars, None);
            assert_eq!(decode(&body, frames, bars, None).as_deref(), Ok(&a[..]), "{frames}x{bars}");
            // A second plane close to the first, coded against it.
            let b: Vec<u16> = a.iter().enumerate().map(|(i, &c)| (c as i32 + (i % 7) as i32 - 3).clamp(0, 4095) as u16).collect();
            let with = encode(&b, frames, bars, Some(&a));
            assert_eq!(decode(&with, frames, bars, Some(&a)).as_deref(), Ok(&b[..]), "{frames}x{bars} with a reference");
        }
    }

    #[test]
    fn extremes_come_back_exactly() {
        let (frames, bars) = (600, 48);
        for fill in [0u16, 4095] {
            let a = vec![fill; frames * bars];
            assert_eq!(decode(&encode(&a, frames, bars, None), frames, bars, None), Ok(a));
        }
        // Alternating full scale and zero: the largest residuals there are.
        let a: Vec<u16> = (0..frames * bars).map(|i| if (i / bars + i % bars) % 2 == 0 { 4095 } else { 0 }).collect();
        assert_eq!(decode(&encode(&a, frames, bars, None), frames, bars, None), Ok(a));
    }

    #[test]
    fn a_reference_makes_a_related_plane_smaller() {
        let (frames, bars) = (1500, 128);
        let a = plane(frames, bars, 3);
        let b: Vec<u16> = a.iter().enumerate().map(|(i, &c)| (c as i32 + (i % 5) as i32 - 2).clamp(0, 4095) as u16).collect();
        let (alone, with) = (encode(&b, frames, bars, None), encode(&b, frames, bars, Some(&a)));
        assert!(with.len() < alone.len(), "{} against {}", with.len(), alone.len());
    }

    /// The wrong reference, or none, is refused by the checksum rather than
    /// decoded into another picture.
    #[test]
    fn another_reference_is_refused() {
        let (frames, bars) = (600, 64);
        let a = plane(frames, bars, 1);
        let b = plane(frames, bars, 2);
        let body = encode(&b, frames, bars, Some(&a));
        assert_eq!(decode(&body, frames, bars, Some(&plane(frames, bars, 5))), Err(Refused::Checksum));
        assert_eq!(decode(&body, frames, bars, None), Err(Refused::Checksum));
        // A reference off by one code in one cell: refused, or (if that cell's
        // difference rounds away in every prediction) the plane itself.
        let mut other = a.clone();
        other[1234] ^= 1;
        assert!(decode(&body, frames, bars, Some(&other)).map_or(true, |p| p == b));
        assert_eq!(decode(&body, frames, bars, Some(&a)), Ok(b));
    }

    /// Any damage is an error, never a panic and never a plane: every cut,
    /// and a flipped bit at many places.
    #[test]
    fn damage_is_refused_never_trusted() {
        let (frames, bars) = (700, 40);
        let a = plane(frames, bars, 9);
        let body = encode(&a, frames, bars, None);
        for cut in (0..body.len()).step_by(body.len() / 211 + 1) {
            assert!(decode(&body[..cut], frames, bars, None).is_err(), "cut at {cut}");
        }
        let mut longer = body.clone();
        longer.push(0);
        assert!(decode(&longer, frames, bars, None).is_err(), "a trailing byte");
        for at in (0..body.len()).step_by(body.len() / 397 + 1) {
            for bit in [0, 3, 7] {
                let mut hurt = body.clone();
                hurt[at] ^= 1 << bit;
                // An error, or (a bit nothing reads, or one that rounds away)
                // the plane itself: never another plane.
                let got = decode(&hurt, frames, bars, None);
                assert!(got.as_ref().map_or(true, |p| *p == a), "bit {bit} of byte {at} of {}", body.len());
            }
        }
        // Another shape than the one encoded.
        assert!(decode(&body, frames + 1, bars, None).is_err());
        assert!(decode(&body, frames, bars + 1, None).is_err());
        assert!(decode(&body, 0, bars, None).is_err());
    }

    #[test]
    fn frequencies_always_sum_to_the_table() {
        let mut hist = vec![0u64; NCTX * NSYM];
        // One dominant symbol and many rare ones; and a context of equals.
        hist[0] = 1_000_000;
        for s in 1..NSYM { hist[s] = 1; }
        for s in 0..NSYM { hist[NSYM + s] = 10; }
        let f = Tables::freqs_for(&hist, 8);
        assert!(Tables::from_freqs(&f, 8).is_ok());
        assert!(f[..NSYM].iter().all(|&v| v >= 1));
    }

    // -----------------------------------------------------------------------
    // Verbatim old encoder & decoder from de4fcb3 (oracle for byte-identity
    // and cross-compatibility tests)
    // -----------------------------------------------------------------------
    fn old_solve(mut a: Vec<f64>, mut b: Vec<f64>, k: usize) -> Vec<f64> {
        for i in 0..k {
            a[i * k + i] += 1e-3 * a[i * k + i].abs() + 1e-6;
        }
        for c in 0..k {
            let mut piv = c;
            for r in c + 1..k {
                if a[r * k + c].abs() > a[piv * k + c].abs() {
                    piv = r;
                }
            }
            if a[piv * k + c].abs() < 1e-12 {
                continue;
            }
            if piv != c {
                for j in 0..k {
                    a.swap(c * k + j, piv * k + j);
                }
                b.swap(c, piv);
            }
            for r in 0..k {
                if r != c {
                    let m = a[r * k + c] / a[c * k + c];
                    if m != 0.0 {
                        for j in c..k {
                            a[r * k + j] -= m * a[c * k + j];
                        }
                        b[r] -= m * b[c];
                    }
                }
            }
        }
        (0..k).map(|i| if a[i * k + i].abs() < 1e-12 { 0.0 } else { b[i] / a[i * k + i] }).collect()
    }

    fn old_fit(codes: &[u16], frames: usize, bars: usize, reference: Option<&[u16]>) -> (Vec<i32>, Vec<i32>) {
        let (ne, no) = (bars.div_ceil(2), bars / 2);
        let at = |t: usize, half: usize, k: isize, back: usize, t0: usize| -> f64 {
            let n = if half % 2 == 0 { ne } else { no } as isize;
            if k < 0 || k >= n {
                return 0.0;
            }
            if half >= 2 {
                return reference.map_or(0.0, |r| r[t * bars + 2 * k as usize + half - 2] as f64);
            }
            if t < t0 + back { 0.0 } else { codes[(t - back) * bars + 2 * k as usize + half] as f64 }
        };
        let cols = par(bars, |f| {
            let (half, k) = (f % 2, (f / 2) as isize);
            let taps = if half == 0 { &TAPS_E } else { &TAPS_O };
            let mut ata = vec![0f64; K * K];
            let mut aty = vec![0f64; K];
            let mut feat = [0f64; K];
            for t in (f % FIT_STEP..frames).step_by(FIT_STEP) {
                let t0 = t / CHUNK_ROWS * CHUNK_ROWS;
                for (j, &(h, back, off)) in taps.iter().enumerate() {
                    feat[j] = at(t, h, k + off, back, t0);
                }
                feat[K - 1] = 1.0;
                let y = codes[t * bars + f] as f64;
                for a in 0..K {
                    let fa = feat[a];
                    if fa == 0.0 {
                        continue;
                    }
                    aty[a] += fa * y;
                    for b in a..K {
                        ata[a * K + b] += fa * feat[b];
                    }
                }
            }
            for a in 0..K {
                for b in 0..a {
                    ata[a * K + b] = ata[b * K + a];
                }
            }
            let w = old_solve(ata, aty, K);
            let scale = (1u32 << QB) as f64;
            let mut q = [0i32; K];
            for j in 0..TAPS {
                q[j] = ((w[j] * scale).round() as i64).clamp(-COEF_MAX, COEF_MAX) as i32;
            }
            q[TAPS] = ((w[TAPS] * scale).round() as i64).clamp(-BIAS_MAX, BIAS_MAX) as i32;
            q
        });
        let (mut ce, mut co) = (vec![0i32; K * ne], vec![0i32; K * no]);
        for (f, q) in cols.iter().enumerate() {
            let (out, n) = if f % 2 == 0 { (&mut ce, ne) } else { (&mut co, no) };
            for j in 0..K {
                out[j * n + f / 2] = q[j];
            }
        }
        (ce, co)
    }

    fn old_pass1(codes: &[u16], bars: usize, reference: Option<&[u16]>, t0: usize, rows: usize, ce: &[i32], co: &[i32]) -> Pass1 {
        let mut m = Model::new(bars, ce, co, reference.is_some());
        let mut syms = Vec::with_capacity(rows * bars);
        let mut raw = BitW { acc: 0, n: 0, out: Vec::new() };
        let mut hist = vec![0u64; NCTX * NSYM];
        for t in t0..t0 + rows {
            let row = &codes[t * bars..(t + 1) * bars];
            if let Some(r) = reference {
                m.set_ref(&r[t * bars..(t + 1) * bars], (t > 0).then(|| &r[(t - 1) * bars..t * bars]));
            }
            for half in 0..2 {
                if half == 0 { m.predict_even() } else { m.predict_odd() }
                let h = if half == 0 { &mut m.even } else { &mut m.odd };
                for k in 0..h.n {
                    let v = row[2 * k + half] as i32;
                    let q = v - h.pred[k];
                    h.cur[PH + k] = v;
                    h.q[PH + k] = q;
                    let (tok, n, bits) = tokenize(q);
                    if n > 0 {
                        raw.put(bits, n);
                    }
                    let ctx = h.ctx[k] as usize;
                    syms.push((ctx * 64) as u16 | tok as u16);
                    hist[ctx * NSYM + tok as usize] += 1;
                }
            }
            m.even.commit();
            m.odd.commit();
        }
        Pass1 { syms, raw: raw.finish(), hist }
    }

    pub(crate) fn old_encode(codes: &[u16], frames: usize, bars: usize, reference: Option<&[u16]>) -> Vec<u8> {
        assert!(frames > 0 && bars > 0 && codes.len() == frames * bars, "v5: shape");
        assert!(reference.is_none_or(|r| r.len() == codes.len()), "v5: reference shape");
        assert!(codes.iter().all(|&c| c as i32 <= MAX), "v5: a code above 12 bits");
        let (ce, co) = old_fit(codes, frames, bars, reference);
        let n = frames.div_ceil(CHUNK_ROWS);
        let parts = par(n, |c| {
            let t0 = c * CHUNK_ROWS;
            let rows = (frames - t0).min(CHUNK_ROWS);
            (old_pass1(codes, bars, reference, t0, rows, &ce, &co), checksum(&codes[t0 * bars..(t0 + rows) * bars]))
        });
        let mut hist = vec![0u64; NCTX * NSYM];
        for (r, _) in &parts {
            for (h, v) in hist.iter_mut().zip(&r.hist) {
                *h += v;
            }
        }
        let freqs = Tables::freqs_for(&hist, PB);
        let tables = Tables::from_freqs(&freqs, PB).expect("frequencies built to sum");
        let rans = par(n, |c| rans_encode(&parts[c].0.syms, &tables));

        let mut out = Vec::with_capacity(HEAD + rans.iter().map(|r| r.len() * 2).sum::<usize>() + (1 << 16));
        out.extend_from_slice(&(CHUNK_ROWS as u32).to_le_bytes());
        out.extend_from_slice(&[PB as u8, 0, 0, 0]);
        for (coefs, n) in [(&ce, bars.div_ceil(2)), (&co, bars / 2)] {
            for k in 0..n {
                for j in 0..TAPS {
                    out.extend_from_slice(&(coefs[j * n + k] as i16).to_le_bytes());
                }
                out.extend_from_slice(&coefs[TAPS * n + k].to_le_bytes());
            }
        }
        for f in &freqs {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for (c, (p, sum)) in parts.iter().enumerate() {
            out.extend_from_slice(&(rans[c].len() as u32).to_le_bytes());
            out.extend_from_slice(&(p.raw.len() as u32).to_le_bytes());
            out.extend_from_slice(&sum.to_le_bytes());
        }
        for (c, (p, _)) in parts.iter().enumerate() {
            for w in &rans[c] {
                out.extend_from_slice(&w.to_le_bytes());
            }
            for w in &p.raw {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn new_encoder_matches_unchanged_de4fcb3_encoder_bytes_and_cross_decodes() {
        for (frames, bars) in [
            (1, 1),
            (3, 5),
            (40, 16),
            (300, 128),
            (513, 257),
            (600, 1024),
        ] {
            let a = plane(frames, bars, 11);
            let old_a = old_encode(&a, frames, bars, None);
            let new_a = encode(&a, frames, bars, None);
            assert_eq!(new_a, old_a, "Mix bytes differ at {frames}x{bars}");
            assert_eq!(decode(&old_a, frames, bars, None).as_deref(), Ok(&a[..]));

            let b: Vec<u16> = a.iter().enumerate().map(|(i, &c)| (c as i32 + (i % 9) as i32 - 4).clamp(0, 4095) as u16).collect();
            let old_b = old_encode(&b, frames, bars, Some(&a));
            let new_b = encode(&b, frames, bars, Some(&a));
            assert_eq!(new_b, old_b, "Ref bytes differ at {frames}x{bars}");
            assert_eq!(decode(&old_b, frames, bars, Some(&a)).as_deref(), Ok(&b[..]));
        }

        // Also test on a 1024x1024 slice of song1's real HP planes if present on disk
        let refs_dir = std::path::Path::new(r"C:\Users\PC\Programming\moosik-v160-home\notes-162-analysis\refs_r8t_N");
        if refs_dir.join("G_mix.u16").exists() {
            let read_slice = |name: &str, rows: usize| -> Vec<u16> {
                use std::io::Read;
                let mut f = std::fs::File::open(refs_dir.join(format!("G_{name}.u16"))).unwrap();
                let mut buf = vec![0u8; rows * 1024 * 2];
                f.read_exact(&mut buf).unwrap();
                buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
            };
            let rows = 1024usize;
            let mix = read_slice("mix", rows);
            let left = read_slice("left", rows);
            let right = read_slice("right", rows);

            let old_m = old_encode(&mix, rows, 1024, None);
            let new_m = encode(&mix, rows, 1024, None);
            assert_eq!(new_m, old_m, "song1 Mix 1024x1024 bytes differ!");
            assert_eq!(decode(&old_m, rows, 1024, None).as_deref(), Ok(&mix[..]));

            let old_l = old_encode(&left, rows, 1024, Some(&mix));
            let new_l = encode(&left, rows, 1024, Some(&mix));
            assert_eq!(new_l, old_l, "song1 Left 1024x1024 bytes differ!");
            assert_eq!(decode(&old_l, rows, 1024, Some(&mix)).as_deref(), Ok(&left[..]));

            let old_r = old_encode(&right, rows, 1024, Some(&mix));
            let new_r = encode(&right, rows, 1024, Some(&mix));
            assert_eq!(new_r, old_r, "song1 Right 1024x1024 bytes differ!");
            assert_eq!(decode(&old_r, rows, 1024, Some(&mix)).as_deref(), Ok(&right[..]));
        }
    }

    #[test]
    #[ignore = "full song1 38896x1024 real HP planes golden byte-equality and same-binary A/B benchmark"]
    fn song1_full_real_hp_planes_byte_identical_and_ab_bench() {
        use std::hint::black_box;
        use std::time::Instant;

        let _iso = crate::spectrum::gpu_calib::isolated_test_env("v5_song1_bench");
        crate::spectrum::gpu_calib::set_workers(6);

        let refs_dir = std::path::Path::new(r"C:\Users\PC\Programming\moosik-v160-home\notes-162-analysis\refs_r8t_N");
        let load_plane = |name: &str| -> Vec<u16> {
            let b = std::fs::read(refs_dir.join(format!("G_{name}.u16"))).expect("read G_*.u16");
            b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let mix = load_plane("mix");
        let left = load_plane("left");
        let right = load_plane("right");
        let bars = 1024usize;
        let frames = mix.len() / bars;
        assert_eq!(mix.len(), frames * bars);
        assert_eq!(left.len(), frames * bars);
        assert_eq!(right.len(), frames * bars);

        // 1. Golden byte-for-byte equality on full song1 Mix, Left, Right planes
        let old_m = old_encode(&mix, frames, bars, None);
        let new_m = encode(&mix, frames, bars, None);
        assert_eq!(new_m, old_m, "Full song1 Mix encoded bytes differ!");
        assert_eq!(decode(&old_m, frames, bars, None).as_deref(), Ok(&mix[..]));

        let old_l = old_encode(&left, frames, bars, Some(&mix));
        let new_l = encode(&left, frames, bars, Some(&mix));
        assert_eq!(new_l, old_l, "Full song1 Left encoded bytes differ!");
        assert_eq!(decode(&old_l, frames, bars, Some(&mix)).as_deref(), Ok(&left[..]));

        let old_r = old_encode(&right, frames, bars, Some(&mix));
        let new_r = encode(&right, frames, bars, Some(&mix));
        assert_eq!(new_r, old_r, "Full song1 Right encoded bytes differ!");
        assert_eq!(decode(&old_r, frames, bars, Some(&mix)).as_deref(), Ok(&right[..]));

        println!(
            "\n=== song1 real HP planes ({frames}x{bars}) GOLDEN BYTE-EQUALITY VERIFIED ==="
        );
        println!(
            "  Mix={} bytes ({:.2} MiB), Left={} bytes ({:.2} MiB), Right={} bytes ({:.2} MiB), Total={:.2} MiB",
            new_m.len(), new_m.len() as f64 / 1048576.0,
            new_l.len(), new_l.len() as f64 / 1048576.0,
            new_r.len(), new_r.len() as f64 / 1048576.0,
            (new_m.len() + new_l.len() + new_r.len()) as f64 / 1048576.0
        );

        // 2. Same-binary interleaved A/B timing (5 alternating pairs) for encoding all 3 planes
        //    and for full publication (encode + verify-decode of Mix + Left + Right)
        let run_old_enc3 = || -> f64 {
            let t0 = Instant::now();
            let bm = black_box(old_encode(black_box(&mix), frames, bars, None));
            let bl = black_box(old_encode(black_box(&left), frames, bars, Some(&mix)));
            let br = black_box(old_encode(black_box(&right), frames, bars, Some(&mix)));
            assert_eq!(bm.len() + bl.len() + br.len(), old_m.len() + old_l.len() + old_r.len());
            t0.elapsed().as_secs_f64() * 1e3
        };

        let run_new_enc3 = || -> f64 {
            let t0 = Instant::now();
            let bm = black_box(encode(black_box(&mix), frames, bars, None));
            let bl = black_box(encode(black_box(&left), frames, bars, Some(&mix)));
            let br = black_box(encode(black_box(&right), frames, bars, Some(&mix)));
            assert_eq!(bm.len() + bl.len() + br.len(), new_m.len() + new_l.len() + new_r.len());
            t0.elapsed().as_secs_f64() * 1e3
        };

        let mut old_enc_ms = Vec::new();
        let mut new_enc_ms = Vec::new();
        for pair in 0..5 {
            let (o, n) = if pair % 2 == 0 {
                (run_old_enc3(), run_new_enc3())
            } else {
                let n = run_new_enc3();
                let o = run_old_enc3();
                (o, n)
            };
            println!("  encode 3 planes pair {}: old={o:.1}ms, new={n:.1}ms", pair + 1);
            old_enc_ms.push(o);
            new_enc_ms.push(n);
        }
        old_enc_ms.sort_by(f64::total_cmp);
        new_enc_ms.sort_by(f64::total_cmp);
        let med_old_e = old_enc_ms[old_enc_ms.len() / 2];
        let med_new_e = new_enc_ms[new_enc_ms.len() / 2];

        // Also measure decode of the 3 planes (safe decoder)
        let t_dec = Instant::now();
        let dm = decode(black_box(&new_m), frames, bars, None).unwrap();
        let dl = decode(black_box(&new_l), frames, bars, Some(&dm)).unwrap();
        let dr = decode(black_box(&new_r), frames, bars, Some(&dm)).unwrap();
        let dec3_ms = t_dec.elapsed().as_secs_f64() * 1e3;
        assert_eq!(dl.len() + dr.len(), 2 * frames * bars);

        println!(
            "  MEDIAN encode 3 planes (Mix+L+R): old={med_old_e:.1}ms -> new={med_new_e:.1}ms ({:.2}x, saves {:.1}ms / -{:.1}%)",
            med_old_e / med_new_e, med_old_e - med_new_e, (1.0 - med_new_e / med_old_e) * 100.0
        );
        println!(
            "  Safe decode 3 planes (Mix+L+R): {dec3_ms:.1}ms | Total to_v5_with (enc+verify-dec): old={:.2}s -> new={:.2}s (saves {:.2}s)",
            (med_old_e + dec3_ms) * 1e-3, (med_new_e + dec3_ms) * 1e-3, (med_old_e - med_new_e) * 1e-3
        );
    }
}
