//! Moosik 1.5.5: exact requested-output f64 Butterfly256/512 leaves.
//!
//! This evaluates dependencies in the unmodified parent's full transforms,
//! with their original vector primitives, constants, operand order and FMA
//! use. It is not a scalar DFT or a different decomposition. f64 vectors hold
//! two complex lanes (the existing f32 counterpart holds four); only lane and
//! transpose addresses change. Unrequested lanes may hold stale scratch, but
//! no primitive crosses complex lanes. Floating-point exception flags and NaN
//! payload/sign are outside the existing `Fft` pruning equality contract.

use super::*;

const MAX_NEEDED: usize = 4;

/// One returned output of the original `[o0,o2,o1,o3]` butterfly-4.
#[inline(always)]
unsafe fn bf4_one<V: AvxVector>(rows: [V; 4], rotation: Rotation90<V>, p: usize) -> V {
    match p {
        0 => V::add(V::add(rows[0], rows[2]), V::add(rows[1], rows[3])),
        1 => V::add(
            V::sub(rows[0], rows[2]),
            V::sub(rows[1], rows[3]).rotate90(rotation),
        ),
        2 => V::sub(V::add(rows[0], rows[2]), V::add(rows[1], rows[3])),
        _ => V::sub(
            V::sub(rows[0], rows[2]),
            V::sub(rows[1], rows[3]).rotate90(rotation),
        ),
    }
}

/// One returned output of the original `[o0,o2,o4,o6,o1,o3,o5,o7]` butterfly-8.
#[inline(always)]
unsafe fn bf8_one<V: AvxVector>(rows: [V; 8], rotation: Rotation90<V>, p: usize) -> V {
    let i = p % 4;
    let a = bf4_one([rows[0], rows[2], rows[4], rows[6]], rotation, i);
    let mut b = bf4_one([rows[1], rows[3], rows[5], rows[7]], rotation, i);
    b = match i {
        1 => avx_vector::apply_butterfly8_twiddle1(b, rotation),
        2 => b.rotate90(rotation),
        3 => avx_vector::apply_butterfly8_twiddle3(b, rotation),
        _ => b,
    };
    if p < 4 {
        V::add(a, b)
    } else {
        V::sub(a, b)
    }
}

/// Output `ix` of `column_butterfly32_loadfn!`, by its original operations.
#[inline(always)]
unsafe fn bf32_one(
    load: impl Fn(usize) -> __m256d,
    tw: [__m256d; 6],
    rot: Rotation90<__m256d>,
    ix: usize,
) -> __m256d {
    let (i, j) = (ix % 4, ix / 4);
    let mut mids = [AvxVector::zero(); 8];
    for (m, mid) in mids.iter_mut().enumerate() {
        let v: __m256d = bf4_one([load(m), load(m + 8), load(m + 16), load(m + 24)], rot, i);
        *mid = match (m, i) {
            (_, 0) | (0, _) => v,
            (1, 1) => AvxVector::mul_complex(v, tw[0]),
            (1, 2) => AvxVector::mul_complex(v, tw[1]),
            (1, _) => AvxVector::mul_complex(v, tw[2]),
            (2, 1) => AvxVector::mul_complex(v, tw[1]),
            (2, 2) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (2, _) => AvxVector::mul_complex(v, tw[4]),
            (3, 1) => AvxVector::mul_complex(v, tw[2]),
            (3, 2) => AvxVector::mul_complex(v, tw[4]),
            (3, _) => AvxVector::mul_complex(v, tw[0].rotate90(rot)),
            (4, 1) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (4, 2) => v.rotate90(rot),
            (4, _) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (5, 1) => AvxVector::mul_complex(v, tw[3]),
            (5, 2) => AvxVector::mul_complex(v, tw[1].rotate90(rot)),
            (5, _) => AvxVector::mul_complex(v, tw[5].rotate90(rot)),
            (6, 1) => AvxVector::mul_complex(v, tw[4]),
            (6, 2) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (6, _) => AvxVector::mul_complex(v, tw[1].neg()),
            (7, 1) => AvxVector::mul_complex(v, tw[5]),
            (7, 2) => AvxVector::mul_complex(v, tw[4].rotate90(rot)),
            (_, _) => AvxVector::mul_complex(v, tw[3].neg()),
        };
    }
    bf8_one(mids, rot, j)
}

impl<T> Butterfly64Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        false
    }
    pub(super) fn pruned_leaf(
        &self,
        _: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
        _: &[usize],
        _: &mut [Complex<f64>],
    ) -> bool {
        false
    }
}

impl<T> Butterfly128Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        false
    }

    pub(super) fn pruned_leaf(
        &self,
        _: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
        _: &[usize],
        _: &mut [Complex<f64>],
    ) -> bool {
        false
    }
}

impl<T> Butterfly256Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        true
    }

    pub(super) fn pruned_leaf(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) -> bool {
        if needed.is_empty() || needed.len() > MAX_NEEDED {
            return false;
        }
        // Instances are available only after the parent's AVX/FMA admission.
        unsafe { self.pruned256(buffer, scratch, needed, out) };
        true
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned256(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) {
        let rot = self.twiddles_butterfly4;
        let input: &[Complex<f64>] = buffer;
        let mut formed = 0u8;
        // Full phase 1 is a butterfly-8 down a 32x8 array. Its packed
        // transpose puts (row r,column c) at scratch[c*8+r].
        for &o in needed {
            let r = o % 8;
            if formed >> r & 1 == 1 {
                continue;
            }
            formed |= 1 << r;
            for columnset in 0..16 {
                let mut rows = [AvxVector::zero(); 8];
                for (q, row) in rows.iter_mut().enumerate() {
                    *row = input.load_complex(columnset * 2 + 32 * q);
                }
                let mut v: __m256d = bf8_one(rows, rot, r);
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                let mut lanes = [Complex { re: 0.0, im: 0.0 }; 2];
                _mm256_storeu_pd(lanes.as_mut_ptr() as *mut f64, v);
                for (lane, value) in lanes.iter().enumerate() {
                    scratch[(columnset * 2 + lane) * 8 + r] = *value;
                }
            }
        }
        // Full phase 2 is column_butterfly32_loadfn!: output o selects
        // row o%8 (vector columnset r/2,lane r%2), butterfly output o/8.
        let s: &[Complex<f64>] = scratch;
        for (slot, &o) in out.iter_mut().zip(needed) {
            let r = o % 8;
            let v = bf32_one(
                |index| s.load_complex((r / 2) * 2 + index * 8),
                self.twiddles_butterfly32,
                rot,
                o / 8,
            );
            let mut lanes = [Complex { re: 0.0, im: 0.0 }; 2];
            _mm256_storeu_pd(lanes.as_mut_ptr() as *mut f64, v);
            *slot = lanes[r % 2];
        }
    }
}

impl<T> Butterfly512Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        true
    }

    pub(super) fn pruned_leaf(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) -> bool {
        if needed.is_empty() || needed.len() > MAX_NEEDED {
            return false;
        }
        unsafe { self.pruned512(buffer, scratch, needed, out) };
        true
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned512(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) {
        let rot = self.twiddles_butterfly4;
        let tw16 = self.twiddles_butterfly16;
        let input: &[Complex<f64>] = buffer;
        let mut formed = 0u16;
        // Full phase 1 is column_butterfly16_loadfn! over the 32x16 array.
        // Row k is output q=k/4 of bf4 over twiddled bf4 outputs i=k%4.
        for &o in needed {
            let k = o % 16;
            if formed >> k & 1 == 1 {
                continue;
            }
            formed |= 1 << k;
            let (i, q) = (k % 4, k / 4);
            for columnset in 0..16 {
                let load = |index| -> __m256d { input.load_complex(columnset * 2 + 32 * index) };
                let mut mids = [AvxVector::zero(); 4];
                for (m, mid) in mids.iter_mut().enumerate() {
                    let v: __m256d =
                        bf4_one([load(m), load(m + 4), load(m + 8), load(m + 12)], rot, i);
                    *mid = match (m, i) {
                        (_, 0) | (0, _) => v,
                        (1, 1) => AvxVector::mul_complex(v, tw16[0]),
                        (1, 2) => avx_vector::apply_butterfly8_twiddle1(v, rot),
                        (1, _) => AvxVector::mul_complex(v, tw16[1]),
                        (2, 1) => avx_vector::apply_butterfly8_twiddle1(v, rot),
                        (2, 2) => v.rotate90(rot),
                        (2, _) => avx_vector::apply_butterfly8_twiddle3(v, rot),
                        (3, 1) => AvxVector::mul_complex(v, tw16[1]),
                        (3, 2) => avx_vector::apply_butterfly8_twiddle3(v, rot),
                        (_, _) => AvxVector::mul_complex(v, tw16[0].neg()),
                    };
                }
                let mut v: __m256d = bf4_one(mids, rot, q);
                if k >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[15 * columnset + k - 1]);
                }
                let mut lanes = [Complex { re: 0.0, im: 0.0 }; 2];
                _mm256_storeu_pd(lanes.as_mut_ptr() as *mut f64, v);
                for (lane, value) in lanes.iter().enumerate() {
                    scratch[(columnset * 2 + lane) * 16 + k] = *value;
                }
            }
        }
        let s: &[Complex<f64>] = scratch;
        for (slot, &o) in out.iter_mut().zip(needed) {
            let k = o % 16;
            let v = bf32_one(
                |index| s.load_complex((k / 2) * 2 + index * 16),
                self.twiddles_butterfly32,
                rot,
                o / 16,
            );
            let mut lanes = [Complex { re: 0.0, im: 0.0 }; 2];
            _mm256_storeu_pd(lanes.as_mut_ptr() as *mut f64, v);
            *slot = lanes[k % 2];
        }
    }
}

#[cfg(test)]
#[path = "avx64_pruned_tests.rs"]
mod tests;
