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

/// Output `ix` of `column_butterfly16_loadfn!`, by its original operations.
#[inline(always)]
unsafe fn bf16_one<V: AvxVector>(
    load: impl Fn(usize) -> V,
    tw16: [V; 2],
    rot: Rotation90<V>,
    ix: usize,
) -> V {
    let (i, q) = (ix % 4, ix / 4);
    let mut mids = [V::zero(); 4];
    for (m, mid) in mids.iter_mut().enumerate() {
        let v = bf4_one([load(m), load(m + 4), load(m + 8), load(m + 12)], rot, i);
        *mid = match (m, i) {
            (_, 0) | (0, _) => v,
            (1, 1) => V::mul_complex(v, tw16[0]),
            (1, 2) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (1, _) => V::mul_complex(v, tw16[1]),
            (2, 1) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (2, 2) => v.rotate90(rot),
            (2, _) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (3, 1) => V::mul_complex(v, tw16[1]),
            (3, 2) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (_, _) => V::mul_complex(v, tw16[0].neg()),
        };
    }
    bf4_one(mids, rot, q)
}

/// Output `ix` of `column_butterfly32_loadfn!`, by its original operations.
#[inline(always)]
unsafe fn bf32_one<V: AvxVector>(
    load: impl Fn(usize) -> V,
    tw: [V; 6],
    rot: Rotation90<V>,
    ix: usize,
) -> V {
    let (i, j) = (ix % 4, ix / 4);
    let mut mids = [V::zero(); 8];
    for (m, mid) in mids.iter_mut().enumerate() {
        let v = bf4_one([load(m), load(m + 8), load(m + 16), load(m + 24)], rot, i);
        *mid = match (m, i) {
            (_, 0) | (0, _) => v,
            (1, 1) => V::mul_complex(v, tw[0]),
            (1, 2) => V::mul_complex(v, tw[1]),
            (1, _) => V::mul_complex(v, tw[2]),
            (2, 1) => V::mul_complex(v, tw[1]),
            (2, 2) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (2, _) => V::mul_complex(v, tw[4]),
            (3, 1) => V::mul_complex(v, tw[2]),
            (3, 2) => V::mul_complex(v, tw[4]),
            (3, _) => V::mul_complex(v, tw[0].rotate90(rot)),
            (4, 1) => avx_vector::apply_butterfly8_twiddle1(v, rot),
            (4, 2) => v.rotate90(rot),
            (4, _) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (5, 1) => V::mul_complex(v, tw[3]),
            (5, 2) => V::mul_complex(v, tw[1].rotate90(rot)),
            (5, _) => V::mul_complex(v, tw[5].rotate90(rot)),
            (6, 1) => V::mul_complex(v, tw[4]),
            (6, 2) => avx_vector::apply_butterfly8_twiddle3(v, rot),
            (6, _) => V::mul_complex(v, tw[1].neg()),
            (7, 1) => V::mul_complex(v, tw[5]),
            (7, 2) => V::mul_complex(v, tw[4].rotate90(rot)),
            (_, _) => V::mul_complex(v, tw[3].neg()),
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
        _: impl AvxArray<f64>,
        _: &mut [Complex<f64>],
        _: &[usize],
        _: &mut [Complex<f64>],
    ) -> bool {
        false
    }
}

impl<T> Butterfly128Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        true
    }

    pub(super) fn pruned_leaf(
        &self,
        input: impl AvxArray<f64>,
        _scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) -> bool {
        if needed.is_empty() || needed.len() > MAX_NEEDED {
            return false;
        }
        unsafe { self.pruned128(input, needed, out) };
        true
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned128(
        &self,
        input: impl AvxArray<f64>,
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) {
        use std::mem::MaybeUninit;

        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();
        let tw16_128 = [
            self.twiddles_butterfly16[0].lo(),
            self.twiddles_butterfly16[1].lo(),
        ];

        let mut row_slot = [0u8; 8];
        let mut row_list = [0u8; MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut mask = 0u8;
        for &o in needed {
            let r = o % 8;
            if (mask >> r) & 1 == 0 {
                mask |= 1 << r;
                row_slot[r] = num_rows as u8;
                row_list[num_rows] = r as u8;
                num_rows += 1;
            }
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f64>; 16]>::uninit(); MAX_NEEDED];
        if num_rows == 1 {
            let r = row_list[0] as usize;
            let dst = row_bufs[0].as_mut_ptr() as *mut f64;
            for columnset in 0..8 {
                let rows = [
                    input.load_complex(columnset * 2),
                    input.load_complex(columnset * 2 + 16),
                    input.load_complex(columnset * 2 + 32),
                    input.load_complex(columnset * 2 + 48),
                    input.load_complex(columnset * 2 + 64),
                    input.load_complex(columnset * 2 + 80),
                    input.load_complex(columnset * 2 + 96),
                    input.load_complex(columnset * 2 + 112),
                ];
                let mut v: __m256d = bf8_one(rows, rot, r);
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                _mm256_storeu_pd(dst.add(columnset * 4), v);
            }
        } else {
            for columnset in 0..8 {
                let rows = [
                    input.load_complex(columnset * 2),
                    input.load_complex(columnset * 2 + 16),
                    input.load_complex(columnset * 2 + 32),
                    input.load_complex(columnset * 2 + 48),
                    input.load_complex(columnset * 2 + 64),
                    input.load_complex(columnset * 2 + 80),
                    input.load_complex(columnset * 2 + 96),
                    input.load_complex(columnset * 2 + 112),
                ];
                let mid = AvxVector::column_butterfly8(rows, rot);
                for s in 0..num_rows {
                    let r = row_list[s] as usize;
                    let mut v = mid[r];
                    if r >= 1 {
                        v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                    }
                    let dst = row_bufs[s].as_mut_ptr() as *mut f64;
                    _mm256_storeu_pd(dst.add(columnset * 4), v);
                }
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o % 8] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f64;
            let v: __m128d = bf16_one(
                |index| _mm_loadu_pd(row_ptr.add(index * 2)),
                tw16_128,
                rot128,
                o / 8,
            );
            _mm_storeu_pd(slot as *mut Complex<f64> as *mut f64, v);
        }
    }
}

impl<T> Butterfly256Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        true
    }

    pub(super) fn pruned_leaf(
        &self,
        input: impl AvxArray<f64>,
        _scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) -> bool {
        if needed.is_empty() || needed.len() > MAX_NEEDED {
            return false;
        }
        // Instances are available only after the parent's AVX/FMA admission.
        unsafe { self.pruned256(input, needed, out) };
        true
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned256(
        &self,
        input: impl AvxArray<f64>,
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) {
        use std::mem::MaybeUninit;

        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();
        let tw32_128 = [
            self.twiddles_butterfly32[0].lo(),
            self.twiddles_butterfly32[1].lo(),
            self.twiddles_butterfly32[2].lo(),
            self.twiddles_butterfly32[3].lo(),
            self.twiddles_butterfly32[4].lo(),
            self.twiddles_butterfly32[5].lo(),
        ];

        let mut row_slot = [0u8; 8];
        let mut row_list = [0u8; MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut mask = 0u8;
        for &o in needed {
            let r = o % 8;
            if (mask >> r) & 1 == 0 {
                mask |= 1 << r;
                row_slot[r] = num_rows as u8;
                row_list[num_rows] = r as u8;
                num_rows += 1;
            }
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f64>; 32]>::uninit(); MAX_NEEDED];
        if num_rows == 1 {
            let r = row_list[0] as usize;
            let dst = row_bufs[0].as_mut_ptr() as *mut f64;
            for columnset in 0..16 {
                let rows = [
                    input.load_complex(columnset * 2),
                    input.load_complex(columnset * 2 + 32),
                    input.load_complex(columnset * 2 + 64),
                    input.load_complex(columnset * 2 + 96),
                    input.load_complex(columnset * 2 + 128),
                    input.load_complex(columnset * 2 + 160),
                    input.load_complex(columnset * 2 + 192),
                    input.load_complex(columnset * 2 + 224),
                ];
                let mut v: __m256d = bf8_one(rows, rot, r);
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                _mm256_storeu_pd(dst.add(columnset * 4), v);
            }
        } else {
            for columnset in 0..16 {
                let rows = [
                    input.load_complex(columnset * 2),
                    input.load_complex(columnset * 2 + 32),
                    input.load_complex(columnset * 2 + 64),
                    input.load_complex(columnset * 2 + 96),
                    input.load_complex(columnset * 2 + 128),
                    input.load_complex(columnset * 2 + 160),
                    input.load_complex(columnset * 2 + 192),
                    input.load_complex(columnset * 2 + 224),
                ];
                let mid = AvxVector::column_butterfly8(rows, rot);
                for s in 0..num_rows {
                    let r = row_list[s] as usize;
                    let mut v = mid[r];
                    if r >= 1 {
                        v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                    }
                    let dst = row_bufs[s].as_mut_ptr() as *mut f64;
                    _mm256_storeu_pd(dst.add(columnset * 4), v);
                }
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o % 8] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f64;
            let v: __m128d = bf32_one(
                |index| _mm_loadu_pd(row_ptr.add(index * 2)),
                tw32_128,
                rot128,
                o / 8,
            );
            _mm_storeu_pd(slot as *mut Complex<f64> as *mut f64, v);
        }
    }
}

impl<T> Butterfly512Avx64<T> {
    pub(super) fn leaf_prunes(&self) -> bool {
        true
    }

    pub(super) fn pruned_leaf(
        &self,
        input: impl AvxArray<f64>,
        _scratch: &mut [Complex<f64>],
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) -> bool {
        if needed.is_empty() || needed.len() > MAX_NEEDED {
            return false;
        }
        unsafe { self.pruned512(input, needed, out) };
        true
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned512(
        &self,
        input: impl AvxArray<f64>,
        needed: &[usize],
        out: &mut [Complex<f64>],
    ) {
        use std::mem::MaybeUninit;

        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();
        let tw16 = self.twiddles_butterfly16;
        let tw32_128 = [
            self.twiddles_butterfly32[0].lo(),
            self.twiddles_butterfly32[1].lo(),
            self.twiddles_butterfly32[2].lo(),
            self.twiddles_butterfly32[3].lo(),
            self.twiddles_butterfly32[4].lo(),
            self.twiddles_butterfly32[5].lo(),
        ];

        let mut row_slot = [0u8; 16];
        let mut row_list = [0u8; MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut k_mask = 0u16;
        let mut i_mask = 0u8;
        for &o in needed {
            let k = o % 16;
            if (k_mask >> k) & 1 == 0 {
                k_mask |= 1 << k;
                i_mask |= 1 << (k % 4);
                row_slot[k] = num_rows as u8;
                row_list[num_rows] = k as u8;
                num_rows += 1;
            }
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f64>; 32]>::uninit(); MAX_NEEDED];
        let twiddle_mid = |v: __m256d, m: usize, i: usize| -> __m256d {
            match (m, i) {
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
            }
        };

        if i_mask.count_ones() == 1 {
            let i = i_mask.trailing_zeros() as usize;
            for columnset in 0..16 {
                let load = |index| -> __m256d { input.load_complex(columnset * 2 + 32 * index) };
                let mut mids = [AvxVector::zero(); 4];
                for m in 0..4 {
                    let v = bf4_one([load(m), load(m + 4), load(m + 8), load(m + 12)], rot, i);
                    mids[m] = twiddle_mid(v, m, i);
                }
                for s in 0..num_rows {
                    let k = row_list[s] as usize;
                    let mut v = bf4_one(mids, rot, k / 4);
                    if k >= 1 {
                        v = AvxVector::mul_complex(v, self.twiddles[15 * columnset + k - 1]);
                    }
                    let dst = row_bufs[s].as_mut_ptr() as *mut f64;
                    _mm256_storeu_pd(dst.add(columnset * 4), v);
                }
            }
        } else {
            for columnset in 0..16 {
                let load = |index| -> __m256d { input.load_complex(columnset * 2 + 32 * index) };
                let mut bf4_all = [[AvxVector::zero(); 4]; 4];
                for m in 0..4 {
                    bf4_all[m] = AvxVector::column_butterfly4(
                        [load(m), load(m + 4), load(m + 8), load(m + 12)],
                        rot,
                    );
                }
                for s in 0..num_rows {
                    let k = row_list[s] as usize;
                    let (i, q) = (k % 4, k / 4);
                    let mids = [
                        bf4_all[0][i],
                        twiddle_mid(bf4_all[1][i], 1, i),
                        twiddle_mid(bf4_all[2][i], 2, i),
                        twiddle_mid(bf4_all[3][i], 3, i),
                    ];
                    let mut v = bf4_one(mids, rot, q);
                    if k >= 1 {
                        v = AvxVector::mul_complex(v, self.twiddles[15 * columnset + k - 1]);
                    }
                    let dst = row_bufs[s].as_mut_ptr() as *mut f64;
                    _mm256_storeu_pd(dst.add(columnset * 4), v);
                }
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o % 16] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f64;
            let v: __m128d = bf32_one(
                |index| _mm_loadu_pd(row_ptr.add(index * 2)),
                tw32_128,
                rot128,
                o / 16,
            );
            _mm_storeu_pd(slot as *mut Complex<f64> as *mut f64, v);
        }
    }
}

#[cfg(test)]
#[path = "avx64_pruned_tests.rs"]
mod tests;
