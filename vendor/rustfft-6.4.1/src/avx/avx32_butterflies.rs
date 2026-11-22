use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_complex::Complex;

use crate::array_utils::DoubleBuf;
use crate::{common::FftNum, twiddles};
use crate::{Direction, Fft, FftDirection, Length};

use super::avx32_utils;
use super::avx_vector::{self, AvxArray};
use super::avx_vector::{AvxArrayMut, AvxVector, AvxVector128, AvxVector256, Rotation90};

// Safety: This macro will call `self::perform_fft_f32()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
macro_rules! boilerplate_fft_simd_butterfly {
    ($struct_name:ident, $len:expr) => {
        impl $struct_name<f32> {
            #[inline]
            pub fn is_supported_by_cpu() -> bool {
                is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")
            }
            #[inline]
            pub fn new(direction: FftDirection) -> Result<Self, ()> {
                if Self::is_supported_by_cpu() {
                    // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
                    Ok(unsafe { Self::new_with_avx(direction) })
                } else {
                    Err(())
                }
            }
        }

        impl<T: FftNum> Fft<T> for $struct_name<f32> {
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    super::avx_fft_helper_immut(
                        simd_input,
                        simd_output,
                        &mut [],
                        self.len(),
                        0,
                        |input, output, _| self.perform_fft_f32(DoubleBuf { input, output }),
                    );
                }
            }
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute_mut(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    super::avx_fft_helper_outofplace(
                        simd_input,
                        simd_output,
                        &mut [],
                        self.len(),
                        0,
                        |input, output, _| self.perform_fft_f32(DoubleBuf { input, output }),
                    );
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
                unsafe {
                    let simd_buffer = crate::array_utils::workaround_transmute_mut(buffer);
                    super::avx_fft_helper_inplace(
                        simd_buffer,
                        &mut [],
                        self.len(),
                        0,
                        |chunk, _| self.perform_fft_f32(chunk),
                    )
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}

// Safety: This macro will call `self::column_butterflies_and_transpose and self::row_butterflies()` which probably has a #[target_feature(enable = "...")] annotation on it.
// Calling functions with that annotation is unsafe, because it doesn't actually check if the CPU has the required features.
// Callers of this macro must guarantee that users can't even obtain an instance of $struct_name if their CPU doesn't have the required CPU features.
macro_rules! boilerplate_fft_simd_butterfly_with_scratch {
    ($struct_name:ident, $len:expr) => {
        impl $struct_name<f32> {
            #[inline]
            pub fn new(direction: FftDirection) -> Result<Self, ()> {
                let has_avx = is_x86_feature_detected!("avx");
                let has_fma = is_x86_feature_detected!("fma");
                if has_avx && has_fma {
                    // Safety: new_internal requires the "avx" feature set. Since we know it's present, we're safe
                    Ok(unsafe { Self::new_with_avx(direction) })
                } else {
                    Err(())
                }
            }
        }
        impl<T: FftNum> $struct_name<T> {
            #[inline]
            fn perform_fft_inplace(
                &self,
                buffer: &mut [Complex<f32>],
                scratch: &mut [Complex<f32>],
            ) {
                // Perform the column FFTs
                // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.column_butterflies_and_transpose(buffer, scratch) };

                // process the row FFTs, and copy from the scratch back to the buffer as we go
                // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
                unsafe {
                    self.row_butterflies(DoubleBuf {
                        input: scratch,
                        output: buffer,
                    })
                };
            }

            #[inline]
            fn perform_fft_immut(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
                // Perform the column FFTs
                // Safety: self.perform_column_butterflies() requres the "avx" and "fma" instruction sets, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.column_butterflies_and_transpose(input, output) };

                // process the row FFTs in-place in the output buffer
                // Safety: self.transpose() requres the "avx" instruction set, and we return Err() in our constructor if the instructions aren't available
                unsafe { self.row_butterflies(output) };
            }
        }
        impl<T: FftNum> Fft<T> for $struct_name<f32> {
            fn process_immutable_with_scratch(
                &self,
                input: &[Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    super::avx_fft_helper_immut(
                        simd_input,
                        simd_output,
                        &mut [],
                        self.len(),
                        0,
                        |input, output, _| self.perform_fft_immut(input, output),
                    );
                }
            }
            fn process_outofplace_with_scratch(
                &self,
                input: &mut [Complex<T>],
                output: &mut [Complex<T>],
                _scratch: &mut [Complex<T>],
            ) {
                unsafe {
                    let simd_input = crate::array_utils::workaround_transmute_mut(input);
                    let simd_output = crate::array_utils::workaround_transmute_mut(output);
                    super::avx_fft_helper_outofplace(
                        simd_input,
                        simd_output,
                        &mut [],
                        self.len(),
                        0,
                        |input, output, _| self.perform_fft_immut(input, output),
                    );
                }
            }
            fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
                unsafe {
                    let simd_buffer = crate::array_utils::workaround_transmute_mut(buffer);
                    let simd_scratch = crate::array_utils::workaround_transmute_mut(scratch);
                    super::avx_fft_helper_inplace(
                        simd_buffer,
                        simd_scratch,
                        self.len(),
                        self.len(),
                        |chunk, scratch| self.perform_fft_inplace(chunk, scratch),
                    )
                }
            }
            /// Moosik (1.5.3 round 13 K0): whether `process_pruned_with_scratch`
            /// skips work for some requests — true for the 256 and 512 leaves,
            /// false for 128 (always the full leaf and gather). Callers such as
            /// the analysis's `InvWork::outputs` take the pruned entry only when
            /// this is true.
            fn prunes(&self) -> bool {
                self.leaf_prunes()
            }
            /// Moosik (1.5.3 round 12-E): the requested-output leaf for 1..=4
            /// requested outputs where one exists (256, 512); otherwise the
            /// full leaf and gather. The contract is checked once, before
            /// any write, either way.
            fn process_pruned_with_scratch(
                &self,
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
                needed: &[usize],
                out: &mut [Complex<T>],
                work: crate::PrunedWork<'_, T>,
            ) {
                crate::check_pruned(self, buffer.len(), scratch.len(), needed, out.len(), &work);
                let done = unsafe {
                    let b: &[Complex<f32>] = crate::array_utils::workaround_transmute(buffer);
                    let s: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(scratch);
                    let o: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(out);
                    self.pruned_leaf(b, s, needed, o)
                };
                if !done {
                    // `pruned_by_full_transform` without its second check.
                    self.process_with_scratch(buffer, scratch);
                    for (o, &i) in out.iter_mut().zip(needed) {
                        *o = buffer[i];
                    }
                }
            }
            fn process_pruned_product_with_scratch(
                &self,
                s: &[Complex<T>],
                h: &[Complex<T>],
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
                needed: &[usize],
                out: &mut [Complex<T>],
                work: crate::PrunedWork<'_, T>,
            ) {
                crate::check_product(self.len(), Some(s.len()), h.len());
                crate::check_pruned(self, buffer.len(), scratch.len(), needed, out.len(), &work);
                let done = unsafe {
                    let ts: &[Complex<f32>] = crate::array_utils::workaround_transmute(s);
                    let th: &[Complex<f32>] = crate::array_utils::workaround_transmute(h);
                    let tb: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(buffer);
                    let sc: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(scratch);
                    let o: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(out);
                    self.pruned_leaf(
                        super::avx_vector::ProductBuf { s: ts, h: th, out: tb },
                        sc,
                        needed,
                        o,
                    )
                };
                if !done {
                    crate::product_into(s, h, buffer);
                    self.process_with_scratch(buffer, scratch);
                    for (o, &i) in out.iter_mut().zip(needed) {
                        *o = buffer[i];
                    }
                }
            }
            fn process_pruned_inplace_product_with_scratch(
                &self,
                h: &[Complex<T>],
                buffer: &mut [Complex<T>],
                scratch: &mut [Complex<T>],
                needed: &[usize],
                out: &mut [Complex<T>],
                work: crate::PrunedWork<'_, T>,
            ) {
                crate::check_product(self.len(), None, h.len());
                crate::check_pruned(self, buffer.len(), scratch.len(), needed, out.len(), &work);
                let done = unsafe {
                    let th: &[Complex<f32>] = crate::array_utils::workaround_transmute(h);
                    let tb: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(buffer);
                    let sc: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(scratch);
                    let o: &mut [Complex<f32>] = crate::array_utils::workaround_transmute_mut(out);
                    self.pruned_leaf(
                        super::avx_vector::InplaceProductBuf { h: th, out: tb },
                        sc,
                        needed,
                        o,
                    )
                };
                if !done {
                    crate::product_in_place(h, buffer);
                    self.process_with_scratch(buffer, scratch);
                    for (o, &i) in out.iter_mut().zip(needed) {
                        *o = buffer[i];
                    }
                }
            }
            #[inline(always)]
            fn get_inplace_scratch_len(&self) -> usize {
                $len
            }
            #[inline(always)]
            fn get_outofplace_scratch_len(&self) -> usize {
                0
            }
            #[inline(always)]
            fn get_immutable_scratch_len(&self) -> usize {
                0
            }
        }
        impl<T> Length for $struct_name<T> {
            #[inline(always)]
            fn len(&self) -> usize {
                $len
            }
        }
        impl<T> Direction for $struct_name<T> {
            #[inline(always)]
            fn fft_direction(&self) -> FftDirection {
                self.direction
            }
        }
    };
}

macro_rules! gen_butterfly_twiddles_interleaved_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 4 + $skip_cols;

            twiddles[index] = AvxVector::make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

macro_rules! gen_butterfly_twiddles_separated_columns {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxVector::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 4 + $skip_cols;

            twiddles[index] = AvxVector::make_mixedradix_twiddle_chunk(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

pub struct Butterfly5Avx<T> {
    twiddles: [__m128; 3],
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly5Avx, 5);
impl Butterfly5Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddle1 = twiddles::compute_twiddle(1, 5, direction);
        let twiddle2 = twiddles::compute_twiddle(2, 5, direction);
        Self {
            twiddles: [
                _mm_set_ps(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm_set_ps(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm_set_ps(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly5Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        let input0 = _mm_castpd_ps(_mm_load1_pd(buffer.input_ptr() as *const f64)); // load the first element of the input, and duplicate it into both complex number slots of input0
        let input12 = buffer.load_partial2_complex(1);
        let input34 = buffer.load_partial2_complex(3);

        // swap elements for inputs 3 and 4
        let input43 = AvxVector::reverse_complex_elements(input34);

        // do some prep work before we can start applying twiddle factors
        let [sum12, diff43] = AvxVector::column_butterfly2([input12, input43]);

        let rotation = AvxVector::make_rotation90(FftDirection::Inverse);
        let rotated43 = AvxVector::rotate90(diff43, rotation);

        let [mid14, mid23] = AvxVector::unpack_complex([sum12, rotated43]);

        // to compute the first output, compute the sum of all elements. mid14[0] and mid23[0] already have the sum of 1+4 and 2+3 respectively, so if we add them, we'll get the sum of all 4
        let sum1234 = AvxVector::add(mid14, mid23);
        let output0 = AvxVector::add(input0, sum1234);

        // apply twiddle factors
        let twiddled14_mid = AvxVector::mul(mid14, self.twiddles[0]);
        let twiddled23_mid = AvxVector::mul(mid14, self.twiddles[1]);
        let twiddled14 = AvxVector::fmadd(mid23, self.twiddles[1], twiddled14_mid);
        let twiddled23 = AvxVector::fmadd(mid23, self.twiddles[2], twiddled23_mid);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled43] = AvxVector::unpack_complex([twiddled14, twiddled23]);
        let [output12, output43] = AvxVector::column_butterfly2([twiddled12, twiddled43]);

        // swap the elements in output43 before writing them out, and add the first input to everything
        let final12 = AvxVector::add(input0, output12);
        let output34 = AvxVector::reverse_complex_elements(output43);
        let final34 = AvxVector::add(input0, output34);

        buffer.store_partial1_complex(output0, 0);
        buffer.store_partial2_complex(final12, 1);
        buffer.store_partial2_complex(final34, 3);
    }
}

pub struct Butterfly7Avx<T> {
    twiddles: [__m128; 5],
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly7Avx, 7);
impl Butterfly7Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddle1 = twiddles::compute_twiddle(1, 7, direction);
        let twiddle2 = twiddles::compute_twiddle(2, 7, direction);
        let twiddle3 = twiddles::compute_twiddle(3, 7, direction);
        Self {
            twiddles: [
                _mm_set_ps(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
                _mm_set_ps(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
                _mm_set_ps(twiddle3.im, twiddle3.im, twiddle3.re, twiddle3.re),
                _mm_set_ps(-twiddle3.im, -twiddle3.im, twiddle3.re, twiddle3.re),
                _mm_set_ps(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
            ],
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly7Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // load the first element of the input, and duplicate it into both complex number slots of input0
        let input0 = _mm_castpd_ps(_mm_load1_pd(buffer.input_ptr() as *const f64));

        // we want to load 3 elements into 123 and 3 elements into 456, but we can only load 4, so we're going to do slightly overlapping reads here
        // we have to reverse 456 immediately after loading, and that'll be easiest if we load the 456 into the latter 3 slots of the register, rather than the front 3 slots
        // as a bonus, that also means we don't need masked reads or anything
        let input123 = buffer.load_complex(1);
        let input456 = buffer.load_complex(3);

        // reverse the order of input456
        let input654 = AvxVector::reverse_complex_elements(input456);

        // do some prep work before we can start applying twiddle factors
        let [sum123, diff654] = AvxVector::column_butterfly2([input123, input654]);
        let rotation = AvxVector::make_rotation90(FftDirection::Inverse);
        let rotated654 = AvxVector::rotate90(diff654, rotation);

        let [mid1634, mid25] = AvxVector::unpack_complex([sum123, rotated654]);

        let mid16 = mid1634.lo();
        let mid25 = mid25.lo();
        let mid34 = mid1634.hi();

        // to compute the first output, compute the sum of all elements. mid16[0], mid25[0], and mid34[0] already have the sum of 1+6, 2+5 and 3+4 respectively, so if we add them, we'll get 1+2+3+4+5+6
        let output0_left = AvxVector::add(mid16, mid25);
        let output0_right = AvxVector::add(input0, mid34);
        let output0 = AvxVector::add(output0_left, output0_right);
        buffer.store_partial1_complex(output0, 0);

        _mm256_zeroupper();

        // apply twiddle factors
        let twiddled16_intermediate1 = AvxVector::mul(mid16, self.twiddles[0]);
        let twiddled25_intermediate1 = AvxVector::mul(mid16, self.twiddles[1]);
        let twiddled34_intermediate1 = AvxVector::mul(mid16, self.twiddles[2]);

        let twiddled16_intermediate2 =
            AvxVector::fmadd(mid25, self.twiddles[1], twiddled16_intermediate1);
        let twiddled25_intermediate2 =
            AvxVector::fmadd(mid25, self.twiddles[3], twiddled25_intermediate1);
        let twiddled34_intermediate2 =
            AvxVector::fmadd(mid25, self.twiddles[4], twiddled34_intermediate1);

        let twiddled16 = AvxVector::fmadd(mid34, self.twiddles[2], twiddled16_intermediate2);
        let twiddled25 = AvxVector::fmadd(mid34, self.twiddles[4], twiddled25_intermediate2);
        let twiddled34 = AvxVector::fmadd(mid34, self.twiddles[1], twiddled34_intermediate2);

        // unpack the data for the last butterfly 2
        let [twiddled12, twiddled65] = AvxVector::unpack_complex([twiddled16, twiddled25]);
        let [twiddled33, twiddled44] = AvxVector::unpack_complex([twiddled34, twiddled34]);

        // we can save one add if we add input0 to twiddled33 now. normally we'd add input0 to the final output, but the arrangement of data makes that a little awkward
        let twiddled033 = AvxVector::add(twiddled33, input0);

        let [output12, output65] = AvxVector::column_butterfly2([twiddled12, twiddled65]);
        let [output033, output044] = AvxVector::column_butterfly2([twiddled033, twiddled44]);
        let output56 = AvxVector::reverse_complex_elements(output65);

        buffer.store_partial2_complex(AvxVector::add(output12, input0), 1);
        buffer.store_partial1_complex(output033, 3);
        buffer.store_partial1_complex(output044, 4);
        buffer.store_partial2_complex(AvxVector::add(output56, input0), 5);
    }
}

pub struct Butterfly11Avx<T> {
    twiddles: [__m256; 10],
    twiddle_lo_4: __m128,
    twiddle_lo_9: __m128,
    twiddle_lo_3: __m128,
    twiddle_lo_8: __m128,
    twiddle_lo_2: __m128,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly11Avx, 11);
impl Butterfly11Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddle1 = twiddles::compute_twiddle(1, 11, direction);
        let twiddle2 = twiddles::compute_twiddle(2, 11, direction);
        let twiddle3 = twiddles::compute_twiddle(3, 11, direction);
        let twiddle4 = twiddles::compute_twiddle(4, 11, direction);
        let twiddle5 = twiddles::compute_twiddle(5, 11, direction);

        let twiddles_lo = [
            _mm_set_ps(twiddle1.im, twiddle1.im, twiddle1.re, twiddle1.re),
            _mm_set_ps(twiddle2.im, twiddle2.im, twiddle2.re, twiddle2.re),
            _mm_set_ps(twiddle3.im, twiddle3.im, twiddle3.re, twiddle3.re),
            _mm_set_ps(twiddle4.im, twiddle4.im, twiddle4.re, twiddle4.re),
            _mm_set_ps(twiddle5.im, twiddle5.im, twiddle5.re, twiddle5.re),
            _mm_set_ps(-twiddle5.im, -twiddle5.im, twiddle5.re, twiddle5.re),
            _mm_set_ps(-twiddle4.im, -twiddle4.im, twiddle4.re, twiddle4.re),
            _mm_set_ps(-twiddle3.im, -twiddle3.im, twiddle3.re, twiddle3.re),
            _mm_set_ps(-twiddle2.im, -twiddle2.im, twiddle2.re, twiddle2.re),
            _mm_set_ps(-twiddle1.im, -twiddle1.im, twiddle1.re, twiddle1.re),
        ];

        Self {
            twiddles: [
                AvxVector256::merge(twiddles_lo[0], twiddles_lo[2]),
                AvxVector256::merge(twiddles_lo[1], twiddles_lo[3]),
                AvxVector256::merge(twiddles_lo[1], twiddles_lo[5]),
                AvxVector256::merge(twiddles_lo[3], twiddles_lo[7]),
                AvxVector256::merge(twiddles_lo[2], twiddles_lo[8]),
                AvxVector256::merge(twiddles_lo[5], twiddles_lo[0]),
                AvxVector256::merge(twiddles_lo[3], twiddles_lo[0]),
                AvxVector256::merge(twiddles_lo[7], twiddles_lo[4]),
                AvxVector256::merge(twiddles_lo[4], twiddles_lo[3]),
                AvxVector256::merge(twiddles_lo[9], twiddles_lo[8]),
            ],
            twiddle_lo_4: twiddles_lo[4],
            twiddle_lo_9: twiddles_lo[9],
            twiddle_lo_3: twiddles_lo[3],
            twiddle_lo_8: twiddles_lo[8],
            twiddle_lo_2: twiddles_lo[2],
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly11Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        let input0 = _mm_castpd_ps(_mm_load1_pd(buffer.input_ptr() as *const f64)); // load the first element of the input, and duplicate it into both complex number slots of input0
        let input1234 = buffer.load_complex(1);
        let input56 = buffer.load_partial2_complex(5);
        let input78910 = buffer.load_complex(7);

        // reverse the order of input78910, and separate
        let [input55, input66] = AvxVector::unpack_complex([input56, input56]);
        let input10987 = AvxVector::reverse_complex_elements(input78910);

        // do some initial butterflies and rotations
        let [sum1234, diff10987] = AvxVector::column_butterfly2([input1234, input10987]);
        let [sum55, diff66] = AvxVector::column_butterfly2([input55, input66]);

        let rotation = AvxVector::make_rotation90(FftDirection::Inverse);
        let rotated10987 = AvxVector::rotate90(diff10987, rotation);
        let rotated66 = AvxVector::rotate90(diff66, rotation.lo());

        // arrange the data into the format to apply twiddles
        let [mid11038, mid2947] = AvxVector::unpack_complex([sum1234, rotated10987]);

        let mid110: __m256 = AvxVector256::merge(mid11038.lo(), mid11038.lo());
        let mid29: __m256 = AvxVector256::merge(mid2947.lo(), mid2947.lo());
        let mid38: __m256 = AvxVector256::merge(mid11038.hi(), mid11038.hi());
        let mid47: __m256 = AvxVector256::merge(mid2947.hi(), mid2947.hi());
        let mid56 = AvxVector::unpacklo_complex([sum55, rotated66]);
        let mid56: __m256 = AvxVector256::merge(mid56, mid56);

        // to compute the first output, compute the sum of all elements. mid16[0], mid25[0], and mid34[0] already have the sum of 1+6, 2+5 and 3+4 respectively, so if we add them, we'll get 1+2+3+4+5+6
        let mid12910 = AvxVector::add(mid110.lo(), mid29.lo());
        let mid3478 = AvxVector::add(mid38.lo(), mid47.lo());
        let output0_left = AvxVector::add(input0, mid56.lo());
        let output0_right = AvxVector::add(mid12910, mid3478);
        let output0 = AvxVector::add(output0_left, output0_right);
        buffer.store_partial1_complex(output0, 0);

        // we need to add the first input to each of our 5 twiddles values -- but right now, input0 is duplicated into both slots
        // but we only want to add it once, so zero the second element
        let zero = _mm_setzero_pd();
        let input0 = _mm_castpd_ps(_mm_move_sd(zero, _mm_castps_pd(input0)));
        let input0 = AvxVector256::merge(input0, input0);

        // apply twiddle factors
        let twiddled11038 = AvxVector::fmadd(mid110, self.twiddles[0], input0);
        let twiddled2947 = AvxVector::fmadd(mid110, self.twiddles[1], input0);
        let twiddled56 = AvxVector::fmadd(mid110.lo(), self.twiddle_lo_4, input0.lo());

        let twiddled11038 = AvxVector::fmadd(mid29, self.twiddles[2], twiddled11038);
        let twiddled2947 = AvxVector::fmadd(mid29, self.twiddles[3], twiddled2947);
        let twiddled56 = AvxVector::fmadd(mid29.lo(), self.twiddle_lo_9, twiddled56);

        let twiddled11038 = AvxVector::fmadd(mid38, self.twiddles[4], twiddled11038);
        let twiddled2947 = AvxVector::fmadd(mid38, self.twiddles[5], twiddled2947);
        let twiddled56 = AvxVector::fmadd(mid38.lo(), self.twiddle_lo_3, twiddled56);

        let twiddled11038 = AvxVector::fmadd(mid47, self.twiddles[6], twiddled11038);
        let twiddled2947 = AvxVector::fmadd(mid47, self.twiddles[7], twiddled2947);
        let twiddled56 = AvxVector::fmadd(mid47.lo(), self.twiddle_lo_8, twiddled56);

        let twiddled11038 = AvxVector::fmadd(mid56, self.twiddles[8], twiddled11038);
        let twiddled2947 = AvxVector::fmadd(mid56, self.twiddles[9], twiddled2947);
        let twiddled56 = AvxVector::fmadd(mid56.lo(), self.twiddle_lo_2, twiddled56);

        // unpack the data for the last butterfly 2
        let [twiddled1234, twiddled10987] =
            AvxVector::unpack_complex([twiddled11038, twiddled2947]);
        let [twiddled55, twiddled66] = AvxVector::unpack_complex([twiddled56, twiddled56]);

        let [output1234, output10987] = AvxVector::column_butterfly2([twiddled1234, twiddled10987]);
        let [output55, output66] = AvxVector::column_butterfly2([twiddled55, twiddled66]);
        let output78910 = AvxVector::reverse_complex_elements(output10987);

        buffer.store_complex(output1234, 1);
        buffer.store_partial1_complex(output55, 5);
        buffer.store_partial1_complex(output66, 6);
        buffer.store_complex(output78910, 7);
    }
}

pub struct Butterfly8Avx<T> {
    twiddles: __m256,
    twiddles_butterfly4: __m256,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly8Avx, 8);
impl Butterfly8Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: AvxVector::make_mixedradix_twiddle_chunk(0, 1, 8, direction),
            twiddles_butterfly4: match direction {
                FftDirection::Forward => [
                    Complex::new(0.0f32, 0.0),
                    Complex::new(0.0, -0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, -0.0),
                ]
                .as_slice()
                .load_complex(0),
                FftDirection::Inverse => [
                    Complex::new(0.0f32, 0.0),
                    Complex::new(-0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-0.0, 0.0),
                ]
                .as_slice()
                .load_complex(0),
            },
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly8Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        let row0 = buffer.load_complex(0);
        let row1 = buffer.load_complex(4);

        // Do our butterfly 2's down the columns
        let [intermediate0, intermediate1_pretwiddle] = AvxVector::column_butterfly2([row0, row1]);

        // Apply the size-8 twiddle factors
        let intermediate1 = AvxVector::mul_complex(intermediate1_pretwiddle, self.twiddles);

        // Rearrange the data before we do our butterfly 4s. This swaps the last 2 elements of butterfly0 with the first two elements of butterfly1
        // The result is that we can then do a 4x butterfly 2, apply twiddles, use unpack instructions to transpose to the final output, then do another 4x butterfly 2
        let permuted0 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x20);
        let permuted1 = _mm256_permute2f128_ps(intermediate0, intermediate1, 0x31);

        // Do the first set of butterfly 2's
        let [postbutterfly0, postbutterfly1_pretwiddle] =
            AvxVector::column_butterfly2([permuted0, permuted1]);

        // Which negative we blend in depends on whether we're forward or direction
        // Our goal is to swap the reals with the imaginaries, then negate either the reals or the imaginaries, based on whether we're an direction or not
        // but we can't use the AvxVector swap_complex_components function, because we only want to swap the odd reals with the odd imaginaries
        let elements_swapped = _mm256_permute_ps(postbutterfly1_pretwiddle, 0xB4);

        // We can negate the elements we want by xoring the row with a pre-set vector
        let postbutterfly1 = AvxVector::xor(elements_swapped, self.twiddles_butterfly4);

        // use unpack instructions to transpose, and to prepare for the final butterfly 2's
        let unpermuted0 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x20);
        let unpermuted1 = _mm256_permute2f128_ps(postbutterfly0, postbutterfly1, 0x31);
        let unpacked = AvxVector::unpack_complex([unpermuted0, unpermuted1]);

        let [output0, output1] = AvxVector::column_butterfly2(unpacked);

        buffer.store_complex(output0, 0);
        buffer.store_complex(output1, 4);
    }
}

pub struct Butterfly9Avx<T> {
    twiddles: __m256,
    twiddles_butterfly3: __m256,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly9Avx, 9);
impl Butterfly9Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddles: [Complex<f32>; 4] = [
            twiddles::compute_twiddle(1, 9, direction),
            twiddles::compute_twiddle(2, 9, direction),
            twiddles::compute_twiddle(2, 9, direction),
            twiddles::compute_twiddle(4, 9, direction),
        ];
        Self {
            twiddles: twiddles.as_slice().load_complex(0),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly9Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // we're going to load these elements in a peculiar way. instead of loading a row into the first 3 element of each register and leaving the last element empty
        // we're leaving the first element empty and putting the data in the last 3 elements. this will let us do 3 total complex multiplies instead of 4.

        let input0_lo = _mm_castpd_ps(_mm_load1_pd(buffer.input_ptr() as *const f64));
        let input0_hi = buffer.load_partial2_complex(1);
        let input0 = AvxVector256::merge(input0_lo, input0_hi);
        let input1 = buffer.load_complex(2);
        let input2 = buffer.load_complex(5);

        // We're going to treat our input as a 3x3 2d array. First, do 3 butterfly 3's down the columns of that array.
        let [mid0, mid1, mid2] =
            AvxVector::column_butterfly3([input0, input1, input2], self.twiddles_butterfly3);

        // merge the twiddle-able data into a single avx vector
        let twiddle_data = _mm256_permute2f128_ps(mid1, mid2, 0x31);
        let twiddled = AvxVector::mul_complex(twiddle_data, self.twiddles);

        // Transpose our 3x3 array. We could use the 4x4 transpose with an empty bottom row, which would result in an empty last column
        // but it turns out that it'll make our packing process later simpler if we duplicate the second row into the last row
        // which will result in duplicating the second column into the last column after the transpose
        let permute0 = _mm256_permute2f128_ps(mid0, mid2, 0x20);
        let permute1 = _mm256_permute2f128_ps(mid1, mid1, 0x20);
        let permute2 = _mm256_permute2f128_ps(mid0, twiddled, 0x31);
        let permute3 = _mm256_permute2f128_ps(twiddled, twiddled, 0x20);

        let transposed0 = AvxVector::unpackhi_complex([permute0, permute1]);
        let [transposed1, transposed2] = AvxVector::unpack_complex([permute2, permute3]);

        // more size 3 buterflies
        let output_rows = AvxVector::column_butterfly3(
            [transposed0, transposed1, transposed2],
            self.twiddles_butterfly3,
        );

        // the elements of row 1 are in pretty much the worst possible order, thankfully we can fix that with just a couple instructions
        let swapped1 = _mm256_permute_ps(output_rows[1], 0x4E); // swap even and odd complex numbers
        let packed1 = _mm256_permute2f128_ps(swapped1, output_rows[2], 0x21);
        buffer.store_complex(packed1, 4);

        // merge just the high element of swapped_lo into the high element of row 0
        let zero_swapped1_lo = AvxVector256::merge(AvxVector::zero(), swapped1.lo());
        let packed0 = _mm256_blend_ps(output_rows[0], zero_swapped1_lo, 0xC0);
        buffer.store_complex(packed0, 0);

        // The last element can just be written on its own
        buffer.store_partial1_complex(output_rows[2].hi(), 8);
    }
}

pub struct Butterfly12Avx<T> {
    twiddles: [__m256; 2],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly12Avx, 12);
impl Butterfly12Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddles = [
            Complex {
                re: 1.0f32,
                im: 0.0,
            },
            Complex { re: 1.0, im: 0.0 },
            twiddles::compute_twiddle(2, 12, direction),
            twiddles::compute_twiddle(4, 12, direction),
            // note that these twiddles are deliberately in a weird order, see perform_fft_f32 for why
            twiddles::compute_twiddle(1, 12, direction),
            twiddles::compute_twiddle(2, 12, direction),
            twiddles::compute_twiddle(3, 12, direction),
            twiddles::compute_twiddle(6, 12, direction),
        ];
        Self {
            twiddles: [
                twiddles.as_slice().load_complex(0),
                twiddles.as_slice().load_complex(4),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly12Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // we're going to load these elements in a peculiar way. instead of loading a row into the first 3 element of each register and leaving the last element empty
        // we're leaving the first element empty and putting the data in the last 3 elements. this will save us a complex multiply.

        // for everything but the first element, we can do overlapping reads. for the first element, an "overlapping read" would have us reading from index -1, so instead we have to shuffle some data around
        let input0_lo = _mm_castpd_ps(_mm_load1_pd(buffer.input_ptr() as *const f64));
        let input0_hi = buffer.load_partial2_complex(1);
        let input_rows = [
            AvxVector256::merge(input0_lo, input0_hi),
            buffer.load_complex(2),
            buffer.load_complex(5),
            buffer.load_complex(8),
        ];

        // 3 butterfly 4's down the columns
        let mut mid = AvxVector::column_butterfly4(input_rows, self.twiddles_butterfly4);

        // Multiply in our twiddle factors. mid2 will be normal, but for mid1 and mid3, we're going to merge the twiddle-able parts into a single vector,
        // and do a single complex multiply on it. this transformation saves a complex multiply and costs nothing,
        // because we needthe second halves of mid1 and mid3 in a single vector for the transpose afterward anyways, so we would have done this permute2f128 operation either way
        mid[2] = AvxVector::mul_complex(mid[2], self.twiddles[0]);
        let merged_mid13 = _mm256_permute2f128_ps(mid[1], mid[3], 0x31);
        let twiddled13 = AvxVector::mul_complex(self.twiddles[1], merged_mid13);

        // Transpose our 3x4 array into a 4x3. we're doing a custom transpose here because we have to re-distribute the merged twiddled23 back out, and we can roll that into the transpose to make it free
        let transposed = {
            let permute0 = _mm256_permute2f128_ps(mid[0], mid[2], 0x20);
            let permute1 = _mm256_permute2f128_ps(mid[1], mid[3], 0x20);
            let permute2 = _mm256_permute2f128_ps(mid[0], mid[2], 0x31);
            let permute3 = twiddled13; // normally we'd need to do a permute here, but we can skip it because we already did it for twiddle factors

            let unpacked1 = AvxVector::unpackhi_complex([permute0, permute1]);
            let [unpacked2, unpacked3] = AvxVector::unpack_complex([permute2, permute3]);

            [unpacked1, unpacked2, unpacked3]
        };

        // Do 4 butterfly 3's down the columns of our transposed array
        let output_rows = AvxVector::column_butterfly3(transposed, self.twiddles_butterfly3);

        buffer.store_complex(output_rows[0], 0);
        buffer.store_complex(output_rows[1], 4);
        buffer.store_complex(output_rows[2], 8);
    }
}

pub struct Butterfly16Avx<T> {
    twiddles: [__m256; 3],
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly16Avx, 16);
impl Butterfly16Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 4, 0, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly16Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            buffer.load_complex(0),
            buffer.load_complex(4),
            buffer.load_complex(8),
            buffer.load_complex(12),
        ];

        // We're going to treat our input as a 4x4 2d array. First, do 4 butterfly 4's down the columns of that array.
        let mut mid = AvxVector::column_butterfly4(rows, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 4x4 array
        let transposed = avx32_utils::transpose_4x4_f32(mid);

        // Do 4 butterfly 4's down the columns of our transposed array
        let output_rows = AvxVector::column_butterfly4(transposed, self.twiddles_butterfly4);

        // Manually unrolling this loop because writing a "for r in 0..4" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        buffer.store_complex(output_rows[0], 0);
        buffer.store_complex(output_rows[1], 4);
        buffer.store_complex(output_rows[2], 8);
        buffer.store_complex(output_rows[3], 12);
    }
}

pub struct Butterfly24Avx<T> {
    twiddles: [__m256; 5],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly24Avx, 24);
impl Butterfly24Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(6, 4, 0, direction),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly24Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // Manually unrolling this loop because writing a "for r in 0..6" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        let rows = [
            buffer.load_complex(0),
            buffer.load_complex(4),
            buffer.load_complex(8),
            buffer.load_complex(12),
            buffer.load_complex(16),
            buffer.load_complex(20),
        ];

        // We're going to treat our input as a 4x6 2d array. First, do 4 butterfly 6's down the columns of that array.
        let mut mid = AvxVector256::column_butterfly6(rows, self.twiddles_butterfly3);

        // apply twiddle factors
        for r in 1..6 {
            mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1]);
        }

        // Transpose our 6x4 array into a 4x6.
        let (transposed0, transposed1) = avx32_utils::transpose_4x6_to_6x4_f32(mid);

        // Do 6 butterfly 4's down the columns of our transposed array
        let output0 = AvxVector::column_butterfly4(transposed0, self.twiddles_butterfly4);
        let output1 = AvxVector::column_butterfly4(transposed1, self.twiddles_butterfly4);

        // the upper two elements of output1 are empty, so only store half the data for it
        for r in 0..4 {
            buffer.store_complex(output0[r], 6 * r);
            buffer.store_partial2_complex(output1[r].lo(), r * 6 + 4);
        }
    }
}

pub struct Butterfly27Avx<T> {
    twiddles: [__m256; 4],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly3: __m256,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly27Avx, 27);
impl Butterfly27Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(3, 9, 1, direction),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, direction),
                AvxVector::broadcast_twiddle(2, 9, direction),
                AvxVector::broadcast_twiddle(4, 9, direction),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly27Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        let mut rows0 = [AvxVector::zero(); 3];
        let mut rows1 = [AvxVector::zero(); 3];
        let mut rows2 = [AvxVector::zero(); 3];
        for r in 0..3 {
            rows0[r] = buffer.load_partial2_complex(r * 9);
            rows1[r] = buffer.load_complex(r * 9 + 1);
            rows2[r] = buffer.load_complex(r * 9 + 5);
        }

        // butterfly 3s down the columns
        let mid0 = AvxVector::column_butterfly3(rows0, self.twiddles_butterfly3.lo());
        let mut mid1 = AvxVector::column_butterfly3(rows1, self.twiddles_butterfly3);
        let mut mid2 = AvxVector::column_butterfly3(rows2, self.twiddles_butterfly3);

        // apply twiddle factors
        mid1[1] = AvxVector::mul_complex(mid1[1], self.twiddles[0]);
        mid2[1] = AvxVector::mul_complex(mid2[1], self.twiddles[1]);
        mid1[2] = AvxVector::mul_complex(mid1[2], self.twiddles[2]);
        mid2[2] = AvxVector::mul_complex(mid2[2], self.twiddles[3]);

        // transpose 9x3 to 3x9. this will be a little awkward because of rows0 containing garbage data, so use a transpose function that knows to ignore it
        let transposed = avx32_utils::transpose_9x3_to_3x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        let output_rows = AvxVector256::column_butterfly9(
            transposed,
            self.twiddles_butterfly9,
            self.twiddles_butterfly3,
        );

        // Our last column is empty, so it's a bit awkward to write out to memory. We could pack it in fewer vectors, but benchmarking shows it's simpler and just as fast to just brute-force it with partial writes
        buffer.store_partial3_complex(output_rows[0], 0);
        buffer.store_partial3_complex(output_rows[1], 3);
        buffer.store_partial3_complex(output_rows[2], 6);
        buffer.store_partial3_complex(output_rows[3], 9);
        buffer.store_partial3_complex(output_rows[4], 12);
        buffer.store_partial3_complex(output_rows[5], 15);
        buffer.store_partial3_complex(output_rows[6], 18);
        buffer.store_partial3_complex(output_rows[7], 21);
        buffer.store_partial3_complex(output_rows[8], 24);
    }
}

pub struct Butterfly32Avx<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly32Avx, 32);
impl Butterfly32Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 8, 0, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly32Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = buffer.load_complex(8 * r);
            rows1[r] = buffer.load_complex(8 * r + 4);
        }

        // We're going to treat our input as a 8x4 2d array. First, do 8 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[2 * r - 2]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2 * r - 1]);
        }

        // Transpose our 8x4 array to an 4x8 array
        let transposed = avx32_utils::transpose_8x4_to_4x8_f32(mid0, mid1);

        // Do 4 butterfly 8's down the columns of our transpsed array
        let output_rows = AvxVector::column_butterfly8(transposed, self.twiddles_butterfly4);

        // Manually unrolling this loop because writing a "for r in 0..8" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        buffer.store_complex(output_rows[0], 0);
        buffer.store_complex(output_rows[1], 1 * 4);
        buffer.store_complex(output_rows[2], 2 * 4);
        buffer.store_complex(output_rows[3], 3 * 4);
        buffer.store_complex(output_rows[4], 4 * 4);
        buffer.store_complex(output_rows[5], 5 * 4);
        buffer.store_complex(output_rows[6], 6 * 4);
        buffer.store_complex(output_rows[7], 7 * 4);
    }
}

pub struct Butterfly36Avx<T> {
    twiddles: [__m256; 6],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly36Avx, 36);
impl Butterfly36Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 9, 1, direction),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, direction),
                AvxVector::broadcast_twiddle(2, 9, direction),
                AvxVector::broadcast_twiddle(4, 9, direction),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly36Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        let mut rows2 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = buffer.load_partial2_complex(r * 9);
            rows1[r] = buffer.load_complex(r * 9 + 1);
            rows2[r] = buffer.load_complex(r * 9 + 5);
        }

        // butterfly 4s down the columns
        let mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4.lo());
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2 * r - 2]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[2 * r - 1]);
        }

        // transpose 9x4 to 4x9. this will be a little awkward because of rows0 containing garbage data, so use a transpose function that knows to ignore it
        let transposed = avx32_utils::transpose_9x4_to_4x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        let output_rows = AvxVector256::column_butterfly9(
            transposed,
            self.twiddles_butterfly9,
            self.twiddles_butterfly3,
        );

        for r in 0..3 {
            buffer.store_complex(output_rows[r * 3], r * 12);
            buffer.store_complex(output_rows[r * 3 + 1], r * 12 + 4);
            buffer.store_complex(output_rows[r * 3 + 2], r * 12 + 8);
        }
    }
}

pub struct Butterfly48Avx<T> {
    twiddles: [__m256; 9],
    twiddles_butterfly3: __m256,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly48Avx, 48);
impl Butterfly48Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(4, 12, 0, direction),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly48Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        let mut rows0 = [AvxVector::zero(); 4];
        let mut rows1 = [AvxVector::zero(); 4];
        let mut rows2 = [AvxVector::zero(); 4];
        for r in 0..4 {
            rows0[r] = buffer.load_complex(12 * r);
            rows1[r] = buffer.load_complex(12 * r + 4);
            rows2[r] = buffer.load_complex(12 * r + 8);
        }

        // We're going to treat our input as a 12x4 2d array. First, do 12 butterfly 4's down the columns of that array.
        let mut mid0 = AvxVector::column_butterfly4(rows0, self.twiddles_butterfly4);
        let mut mid1 = AvxVector::column_butterfly4(rows1, self.twiddles_butterfly4);
        let mut mid2 = AvxVector::column_butterfly4(rows2, self.twiddles_butterfly4);

        // apply twiddle factors
        for r in 1..4 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[3 * r - 3]);
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[3 * r - 2]);
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[3 * r - 1]);
        }

        // Transpose our 12x4 array into a 4x12.
        let transposed = avx32_utils::transpose_12x4_to_4x12_f32(mid0, mid1, mid2);

        // Do 4 butterfly 12's down the columns of our transposed array
        let output_rows = AvxVector256::column_butterfly12(
            transposed,
            self.twiddles_butterfly3,
            self.twiddles_butterfly4,
        );

        // Manually unrolling this loop because writing a "for r in 0..12" loop results in slow codegen that makes the whole thing take 1.5x longer :(
        buffer.store_complex(output_rows[0], 0);
        buffer.store_complex(output_rows[1], 4);
        buffer.store_complex(output_rows[2], 8);
        buffer.store_complex(output_rows[3], 12);
        buffer.store_complex(output_rows[4], 16);
        buffer.store_complex(output_rows[5], 20);
        buffer.store_complex(output_rows[6], 24);
        buffer.store_complex(output_rows[7], 28);
        buffer.store_complex(output_rows[8], 32);
        buffer.store_complex(output_rows[9], 36);
        buffer.store_complex(output_rows[10], 40);
        buffer.store_complex(output_rows[11], 44);
    }
}

pub struct Butterfly54Avx<T> {
    twiddles: [__m256; 10],
    twiddles_butterfly9: [__m256; 3],
    twiddles_butterfly9_lo: [__m256; 2],
    twiddles_butterfly3: __m256,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly54Avx, 54);
impl Butterfly54Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let twiddle1 = __m128::broadcast_twiddle(1, 9, direction);
        let twiddle2 = __m128::broadcast_twiddle(2, 9, direction);
        let twiddle4 = __m128::broadcast_twiddle(4, 9, direction);

        Self {
            twiddles: gen_butterfly_twiddles_interleaved_columns!(6, 9, 1, direction),
            twiddles_butterfly9: [
                AvxVector::broadcast_twiddle(1, 9, direction),
                AvxVector::broadcast_twiddle(2, 9, direction),
                AvxVector::broadcast_twiddle(4, 9, direction),
            ],
            twiddles_butterfly9_lo: [
                AvxVector256::merge(twiddle1, twiddle2),
                AvxVector256::merge(twiddle2, twiddle4),
            ],
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly54Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // we're going to load our data in a peculiar way. we're going to load the first column on its own as a column of __m128.
        // it's faster to just load the first 2 columns into these m128s than trying to worry about masks, etc, so the second column will piggyback along and we just won't use it
        //
        // we have too much data to fit into registers all at once, so split up our data processing so that we entirely finish with one "rows_" array before moving to the next
        let mut rows0 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows0[r * 2] = buffer.load_partial2_complex(r * 18);
            rows0[r * 2 + 1] = buffer.load_partial2_complex(r * 18 + 9);
        }
        let mid0 = AvxVector128::column_butterfly6(rows0, self.twiddles_butterfly3);

        // next set of butterfly 6's
        let mut rows1 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows1[r * 2] = buffer.load_complex(r * 18 + 1);
            rows1[r * 2 + 1] = buffer.load_complex(r * 18 + 10);
        }
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[2 * r - 2]);
        }

        // final set of butterfly 6's
        let mut rows2 = [AvxVector::zero(); 6];
        for r in 0..3 {
            rows2[r * 2] = buffer.load_complex(r * 18 + 5);
            rows2[r * 2 + 1] = buffer.load_complex(r * 18 + 14);
        }
        let mut mid2 = AvxVector256::column_butterfly6(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[2 * r - 1]);
        }

        // transpose 9x6 to 6x9. this will be a little awkward because of rows0 containing garbage data, so use a transpose function that knows to ignore it
        let (transposed0, transposed1) =
            avx32_utils::transpose_9x6_to_6x9_emptycolumn1_f32(mid0, mid1, mid2);

        // butterfly 9s down the rows
        // process the other half
        let output_rows1 = AvxVector128::column_butterfly9(
            transposed1,
            self.twiddles_butterfly9_lo,
            self.twiddles_butterfly3,
        );
        for r in 0..9 {
            buffer.store_partial2_complex(output_rows1[r], r * 6 + 4);
        }

        // we have too much data to fit into registers all at once, do one set of butterfly 9's and output them before even starting on the others, to make it easier for the compiler to figure out what to spill
        let output_rows0 = AvxVector256::column_butterfly9(
            transposed0,
            self.twiddles_butterfly9,
            self.twiddles_butterfly3,
        );
        for r in 0..9 {
            buffer.store_complex(output_rows0[r], r * 6);
        }
    }
}

pub struct Butterfly64Avx<T> {
    twiddles: [__m256; 14],
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly64Avx, 64);
impl Butterfly64Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 8, 0, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly64Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // We're going to treat our input as a 8x8 2d array. First, do 8 butterfly 8's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing a sizeable chunk and not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [AvxVector::zero(); 8];
        for r in 0..8 {
            rows0[r] = buffer.load_complex(8 * r);
        }
        let mut mid0 = AvxVector::column_butterfly8(rows0, self.twiddles_butterfly4);
        for r in 1..8 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[r - 1]);
        }

        // One half is done, so the compiler can spill everything above this. Now do the other set of columns
        let mut rows1 = [AvxVector::zero(); 8];
        for r in 0..8 {
            rows1[r] = buffer.load_complex(8 * r + 4);
        }
        let mut mid1 = AvxVector::column_butterfly8(rows1, self.twiddles_butterfly4);
        for r in 1..8 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[r - 1 + 7]);
        }

        // Transpose our 8x8 array
        let (transposed0, transposed1) = avx32_utils::transpose_8x8_f32(mid0, mid1);

        // Do 8 butterfly 8's down the columns of our transposed array, and store the results
        // Same thing as above - Do the half of the butterfly 8's separately to give the compiler a better hint about what to spill
        let output0 = AvxVector::column_butterfly8(transposed0, self.twiddles_butterfly4);
        for r in 0..8 {
            buffer.store_complex(output0[r], 8 * r);
        }

        let output1 = AvxVector::column_butterfly8(transposed1, self.twiddles_butterfly4);
        for r in 0..8 {
            buffer.store_complex(output1[r], 8 * r + 4);
        }
    }
}

pub struct Butterfly72Avx<T> {
    twiddles: [__m256; 15],
    twiddles_butterfly4: Rotation90<__m256>,
    twiddles_butterfly3: __m256,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly!(Butterfly72Avx, 72);
impl Butterfly72Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(6, 12, 0, direction),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            twiddles_butterfly3: AvxVector::broadcast_twiddle(1, 3, direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly72Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn perform_fft_f32(&self, mut buffer: impl AvxArrayMut<f32>) {
        // We're going to treat our input as a 12x6 2d array. First, do butterfly 6's down the columns of that array.
        // We can't fit the whole problem into AVX registers at once, so we'll have to spill some things.
        // By computing a sizeable chunk and not referencing any of it for a while, we're making it easy for the compiler to decide what to spill
        let mut rows0 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows0[r] = buffer.load_complex(12 * r);
        }
        let mut mid0 = AvxVector256::column_butterfly6(rows0, self.twiddles_butterfly3);
        for r in 1..6 {
            mid0[r] = AvxVector::mul_complex(mid0[r], self.twiddles[r - 1]);
        }

        // One third is done, so the compiler can spill everything above this. Now do the middle set of columns
        let mut rows1 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows1[r] = buffer.load_complex(12 * r + 4);
        }
        let mut mid1 = AvxVector256::column_butterfly6(rows1, self.twiddles_butterfly3);
        for r in 1..6 {
            mid1[r] = AvxVector::mul_complex(mid1[r], self.twiddles[r - 1 + 5]);
        }

        // two thirds are done, so the compiler can spill everything above this. Now do the final set of columns
        let mut rows2 = [AvxVector::zero(); 6];
        for r in 0..6 {
            rows2[r] = buffer.load_complex(12 * r + 8);
        }
        let mut mid2 = AvxVector256::column_butterfly6(rows2, self.twiddles_butterfly3);
        for r in 1..6 {
            mid2[r] = AvxVector::mul_complex(mid2[r], self.twiddles[r - 1 + 10]);
        }

        // Transpose our 12x6 array to 6x12 array
        let (transposed0, transposed1) = avx32_utils::transpose_12x6_to_6x12_f32(mid0, mid1, mid2);

        // Do butterfly 12's down the columns of our transposed array, and store the results
        // Same thing as above - Do the half of the butterfly 12's separately to give the compiler a better hint about what to spill
        let output0 = AvxVector128::column_butterfly12(
            transposed0,
            self.twiddles_butterfly3,
            self.twiddles_butterfly4,
        );
        for r in 0..12 {
            buffer.store_partial2_complex(output0[r], 6 * r);
        }

        let output1 = AvxVector256::column_butterfly12(
            transposed1,
            self.twiddles_butterfly3,
            self.twiddles_butterfly4,
        );
        for r in 0..12 {
            buffer.store_complex(output1[r], 6 * r + 2);
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
struct Bf16PackedTwiddles {
    i1: __m256,
    i3: __m256,
}

#[target_feature(enable = "avx")]
unsafe fn pack_bf16_twiddles(tw16: [__m256; 2]) -> Bf16PackedTwiddles {
    Bf16PackedTwiddles {
        i1: pack4_complex(tw16[0], tw16[0], tw16[1], tw16[1]),
        i3: pack4_complex(tw16[1], tw16[1], tw16[0].neg(), tw16[0].neg()),
    }
}

pub struct Butterfly128Avx<T> {
    twiddles: [__m256; 28],
    twiddles_butterfly16: [__m256; 2],
    packed_bf16: Bf16PackedTwiddles,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly128Avx, 128);
impl Butterfly128Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let tw16 = [
            AvxVector::broadcast_twiddle(1, 16, direction),
            AvxVector::broadcast_twiddle(3, 16, direction),
        ];
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 16, 0, direction),
            twiddles_butterfly16: tw16,
            packed_bf16: pack_bf16_twiddles(tw16),
            twiddles_butterfly4: AvxVector::make_rotation90(direction),
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly128Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(
        &self,
        input: &[Complex<f32>],
        mut output: &mut [Complex<f32>],
    ) {
        // A size-128 FFT is way too big to fit in registers, so instead we're going to compute it in two phases, storing in scratch in between.

        // First phase is to treat this size-128 array like a 16x8 2D array, and do butterfly 8's down the columns
        // Then, apply twiddle factors, and finally transpose into the scratch space

        // But again, we don't have enough registers to load it all at once, so only load one column of AVX vectors at a time
        for columnset in 0..4 {
            let mut rows = [AvxVector::zero(); 8];
            for r in 0..8 {
                rows[r] = input.load_complex(columnset * 4 + 16 * r);
            }
            // apply butterfly 8
            let mut mid = AvxVector::column_butterfly8(rows, self.twiddles_butterfly4);

            // apply twiddle factors
            for r in 1..8 {
                mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1 + 7 * columnset]);
            }

            // transpose
            let transposed = AvxVector::transpose8_packed(mid);

            // write out
            for i in 0..4 {
                output.store_complex(transposed[i * 2], columnset * 32 + i * 8);
                output.store_complex(transposed[i * 2 + 1], columnset * 32 + i * 8 + 4);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, mut buffer: impl AvxArrayMut<f32>) {
        // Second phase: Butterfly 16's down the columns of our transposed array.
        // Thankfully, during the first phase, we set everything up so that all we have to do here is compute the size-16 FFT columns and write them back out where we got them
        // We're also using a customized butterfly16 function that is smarter about when it loads/stores data, to reduce register spilling
        for columnset in 0usize..2 {
            column_butterfly16_loadfn!(
                |index: usize| buffer.load_complex(columnset * 4 + index * 8),
                |data, index| buffer.store_complex(data, columnset * 4 + index * 8),
                self.twiddles_butterfly16,
                self.twiddles_butterfly4
            );
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
struct Bf32PackedTwiddles {
    lo: [__m256; 3],
    hi: [__m256; 3],
}

#[target_feature(enable = "avx")]
unsafe fn pack4_complex(c0: __m256, c1: __m256, c2: __m256, c3: __m256) -> __m256 {
    let t01 = _mm256_unpacklo_pd(_mm256_castps_pd(c0), _mm256_castps_pd(c1));
    let t23 = _mm256_unpacklo_pd(_mm256_castps_pd(c2), _mm256_castps_pd(c3));
    _mm256_castpd_ps(_mm256_permute2f128_pd::<0x20>(t01, t23))
}

#[target_feature(enable = "avx")]
unsafe fn pack_bf32_twiddles(tw: [__m256; 6], rot: Rotation90<__m256>) -> Bf32PackedTwiddles {
    Bf32PackedTwiddles {
        lo: [
            pack4_complex(tw[0], tw[0], tw[1], tw[2]),
            pack4_complex(tw[0], tw[1], tw[0], tw[4]),
            pack4_complex(tw[0], tw[2], tw[4], tw[0].rotate90(rot)),
        ],
        hi: [
            pack4_complex(tw[0], tw[3], tw[4], tw[5]),
            pack4_complex(tw[0], tw[1].rotate90(rot), tw[0], tw[4].rotate90(rot)),
            pack4_complex(tw[0], tw[5].rotate90(rot), tw[1].neg(), tw[3].neg()),
        ],
    }
}

pub struct Butterfly256Avx<T> {
    twiddles: [__m256; 56],
    twiddles_butterfly32: [__m256; 6],
    packed_bf32: Bf32PackedTwiddles,
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly256Avx, 256);
impl Butterfly256Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let tw32 = [
            AvxVector::broadcast_twiddle(1, 32, direction),
            AvxVector::broadcast_twiddle(2, 32, direction),
            AvxVector::broadcast_twiddle(3, 32, direction),
            AvxVector::broadcast_twiddle(5, 32, direction),
            AvxVector::broadcast_twiddle(6, 32, direction),
            AvxVector::broadcast_twiddle(7, 32, direction),
        ];
        let rot = AvxVector::make_rotation90(direction);
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(8, 32, 0, direction),
            twiddles_butterfly32: tw32,
            packed_bf32: pack_bf32_twiddles(tw32, rot),
            twiddles_butterfly4: rot,
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly256Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(
        &self,
        input: &[Complex<f32>],
        mut output: &mut [Complex<f32>],
    ) {
        // A size-256 FFT is way too big to fit in registers, so instead we're going to compute it in two phases, storing in scratch in between.

        // First phase is to treeat this size-256 array like a 32x8 2D array, and do butterfly 8's down the columns
        // Then, apply twiddle factors, and finally transpose into the scratch space

        // But again, we don't have enough registers to load it all at once, so only load one column of AVX vectors at a time
        for columnset in 0..8 {
            let mut rows = [AvxVector::zero(); 8];
            for r in 0..8 {
                rows[r] = input.load_complex(columnset * 4 + 32 * r);
            }
            let mut mid = AvxVector::column_butterfly8(rows, self.twiddles_butterfly4);
            for r in 1..8 {
                mid[r] = AvxVector::mul_complex(mid[r], self.twiddles[r - 1 + 7 * columnset]);
            }

            // Before writing to the scratch, transpose this chunk of the array
            let transposed = AvxVector::transpose8_packed(mid);

            for i in 0..4 {
                output.store_complex(transposed[i * 2], columnset * 32 + i * 8);
                output.store_complex(transposed[i * 2 + 1], columnset * 32 + i * 8 + 4);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, mut buffer: impl AvxArrayMut<f32>) {
        // Second phase: Butterfly 32's down the columns of our transposed array.
        // Thankfully, during the first phase, we set everything up so that all we have to do here is compute the size-32 FFT columns and write them back out where we got them
        // We're also using a customized butterfly32 function that is smarter about when it loads/stores data, to reduce register spilling
        for columnset in 0..2 {
            column_butterfly32_loadfn!(
                |index: usize| buffer.load_complex(columnset * 4 + index * 8),
                |data, index| buffer.store_complex(data, columnset * 4 + index * 8),
                self.twiddles_butterfly32,
                self.twiddles_butterfly4
            );
        }
    }
}

pub struct Butterfly512Avx<T> {
    twiddles: [__m256; 120],
    twiddles_butterfly32: [__m256; 6],
    packed_bf32: Bf32PackedTwiddles,
    twiddles_butterfly16: [__m256; 2],
    twiddles_butterfly4: Rotation90<__m256>,
    direction: FftDirection,
    _phantom_t: std::marker::PhantomData<T>,
}
boilerplate_fft_simd_butterfly_with_scratch!(Butterfly512Avx, 512);
impl Butterfly512Avx<f32> {
    #[target_feature(enable = "avx")]
    unsafe fn new_with_avx(direction: FftDirection) -> Self {
        let tw32 = [
            AvxVector::broadcast_twiddle(1, 32, direction),
            AvxVector::broadcast_twiddle(2, 32, direction),
            AvxVector::broadcast_twiddle(3, 32, direction),
            AvxVector::broadcast_twiddle(5, 32, direction),
            AvxVector::broadcast_twiddle(6, 32, direction),
            AvxVector::broadcast_twiddle(7, 32, direction),
        ];
        let rot = AvxVector::make_rotation90(direction);
        Self {
            twiddles: gen_butterfly_twiddles_separated_columns!(16, 32, 0, direction),
            twiddles_butterfly32: tw32,
            packed_bf32: pack_bf32_twiddles(tw32, rot),
            twiddles_butterfly16: [
                AvxVector::broadcast_twiddle(1, 16, direction),
                AvxVector::broadcast_twiddle(3, 16, direction),
            ],
            twiddles_butterfly4: rot,
            direction,
            _phantom_t: PhantomData,
        }
    }
}
impl<T> Butterfly512Avx<T> {
    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn column_butterflies_and_transpose(
        &self,
        input: &[Complex<f32>],
        mut output: &mut [Complex<f32>],
    ) {
        // A size-512 FFT is way too big to fit in registers, so instead we're going to compute it in two phases, storing in scratch in between.

        // First phase is to treat this size-512 array like a 32x16 2D array, and do butterfly 16's down the columns
        // Then, apply twiddle factors, and finally transpose into the scratch space

        // But again, we don't have enough registers to load it all at once, so only load one column of AVX vectors at a time
        // We're also using a customized butterfly16 function that is smarter about when it loads/stores data, to reduce register spilling
        const TWIDDLES_PER_COLUMN: usize = 15;
        for (columnset, twiddle_chunk) in
            self.twiddles.chunks_exact(TWIDDLES_PER_COLUMN).enumerate()
        {
            // Sadly we have to use MaybeUninit here. If we init an array like normal with AvxVector::Zero(), the compiler can't seem to figure out that it can
            // eliminate the dead stores of zeroes to the stack. By using uninit here, we avoid those unnecessary writes
            let mut mid_uninit: [MaybeUninit<__m256>; 16] = [MaybeUninit::<__m256>::uninit(); 16];

            column_butterfly16_loadfn!(
                |index: usize| input.load_complex(columnset * 4 + 32 * index),
                |data, index: usize| {
                    mid_uninit[index].as_mut_ptr().write(data);
                },
                self.twiddles_butterfly16,
                self.twiddles_butterfly4
            );

            // Apply twiddle factors, transpose, and store. Traditionally we apply all the twiddle factors at once and then do all the transposes at once,
            // But our data is pushing the limit of what we can store in registers, so the idea here is to get the data out the door with as few spills to the stack as possible
            for chunk in 0..4 {
                let twiddled = [
                    if chunk > 0 {
                        AvxVector::mul_complex(
                            mid_uninit[4 * chunk].assume_init(),
                            twiddle_chunk[4 * chunk - 1],
                        )
                    } else {
                        mid_uninit[4 * chunk].assume_init()
                    },
                    AvxVector::mul_complex(
                        mid_uninit[4 * chunk + 1].assume_init(),
                        twiddle_chunk[4 * chunk],
                    ),
                    AvxVector::mul_complex(
                        mid_uninit[4 * chunk + 2].assume_init(),
                        twiddle_chunk[4 * chunk + 1],
                    ),
                    AvxVector::mul_complex(
                        mid_uninit[4 * chunk + 3].assume_init(),
                        twiddle_chunk[4 * chunk + 2],
                    ),
                ];

                let transposed = AvxVector::transpose4_packed(twiddled);

                output.store_complex(transposed[0], columnset * 64 + 0 * 16 + 4 * chunk);
                output.store_complex(transposed[1], columnset * 64 + 1 * 16 + 4 * chunk);
                output.store_complex(transposed[2], columnset * 64 + 2 * 16 + 4 * chunk);
                output.store_complex(transposed[3], columnset * 64 + 3 * 16 + 4 * chunk);
            }
        }
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn row_butterflies(&self, mut buffer: impl AvxArrayMut<f32>) {
        // Second phase: Butterfly 32's down the columns of our transposed array.
        // Thankfully, during the first phase, we set everything up so that all we have to do here is compute the size-32 FFT columns and write them back out where we got them
        // We're also using a customized butterfly32 function that is smarter about when it loads/stores data, to reduce register spilling
        for columnset in 0..4 {
            column_butterfly32_loadfn!(
                |index: usize| buffer.load_complex(columnset * 4 + index * 16),
                |data, index| buffer.store_complex(data, columnset * 4 + index * 16),
                self.twiddles_butterfly32,
                self.twiddles_butterfly4
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    macro_rules! test_avx_butterfly {
        ($test_name:ident, $struct_name:ident, $size:expr) => (
            #[test]
            fn $test_name() {
                let butterfly = $struct_name::<f32>::new(FftDirection::Forward).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly as &dyn Fft<f32>, $size, FftDirection::Forward);

                let butterfly_inverse = $struct_name::<f32>::new(FftDirection::Inverse).expect("Can't run test because this machine doesn't have the required instruction sets");
                check_fft_algorithm(&butterfly_inverse as &dyn Fft<f32>, $size, FftDirection::Inverse);
            }
        )
    }
    test_avx_butterfly!(test_avx_butterfly5, Butterfly5Avx, 5);
    test_avx_butterfly!(test_avx_butterfly7, Butterfly7Avx, 7);
    test_avx_butterfly!(test_avx_butterfly8, Butterfly8Avx, 8);
    test_avx_butterfly!(test_avx_butterfly9, Butterfly9Avx, 9);
    test_avx_butterfly!(test_avx_butterfly11, Butterfly11Avx, 11);
    test_avx_butterfly!(test_avx_butterfly12, Butterfly12Avx, 12);
    test_avx_butterfly!(test_avx_butterfly16, Butterfly16Avx, 16);
    test_avx_butterfly!(test_avx_butterfly24, Butterfly24Avx, 24);
    test_avx_butterfly!(test_avx_butterfly27, Butterfly27Avx, 27);
    test_avx_butterfly!(test_avx_butterfly32, Butterfly32Avx, 32);
    test_avx_butterfly!(test_avx_butterfly36, Butterfly36Avx, 36);
    test_avx_butterfly!(test_avx_butterfly48, Butterfly48Avx, 48);
    test_avx_butterfly!(test_avx_butterfly54, Butterfly54Avx, 54);
    test_avx_butterfly!(test_avx_butterfly64, Butterfly64Avx, 64);
    test_avx_butterfly!(test_avx_butterfly72, Butterfly72Avx, 72);
    test_avx_butterfly!(test_avx_butterfly128, Butterfly128Avx, 128);
    test_avx_butterfly!(test_avx_butterfly256, Butterfly256Avx, 256);
    test_avx_butterfly!(test_avx_butterfly512, Butterfly512Avx, 512);

    #[test]
    fn f32_pruned_matches_full_bit_for_bit() {
        use crate::{FftPlanner, PrunedWork};

        fn assert_bits_f32(a: Complex<f32>, b: Complex<f32>, label: &str) {
            let eq = |x: f32, y: f32| if y.is_nan() { x.is_nan() } else { x.to_bits() == y.to_bits() };
            assert!(
                eq(a.re, b.re) && eq(a.im, b.im),
                "{label}: actual={a:?} [{:08x},{:08x}] expected={b:?} [{:08x},{:08x}]",
                a.re.to_bits(),
                a.im.to_bits(),
                b.re.to_bits(),
                b.im.to_bits()
            );
        }

        let mut planner = FftPlanner::<f32>::new();
        for dir in [FftDirection::Forward, FftDirection::Inverse] {
            for &n in &[128usize, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
                let fft = planner.plan_fft(n, dir);
                let s: Vec<Complex<f32>> = (0..n)
                    .map(|i| {
                        Complex::new(
                            ((i as f32 * 0.37).sin() + if i == n / 5 { f32::INFINITY } else { 0.0 }) * 1.1,
                            ((i as f32 * 0.19).cos() - 0.25) * 0.9,
                        )
                    })
                    .collect();
                let s_finite: Vec<Complex<f32>> = (0..n)
                    .map(|i| Complex::new((i as f32 * 0.37).sin() * 1.1, ((i as f32 * 0.19).cos() - 0.25) * 0.9))
                    .collect();
                let h: Vec<Complex<f32>> = (0..n)
                    .map(|i| Complex::new((i as f32 * 0.13).cos() * 0.7, (i as f32 * 0.29).sin() * 0.6))
                    .collect();

                let mut full = s_finite.clone();
                let mut full_scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
                fft.process_with_scratch(&mut full, &mut full_scratch);

                let mut prod_full: Vec<Complex<f32>> = s_finite.iter().zip(&h).map(|(a, b)| {
                    let mut x = *a;
                    x *= *b;
                    x
                }).collect();
                fft.process_with_scratch(&mut prod_full, &mut full_scratch);

                let cases: Vec<Vec<usize>> = if n <= 512 {
                    vec![
                        vec![0],
                        vec![n - 1],
                        vec![3, 19],
                        vec![1, 5, 17, 63],
                        (0..n).step_by(41).take(12).collect(),
                    ]
                } else {
                    vec![
                        vec![7],
                        vec![n / 3, n - 5],
                        (100..n).step_by(267).collect(),
                    ]
                };

                for needed in cases {
                    let u = fft.pruned_work_len(needed.len());
                    let mut work_idx = vec![usize::MAX; 2 * u];
                    let mut work_val = vec![Complex::new(f32::NAN, f32::INFINITY); u];
                    let mut scratch = vec![Complex::new(f32::NAN, -f32::INFINITY); fft.get_inplace_scratch_len()];
                    let mut out = vec![Complex::default(); needed.len()];

                    let mut buf = s_finite.clone();
                    fft.process_pruned_with_scratch(
                        &mut buf,
                        &mut scratch,
                        &needed,
                        &mut out,
                        PrunedWork { idx: &mut work_idx, val: &mut work_val },
                    );
                    for (&idx_o, &got) in needed.iter().zip(&out) {
                        assert_bits_f32(got, full[idx_o], &format!("n={n} dir={dir:?} idx={idx_o} needed_len={}", needed.len()));
                    }

                    let mut buf = vec![Complex::new(f32::NAN, -f32::INFINITY); n];
                    fft.process_pruned_product_with_scratch(
                        &s_finite,
                        &h,
                        &mut buf,
                        &mut scratch,
                        &needed,
                        &mut out,
                        PrunedWork { idx: &mut work_idx, val: &mut work_val },
                    );
                    for (&idx_o, &got) in needed.iter().zip(&out) {
                        assert_bits_f32(got, prod_full[idx_o], &format!("prod n={n} dir={dir:?} idx={idx_o}"));
                    }
                }
                let _ = s;
            }
        }
    }
}

// ── Moosik (1.5.3 round 12-E): exact requested-output Butterfly256/512 ────
//
// Requested-output leaves (used by `process_pruned_with_scratch` for 1..=4
// requested outputs; otherwise the full leaf and gather). For Butterfly256: for each requested output, only the
// operations it depends on in `column_butterflies_and_transpose` +
// `row_butterflies`, each one the original primitive (`AvxVector::add`,
// `sub`, `rotate90`, `mul_complex`, `apply_butterfly8_twiddle1/3`, `neg`) on
// the original operands in the original operand order, with the original
// constant vectors. The primitives are lane-wise (no lane crosses another
// except within one complex value), so a value computed in one lane of a
// vector whose other lanes hold anything is the same bits. What is dropped:
// the outputs of every butterfly no requested output reads, their twiddle
// products, the 8x8 transpose (replaced by storing the live lanes where the
// transpose would have put them), and the half of the row pass whose lanes
// are not requested.

/// Output `p` of `AvxVector::column_butterfly4(rows, rotation)` (its returned
/// order `[o0, o2, o1, o3]`), by the same operations.
#[inline(always)]
unsafe fn bf4_one<V: AvxVector>(rows: [V; 4], rotation: Rotation90<V>, p: usize) -> V {
    match p {
        // [mid0, mid2] = bf2([r0, r2]); [mid1, mid3] = bf2([r1, r3]);
        // o0 = add(mid0, mid1); o1 = sub(mid0, mid1);
        // o2 = add(mid2, rot(mid3)); o3 = sub(mid2, rot(mid3)).
        0 => V::add(V::add(rows[0], rows[2]), V::add(rows[1], rows[3])),
        1 => V::add(V::sub(rows[0], rows[2]), V::sub(rows[1], rows[3]).rotate90(rotation)),
        2 => V::sub(V::add(rows[0], rows[2]), V::add(rows[1], rows[3])),
        _ => V::sub(V::sub(rows[0], rows[2]), V::sub(rows[1], rows[3]).rotate90(rotation)),
    }
}

/// Output `p` of `AvxVector::column_butterfly8(rows, rotation)` (returned
/// order `[o0, o2, o4, o6, o1, o3, o5, o7]`), by the same operations.
#[inline(always)]
pub(super) unsafe fn bf8_one<V: AvxVector>(rows: [V; 8], rotation: Rotation90<V>, p: usize) -> V {
    let i = p % 4;
    let a = bf4_one([rows[0], rows[2], rows[4], rows[6]], rotation, i);
    let mut b = bf4_one([rows[1], rows[3], rows[5], rows[7]], rotation, i);
    b = match i {
        1 => avx_vector::apply_butterfly8_twiddle1(b, rotation),
        2 => b.rotate90(rotation),
        3 => avx_vector::apply_butterfly8_twiddle3(b, rotation),
        _ => b,
    };
    if p < 4 { V::add(a, b) } else { V::sub(a, b) }
}

/// Most requested outputs the requested-output leaves take; larger (and
/// empty) sets use the full leaf and gather.
pub(crate) const PRUNED_LEAF_MAX_NEEDED: usize = 4;

impl<T> Butterfly256Avx<T> {
    /// This leaf has a requested-output path.
    #[inline(always)]
    fn leaf_prunes(&self) -> bool {
        true
    }

    /// `true` when the requested-output leaf computed `out` (1..=4 requested
    /// outputs); `false` sends the caller to the full leaf and gather.
    /// `needed` has already been checked (strictly increasing, < 256).
    #[inline]
    fn pruned_leaf(
        &self,
        input: impl AvxArray<f32>,
        scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) -> bool {
        let ok = !needed.is_empty() && needed.len() <= PRUNED_LEAF_MAX_NEEDED;
        #[cfg(feature = "moosik-diagnostics")]
        crate::moosik_diagnostics::count(if ok { 0 } else { 1 });
        if ok {
            // Safety: instances exist only after AVX and FMA detection (`new`).
            unsafe { self.pruned256(input, scratch, needed, out) };
        }
        ok
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned256(
        &self,
        input: impl AvxArray<f32>,
        _scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) {
        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();

        let mut row_slot = [0u8; 8];
        let mut row_list = [0u8; PRUNED_LEAF_MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut mask = 0u8;
        for &o in needed {
            let r = o & 7;
            if (mask >> r) & 1 == 0 {
                mask |= 1 << r;
                row_slot[r] = num_rows as u8;
                row_list[num_rows] = r as u8;
                num_rows += 1;
            }
        }

        if num_rows == 1 {
            let r = row_list[0] as usize;
            let mut cols = [AvxVector::zero(); 8];
            for (columnset, dst_col) in cols.iter_mut().enumerate() {
                let rows = [
                    input.load_complex(columnset * 4),
                    input.load_complex(columnset * 4 + 32),
                    input.load_complex(columnset * 4 + 64),
                    input.load_complex(columnset * 4 + 96),
                    input.load_complex(columnset * 4 + 128),
                    input.load_complex(columnset * 4 + 160),
                    input.load_complex(columnset * 4 + 192),
                    input.load_complex(columnset * 4 + 224),
                ];
                let mut v: __m256 = bf8_one(rows, rot, r);
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                *dst_col = v;
            }
            for (slot, &o) in out.iter_mut().zip(needed) {
                let v = bf32_one_from_cols(cols, rot, rot128, &self.packed_bf32, o >> 3);
                _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
            }
            return;
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f32>; 32]>::uninit(); PRUNED_LEAF_MAX_NEEDED];
        for columnset in 0..8 {
            let rows = [
                input.load_complex(columnset * 4),
                input.load_complex(columnset * 4 + 32),
                input.load_complex(columnset * 4 + 64),
                input.load_complex(columnset * 4 + 96),
                input.load_complex(columnset * 4 + 128),
                input.load_complex(columnset * 4 + 160),
                input.load_complex(columnset * 4 + 192),
                input.load_complex(columnset * 4 + 224),
            ];
            let mid = AvxVector::column_butterfly8(rows, rot);
            for s in 0..num_rows {
                let r = row_list[s] as usize;
                let mut v = mid[r];
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                let dst = row_bufs[s].as_mut_ptr() as *mut f32;
                _mm256_storeu_ps(dst.add(columnset * 8), v);
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o & 7] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f32;
            let cols = [
                _mm256_loadu_ps(row_ptr),
                _mm256_loadu_ps(row_ptr.add(8)),
                _mm256_loadu_ps(row_ptr.add(16)),
                _mm256_loadu_ps(row_ptr.add(24)),
                _mm256_loadu_ps(row_ptr.add(32)),
                _mm256_loadu_ps(row_ptr.add(40)),
                _mm256_loadu_ps(row_ptr.add(48)),
                _mm256_loadu_ps(row_ptr.add(56)),
            ];
            let v = bf32_one_from_cols(cols, rot, rot128, &self.packed_bf32, o >> 3);
            _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
        }
    }
}

/// The butterfly-16 output `ix` over 4 column vectors `[col0..col3]` (4 complex
/// lanes each), computing `bf4` down 4-element strides in one YMM register and
/// collapsing the remaining 4-point stage in XMM registers without stack reloads.
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn bf16_one_from_cols(
    cols: [__m256; 4],
    rot: Rotation90<__m256>,
    rot128: Rotation90<__m128>,
    tw: &Bf16PackedTwiddles,
    ix: usize,
) -> __m128 {
    let i = ix & 3;
    let q = ix >> 2;
    let v = bf4_one(cols, rot, i);
    let mid = match i {
        0 => v,
        1 => {
            let m = AvxVector::mul_complex(v, tw.i1);
            let w = avx_vector::apply_butterfly8_twiddle1(v, rot);
            let m01 = _mm256_blend_ps::<0x03>(m, v);
            _mm256_blend_ps::<0x30>(m01, w)
        }
        2 => {
            let t1 = avx_vector::apply_butterfly8_twiddle1(v, rot);
            let t2 = v.rotate90(rot);
            let t3 = avx_vector::apply_butterfly8_twiddle3(v, rot);
            let a = _mm256_blend_ps::<0x0C>(v, t1);
            let b = _mm256_blend_ps::<0x30>(a, t2);
            _mm256_blend_ps::<0xC0>(b, t3)
        }
        _ => {
            let m = AvxVector::mul_complex(v, tw.i3);
            let w = avx_vector::apply_butterfly8_twiddle3(v, rot);
            let m01 = _mm256_blend_ps::<0x03>(m, v);
            _mm256_blend_ps::<0x30>(m01, w)
        }
    };
    let lo = _mm256_castps256_ps128(mid);
    let hi = _mm256_extractf128_ps::<1>(mid);
    let p0 = lo;
    let p1 = _mm_movehl_ps(lo, lo);
    let p2 = hi;
    let p3 = _mm_movehl_ps(hi, hi);
    bf4_one([p0, p1, p2, p3], rot128, q)
}

impl<T> Butterfly128Avx<T> {
    /// This leaf has a requested-output path.
    #[inline(always)]
    fn leaf_prunes(&self) -> bool {
        true
    }

    /// As `Butterfly256Avx::pruned_leaf`.
    #[inline]
    fn pruned_leaf(
        &self,
        input: impl AvxArray<f32>,
        scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) -> bool {
        let ok = !needed.is_empty() && needed.len() <= PRUNED_LEAF_MAX_NEEDED;
        if ok {
            // Safety: instances exist only after AVX and FMA detection (`new`).
            unsafe { self.pruned128(input, scratch, needed, out) };
        }
        ok
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned128(
        &self,
        input: impl AvxArray<f32>,
        _scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) {
        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();

        let mut row_slot = [0u8; 8];
        let mut row_list = [0u8; PRUNED_LEAF_MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut mask = 0u8;
        for &o in needed {
            let r = o & 7;
            if (mask >> r) & 1 == 0 {
                mask |= 1 << r;
                row_slot[r] = num_rows as u8;
                row_list[num_rows] = r as u8;
                num_rows += 1;
            }
        }

        if num_rows == 1 {
            let r = row_list[0] as usize;
            let mut cols = [AvxVector::zero(); 4];
            for (columnset, dst_col) in cols.iter_mut().enumerate() {
                let rows = [
                    input.load_complex(columnset * 4),
                    input.load_complex(columnset * 4 + 16),
                    input.load_complex(columnset * 4 + 32),
                    input.load_complex(columnset * 4 + 48),
                    input.load_complex(columnset * 4 + 64),
                    input.load_complex(columnset * 4 + 80),
                    input.load_complex(columnset * 4 + 96),
                    input.load_complex(columnset * 4 + 112),
                ];
                let mut v: __m256 = bf8_one(rows, rot, r);
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                *dst_col = v;
            }
            for (slot, &o) in out.iter_mut().zip(needed) {
                let v = bf16_one_from_cols(cols, rot, rot128, &self.packed_bf16, o >> 3);
                _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
            }
            return;
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f32>; 16]>::uninit(); PRUNED_LEAF_MAX_NEEDED];
        for columnset in 0..4 {
            let rows = [
                input.load_complex(columnset * 4),
                input.load_complex(columnset * 4 + 16),
                input.load_complex(columnset * 4 + 32),
                input.load_complex(columnset * 4 + 48),
                input.load_complex(columnset * 4 + 64),
                input.load_complex(columnset * 4 + 80),
                input.load_complex(columnset * 4 + 96),
                input.load_complex(columnset * 4 + 112),
            ];
            let mid = AvxVector::column_butterfly8(rows, rot);
            for s in 0..num_rows {
                let r = row_list[s] as usize;
                let mut v = mid[r];
                if r >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[r - 1 + 7 * columnset]);
                }
                let dst = row_bufs[s].as_mut_ptr() as *mut f32;
                _mm256_storeu_ps(dst.add(columnset * 8), v);
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o & 7] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f32;
            let cols = [
                _mm256_loadu_ps(row_ptr),
                _mm256_loadu_ps(row_ptr.add(8)),
                _mm256_loadu_ps(row_ptr.add(16)),
                _mm256_loadu_ps(row_ptr.add(24)),
            ];
            let v = bf16_one_from_cols(cols, rot, rot128, &self.packed_bf16, o >> 3);
            _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
        }
    }
}

/// The butterfly-32 output `ix` over 8 column vectors `[col0..col7]` (4 complex
/// lanes each), computing `bf4` on even/odd 128-stride groups in YMMs and
/// collapsing the 8-point stage in a single XMM vector without scalar reloads.
#[inline]
#[target_feature(enable = "avx", enable = "fma")]
unsafe fn bf32_one_from_cols(
    cols: [__m256; 8],
    rot: Rotation90<__m256>,
    rot128: Rotation90<__m128>,
    tw: &Bf32PackedTwiddles,
    ix: usize,
) -> __m128 {
    let i = ix & 3;
    let j = ix >> 2;
    let v_lo = bf4_one([cols[0], cols[2], cols[4], cols[6]], rot, i);
    let v_hi = bf4_one([cols[1], cols[3], cols[5], cols[7]], rot, i);
    let (mid_lo, mid_hi) = match i {
        0 => (v_lo, v_hi),
        1 => (
            _mm256_blend_ps::<0xFC>(v_lo, AvxVector::mul_complex(v_lo, tw.lo[0])),
            _mm256_blend_ps::<0x03>(
                AvxVector::mul_complex(v_hi, tw.hi[0]),
                avx_vector::apply_butterfly8_twiddle1(v_hi, rot),
            ),
        ),
        2 => (
            _mm256_blend_ps::<0x30>(
                _mm256_blend_ps::<0xCC>(v_lo, AvxVector::mul_complex(v_lo, tw.lo[1])),
                avx_vector::apply_butterfly8_twiddle1(v_lo, rot),
            ),
            _mm256_blend_ps::<0x30>(
                _mm256_blend_ps::<0x03>(
                    AvxVector::mul_complex(v_hi, tw.hi[1]),
                    v_hi.rotate90(rot),
                ),
                avx_vector::apply_butterfly8_twiddle3(v_hi, rot),
            ),
        ),
        _ => (
            _mm256_blend_ps::<0xFC>(v_lo, AvxVector::mul_complex(v_lo, tw.lo[2])),
            _mm256_blend_ps::<0x03>(
                AvxVector::mul_complex(v_hi, tw.hi[2]),
                avx_vector::apply_butterfly8_twiddle3(v_hi, rot),
            ),
        ),
    };
    let p0 = _mm256_castps256_ps128(mid_lo);
    let p1 = _mm256_extractf128_ps::<1>(mid_lo);
    let p2 = _mm256_castps256_ps128(mid_hi);
    let p3 = _mm256_extractf128_ps::<1>(mid_hi);
    let ab = bf4_one([p0, p1, p2, p3], rot128, j & 3);
    let mut b = _mm_movehl_ps(ab, ab);
    b = match j & 3 {
        1 => avx_vector::apply_butterfly8_twiddle1(b, rot128),
        2 => b.rotate90(rot128),
        3 => avx_vector::apply_butterfly8_twiddle3(b, rot128),
        _ => b,
    };
    if j < 4 { AvxVector::add(ab, b) } else { AvxVector::sub(ab, b) }
}

impl<T> Butterfly512Avx<T> {
    /// This leaf has a requested-output path.
    #[inline(always)]
    fn leaf_prunes(&self) -> bool {
        true
    }

    /// As `Butterfly256Avx::pruned_leaf`.
    #[inline]
    fn pruned_leaf(
        &self,
        input: impl AvxArray<f32>,
        scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) -> bool {
        let ok = !needed.is_empty() && needed.len() <= PRUNED_LEAF_MAX_NEEDED;
        #[cfg(feature = "moosik-diagnostics")]
        crate::moosik_diagnostics::count(if ok { 2 } else { 3 });
        if ok {
            // Safety: instances exist only after AVX and FMA detection (`new`).
            unsafe { self.pruned512(input, scratch, needed, out) };
        }
        ok
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn pruned512(
        &self,
        input: impl AvxArray<f32>,
        _scratch: &mut [Complex<f32>],
        needed: &[usize],
        out: &mut [Complex<f32>],
    ) {
        let rot = self.twiddles_butterfly4;
        let rot128 = rot.lo();
        let tw16 = self.twiddles_butterfly16;

        let mut row_slot = [0u8; 16];
        let mut row_list = [0u8; PRUNED_LEAF_MAX_NEEDED];
        let mut num_rows = 0usize;
        let mut k_mask = 0u16;
        let mut i_mask = 0u8;
        for &o in needed {
            let k = o & 15;
            if (k_mask >> k) & 1 == 0 {
                k_mask |= 1 << k;
                i_mask |= 1 << (k & 3);
                row_slot[k] = num_rows as u8;
                row_list[num_rows] = k as u8;
                num_rows += 1;
            }
        }

        let twiddle_mid = |v: __m256, m: usize, i: usize| -> __m256 {
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

        if num_rows == 1 {
            let k = row_list[0] as usize;
            let (i, q) = (k & 3, k >> 2);
            let mut cols = [AvxVector::zero(); 8];
            for (columnset, dst_col) in cols.iter_mut().enumerate() {
                let load = |index: usize| -> __m256 { input.load_complex(columnset * 4 + 32 * index) };
                let mut mids = [AvxVector::zero(); 4];
                for m in 0..4 {
                    let v = bf4_one([load(m), load(m + 4), load(m + 8), load(m + 12)], rot, i);
                    mids[m] = twiddle_mid(v, m, i);
                }
                let mut v = bf4_one(mids, rot, q);
                if k >= 1 {
                    v = AvxVector::mul_complex(v, self.twiddles[15 * columnset + k - 1]);
                }
                *dst_col = v;
            }
            for (slot, &o) in out.iter_mut().zip(needed) {
                let v = bf32_one_from_cols(cols, rot, rot128, &self.packed_bf32, o >> 4);
                _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
            }
            return;
        }

        let mut row_bufs = [MaybeUninit::<[Complex<f32>; 32]>::uninit(); PRUNED_LEAF_MAX_NEEDED];
        if i_mask.count_ones() == 1 {
            let i = i_mask.trailing_zeros() as usize;
            for columnset in 0..8 {
                let load = |index: usize| -> __m256 { input.load_complex(columnset * 4 + 32 * index) };
                let mut mids = [AvxVector::zero(); 4];
                for m in 0..4 {
                    let v = bf4_one([load(m), load(m + 4), load(m + 8), load(m + 12)], rot, i);
                    mids[m] = twiddle_mid(v, m, i);
                }
                for s in 0..num_rows {
                    let k = row_list[s] as usize;
                    let mut v = bf4_one(mids, rot, k >> 2);
                    if k >= 1 {
                        v = AvxVector::mul_complex(v, self.twiddles[15 * columnset + k - 1]);
                    }
                    let dst = row_bufs[s].as_mut_ptr() as *mut f32;
                    _mm256_storeu_ps(dst.add(columnset * 8), v);
                }
            }
        } else {
            for columnset in 0..8 {
                let load = |index: usize| -> __m256 { input.load_complex(columnset * 4 + 32 * index) };
                let mut bf4_all = [[AvxVector::zero(); 4]; 4];
                for m in 0..4 {
                    bf4_all[m] = AvxVector::column_butterfly4(
                        [load(m), load(m + 4), load(m + 8), load(m + 12)],
                        rot,
                    );
                }
                for s in 0..num_rows {
                    let k = row_list[s] as usize;
                    let (i, q) = (k & 3, k >> 2);
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
                    let dst = row_bufs[s].as_mut_ptr() as *mut f32;
                    _mm256_storeu_ps(dst.add(columnset * 8), v);
                }
            }
        }

        for (slot, &o) in out.iter_mut().zip(needed) {
            let s = row_slot[o & 15] as usize;
            let row_ptr = row_bufs[s].as_ptr() as *const f32;
            let cols = [
                _mm256_loadu_ps(row_ptr),
                _mm256_loadu_ps(row_ptr.add(8)),
                _mm256_loadu_ps(row_ptr.add(16)),
                _mm256_loadu_ps(row_ptr.add(24)),
                _mm256_loadu_ps(row_ptr.add(32)),
                _mm256_loadu_ps(row_ptr.add(40)),
                _mm256_loadu_ps(row_ptr.add(48)),
                _mm256_loadu_ps(row_ptr.add(56)),
            ];
            let v = bf32_one_from_cols(cols, rot, rot128, &self.packed_bf32, o >> 4);
            _mm_store_sd(slot as *mut Complex<f32> as *mut f64, _mm_castps_pd(v));
        }
    }
}

/// Moosik 1.6.1: the stored coefficients of an original `Butterfly512Avx<f32>`,
/// read back as f32 bits for the exact GPU lane, which evaluates the same
/// graph with the same constants. Read-only; no FFT body changes.
pub struct ExactGpuLeafWords {
    /// `twiddles_butterfly32` (one lane each), the rotation mask, half_root2.
    pub phase2: [u32; 16],
    /// `twiddles_butterfly16`, one lane each.
    pub tw16: [u32; 4],
    /// The separated-column twiddles laid out per element (1 + 0i where none).
    pub outer: Vec<[u32; 2]>,
}

impl ExactGpuLeafWords {
    pub(crate) fn new(direction: FftDirection) -> Option<Self> {
        let p = Butterfly512Avx::<f32>::new(direction).ok()?;
        Some(unsafe { Self::with_avx(&p) })
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn with_avx(p: &Butterfly512Avx<f32>) -> Self {
        let lane0 = |v: __m256| {
            let mut bits = [0u32; 8];
            _mm256_storeu_ps(bits.as_mut_ptr().cast(), v);
            for lane in 1..4 {
                assert_eq!(&bits[0..2], &bits[lane * 2..lane * 2 + 2], "broadcast twiddle lanes differ");
            }
            [bits[0], bits[1]]
        };
        let mut phase2 = [0u32; 16];
        for (k, &v) in p.twiddles_butterfly32.iter().enumerate() {
            phase2[k * 2..k * 2 + 2].copy_from_slice(&lane0(v));
        }
        let masks = p.twiddles_butterfly4.moosik_mask_bits();
        for lane in 1..4 {
            assert_eq!(&masks[0..2], &masks[lane * 2..lane * 2 + 2], "rotation mask lanes differ");
        }
        phase2[12..14].copy_from_slice(&masks[0..2]);
        let h: __m256 = AvxVector::half_root2();
        let mut hwords = [0u32; 8];
        _mm256_storeu_ps(hwords.as_mut_ptr().cast(), h);
        assert!(hwords.iter().all(|&v| v == hwords[0]));
        phase2[14] = hwords[0];
        let mut tw16 = [0u32; 4];
        for (i, &v) in p.twiddles_butterfly16.iter().enumerate() {
            tw16[i * 2..i * 2 + 2].copy_from_slice(&lane0(v));
        }
        let stride = 512 / 32;
        let mut outer = vec![[1.0f32.to_bits(), 0]; 512];
        for c in 0..32 {
            for k in 1..stride {
                let v = p.twiddles[(stride - 1) * (c / 4) + k - 1];
                let mut bits = [0u32; 8];
                _mm256_storeu_ps(bits.as_mut_ptr().cast(), v);
                outer[c * stride + k].copy_from_slice(&bits[(c % 4) * 2..(c % 4) * 2 + 2]);
            }
        }
        Self { phase2, tw16, outer }
    }
}
