//! Optional, bounded kernel preparation and inverse work on a current analysis pool.
//!
//! This helper owns source buffers and all inverse workers instead of keeping
//! them alongside the serial route's buffers. Kernel storage is owned by the
//! caller but charged here. Coefficients are gathered into one preallocated
//! `[bin][channel][cap]` slab, then consumed serially by the caller: no mapping,
//! progress or observer callback crosses a worker boundary. FFT geometry and
//! per-coefficient arithmetic are the serial route's, unchanged.

use super::{
    Bin, Error, Limits, Requested, Window, checked_product, fill_input, frame_center, poll,
    zeroed,
};
use rayon::prelude::*;
use rustfft::{Fft, num_complex::Complex};
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

type C64 = Complex<f64>;

// This admission threshold changes only where the existing kernel work runs.
// Low-N groups retain exactly the same plans, products, output clock and
// workspace budget; sub-512 transforms still use the serial route.
const MIN_PARALLEL_N: usize = 512;

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub slots: usize,
    pub batch_size: usize,
    pub source_blocks: usize,
    /// Dense source/product/scratch/Requested/slab payload plus kernel batch.
    /// Allocator overhead, small metadata and rustfft plan storage are excluded,
    /// as they are by the parent core's existing explicit-workspace contract.
    pub workspace_bytes: usize,
    single_product: bool,
    n: usize,
    channels: usize,
    scratch_len: usize,
    coefficient_cap: usize,
    requested_cap: Option<usize>,
    frames: usize,
    half: usize,
    sample_rate: u32,
    frame_rate: f32,
}

impl Plan {
    /// `None` means use the original serial implementation. In particular,
    /// this never creates or implicitly selects Rayon’s global pool.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn for_group(
        n: usize,
        channels: usize,
        bins: usize,
        frames: usize,
        half: usize,
        sample_rate: u32,
        frame_rate: f32,
        scratch_len: usize,
        inverse: &dyn Fft<f64>,
        limits: Limits,
        allow_pruning: bool,
    ) -> Option<Self> {
        rayon::current_thread_index()?;
        Self::with_workers_mode(
            n,
            channels,
            bins,
            frames,
            half,
            sample_rate,
            frame_rate,
            scratch_len,
            inverse,
            limits,
            allow_pruning,
            rayon::current_num_threads(),
            false,
        )
    }

    /// Parallel plan extended with multi-block source buffering inside the same
    /// explicit workspace budget so kernels stay resident in shared cache across
    /// blocks and source FFTs run across all workers.
    #[allow(clippy::too_many_arguments)]
    pub fn for_group_batched(
        n: usize,
        channels: usize,
        bins: usize,
        frames: usize,
        half: usize,
        sample_rate: u32,
        frame_rate: f32,
        scratch_len: usize,
        inverse: &dyn Fft<f64>,
        limits: Limits,
        allow_pruning: bool,
    ) -> Option<Self> {
        rayon::current_thread_index()?;
        let mut plan = Self::with_workers_mode(
            n,
            channels,
            bins,
            frames,
            half,
            sample_rate,
            frame_rate,
            scratch_len,
            inverse,
            limits,
            allow_pruning,
            rayon::current_num_threads(),
            true,
        )?;
        let num_batches = bins.div_ceil(plan.batch_size).max(1);
        let balanced_batch = bins.div_ceil(num_batches).max(plan.slots).min(plan.batch_size);
        let dense = n.checked_mul(size_of::<C64>())?;
        let sources_1 = channels.checked_mul(dense)?;
        let requested = match plan.requested_cap {
            Some(c) => Requested::bytes(inverse, c, channels).ok()?,
            None => 0,
        };
        let single_slot_bytes = dense
            .checked_add(scratch_len.checked_mul(size_of::<C64>())?)?
            .checked_add(requested)?;
        let slab_per_bin = channels
            .checked_mul(plan.coefficient_cap)?
            .checked_mul(size_of::<C64>())?;
        let base_bytes = sources_1
            .checked_add(plan.slots.checked_mul(single_slot_bytes)?)?
            .checked_add(balanced_batch.checked_mul(dense.checked_add(slab_per_bin)?)?)?;
        if base_bytes > limits.workspace_bytes {
            return Some(plan);
        }
        let headroom = limits.workspace_bytes - base_bytes;
        let payload = n - 2 * half;
        let est_frames_per_block = ((payload as f64 * f64::from(frame_rate)
            / f64::from(sample_rate))
        .floor() as usize)
            .max(1);
        let est_blocks = frames.div_ceil(est_frames_per_block).max(1);
        if num_batches > 1 && dense >= 2 * 1024 * 1024 && est_blocks >= 2 && est_blocks <= 64 {
            let mut actual_blocks = 0usize;
            let mut f = 0usize;
            while f < frames && actual_blocks <= 64 {
                let start = frame_center(f, sample_rate, frame_rate);
                let mut end = f + 1;
                while end < frames
                    && frame_center(end, sample_rate, frame_rate) - start < payload
                {
                    end += 1;
                }
                actual_blocks += 1;
                f = end;
            }
            if actual_blocks >= 2 {
                if let (Some(need_src), Some(slots_total), Some(per_bin_all)) = (
                    actual_blocks.checked_mul(sources_1),
                    plan.slots.checked_mul(single_slot_bytes),
                    actual_blocks.checked_mul(slab_per_bin).and_then(|s| dense.checked_add(s)),
                ) {
                    let fixed_all = need_src.saturating_add(slots_total);
                    if let Some(avail) = limits.workspace_bytes.checked_sub(fixed_all) {
                        let cap_bins = (avail / per_bin_all).min(bins);
                        if cap_bins >= bins.div_ceil(3).max(plan.slots * 3) {
                            let new_batches = bins.div_ceil(cap_bins).max(1);
                            let balanced = bins.div_ceil(new_batches).min(cap_bins);
                            if let Some(total_bytes) = balanced
                                .checked_mul(per_bin_all)
                                .and_then(|b| fixed_all.checked_add(b))
                            {
                                if total_bytes <= limits.workspace_bytes {
                                    plan.batch_size = balanced;
                                    plan.source_blocks = actual_blocks;
                                    plan.single_product = true;
                                    plan.workspace_bytes = total_bytes;
                                    return Some(plan);
                                }
                            }
                        }
                    }
                }
            }
        }
        let target_wave = if dense <= 16 * 1024 {
            256
        } else if dense <= 64 * 1024 {
            64
        } else if dense <= 256 * 1024 {
            16
        } else {
            8
        };
        let max_wave = target_wave.min(est_blocks);
        let per_extra_block = sources_1.checked_add(balanced_batch.checked_mul(slab_per_bin)?)?;
        let extra = if per_extra_block > 0 {
            (headroom / per_extra_block).min(max_wave.saturating_sub(1))
        } else {
            0
        };
        plan.batch_size = balanced_batch;
        plan.source_blocks = 1 + extra;
        plan.single_product = true;
        plan.workspace_bytes = base_bytes + extra * per_extra_block;
        Some(plan)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn with_workers(
        n: usize,
        channels: usize,
        bins: usize,
        frames: usize,
        half: usize,
        sample_rate: u32,
        frame_rate: f32,
        scratch_len: usize,
        inverse: &dyn Fft<f64>,
        limits: Limits,
        allow_pruning: bool,
        workers: usize,
    ) -> Option<Self> {
        Self::with_workers_mode(
            n,
            channels,
            bins,
            frames,
            half,
            sample_rate,
            frame_rate,
            scratch_len,
            inverse,
            limits,
            allow_pruning,
            workers,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_workers_mode(
        n: usize,
        channels: usize,
        bins: usize,
        frames: usize,
        half: usize,
        sample_rate: u32,
        frame_rate: f32,
        scratch_len: usize,
        inverse: &dyn Fft<f64>,
        limits: Limits,
        allow_pruning: bool,
        workers: usize,
        single_product: bool,
    ) -> Option<Self> {
        if n < MIN_PARALLEL_N
            || inverse.len() != n
            || !(1..=2).contains(&channels)
            || bins < 2
            || frames == 0
            || workers < 2
            || sample_rate == 0
            || !frame_rate.is_finite()
            || frame_rate <= 0.0
            || scratch_len < inverse.get_inplace_scratch_len()
        {
            return None;
        }
        let payload = n.checked_sub(half.checked_mul(2)?)?;
        if payload == 0 {
            return None;
        }
        // Unlike the pruning request cap, this must fit *all* outputs in a
        // block, including dense/duplicate schedules that use a whole inverse.
        let cap = ((payload as f64 * f64::from(frame_rate) / f64::from(sample_rate)).ceil()
            as usize)
            .saturating_add(1)
            .min(frames);
        let dense = n.checked_mul(size_of::<C64>())?;
        let sources = channels.checked_mul(dense)?;
        let product_channels = if single_product { 1 } else { channels };
        let product_and_scratch = product_channels
            .checked_mul(dense)?
            .checked_add(scratch_len.checked_mul(size_of::<C64>())?)?;
        let slab_per_bin = channels.checked_mul(cap)?.checked_mul(size_of::<C64>())?;
        let per_bin = dense.checked_add(slab_per_bin)?;
        let sparse_cap = cap.min(n / 4);
        // Optional bookkeeping never makes a previously admitted transform
        // fail. Try without it as well before leaving the whole group serial.
        for requested_cap in [
            (allow_pruning && inverse.prunes()).then_some(sparse_cap),
            None,
        ] {
            let requested = match requested_cap {
                Some(c) => Requested::bytes(inverse, c, channels).ok()?,
                None => 0,
            };
            let slot_bytes = product_and_scratch.checked_add(requested)?;
            let max_slots = if dense >= 4 * 1024 * 1024 {
                workers.min(4).min(bins)
            } else {
                workers.min(bins)
            };
            for slots in (2..=max_slots).rev() {
                let fixed = sources.checked_add(slots.checked_mul(slot_bytes)?)?;
                let Some(available) = limits.workspace_bytes.checked_sub(fixed) else {
                    continue;
                };
                let batch_size = (available / per_bin).min(bins);
                if batch_size < slots {
                    continue;
                }
                return Some(Self {
                    slots,
                    batch_size,
                    source_blocks: 1,
                    workspace_bytes: fixed.checked_add(batch_size.checked_mul(per_bin)?)?,
                    single_product,
                    n,
                    channels,
                    scratch_len,
                    coefficient_cap: cap,
                    requested_cap,
                    frames,
                    half,
                    sample_rate,
                    frame_rate,
                });
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Block {
    pub first: usize,
    pub end: usize,
    pub sample_rate: u32,
    pub frame_rate: f32,
    pub half: usize,
    pub start: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stats {
    pub inverse_ffts: u64,
    pub inverses_pruned: u64,
    pub inverses_full: u64,
    pub pruned_output_values: u64,
    /// Sum of inverse task wall times; tasks overlap, so this is not CPU time.
    pub worker_wall_ns: u128,
    pub worker_tasks: u64,
}

impl Stats {
    fn add(self, rhs: Self) -> Self {
        Self {
            inverse_ffts: self.inverse_ffts + rhs.inverse_ffts,
            inverses_pruned: self.inverses_pruned + rhs.inverses_pruned,
            inverses_full: self.inverses_full + rhs.inverses_full,
            pruned_output_values: self.pruned_output_values + rhs.pruned_output_values,
            worker_wall_ns: self.worker_wall_ns + rhs.worker_wall_ns,
            worker_tasks: self.worker_tasks + rhs.worker_tasks,
        }
    }
}

struct WaveBlock {
    block: Block,
    count: usize,
    prune: bool,
    idx_start: usize,
    idx_len: usize,
}

#[derive(Clone, Copy)]
struct SyncSliceMut {
    ptr: *mut C64,
    len: usize,
}
// Safety: workers access strictly disjoint indices/sub-slices coordinated by atomic task IDs.
unsafe impl Send for SyncSliceMut {}
unsafe impl Sync for SyncSliceMut {}

#[derive(Clone, Copy)]
struct SyncSliceConst {
    ptr: *const C64,
    len: usize,
}
unsafe impl Send for SyncSliceConst {}
unsafe impl Sync for SyncSliceConst {}

#[derive(Clone, Copy)]
struct SyncVecPtr {
    ptr: *mut Vec<C64>,
}
unsafe impl Send for SyncVecPtr {}
unsafe impl Sync for SyncVecPtr {}

struct Slot {
    product_left: Vec<C64>,
    #[allow(dead_code)]
    product_right: Vec<C64>,
    scratch: Vec<C64>,
    requested: Option<Requested>,
    stats: Stats,
}

impl Slot {
    fn new(plan: Plan, inverse: &dyn Fft<f64>) -> Result<Self, Error> {
        Ok(Self {
            product_left: zeroed(plan.n)?,
            product_right: zeroed(
                if !plan.single_product && plan.channels == 2 {
                    plan.n
                } else {
                    0
                },
            )?,
            scratch: zeroed(plan.scratch_len)?,
            requested: plan
                .requested_cap
                .map(|c| Requested::new(inverse, c, plan.channels))
                .transpose()?,
            stats: Stats::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_kernels(
        &mut self,
        bins: &[Bin],
        destination: &mut [Vec<C64>],
        forward: &dyn Fft<f64>,
        plan: Plan,
        window: Window,
        should_continue: &(dyn Fn() -> bool + Sync),
        #[cfg(test)] after_publish: Option<&(dyn Fn(usize) + Sync)>,
    ) -> Result<u64, Error> {
        let mut entered = 0;
        for (&bin, out) in bins.iter().zip(destination.iter_mut()) {
            poll(should_continue)?;
            let mut values = super::kernel(bin, plan.n, plan.sample_rate, window, should_continue)?;
            poll(should_continue)?;
            entered += 1; // Count the actual call, not a planned kernel.
            forward.process_with_scratch(&mut values, &mut self.scratch);
            *out = values; // Empty placeholder takes ownership; no dense clone.
            #[cfg(test)]
            if let Some(after_publish) = after_publish { after_publish(bin.index); }
            poll(should_continue)?;
        }
        Ok(entered)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_task(
        &mut self,
        source: &[C64],
        kernel: &[C64],
        out: &mut [C64],
        inverse: &dyn Fft<f64>,
        plan: Plan,
        wb: &WaveBlock,
        indices: &[usize],
        aborted: &AtomicBool,
        should_continue: &(dyn Fn() -> bool + Sync),
    ) -> Result<(), Error> {
        if wb.prune {
            let r = self.requested.as_mut().expect("prepared above");
            self.stats.inverse_ffts += 1;
            self.stats.inverses_pruned += 1;
            self.stats.pruned_output_values += wb.count as u64;
            inverse.process_pruned_product_with_scratch(
                source,
                kernel,
                &mut self.product_left,
                &mut self.scratch,
                indices,
                out,
                rustfft::PrunedWork {
                    idx: &mut r.work_indices,
                    val: &mut r.work_values,
                },
            );
        } else {
            // Frozen Mul semantics, not a new multiplication/FMA route.
            for i in 0..plan.n {
                if i % 4096 == 0 && !should_continue() {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(Error::Aborted);
                }
                self.product_left[i] = source[i] * kernel[i];
            }
            self.stats.inverse_ffts += 1;
            self.stats.inverses_full += 1;
            inverse.process_with_scratch(&mut self.product_left, &mut self.scratch);
            for (offset, frame) in (wb.block.first..wb.block.end).enumerate() {
                let at = wb.block.half
                    + frame_center(frame, wb.block.sample_rate, wb.block.frame_rate)
                    - wb.block.start;
                out[offset] = self.product_left[at];
            }
        }
        Ok(())
    }
}

pub(super) struct Workspace {
    plan: Plan,
    source_left: Vec<C64>,
    source_right: Vec<C64>,
    extra_sources: Vec<(Vec<C64>, Vec<C64>)>,
    wave_meta: Vec<WaveBlock>,
    wave_indices: Vec<usize>,
    frame_loc_scratch: Vec<(usize, usize)>,
    bin_indices_scratch: Vec<usize>,
    slots: Vec<Slot>,
    coefficients: Vec<C64>,
    active_bins: usize,
    active_blocks: usize,
    active_count: usize,
    phase_timing: bool,
}

impl Workspace {
    pub fn new(mut plan: Plan, inverse: &dyn Fft<f64>) -> Result<Self, Error> {
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(plan.slots)
            .map_err(|_| Error::Allocation)?;
        for _ in 0..plan.slots {
            slots.push(Slot::new(plan, inverse)?);
        }
        let source_left = zeroed(plan.n)?;
        let source_right = zeroed(if plan.channels == 2 { plan.n } else { 0 })?;

        let try_wave = |blocks: usize| -> Result<(Vec<(Vec<C64>, Vec<C64>)>, Vec<C64>), Error> {
            let mut extra = Vec::new();
            extra
                .try_reserve_exact(blocks.saturating_sub(1))
                .map_err(|_| Error::Allocation)?;
            for _ in 1..blocks {
                let l = zeroed(plan.n)?;
                let r = zeroed(if plan.channels == 2 { plan.n } else { 0 })?;
                extra.push((l, r));
            }
            let coeffs = zeroed(checked_product(
                checked_product(plan.batch_size, blocks)?,
                checked_product(plan.channels, plan.coefficient_cap)?,
            )?)?;
            Ok((extra, coeffs))
        };

        let (extra_sources, coefficients) = if plan.source_blocks > 1 {
            match try_wave(plan.source_blocks) {
                Ok(pair) => pair,
                Err(Error::Allocation) => {
                    let per_block_src = plan.channels * plan.n * size_of::<C64>();
                    let per_block_slab =
                        plan.batch_size * plan.channels * plan.coefficient_cap * size_of::<C64>();
                    let saved = (plan.source_blocks - 1) * (per_block_src + per_block_slab);
                    plan.source_blocks = 1;
                    plan.workspace_bytes = plan.workspace_bytes.saturating_sub(saved);
                    try_wave(1)?
                }
                Err(e) => return Err(e),
            }
        } else {
            try_wave(1)?
        };

        Ok(Self {
            plan,
            source_left,
            source_right,
            extra_sources,
            wave_meta: Vec::new(),
            wave_indices: Vec::new(),
            frame_loc_scratch: Vec::new(),
            bin_indices_scratch: Vec::new(),
            slots,
            coefficients,
            active_bins: 0,
            active_blocks: 0,
            active_count: 0,
            phase_timing: false,
        })
    }

    pub fn workspace_bytes(&self) -> usize {
        self.plan.workspace_bytes
    }

    pub fn set_phase_timing(&mut self, enabled: bool) {
        self.phase_timing = enabled;
    }

    /// Reuse slot zero's scratch while preparing the immutable source FFTs.
    #[cfg(test)]
    pub fn source_buffers(&mut self) -> (&mut [C64], &mut [C64], &mut [C64]) {
        (
            &mut self.source_left,
            &mut self.source_right,
            &mut self.slots[0].scratch,
        )
    }

    /// Fill and forward-transform `blocks.len()` source blocks across workers.
    pub fn prepare_sources_wave(
        &mut self,
        blocks: &[Block],
        left: &[f32],
        right: Option<&[f32]>,
        forward: &dyn Fft<f64>,
        should_continue: &(dyn Fn() -> bool + Sync),
    ) -> Result<(), Error> {
        poll(should_continue)?;
        if blocks.is_empty()
            || blocks.len() > self.plan.source_blocks
            || forward.len() != self.plan.n
            || right.is_some() != (self.plan.channels == 2)
        {
            return Err(Error::Invalid("CQT source wave does not match its plan"));
        }
        let n = self.plan.n;
        let channels = self.plan.channels;
        let total_tasks = blocks.len() * channels;
        if total_tasks == 1 || rayon::current_thread_index().is_none() || self.slots.len() < 2 {
            let scratch = &mut self.slots[0].scratch;
            for (b_idx, block) in blocks.iter().enumerate() {
                let (dst_l, dst_r) = if b_idx == 0 {
                    (&mut self.source_left, &mut self.source_right)
                } else {
                    let pair = &mut self.extra_sources[b_idx - 1];
                    (&mut pair.0, &mut pair.1)
                };
                fill_input(dst_l, left, block.start, block.half, should_continue)?;
                forward.process_with_scratch(dst_l, scratch);
                if let Some(right) = right {
                    fill_input(dst_r, right, block.start, block.half, should_continue)?;
                    forward.process_with_scratch(dst_r, scratch);
                }
            }
            return Ok(());
        }

        let mut targets = Vec::with_capacity(total_tasks);
        for (b_idx, block) in blocks.iter().enumerate() {
            let (dst_l, dst_r) = if b_idx == 0 {
                (&mut self.source_left, &mut self.source_right)
            } else {
                let pair = &mut self.extra_sources[b_idx - 1];
                (&mut pair.0, &mut pair.1)
            };
            targets.push((
                SyncSliceMut {
                    ptr: dst_l.as_mut_ptr(),
                    len: n,
                },
                block.start,
                block.half,
                false,
            ));
            if channels == 2 {
                targets.push((
                    SyncSliceMut {
                        ptr: dst_r.as_mut_ptr(),
                        len: n,
                    },
                    block.start,
                    block.half,
                    true,
                ));
            }
        }
        let next_task = AtomicUsize::new(0);
        let aborted = AtomicBool::new(false);
        let targets_ref = &targets;
        self.slots.par_iter_mut().try_for_each(|slot| {
            loop {
                if aborted.load(Ordering::Relaxed) {
                    return Err(Error::Aborted);
                }
                if !should_continue() {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(Error::Aborted);
                }
                let id = next_task.fetch_add(1, Ordering::Relaxed);
                if id >= targets_ref.len() {
                    return Ok(());
                }
                let (target, start, half, is_right) = targets_ref[id];
                let dst = unsafe { std::slice::from_raw_parts_mut(target.ptr, target.len) };
                let src = if is_right {
                    right.expect("stereo channel present")
                } else {
                    left
                };
                fill_input(dst, src, start, half, should_continue)?;
                if !should_continue() {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(Error::Aborted);
                }
                forward.process_with_scratch(dst, &mut slot.scratch);
            }
        })?;
        poll(should_continue)?;
        Ok(())
    }

    /// Build the ordered bank using only already-admitted slot scratch. The
    /// returned dense vectors are exactly the kernel batch charged by Plan.
    /// All preparation joins before source FFTs may borrow slot zero's scratch.
    pub fn prepare_kernels(
        &mut self,
        bins: &[Bin],
        forward: &dyn Fft<f64>,
        window: Window,
        should_continue: &(dyn Fn() -> bool + Sync),
    ) -> Result<(Vec<Vec<C64>>, u64), Error> {
        self.prepare_kernels_observed(bins, forward, window, should_continue,
            #[cfg(test)] None,
        )
    }

    fn prepare_kernels_observed(
        &mut self,
        bins: &[Bin],
        forward: &dyn Fft<f64>,
        window: Window,
        should_continue: &(dyn Fn() -> bool + Sync),
        #[cfg(test)] after_publish: Option<&(dyn Fn(usize) + Sync)>,
    ) -> Result<(Vec<Vec<C64>>, u64), Error> {
        self.active_bins = 0;
        self.active_blocks = 0;
        self.active_count = 0;
        poll(should_continue)?;
        // Plan admission checks inverse scratch. The parent normally passes
        // max(forward, inverse); validate the *forward* requirement as well,
        // including every allocated slot, before building any dense kernel.
        if bins.is_empty()
            || bins.len() > self.plan.batch_size
            || forward.len() != self.plan.n
            || forward.fft_direction() != rustfft::FftDirection::Forward
            || self.slots.len() != self.plan.slots
            || self.slots.iter().any(|slot| slot.scratch.len() < forward.get_inplace_scratch_len())
            || bins.iter().any(|bin| bin.half > self.plan.half
                || !bin.frequency.is_finite() || bin.frequency <= 0.0
                || bin.frequency >= f64::from(self.plan.sample_rate) * 0.5)
        {
            return Err(Error::Invalid("CQT kernel preparation does not match its admitted workspace"));
        }
        let mut kernels = Vec::new();
        kernels.try_reserve_exact(bins.len()).map_err(|_| Error::Allocation)?;
        kernels.resize_with(bins.len(), Vec::new);
        let plan = self.plan;
        let entered = if rayon::current_thread_index().is_some() && self.plan.slots > 1 {
            let next_bin = AtomicUsize::new(0);
            let aborted = AtomicBool::new(false);
            let entered_sum = std::sync::atomic::AtomicU64::new(0);
            let dst_ptr = SyncVecPtr { ptr: kernels.as_mut_ptr() };
            self.slots.par_iter_mut().try_for_each(|slot| {
                let ptr = dst_ptr;
                let mut local_entered = 0u64;
                loop {
                    if aborted.load(Ordering::Relaxed) || !should_continue() {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Aborted);
                    }
                    let id = next_bin.fetch_add(1, Ordering::Relaxed);
                    if id >= bins.len() {
                        break;
                    }
                    let bin = bins[id];
                    let mut values = match super::kernel(bin, plan.n, plan.sample_rate, window, should_continue) {
                        Ok(v) => v,
                        Err(Error::Aborted) => {
                            aborted.store(true, Ordering::Relaxed);
                            return Err(Error::Aborted);
                        }
                        Err(e) => return Err(e),
                    };
                    if !should_continue() {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Aborted);
                    }
                    local_entered += 1;
                    forward.process_with_scratch(&mut values, &mut slot.scratch);
                    unsafe {
                        *ptr.ptr.add(id) = values;
                    }
                    #[cfg(test)]
                    if let Some(after_publish) = after_publish {
                        after_publish(bin.index);
                    }
                    if !should_continue() {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Aborted);
                    }
                }
                entered_sum.fetch_add(local_entered, Ordering::Relaxed);
                Ok(())
            })?;
            entered_sum.load(Ordering::Relaxed)
        } else {
            // A moved workspace does not implicitly start Rayon's global pool.
            let chunk = bins.len().div_ceil(self.plan.slots);
            let mut entered = 0;
            for ((slot, bins), destination) in self.slots.iter_mut()
                .zip(bins.chunks(chunk)).zip(kernels.chunks_mut(chunk))
            {
                entered += slot.prepare_kernels(bins, destination, forward, plan, window,
                    should_continue, #[cfg(test)] after_publish,
                )?;
            }
            entered
        };
        poll(should_continue)?;
        Ok((kernels, entered))
    }

    /// No explicit FFT/slab buffer allocation, mapping or progress callback
    /// occurs here. Rayon may allocate its own scheduling metadata.
    /// On cancellation no coefficient slab is made available to the caller.
    #[cfg(test)]
    pub fn process_block(
        &mut self,
        kernels: &[Vec<C64>],
        inverse: &dyn Fft<f64>,
        block: Block,
        should_continue: &(dyn Fn() -> bool + Sync),
    ) -> Result<Stats, Error> {
        self.process_wave(kernels, inverse, std::slice::from_ref(&block), should_continue)
    }

    /// Execute `kernels` x `blocks` x `channels` using kernel-outer dynamic
    /// work stealing so each kernel stays resident in shared L3 across blocks
    /// and channels, with zero static tail imbalance across workers.
    pub fn process_wave(
        &mut self,
        kernels: &[Vec<C64>],
        inverse: &dyn Fft<f64>,
        blocks: &[Block],
        should_continue: &(dyn Fn() -> bool + Sync),
    ) -> Result<Stats, Error> {
        self.active_bins = 0;
        self.active_blocks = 0;
        self.active_count = 0;
        poll(should_continue)?;
        if blocks.is_empty()
            || blocks.len() > self.plan.source_blocks
            || kernels.is_empty()
            || kernels.len() > self.plan.batch_size
            || kernels.iter().any(|h| h.len() != self.plan.n)
            || inverse.len() != self.plan.n
        {
            return Err(Error::Invalid("CQT parallel wave does not match its plan"));
        }
        let max_requested = self.slots[0].requested.as_ref().map_or(0, |r| r.indices.len());
        self.wave_meta.clear();
        self.wave_indices.clear();
        for &block in blocks {
            let count = block
                .end
                .checked_sub(block.first)
                .ok_or(Error::Invalid("CQT block runs backwards"))?;
            if count == 0
                || block.end > self.plan.frames
                || count > self.plan.coefficient_cap
                || block.half != self.plan.half
                || block.sample_rate != self.plan.sample_rate
                || block.frame_rate.to_bits() != self.plan.frame_rate.to_bits()
                || block.start != frame_center(block.first, block.sample_rate, block.frame_rate)
            {
                return Err(Error::Invalid("CQT parallel block does not match its plan"));
            }
            let last = block
                .half
                .checked_add(
                    frame_center(block.end - 1, block.sample_rate, block.frame_rate)
                        .checked_sub(block.start)
                        .ok_or(Error::Invalid("CQT block center precedes its input"))?,
                )
                .ok_or(Error::Resource("CQT block center overflows"))?;
            if last >= self.plan.n {
                return Err(Error::Invalid(
                    "CQT requested sample is outside its transform",
                ));
            }
            let idx_start = self.wave_indices.len();
            let prune = if max_requested > 0 && count <= max_requested {
                for frame in block.first..block.end {
                    self.wave_indices.push(
                        block.half
                            + frame_center(frame, block.sample_rate, block.frame_rate)
                            - block.start,
                    );
                }
                self.wave_indices[idx_start..]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            } else {
                false
            };
            let idx_len = if prune {
                self.wave_indices.len() - idx_start
            } else {
                self.wave_indices.truncate(idx_start);
                0
            };
            self.wave_meta.push(WaveBlock {
                block,
                count,
                prune,
                idx_start,
                idx_len,
            });
        }
        for slot in &mut self.slots {
            slot.stats = Stats::default();
        }
        let active_bins = kernels.len();
        let active_blocks = blocks.len();
        let channels = self.plan.channels;
        let cap = self.plan.coefficient_cap;
        let needed_len = active_blocks * active_bins * channels * cap;
        let out_slab = SyncSliceMut {
            ptr: self.coefficients.as_mut_ptr(),
            len: needed_len,
        };
        assert!(needed_len <= self.coefficients.len());

        let mut sources = Vec::with_capacity(active_blocks * channels);
        for b_idx in 0..active_blocks {
            let (l, r) = if b_idx == 0 {
                (&self.source_left, &self.source_right)
            } else {
                let p = &self.extra_sources[b_idx - 1];
                (&p.0, &p.1)
            };
            sources.push(SyncSliceConst {
                ptr: l.as_ptr(),
                len: l.len(),
            });
            if channels == 2 {
                sources.push(SyncSliceConst {
                    ptr: r.as_ptr(),
                    len: r.len(),
                });
            }
        }

        let plan = self.plan;
        let phase_timing = self.phase_timing;
        let wave_meta = &self.wave_meta;
        let wave_indices = &self.wave_indices;
        let sources_ref = &sources;
        let blocks_ch = active_blocks * channels;
        let total_tasks = active_bins * blocks_ch;
        let dense = plan.n * size_of::<C64>();
        // Process L3-sized source tiles across all kernels in the batch so large source
        // spectra stay resident in shared L3 across bins instead of thrashing.
        let tile_src = if blocks_ch.saturating_mul(dense) > 2 * 1024 * 1024 && active_bins >= 8 {
            ((4 * 1024 * 1024) / dense).max(2).min(blocks_ch)
        } else {
            blocks_ch
        };
        let full_tiles = blocks_ch / tile_src;
        let per_tile = tile_src * active_bins;
        let full_tasks = full_tiles * per_tile;
        let tail_src = blocks_ch - full_tiles * tile_src;
        let next_task = AtomicUsize::new(0);
        let aborted = AtomicBool::new(false);

        let run_worker = |slot: &mut Slot| -> Result<(), Error> {
            let slab = out_slab;
            let phase = super::phases::clock(phase_timing);
            let mut did_work = false;
            loop {
                if aborted.load(Ordering::Relaxed) {
                    return Err(Error::Aborted);
                }
                if !should_continue() {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(Error::Aborted);
                }
                let id = next_task.fetch_add(1, Ordering::Relaxed);
                if id >= total_tasks {
                    break;
                }
                did_work = true;
                let (bin, rem) = if id < full_tasks {
                    let t = id / per_tile;
                    let in_tile = id % per_tile;
                    (in_tile / tile_src, t * tile_src + (in_tile % tile_src))
                } else {
                    let in_tail = id - full_tasks;
                    (in_tail / tail_src, full_tiles * tile_src + (in_tail % tail_src))
                };
                let b_idx = rem / channels;
                let ch = rem % channels;
                let wb = &wave_meta[b_idx];
                let src_slice = sources_ref[rem];
                let source = unsafe { std::slice::from_raw_parts(src_slice.ptr, src_slice.len) };
                let out_off = ((b_idx * active_bins + bin) * channels + ch) * cap;
                assert!(out_off + wb.count <= slab.len);
                let out = unsafe { std::slice::from_raw_parts_mut(slab.ptr.add(out_off), wb.count) };
                slot.run_task(
                    source,
                    &kernels[bin],
                    out,
                    inverse,
                    plan,
                    wb,
                    &wave_indices[wb.idx_start..wb.idx_start + wb.idx_len],
                    &aborted,
                    should_continue,
                )?;
            }
            if phase_timing && did_work {
                slot.stats.worker_wall_ns += super::phases::elapsed(phase);
                slot.stats.worker_tasks += 1;
            }
            Ok(())
        };

        // A moved workspace does not implicitly start Rayon's global pool.
        if rayon::current_thread_index().is_some() && self.plan.slots > 1 {
            self.slots.par_iter_mut().try_for_each(run_worker)?;
        } else {
            run_worker(&mut self.slots[0])?;
        }
        poll(should_continue)?;
        self.active_bins = active_bins;
        self.active_blocks = active_blocks;
        self.active_count = self.wave_meta[0].count;
        Ok(self
            .slots
            .iter()
            .fold(Stats::default(), |s, slot| s.add(slot.stats)))
    }

    pub fn wave_coefficients(&self, block: usize, bin: usize, channel: usize, count: usize) -> &[C64] {
        assert!(block < self.active_blocks && bin < self.active_bins && channel < self.plan.channels);
        let at = ((block * self.active_bins + bin) * self.plan.channels + channel)
            * self.plan.coefficient_cap;
        &self.coefficients[at..at + count]
    }

    /// Precompute normalized mixed and channel magnitudes across workers in-place
    /// in `self.coefficients` (`f64::from(f32)` is lossless bit-for-bit).
    /// Stereo stores `(mix, left)` in channel 0 `(re, im)` and `right` in channel 1 `.re`.
    pub(super) fn fold_wave_magnitudes(&mut self, norm: f64) {
        let active_bins = self.active_bins;
        let channels = self.plan.channels;
        let cap = self.plan.coefficient_cap;
        let stride = channels * cap;
        let slice_len = self.active_blocks * active_bins * stride;
        let wave_meta = &self.wave_meta;
        let coeffs = &mut self.coefficients[..slice_len];
        let work = |(idx, chunk): (usize, &mut [C64])| {
            let blk_idx = idx / active_bins;
            let count = wave_meta[blk_idx].count;
            if channels == 2 {
                let (l_slice, r_slice) = chunk.split_at_mut(cap);
                for rel in 0..count {
                    let l = l_slice[rel] * norm;
                    let r = r_slice[rel] * norm;
                    let mix_mag = ((l + r) * 0.5).norm() as f32;
                    let l_mag = l.norm() as f32;
                    let r_mag = r.norm() as f32;
                    l_slice[rel] = C64::new(f64::from(mix_mag), f64::from(l_mag));
                    r_slice[rel].re = f64::from(r_mag);
                }
            } else {
                for v in &mut chunk[..count] {
                    let l = *v * norm;
                    v.re = f64::from(l.norm() as f32);
                }
            }
        };
        if self.active_blocks * self.active_bins >= 256
            && rayon::current_thread_index().is_some()
            && self.plan.slots > 1
        {
            coeffs.par_chunks_mut(stride).enumerate().for_each(work);
        } else {
            coeffs.chunks_mut(stride).enumerate().for_each(work);
        }
    }

    pub(super) fn write_wave_codes<M: Fn(f32) -> u16 + Sync>(
        &mut self,
        codes: &mut [Vec<u16>; 3],
        bins: &[Bin],
        bars: usize,
        norm: f64,
        map: &M,
    ) {
        let active_bins = self.active_bins;
        let active_blocks = self.active_blocks;
        if active_bins == 0 || active_blocks == 0 {
            return;
        }
        debug_assert_eq!(bins.len(), active_bins);
        let Self {
            plan,
            wave_meta,
            frame_loc_scratch,
            bin_indices_scratch,
            coefficients: coeffs,
            ..
        } = self;
        let channels = plan.channels;
        let cap = plan.coefficient_cap;
        let stride = channels * cap;
        let wave_meta = &wave_meta[..active_blocks];
        let w_first = wave_meta[0].block.first;
        let w_last = wave_meta[active_blocks - 1].block.end;
        let total_frames = w_last - w_first;
        frame_loc_scratch.clear();
        frame_loc_scratch.reserve(total_frames.saturating_sub(frame_loc_scratch.capacity()));
        for (blk_idx, wb) in wave_meta.iter().enumerate() {
            for rel in 0..wb.count {
                frame_loc_scratch.push((blk_idx, rel));
            }
        }
        bin_indices_scratch.clear();
        bin_indices_scratch.extend(bins.iter().map(|b| b.index));
        let frame_loc = &frame_loc_scratch[..];
        let bin_indices = &bin_indices_scratch[..];
        let start_elem = w_first * bars;
        let end_elem = w_last * bars;
        let use_par = total_frames * active_bins >= 2048
            && rayon::current_thread_index().is_some()
            && plan.slots > 1;
        let min_rows = (2048 / active_bins.max(1)).clamp(1, 32);

        if channels == 2 {
            let (c0, rest) = codes.split_at_mut(1);
            let (c1, c2) = rest.split_at_mut(1);
            let s0 = &mut c0[0][start_elem..end_elem];
            let s1 = &mut c1[0][start_elem..end_elem];
            let s2 = &mut c2[0][start_elem..end_elem];
            let write_row = |(f_off, ((row0, row1), row2)): (usize, ((&mut [u16], &mut [u16]), &mut [u16]))| {
                let (blk_idx, rel) = frame_loc[f_off];
                let blk_base = blk_idx * active_bins * stride;
                for (k, &bar_idx) in bin_indices.iter().enumerate() {
                    let base = blk_base + k * stride;
                    let l = coeffs[base + rel] * norm;
                    let r = coeffs[base + cap + rel] * norm;
                    let m = (l + r) * 0.5;
                    // `norm` (a hypot), as the serial writer takes it: a sqrt
                    // of the squares can round differently.
                    row0[bar_idx] = map(m.norm() as f32);
                    row1[bar_idx] = map(l.norm() as f32);
                    row2[bar_idx] = map(r.norm() as f32);
                }
            };
            if use_par {
                s0.par_chunks_mut(bars)
                    .zip(s1.par_chunks_mut(bars))
                    .zip(s2.par_chunks_mut(bars))
                    .with_min_len(min_rows)
                    .enumerate()
                    .for_each(write_row);
            } else {
                s0.chunks_mut(bars)
                    .zip(s1.chunks_mut(bars))
                    .zip(s2.chunks_mut(bars))
                    .enumerate()
                    .for_each(write_row);
            }
        } else {
            let s0 = &mut codes[0][start_elem..end_elem];
            let write_row = |(f_off, row0): (usize, &mut [u16])| {
                let (blk_idx, rel) = frame_loc[f_off];
                let blk_base = blk_idx * active_bins * stride;
                for (k, &bar_idx) in bin_indices.iter().enumerate() {
                    let l = coeffs[blk_base + k * stride + rel] * norm;
                    row0[bar_idx] = map(l.norm() as f32);
                }
            };
            if use_par {
                s0.par_chunks_mut(bars)
                    .with_min_len(min_rows)
                    .enumerate()
                    .for_each(write_row);
            } else {
                s0.chunks_mut(bars)
                    .enumerate()
                    .for_each(write_row);
            }
        }
    }

    /// Unnormalized complex samples, in the original frame order. The caller
    /// retains its exact `* (1/N)`, Mix, norm, cast and map operation order.
    #[cfg(test)]
    pub fn coefficients(&self, bin: usize, channel: usize, count: usize) -> &[C64] {
        assert!(count <= self.active_count);
        self.wave_coefficients(0, bin, channel, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{FftDirection, FftPlannerScalar};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn preparation_bins(n: usize, count: usize, window: Window) -> Vec<Bin> {
        let config = super::super::Config { bins_per_octave: 12, window, frame_rate: 180.0, bass_limit_s: None };
        let base = config.support_q() * 48_000.0 / (2.0 * (n * 3 / 16) as f64);
        let frequencies: Vec<_> = (0..count).map(|i| base * (1.0 + i as f64 * 0.002)).collect();
        let mut groups = super::super::layout(&frequencies, 48_000, &config, Limits::default()).unwrap();
        assert_eq!(groups.len(), 1);
        groups.remove(&n).expect("preparation fixture must select its named N")
    }

    fn assert_kernel_bits(left: &[Vec<C64>], right: &[Vec<C64>]) {
        assert_eq!(left.len(), right.len());
        for (bin, (left, right)) in left.iter().zip(right).enumerate() {
            assert_eq!(left.len(), right.len());
            for (tap, (left, right)) in left.iter().zip(right).enumerate() {
                assert_eq!((left.re.to_bits(), left.im.to_bits()),
                    (right.re.to_bits(), right.im.to_bits()), "kernel {bin}, FFT position {tap}");
            }
        }
    }

    #[test]
    fn prepared_kernels_keep_order_exact_bits_and_the_admitted_scratch_allocations() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        for n in [512, 4096, 32768] {
            let mut planner = rustfft::FftPlanner::<f64>::new();
            let forward = planner.plan_fft_forward(n);
            let inverse = planner.plan_fft_inverse(n);
            let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
            for window in [Window::Hann, Window::BlackmanHarris] {
                for count in [7, 13] {
                    let bins = preparation_bins(n, count, window);
                    let half = bins.iter().map(|bin| bin.half).max().unwrap();
                    let mut serial_scratch = vec![C64::default(); scratch_len];
                    let expected: Vec<_> = bins.iter().map(|&bin| {
                        let mut values = super::super::kernel(bin, n, 48_000, window, &|| true).unwrap();
                        forward.process_with_scratch(&mut values, &mut serial_scratch);
                        values
                    }).collect();
                    for channels in [1, 2] {
                        let mut workspace = pool.install(|| {
                            let ample = Plan::for_group(n, channels, count, 64, half, 48_000, 180.0,
                                scratch_len, &*inverse, Limits::default(), true).unwrap();
                            // Re-admit at exactly this layout's payload limit,
                            // not an unrelated large allowance.
                            let limits = Limits { workspace_bytes: ample.workspace_bytes, ..Limits::default() };
                            let p = Plan::for_group(n, channels, count, 64, half, 48_000, 180.0,
                                scratch_len, &*inverse, limits, true).unwrap();
                            assert_eq!((p.slots, p.batch_size), (6, count));
                            assert!(p.workspace_bytes <= limits.workspace_bytes);
                            let mut workspace = Workspace::new(p, &*inverse).unwrap();
                            let allocations: Vec<_> = workspace.slots.iter().map(|slot|
                                (slot.scratch.as_ptr() as usize, slot.scratch.len(), slot.scratch.capacity())).collect();
                            for active in [count, 5, 1] {
                                let (got, entered) = workspace.prepare_kernels(&bins[..active], &*forward, window, &|| true).unwrap();
                                assert_eq!(entered, active as u64);
                                assert_kernel_bits(&got, &expected[..active]);
                                assert_eq!(allocations, workspace.slots.iter().map(|slot|
                                    (slot.scratch.as_ptr() as usize, slot.scratch.len(), slot.scratch.capacity())).collect::<Vec<_>>());
                                assert!(workspace.slots.iter().all(|slot| slot.stats == Stats::default()),
                                    "kernel prep polluted inverse/phase counters");
                            }
                            workspace
                        });
                        // Moving an admitted workspace out of its pool must
                        // not silently submit preparation to a global pool.
                        let (got, entered) = workspace.prepare_kernels(&bins, &*forward, window,
                            &|| { assert!(rayon::current_thread_index().is_none()); true }).unwrap();
                        assert_eq!(entered, count as u64);
                        assert_kernel_bits(&got, &expected);
                    }
                }
            }
        }
    }

    struct MustNotRunForward {
        n: usize,
        scratch: usize,
        calls: AtomicUsize,
    }

    impl rustfft::Length for MustNotRunForward { fn len(&self) -> usize { self.n } }
    impl rustfft::Direction for MustNotRunForward {
        fn fft_direction(&self) -> FftDirection { FftDirection::Forward }
    }
    impl Fft<f64> for MustNotRunForward {
        fn process_with_scratch(&self, _: &mut [C64], _: &mut [C64]) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            panic!("invalid preparation entered its FFT");
        }
        fn process_outofplace_with_scratch(&self, _: &mut [C64], _: &mut [C64], _: &mut [C64]) { panic!("unused"); }
        fn process_immutable_with_scratch(&self, _: &[C64], _: &mut [C64], _: &mut [C64]) { panic!("unused"); }
        fn get_inplace_scratch_len(&self) -> usize { self.scratch }
        fn get_outofplace_scratch_len(&self) -> usize { self.scratch }
        fn get_immutable_scratch_len(&self) -> usize { self.scratch }
    }

    #[test]
    fn preparation_validates_every_forward_scratch_and_plan_before_work() {
        let mut planner = FftPlannerScalar::<f64>::new();
        let real_forward = planner.plan_fft(4096, FftDirection::Forward);
        let inverse = planner.plan_fft(4096, FftDirection::Inverse);
        let bins = preparation_bins(4096, 4, Window::Hann);
        let p = Plan::with_workers(4096, 2, bins.len(), 64, bins.iter().map(|b| b.half).max().unwrap(),
            48_000, 180.0, inverse.get_inplace_scratch_len().max(real_forward.get_inplace_scratch_len()).max(1), &*inverse,
            Limits { workspace_bytes: 64 << 20, ..Limits::default() }, true, 6).unwrap();
        let mut workspace = Workspace::new(p, &*inverse).unwrap();
        let forward = MustNotRunForward { n: 4096, scratch: p.scratch_len, calls: AtomicUsize::new(0) };
        let short = workspace.slots.last_mut().unwrap();
        short.scratch.truncate(p.scratch_len - 1);
        workspace.active_bins = 1;
        workspace.active_count = 1;
        // A one-bin tail would use only slot zero: still validate the final
        // allocated slot rather than accidentally testing only active slots.
        assert!(matches!(workspace.prepare_kernels(&bins[..1], &forward, Window::Hann, &|| true), Err(Error::Invalid(_))));
        assert_eq!((workspace.active_bins, workspace.active_count), (0, 0));
        workspace.slots.last_mut().unwrap().scratch.resize(p.scratch_len, C64::default());
        let (valid, entered) = workspace.prepare_kernels(&bins, &*real_forward, Window::Hann, &|| true).unwrap();
        assert_eq!((valid.len(), entered), (bins.len(), bins.len() as u64),
            "matching plan/bins and repaired scratch must be a valid positive control");
        let wrong_length = MustNotRunForward { n: 2048, scratch: p.scratch_len, calls: AtomicUsize::new(0) };
        assert!(matches!(workspace.prepare_kernels(&bins, &wrong_length, Window::Hann, &|| true), Err(Error::Invalid(_))));
        assert!(matches!(workspace.prepare_kernels(&bins, &*inverse, Window::Hann, &|| true), Err(Error::Invalid(_))));
        assert!(matches!(workspace.prepare_kernels(&[], &forward, Window::Hann, &|| true), Err(Error::Invalid(_))));
        let too_many = vec![bins[0]; p.batch_size + 1];
        assert!(matches!(workspace.prepare_kernels(&too_many, &forward, Window::Hann, &|| true), Err(Error::Invalid(_))));
        assert_eq!(forward.calls.load(Ordering::SeqCst), 0);
        assert_eq!(wrong_length.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancelled_partial_preparation_returns_no_bank_and_reuses_the_workspace() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        let bins = preparation_bins(4096, 13, Window::BlackmanHarris);
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(4096);
        let inverse = planner.plan_fft_inverse(4096);
        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        pool.install(|| {
            let p = Plan::for_group(4096, 2, bins.len(), 64, bins.iter().map(|b| b.half).max().unwrap(),
                48_000, 180.0, scratch_len, &*inverse, Limits::default(), true).unwrap();
            assert_eq!(p.slots, 6);
            let mut workspace = Workspace::new(p, &*inverse).unwrap();
            let completed = AtomicUsize::new(0);
            let abort = AtomicBool::new(false);
            let after_publish = |_: usize| {
                if completed.fetch_add(1, Ordering::SeqCst) == 0 {
                    abort.store(true, Ordering::SeqCst);
                } else {
                    while !abort.load(Ordering::SeqCst) { std::hint::spin_loop(); }
                }
            };
            assert!(matches!(workspace.prepare_kernels_observed(&bins, &*forward, Window::BlackmanHarris,
                &|| !abort.load(Ordering::SeqCst), Some(&after_publish)), Err(Error::Aborted)));
            assert!(completed.load(Ordering::SeqCst) >= 1 && completed.load(Ordering::SeqCst) < bins.len());
            assert_eq!((workspace.active_bins, workspace.active_count), (0, 0));

            // One-bin tail: let the slot's final poll succeed after publishing,
            // then refuse at the caller's post-join poll. Without that final
            // check this would return a complete but superseded bank.
            let published = AtomicBool::new(false);
            let post_checks = AtomicUsize::new(0);
            assert!(matches!(workspace.prepare_kernels_observed(&bins[..1], &*forward, Window::BlackmanHarris,
                &|| !published.load(Ordering::SeqCst) || post_checks.fetch_add(1, Ordering::SeqCst) == 0,
                Some(&|_| published.store(true, Ordering::SeqCst))), Err(Error::Aborted)));
            assert_eq!(post_checks.load(Ordering::SeqCst), 2);

            let (got, entered) = workspace.prepare_kernels(&bins, &*forward, Window::BlackmanHarris, &|| true).unwrap();
            assert_eq!(entered, bins.len() as u64);
            let mut scratch = vec![C64::default(); scratch_len];
            let expected: Vec<_> = bins.iter().map(|&bin| {
                let mut values = super::super::kernel(bin, 4096, 48_000, Window::BlackmanHarris, &|| true).unwrap();
                forward.process_with_scratch(&mut values, &mut scratch);
                values
            }).collect();
            assert_kernel_bits(&got, &expected);
            assert!(workspace.slots.iter().all(|slot| slot.stats == Stats::default()));
        });
    }

    fn plan(inverse: &dyn Fft<f64>, budget: usize) -> Option<Plan> {
        Plan::with_workers(
            4096,
            2,
            4,
            64,
            256,
            48_000,
            180.0,
            inverse.get_inplace_scratch_len(),
            inverse,
            Limits {
                workspace_bytes: budget,
                ..Default::default()
            },
            false,
            6,
        )
    }

    #[test]
    fn joint_budget_never_admits_extra_workers_beside_a_serial_only_budget() {
        let inverse = FftPlannerScalar::<f64>::new().plan_fft(4096, FftDirection::Inverse);
        let serial_minimum = (4096 * 5 + inverse.get_inplace_scratch_len()) * size_of::<C64>();
        assert!(
            plan(&*inverse, serial_minimum).is_none(),
            "a one-slot budget admitted parallel buffers"
        );
        let admitted = plan(&*inverse, 64 << 20).expect("ample budget refused");
        assert!(admitted.slots >= 2 && admitted.slots <= 4);
        assert!(admitted.workspace_bytes <= 64 << 20);
        assert!(admitted.batch_size >= admitted.slots);
        assert!(
            Plan::for_group(
                4096,
                2,
                4,
                64,
                256,
                48_000,
                180.0,
                inverse.get_inplace_scratch_len(),
                &*inverse,
                Limits::default(),
                false
            )
            .is_none(),
            "a plain caller implicitly admitted the global pool"
        );
    }

    #[test]
    fn cancellation_hides_partial_coefficients_and_the_workspace_can_be_reused() {
        let inverse = FftPlannerScalar::<f64>::new().plan_fft(4096, FftDirection::Inverse);
        let p = plan(&*inverse, 64 << 20).unwrap();
        let mut workspace = Workspace::new(p, &*inverse).unwrap();
        let kernels = vec![vec![C64::new(1.0, -0.25); 4096]; 4];
        let block = Block {
            first: 0,
            end: 4,
            sample_rate: 48_000,
            frame_rate: 180.0,
            half: 256,
            start: 0,
        };
        let calls = AtomicUsize::new(0);
        assert_eq!(
            workspace.process_block(&kernels, &*inverse, block, &|| calls
                .fetch_add(1, Ordering::SeqCst)
                == 0),
            Err(Error::Aborted)
        );
        assert!(calls.load(Ordering::SeqCst) >= 2);
        assert_eq!(workspace.active_bins, 0);
        let stats = workspace
            .process_block(&kernels, &*inverse, block, &|| true)
            .unwrap();
        assert_eq!(stats.inverse_ffts, 8);
        assert_eq!(stats.inverses_full, 8);
        assert_eq!(stats.inverses_pruned, 0);
        assert_eq!(workspace.coefficients(0, 0, 4), &[C64::new(0.0, 0.0); 4]);
    }

    #[test]
    fn explicit_pool_parallel_coefficients_match_the_same_full_plan_bit_for_bit() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let inverse = rustfft::FftPlanner::<f64>::new().plan_fft_inverse(4096);
        let scratch_len = inverse.get_inplace_scratch_len();
        for (sample_rate, frame_rate, first, end, frames) in [
            (48_000, 180.0, 25, 29, 64),
            // A valid public-core clock with duplicate sample indices: the
            // whole inverse, not the strictly-increasing pruning API, applies.
            (100, 360.0, 0, 10, 1024),
        ] {
            for allow_pruning in [false, true] {
                pool.install(|| {
                    let p = Plan::for_group(4096, 2, 4, frames, 256, sample_rate, frame_rate,
                        scratch_len, &*inverse, Limits::default(), allow_pruning).unwrap();
                    assert_eq!(p.slots, 3, "did not use the explicitly current pool size");
                    let mut workspace = Workspace::new(p, &*inverse).unwrap();
                    let (left, right, _) = workspace.source_buffers();
                    for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
                        *l = C64::new((i as f64 * 0.013).sin(), (i as f64 * 0.017).cos());
                        *r = C64::new((i as f64 * 0.021).cos(), -(i as f64 * 0.019).sin());
                    }
                    let left = left.to_vec();
                    let right = right.to_vec();
                    let kernels: Vec<Vec<C64>> = (0..4).map(|bin| (0..4096).map(|i| {
                        let phase = (i * (bin + 1)) as f64 * 0.007;
                        C64::new(phase.cos() * 0.1, phase.sin() * 0.2)
                    }).collect()).collect();
                    let block = Block {
                        first, end, sample_rate, frame_rate, half: 256,
                        start: frame_center(first, sample_rate, frame_rate),
                    };
                    // An uneven partial kernel batch on the same allocated
                    // workspace exercises the final slab chunk and idle slots.
                    for bins in [4, 2, 3] {
                        let stats = workspace.process_block(&kernels[..bins], &*inverse, block, &|| true).unwrap();
                        assert_eq!(stats.inverse_ffts, (bins * 2) as u64);
                        assert_eq!(stats.inverses_full + stats.inverses_pruned, stats.inverse_ffts);
                        if !allow_pruning || sample_rate == 100 {
                            assert_eq!(stats.inverses_pruned, 0, "a whole/duplicate schedule was pruned");
                        }
                        for (bin, kernel) in kernels.iter().enumerate().take(bins) {
                            for (channel, source) in [&left, &right].into_iter().enumerate() {
                                let mut whole: Vec<C64> = source.iter().zip(kernel).map(|(&s, &h)| s * h).collect();
                                inverse.process_with_scratch(&mut whole, &mut vec![C64::default(); scratch_len]);
                                let got = workspace.coefficients(bin, channel, end - first);
                                for (offset, frame) in (first..end).enumerate() {
                                    let at = block.half + frame_center(frame, sample_rate, frame_rate) - block.start;
                                    assert_eq!((got[offset].re.to_bits(), got[offset].im.to_bits()),
                                        (whole[at].re.to_bits(), whole[at].im.to_bits()),
                                        "prune={allow_pruning} sr={sample_rate} bins={bins} bin={bin} channel={channel} frame={frame}");
                                }
                            }
                        }
                    }
                });
            }
        }
    }

    #[test]
    fn genuine_low_n_kernels_keep_full_bits_codes_budget_and_cancellation_with_partial_batches() {
        use super::super::{Config, Window, fill_input, frame_count, kernel, layout};
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        let sr = 48_000;
        let cfg = Config { bins_per_octave: 12, window: Window::Blackman, frame_rate: 179.5, bass_limit_s: None };
        let left: Vec<_> = (0..8193).map(|i| {
            (0.57 * (i as f64 * 0.041).sin() + 0.13 * (i as f64 * 0.371).cos()) as f32
        }).collect();
        let right: Vec<_> = (0..8193).map(|i| {
            (0.43 * (i as f64 * 0.073 + 0.31).cos() - 0.19 * (i as f64 * 0.219).sin()) as f32
        }).collect();
        let frames = frame_count(left.len(), sr, cfg.frame_rate).unwrap();
        for n in [512, 1024, 2048] {
            let mut planner = rustfft::FftPlanner::<f64>::new();
            let forward = planner.plan_fft_forward(n);
            let inverse = planner.plan_fft_inverse(n);
            let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
            let nominal_half = n * 3 / 16;
            let base = cfg.support_q() * f64::from(sr) / (2.0 * nominal_half as f64);
            for bins in [7, 13] {
                let frequencies: Vec<_> = (0..bins).map(|i| base * (1.0 + i as f64 * 0.002)).collect();
                let groups = layout(&frequencies, sr, &cfg, Limits::default()).unwrap();
                assert_eq!(groups.len(), 1);
                let bank = groups.get(&n).expect("genuine CQT fixture must select its named low N");
                let half = bank.iter().map(|b| b.half).max().unwrap();
                let mut scratch = vec![C64::default(); scratch_len];
                let kernels: Vec<_> = bank.iter().map(|&bin| {
                    let mut values = kernel(bin, n, sr, cfg.window, &|| true).unwrap();
                    forward.process_with_scratch(&mut values, &mut scratch);
                    values
                }).collect();
                for channels in [1, 2] {
                    assert!(Plan::for_group(n, channels, bins, frames, half, sr, cfg.frame_rate,
                        scratch_len, &*inverse, Limits::default(), true).is_none(),
                        "low-N caller outside a pool implicitly admitted global work");
                    for allow_pruning in [false, true] {
                        pool.install(|| {
                            let serial_minimum = ((2 * channels + 1) * n + scratch_len) * size_of::<C64>();
                            assert!(Plan::for_group(n, channels, bins, frames, half, sr, cfg.frame_rate,
                                scratch_len, &*inverse, Limits { workspace_bytes: serial_minimum,
                                    ..Limits::default() }, allow_pruning).is_none(),
                                "a serial-only low-N budget admitted extra worker buffers");
                            let limits = Limits { workspace_bytes: 8 << 20, ..Limits::default() };
                            let p = Plan::for_group(n, channels, bins, frames, half, sr, cfg.frame_rate,
                                scratch_len, &*inverse, limits, allow_pruning).unwrap();
                            assert_eq!(p.slots, 6);
                            assert!(p.workspace_bytes <= limits.workspace_bytes && p.batch_size >= bins);
                            let mut workspace = Workspace::new(p, &*inverse).unwrap();
                            for first in [0, 5, frames - 1] {
                                let start = frame_center(first, sr, cfg.frame_rate);
                                let mut end = first + 1;
                                while end < frames && frame_center(end, sr, cfg.frame_rate) - start < n - 2 * half {
                                    end += 1;
                                }
                                let count = end - first;
                                let block = Block { first, end, sample_rate: sr, frame_rate: cfg.frame_rate, half, start };
                                let (source_left, source_right, scratch) = workspace.source_buffers();
                                fill_input(source_left, &left, start, half, &|| true).unwrap();
                                forward.process_with_scratch(source_left, scratch);
                                if channels == 2 {
                                    fill_input(source_right, &right, start, half, &|| true).unwrap();
                                    forward.process_with_scratch(source_right, scratch);
                                }
                                let left_fft = source_left.to_vec();
                                let right_fft = source_right.to_vec();
                                // Full 7/13-bin batches and 5/1-bin tails reuse
                                // the same allocated six-slot workspace.
                                for active in [bins, 5, 1] {
                                    let stats = workspace.process_block(&kernels[..active], &*inverse, block, &|| true).unwrap();
                                    assert_eq!(stats.inverse_ffts, (active * channels) as u64);
                                    assert_eq!(stats.inverses_pruned + stats.inverses_full, stats.inverse_ffts);
                                    if !allow_pruning { assert_eq!(stats.inverses_pruned, 0); }
                                    if allow_pruning && inverse.prunes() { assert!(stats.inverses_pruned > 0); }
                                    for (bin, kernel) in kernels.iter().enumerate().take(active) {
                                        let mut whole_left: Vec<_> = left_fft.iter().zip(kernel).map(|(&s, &h)| s * h).collect();
                                        let mut whole_right: Vec<_> = right_fft.iter().zip(kernel).map(|(&s, &h)| s * h).collect();
                                        let mut scratch = vec![C64::default(); scratch_len];
                                        inverse.process_with_scratch(&mut whole_left, &mut scratch);
                                        if channels == 2 { inverse.process_with_scratch(&mut whole_right, &mut scratch); }
                                        let got_left = workspace.coefficients(bin, 0, count);
                                        let got_right = (channels == 2).then(|| workspace.coefficients(bin, 1, count));
                                        let norm = 1.0 / n as f64;
                                        for (offset, frame) in (first..end).enumerate() {
                                            let at = half + frame_center(frame, sr, cfg.frame_rate) - start;
                                            let l = got_left[offset] * norm;
                                            let expected_l = whole_left[at] * norm;
                                            assert_eq!((l.re.to_bits(), l.im.to_bits()),
                                                (expected_l.re.to_bits(), expected_l.im.to_bits()),
                                                "N={n} bins={bins}/{active} channels={channels} pruning={allow_pruning}");
                                            assert_eq!(crate::spectrum::aslt_code(l.norm() as f32),
                                                crate::spectrum::aslt_code(expected_l.norm() as f32));
                                            if let Some(values) = got_right {
                                                let r = values[offset] * norm;
                                                let expected_r = whole_right[at] * norm;
                                                assert_eq!((r.re.to_bits(), r.im.to_bits()),
                                                    (expected_r.re.to_bits(), expected_r.im.to_bits()));
                                                assert_eq!(crate::spectrum::aslt_code(r.norm() as f32),
                                                    crate::spectrum::aslt_code(expected_r.norm() as f32));
                                                assert_eq!(crate::spectrum::aslt_code(((l + r) * 0.5).norm() as f32),
                                                    crate::spectrum::aslt_code(((expected_l + expected_r) * 0.5).norm() as f32));
                                            }
                                        }
                                    }
                                }
                                let calls = AtomicUsize::new(0);
                                assert_eq!(workspace.process_block(&kernels, &*inverse, block,
                                    &|| calls.fetch_add(1, Ordering::SeqCst) == 0), Err(Error::Aborted));
                                assert_eq!(workspace.active_bins, 0);
                                assert_eq!(workspace.active_count, 0);
                                assert_eq!(workspace.process_block(&kernels, &*inverse, block, &|| true).unwrap().inverse_ffts,
                                    (bins * channels) as u64, "cancelled low-N workspace was not reusable");
                            }
                        });
                    }
                }
            }
        }
        let inverse = rustfft::FftPlanner::<f64>::new().plan_fft_inverse(256);
        pool.install(|| assert!(Plan::for_group(256, 2, 7, frames, 48, sr, cfg.frame_rate,
            inverse.get_inplace_scratch_len(), &*inverse, Limits::default(), true).is_none(),
            "sub-512 group unexpectedly entered the parallel candidate"));
    }
}
