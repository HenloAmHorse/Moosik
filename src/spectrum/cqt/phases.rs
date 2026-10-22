//! Opt-in coarse diagnostics, not a transform or cache-identity setting.
//!
//! One environment lookup occurs at the public analysis boundary. Disabled
//! clocks are None: they call neither Instant::now/elapsed nor allocate group
//! records. Enabled regions use one clock per group, batch, block or inverse
//! task, never per sample/frame/member. Fold regions deliberately include
//! progress/observer/map callbacks; serial fallback combines inverse+fold.
//! Worker durations overlap and are wall-time sums, NOT process CPU seconds.

use super::WorkStats;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Timings {
    pub enabled: bool,
    pub total_wall_ns: u128,
    pub setup_wall_ns: u128,
    pub planning_wall_ns: u128,
    pub allocation_wall_ns: u128,
    pub kernel_prepare_fft_wall_ns: u128,
    pub source_fill_fft_wall_ns: u128,
    pub parallel_inverse_region_wall_ns: u128,
    /// Sum of exclusive-slot task wall durations; overlaps across slots.
    pub parallel_inverse_worker_ns: u128,
    pub parallel_fold_progress_wall_ns: u128,
    pub serial_inverse_fold_progress_wall_ns: u128,
    pub unclassified_wall_ns: u128,
    pub groups: u64,
    pub kernel_batches: u64,
    pub source_blocks: u64,
    pub parallel_blocks: u64,
    pub serial_blocks: u64,
    pub parallel_tasks: u64,
    pub max_parallel_tasks_per_block: u64,
}

impl Timings {
    fn accounted_wall_ns(&self) -> u128 {
        self.setup_wall_ns
            + self.planning_wall_ns
            + self.allocation_wall_ns
            + self.kernel_prepare_fft_wall_ns
            + self.source_fill_fft_wall_ns
            + self.parallel_inverse_region_wall_ns
            + self.parallel_fold_progress_wall_ns
            + self.serial_inverse_fold_progress_wall_ns
    }

    fn finish_wall(&mut self, total: u128) {
        self.total_wall_ns = total;
        let accounted = self.accounted_wall_ns();
        debug_assert!(accounted <= total, "coarse phase regions overlap");
        self.unclassified_wall_ns = total.saturating_sub(accounted);
    }

    fn difference(self, before: Self) -> Self {
        Self {
            enabled: self.enabled,
            total_wall_ns: 0,
            setup_wall_ns: self.setup_wall_ns - before.setup_wall_ns,
            planning_wall_ns: self.planning_wall_ns - before.planning_wall_ns,
            allocation_wall_ns: self.allocation_wall_ns - before.allocation_wall_ns,
            kernel_prepare_fft_wall_ns: self.kernel_prepare_fft_wall_ns
                - before.kernel_prepare_fft_wall_ns,
            source_fill_fft_wall_ns: self.source_fill_fft_wall_ns - before.source_fill_fft_wall_ns,
            parallel_inverse_region_wall_ns: self.parallel_inverse_region_wall_ns
                - before.parallel_inverse_region_wall_ns,
            parallel_inverse_worker_ns: self.parallel_inverse_worker_ns
                - before.parallel_inverse_worker_ns,
            parallel_fold_progress_wall_ns: self.parallel_fold_progress_wall_ns
                - before.parallel_fold_progress_wall_ns,
            serial_inverse_fold_progress_wall_ns: self.serial_inverse_fold_progress_wall_ns
                - before.serial_inverse_fold_progress_wall_ns,
            unclassified_wall_ns: 0,
            groups: self.groups - before.groups,
            kernel_batches: self.kernel_batches - before.kernel_batches,
            source_blocks: self.source_blocks - before.source_blocks,
            parallel_blocks: self.parallel_blocks - before.parallel_blocks,
            serial_blocks: self.serial_blocks - before.serial_blocks,
            parallel_tasks: self.parallel_tasks - before.parallel_tasks,
            // Per-group maxima are supplied by its block recorder below.
            max_parallel_tasks_per_block: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Region {
    Setup,
    Planning,
    Allocation,
    Kernel,
    Source,
    ParallelInverse,
    ParallelFold,
    SerialCombined,
}

#[inline]
pub(super) fn clock(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

#[inline]
pub(super) fn elapsed(start: Option<Instant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_nanos())
}

#[derive(Serialize)]
struct Group {
    fft_len: usize,
    bins: usize,
    inverse_slots: usize,
    batch_capacity: usize,
    kernel_ffts: u64,
    input_ffts: u64,
    inverse_ffts: u64,
    requested_inverse_calls: u64,
    whole_inverse_calls: u64,
    workspace_bytes: usize,
    phases: Timings,
}

pub(super) struct GroupStart {
    start: Instant,
    before: Timings,
    work_before: WorkStats,
    fft_len: usize,
    bins: usize,
}

pub(super) struct Profile {
    pub totals: Timings,
    start: Option<Instant>,
    groups: Vec<Group>,
    group_max_tasks: u64,
}

impl Profile {
    pub fn new(enabled: bool) -> Self {
        Self {
            totals: Timings {
                enabled,
                ..Timings::default()
            },
            start: clock(enabled),
            groups: Vec::new(),
            group_max_tasks: 0,
        }
    }

    #[inline]
    pub fn clock(&self) -> Option<Instant> {
        clock(self.totals.enabled)
    }

    #[inline]
    pub fn record(&mut self, region: Region, start: Option<Instant>) {
        let Some(start) = start else { return };
        let nanos = start.elapsed().as_nanos();
        let target = match region {
            Region::Setup => &mut self.totals.setup_wall_ns,
            Region::Planning => &mut self.totals.planning_wall_ns,
            Region::Allocation => &mut self.totals.allocation_wall_ns,
            Region::Kernel => &mut self.totals.kernel_prepare_fft_wall_ns,
            Region::Source => &mut self.totals.source_fill_fft_wall_ns,
            Region::ParallelInverse => &mut self.totals.parallel_inverse_region_wall_ns,
            Region::ParallelFold => &mut self.totals.parallel_fold_progress_wall_ns,
            Region::SerialCombined => &mut self.totals.serial_inverse_fold_progress_wall_ns,
        };
        *target += nanos;
        match region {
            Region::Kernel => self.totals.kernel_batches += 1,
            Region::Source => self.totals.source_blocks += 1,
            Region::ParallelInverse => self.totals.parallel_blocks += 1,
            Region::SerialCombined => self.totals.serial_blocks += 1,
            _ => {}
        }
    }

    pub fn begin_group(
        &mut self,
        fft_len: usize,
        bins: usize,
        work: WorkStats,
    ) -> Option<GroupStart> {
        let start = self.clock()?;
        let before = self.totals;
        self.totals.groups += 1;
        self.group_max_tasks = 0;
        Some(GroupStart {
            start,
            before,
            work_before: work,
            fft_len,
            bins,
        })
    }

    pub fn workers(&mut self, nanos: u128, tasks: u64) {
        if self.totals.enabled {
            self.totals.parallel_inverse_worker_ns += nanos;
            self.totals.parallel_tasks += tasks;
            self.totals.max_parallel_tasks_per_block =
                self.totals.max_parallel_tasks_per_block.max(tasks);
            self.group_max_tasks = self.group_max_tasks.max(tasks);
        }
    }

    pub fn finish_group(
        &mut self,
        start: Option<GroupStart>,
        slots: usize,
        batch: usize,
        workspace: usize,
        work: WorkStats,
    ) {
        let Some(start) = start else { return };
        let mut phases = self.totals.difference(start.before);
        phases.max_parallel_tasks_per_block = self.group_max_tasks;
        phases.finish_wall(start.start.elapsed().as_nanos());
        self.groups.push(Group {
            fft_len: start.fft_len,
            bins: start.bins,
            inverse_slots: slots,
            batch_capacity: batch,
            kernel_ffts: work.kernel_ffts - start.work_before.kernel_ffts,
            input_ffts: work.input_ffts - start.work_before.input_ffts,
            inverse_ffts: work.inverse_ffts - start.work_before.inverse_ffts,
            requested_inverse_calls: work.inverses_pruned - start.work_before.inverses_pruned,
            whole_inverse_calls: work.inverses_full - start.work_before.inverses_full,
            workspace_bytes: workspace,
            phases,
        });
    }

    pub fn finish(&mut self) -> Timings {
        if self.totals.enabled {
            self.totals.finish_wall(elapsed(self.start));
        }
        self.totals
    }

    pub fn report(&self, sample_rate: u32, frames: usize, bars: usize, bpo: u16, channels: usize) {
        if self.totals.enabled {
            eprintln!(
                "CQT_PHASES {}",
                serde_json::json!({
                    "schema": "moosik-cqt-coarse-phases-v1",
                    "sample_rate": sample_rate, "frames": frames, "bars": bars,
                    "bins_per_octave": bpo, "channels": channels,
                    "scope": "transform-core; kernel=allocation+trig+kernel-FFT; source=fill+forward-FFT; fold includes progress/map/observer; serial combines inverse+fold+progress",
                    "worker_scope": "overlapping task wall-time sum, not CPU-time; includes requested-index setup and coefficient gather",
                    "unclassified_scope": "control, validation, drops and timer bookkeeping outside named regions; JSON serialization/stderr excluded from core total",
                    "totals": self.totals, "groups": self.groups,
                })
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profile_has_no_clocks_or_allocated_group_records() {
        let mut profile = Profile::new(false);
        assert!(profile.clock().is_none());
        assert!(profile.begin_group(4096, 8, WorkStats::default()).is_none());
        profile.record(Region::Setup, profile.clock());
        profile.workers(500, 4);
        assert_eq!(profile.finish(), Timings::default());
        assert_eq!(profile.groups.capacity(), 0);
    }

    #[test]
    fn worker_overlap_is_not_added_to_exclusive_wall_accounting() {
        let mut timings = Timings {
            enabled: true,
            planning_wall_ns: 5,
            parallel_inverse_region_wall_ns: 10,
            parallel_inverse_worker_ns: 60,
            ..Timings::default()
        };
        timings.finish_wall(20);
        assert_eq!(timings.accounted_wall_ns(), 15);
        assert_eq!(timings.unclassified_wall_ns, 5);
        let mut later = timings;
        later.planning_wall_ns += 7;
        assert_eq!(later.difference(timings).planning_wall_ns, 7);
        assert_eq!(later.difference(timings).parallel_inverse_worker_ns, 0);
    }
}
