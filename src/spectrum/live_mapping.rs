//! The FFT-overlap geometry is constant between settings changes too.
//! Retain the shipped bin order, weight expressions and per-bar division;
//! only move audio-independent work out of the accepted live tick.
use super::*;

pub(super) struct Plan {
    key: CqtKey,
    gaussian: bool,
    bars: Vec<Bar>,
}

enum Bar {
    SubBin(f32),
    Weighted {
        first: usize,
        weights: Vec<f32>,
        total: f32,
    },
}

impl SpectrumAnalyzer {
    pub(super) fn ensure_overlap_plan(&mut self, sr: u32, n: usize, padded: usize) {
        let scale = self.analysis_scale();
        let hi = self.live_mapping_max_freq(sr);
        let key = CqtKey::new(n, padded, sr, self.min_freq, hi, scale);
        let gaussian = !matches!(self.bar_mapping, BarMappingMode::FlatOverlap);
        if self
            .overlap_plan
            .as_ref()
            .is_some_and(|p| p.key == key && p.gaussian == gaussian)
        {
            return;
        }
        let top = (padded / 2) as f32 - 1.0;
        // Bar `bar`'s upper edge and bar `bar + 1`'s lower edge are the same
        // expression on the same `f32`, so each of the n + 1 inversions is
        // computed once and shared, bit for bit. They dominate a cold build.
        let edges: Vec<f32> = (0..=n)
            .map(|edge| scale.freq_at(edge as f32 / n as f32, self.min_freq, hi))
            .collect();
        let bars = (0..n)
            .map(|bar| {
                let freq_lo = edges[bar];
                let freq_hi = edges[bar + 1];
                let lo = (freq_lo * padded as f32 / sr as f32).clamp(1.0, top);
                let hi = (freq_hi * padded as f32 / sr as f32).clamp(lo, top);
                if hi - lo <= 1.0 {
                    Bar::SubBin((lo + hi) * 0.5)
                } else {
                    let first = lo.floor() as usize;
                    let last = (hi.ceil() as usize).min(padded / 2 - 1);
                    let bc = (lo + hi) * 0.5;
                    let sigma = ((hi - lo) * 0.5).max(0.5);
                    let mut total = 0.0f32;
                    let weights = (first..=last)
                        .map(|b| {
                            let w = if gaussian {
                                let center_b = b as f32 + 0.5;
                                (-(center_b - bc).powi(2) / (2.0 * sigma * sigma)).exp()
                            } else {
                                (hi.min(b as f32 + 1.0) - lo.max(b as f32)).max(0.0)
                            };
                            total += w;
                            w
                        })
                        .collect();
                    Bar::Weighted {
                        first,
                        weights,
                        total,
                    }
                }
            })
            .collect();
        self.overlap_plan = Some(Plan {
            key,
            gaussian,
            bars,
        });
    }

    pub(super) fn overlap_bars(&self, norms: &[f32]) -> Vec<f32> {
        let plan = self
            .overlap_plan
            .as_ref()
            .expect("geometry prepared before gather");
        plan.bars
            .iter()
            .enumerate()
            .map(|(i, bar)| {
                let mag = match bar {
                    Bar::SubBin(center) => interp_sub_bin(norms, *center, &self.interp_mode),
                    Bar::Weighted {
                        first,
                        weights,
                        total,
                    } => {
                        let mut sum = 0.0f32;
                        for (norm, weight) in
                            norms[*first..*first + weights.len()].iter().zip(weights)
                        {
                            sum += norm * weight;
                        }
                        if *total > 0.0 { sum / total } else { 0.0 }
                    }
                };
                self.bar_db(mag, i, self.fft_size as f32)
            })
            .collect()
    }
}
