//! Requested versus effective Superlet numerics, and the worker-only HP call.
//! Standard cache bytes and Standard's calibration remain independent of HP.
use super::*;

#[derive(Default)]
pub struct State {
    pub enabled: bool,
    choice: Option<(AnalysisSettings, String)>,
    disabled: Option<String>,
    pub(super) current: Option<PathBuf>,
    #[cfg(test)]
    pub(crate) hook: Option<Hook>,
}

impl State {
    pub(super) fn requested(&self, settings: AnalysisSettings) -> AnalysisSettings {
        settings.with_numerics(if self.enabled { Numerics::HighAccuracyGpu } else { Numerics::Standard })
    }

    pub(super) fn effective(&self, requested: AnalysisSettings) -> AnalysisSettings {
        if requested.numerics() == Numerics::HighAccuracyGpu
            && (self.disabled.is_some() || self.choice.as_ref().is_some_and(|(s, _)| *s == requested))
        {
            requested.with_numerics(Numerics::Standard)
        } else { requested }
    }

    pub(super) fn standard(&mut self, requested: AnalysisSettings, reason: String) {
        self.choice = Some((requested, reason));
    }

    pub(super) fn disable(&mut self, reason: String) { self.disabled = Some(reason); }

    pub(super) fn reason(&self, requested: &AnalysisSettings) -> Option<&str> {
        self.disabled.as_deref().or_else(|| self.choice.as_ref()
            .filter(|(s, _)| s == requested).map(|(_, reason)| reason.as_str()))
    }

    pub(super) fn attempt(&self) -> Attempt {
        Attempt {
            #[cfg(test)]
            hook: self.hook.clone(),
        }
    }
}

/// Per-job injected backend in tests; ordinary builds always call the reviewed
/// engine. No process-global routing switch can alter a concurrent analyser.
#[derive(Default, Clone)]
pub(super) struct Attempt {
    #[cfg(test)]
    hook: Option<Hook>,
}

#[cfg(test)]
pub(crate) type Hook = Arc<dyn Fn(&[f32], &[f32], u32, &AtomicBool,
    &(dyn Fn(usize) + Sync)) -> Result<([Vec<u16>; 3], usize), aslt::hp::HpError> + Send + Sync>;

impl Attempt {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(&self, left: &[f32], right: &[f32], sr: u32, bars: usize,
        min: f32, max: f32, cfg: &aslt::AsltConfig, fps: f32, abort: &AtomicBool,
        progress: &(dyn Fn(usize) + Sync)) -> Result<([Vec<u16>; 3], usize), aslt::hp::HpError>
    {
        aslt::hp::admits(bars, min, max, cfg, fps).map_err(aslt::hp::HpError::Unavailable)?;
        if abort.load(Ordering::Relaxed) { return Err(aslt::hp::HpError::Cancelled); }
        #[cfg(test)]
        if let Some(hook) = &self.hook { return hook(left, right, sr, abort, progress); }
        aslt::hp::analyze_on_this_thread(left, right, sr, bars, min, max, cfg, fps, abort, progress)
    }
}

impl SpectrumAnalyzer {
    pub fn set_hp_enabled(&mut self, enabled: bool) {
        if self.hp.enabled != enabled {
            self.retire_job();
            self.pending_request = None;
            self.hp.choice = None;
            self.hp.enabled = enabled;
        }
    }

    /// Explicitly opt this track back into HP. A quarantined session still
    /// refuses HP: only restarting the player resets a device failure.
    pub fn request_hp_reanalysis(&mut self, path: PathBuf) {
        self.hp.choice = None;
        self.start_fresh_preprocess(path);
    }

    pub fn hp_label(&self) -> Option<String> {
        if !self.hp.enabled || self.bar_mapping != BarMappingMode::Superlet { return None; }
        let path = self.hp.current.as_deref()?;
        let requested = self.requested_settings(path);
        let reason = self.hp.reason(&requested);
        let actual = if self.pre_record.as_deref().is_some_and(|s| s.contains(";num=gpu-hp1")) {
            "High-accuracy GPU numerics"
        } else if self.pre_record.is_some() { "Standard numerics" }
        else if !self.pre_frames.is_empty() { "Legacy analysis (numerics unverified)" }
        else if self.analysis_busy() { "High-accuracy GPU requested" }
        else { "High-accuracy GPU enabled" };
        Some(match reason { Some(r) => format!("{actual} — {r}"), None => actual.to_owned() })
    }
}
