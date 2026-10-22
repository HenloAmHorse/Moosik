//! Controls that apply to the genuine, pre-processed constant-Q transform.
//!
//! Display-bar density and filter support are deliberately separate controls:
//! increasing the number of displayed samples does not narrow a filter.

use eframe::egui;

use super::{BarMappingMode, SpectrumWindow, cqt};

impl SpectrumWindow {
    /// Edit the active CQT config without starting work or touching the analyser.
    /// The parent applies the returned quality change through its normal cache
    /// and generation path, once all controls have finished editing.
    pub(super) fn true_cqt_settings_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let BarMappingMode::TrueCqt(mut cfg) = self.bar_mapping.clone() else {
            return false;
        };
        let before = cfg;

        ui.label("True CQT · pre-process only · logarithmic frequency grid");
        ui.horizontal_wrapped(|ui| {
            ui.label("Resolution:");
            ui.add(
                egui::Slider::new(&mut cfg.bins_per_octave, 12..=192)
                    .integer()
                    .suffix(" bins/octave"),
            )
            .on_hover_text(
                "Sets the constant-Q filters' support, independently of the display \
                 bar count. Superlet settings do not apply. More bins per octave \
                 use longer windows: finer frequency discrimination, but more temporal \
                 smearing and more work. More is not universally better. Increasing \
                 display bars alone does not add independent frequency resolution.",
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Window:");
            for window in [
                cqt::Window::Hann,
                cqt::Window::Hamming,
                cqt::Window::Blackman,
                cqt::Window::BlackmanHarris,
            ] {
                ui.selectable_value(&mut cfg.window, window, window.label());
            }
        })
        .response
        .on_hover_text(
            "The per-frequency analysis window, not the live FFT window. Window \
             shapes trade peak width against leakage and sidelobe suppression; \
             there is no universally best shape.",
        );

        ui.horizontal_wrapped(|ui| {
            ui.label("Analysis frame rate:");
            ui.add(
                egui::Slider::new(&mut cfg.frame_rate, 15.0..=240.0)
                    // Preserve a validated profile's wider supported range
                    // (1..=360) until the user actually edits this control.
                    .clamping(egui::SliderClamping::Edits)
                    .suffix(" frames/s"),
            )
            .on_hover_text(
                "Stored analysis rows per second, not the screen's repaint rate. \
                 Higher rates cost more time and cache space; they sample the same \
                 frequency-dependent windows more often, without shortening them. \
                 Frame centres follow this clock even when the sample hop varies.",
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Bass response:");
            let mut responsive = cfg.bass_limit_s.is_some();
            ui.selectable_value(&mut responsive, false, "Full").on_hover_text(
                "Full constant-Q support at every frequency: the sharpest bass, and the slowest.",
            );
            ui.selectable_value(&mut responsive, true, "Responsive").on_hover_text(
                "A separate variable-Q transform: each filter's bandwidth is f/Q + 1/limit, \
                 so bass windows stay shorter than the limit while higher frequencies keep \
                 nearly the same support as Full. Faster bass motion costs bass frequency \
                 discrimination. It is cached separately and never replaces Full results.",
            );
            if responsive != cfg.bass_limit_s.is_some() {
                cfg.bass_limit_s = responsive.then_some(cqt::DEFAULT_RESPONSIVE_LIMIT_S);
            }
            if let Some(limit) = cfg.bass_limit_s.as_mut() {
                ui.add(
                    egui::Slider::new(limit, 0.1..=2.0)
                        // Keep a validated profile's wider range until edited.
                        .clamping(egui::SliderClamping::Edits)
                        .logarithmic(true)
                        .suffix(" s limit"),
                )
                .on_hover_text(
                    "Bass windows stay shorter than this. Shorter reacts faster and blurs bass pitch more.",
                );
            }
        });

        // A support-length estimate, not a claim about the window's measured
        // bandwidth, temporal resolution, or onset precision.
        let q = cfg.support_q();
        let bass_hz = self.min_freq as f64;
        if bass_hz.is_finite() && bass_hz > 0.0 {
            let support_s = cfg.support_seconds(bass_hz);
            ui.label(format!(
                "Support Q ≈ {q:.1} · at {bass_hz:.1} Hz: ≈ {support_s:.2} s window, \
                 ≈ {:.2} s on either side of its centre",
                support_s * 0.5,
            ));
        }
        ui.label(if cfg.bass_limit_s.is_some() {
            "Responsive: windows are centred, and bass support is limited as shown. \
             This analysis does not change playback audio. FFT and Superlet \
             settings do not change CQT. While it is not ready, live PCM can \
             supply a separate, clearly labelled LIVE FFT PREVIEW."
        } else {
            "Windows are centred and keep their full low-frequency support. \
             This analysis does not change playback audio. FFT and Superlet \
             settings do not change CQT. While it is not ready, live PCM can \
             supply a separate, clearly labelled LIVE FFT PREVIEW."
        });

        if cfg == before {
            return false;
        }
        if let Err(error) = cfg.validate() {
            ui.colored_label(
                ui.visuals().error_fg_color,
                format!("CQT settings: {error}"),
            );
            return false;
        }
        self.last_true_cqt = cfg;
        self.bar_mapping = BarMappingMode::TrueCqt(cfg);
        true
    }
}
