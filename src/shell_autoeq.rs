//! Headphone EQ (1.6.2): AutoEq's headphones, searched as you type,
//! equalized by autoeq.app, then used in Moosik's EQ or saved for other apps.
//! Settings → Pro tools and → Headphones open it, and so does Analyze's EQ.
//! Every request runs on a thread.

use super::*;
use crate::autoeq::{self, Catalog, Kind, Made, Measurement, Settings};
use crate::spectrum::eq::EqBand;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};

/// What a request's thread brings back.
enum Done {
    List(Result<Catalog, String>),
    /// Into Moosik's EQ: the headphones, preamp, bands, and a note.
    Used(Result<(String, f32, Vec<EqBand>, String), String>),
    /// Files for other apps: the folder, the files written, what failed.
    Saved(Result<(PathBuf, usize, Vec<String>), String>),
}

#[derive(Default)]
pub(crate) struct HeadphoneEq {
    pub(crate) open: bool,
    catalog: Option<Arc<Catalog>>,
    rx: Option<mpsc::Receiver<Done>>,
    query: String,
    hits: Vec<usize>,
    searched: Option<String>,
    pick: Option<usize>,
    meas: usize,
    target: usize,
    settings: Option<Settings>,
    status: String,
    /// Moosik's EQ when "Use in Moosik" was pressed: the answer goes in only
    /// onto that EQ, never over a change made while autoeq.app answered.
    asked: Option<(Vec<EqBand>, f32, bool)>,
}

/// autoeq.app's parametric EQ for Moosik, kept for next time; else the one
/// kept from before.
fn use_in_moosik(dir: PathBuf, name: String, m: Measurement, s: Settings) -> Done {
    let fresh = autoeq::equalize(&name, &m, &s, Kind::Parametric).and_then(|made| {
        let Made::Filters { preamp, filters } = &made else { return Err("autoeq.app answered with another kind of equalizer".into()) };
        // Kept as well: a headphone done once also works offline.
        let _ = autoeq::write(&dir, &name, Kind::Parametric, &made);
        Ok((*preamp as f32, autoeq::bands(filters)))
    });
    Done::Used(match fresh {
        Ok((preamp, bands)) => Ok((name, preamp, bands, String::new())),
        Err(e) => match autoeq::kept(&dir, &name) {
            Some((preamp, bands)) => Ok((name, preamp, bands, format!(" ({e}; the profile made for it before was used.)"))),
            None => Err(e),
        },
    })
}

/// Every kind for other apps, and the read-me.
fn save_for_apps(dir: PathBuf, name: String, m: Measurement, s: Settings) -> Done {
    let mut folder = None;
    let (mut written, mut failed) = (0, Vec::new());
    for kind in [Kind::Parametric, Kind::Graphic, Kind::Fixed10, Kind::Fixed31, Kind::Uapp, Kind::TenPeaks] {
        match autoeq::equalize(&name, &m, &s, kind).and_then(|made| autoeq::write(&dir, &name, kind, &made)) {
            Ok(f) => {
                folder = Some(f);
                written += 1;
            }
            Err(e) => failed.push(format!("{kind:?}: {e}")),
        }
    }
    Done::Saved(match folder {
        Some(f) => autoeq::write_readme(&dir, &name, &m, &s).map(|()| (f, written, failed)),
        None => Err(failed.first().cloned().unwrap_or_else(|| "nothing was made".into())),
    })
}

impl MoosikApp {
    fn heq_start(&mut self, f: impl FnOnce() -> Done + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        match std::thread::Builder::new().name("autoeq".into()).spawn(move || { let _ = tx.send(f()); }) {
            Ok(_) => self.shell.heq.rx = Some(rx),
            Err(_) => self.shell.heq.status = "Could not start the request.".into(),
        }
    }

    fn heq_poll(&mut self) {
        let Some(rx) = &self.shell.heq.rx else { return };
        let done = match rx.try_recv() {
            Ok(d) => d,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.shell.heq.rx = None;
                return;
            }
        };
        self.shell.heq.rx = None;
        let asked = self.shell.heq.asked.take();
        let status = match done {
            Done::List(Ok(c)) => {
                self.shell.heq.catalog = Some(Arc::new(c));
                self.shell.heq.searched = None;
                String::new()
            }
            Done::List(Err(e)) => format!("AutoEq's headphone list could not be loaded: {e}"),
            Done::Used(Ok((name, _, _, _))) if asked != Some(self.eq_now()) => {
                format!("The EQ was changed while autoeq.app was answering, so {name} was not put in. Use in Moosik puts it in now.")
            }
            Done::Used(Ok((name, preamp, bands, note))) => {
                let preset = format!("{name} · AutoEq");
                self.spectrum_window.apply_correction(preset.clone(), preamp, bands);
                let bp = if self.engine.as_ref().is_some_and(|e| e.on_bp_stream()) {
                    " Bit-perfect output is on, so the EQ is bypassed: turn it off under Output to hear it."
                } else {
                    ""
                };
                format!("In Moosik's EQ as the preset \"{preset}\", switched on.{note}{bp}")
            }
            Done::Saved(Ok((folder, n, failed))) => {
                let more = if failed.is_empty() { String::new() } else { format!(" Not made: {}.", failed.join("; ")) };
                format!("Saved {n} equalizers and a read-me in {}.{more}", folder.display())
            }
            Done::Used(Err(e)) | Done::Saved(Err(e)) => e,
        };
        self.shell.heq.status = status;
    }

    /// Moosik's EQ as "Use in Moosik" sets it: the bands, the preamp, on.
    fn eq_now(&self) -> (Vec<EqBand>, f32, bool) {
        let eq = self.spectrum_window.eq_state.lock().unwrap();
        (eq.bands.clone(), eq.preamp_db, eq.enabled)
    }

    /// The window, when open (both layouts). Analyze's EQ asks for it too.
    /// A request's answer is taken as it arrives, the window open or not.
    pub(crate) fn headphone_eq_window(&mut self, ctx: &egui::Context) {
        if std::mem::take(&mut self.spectrum_window.want_headphone_eq) {
            self.shell.heq.open = true;
        }
        self.heq_poll();
        if self.shell.heq.open {
            let mut open = true;
            egui::Window::new("Headphone EQ (AutoEq)")
                .open(&mut open)
                .default_width(460.0)
                .resizable(true)
                .collapsible(false)
                .show(ctx, |ui| self.heq_ui(ui));
            self.shell.heq.open &= open;
        }
        if self.shell.heq.rx.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn heq_ui(&mut self, ui: &mut egui::Ui) {
        let dim = ui.visuals().weak_text_color();
        let small = |t: String| RichText::new(t).size(11.5).color(dim);
        let busy = self.shell.heq.rx.is_some();
        let Some(cat) = self.shell.heq.catalog.clone() else {
            if busy {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(small("Loading AutoEq's headphone list…".into()));
                });
            } else if self.shell.heq.status.is_empty() {
                let dir = autoeq::dir();
                self.heq_start(move || {
                    Done::List(autoeq::load(&dir).map(Ok::<Catalog, String>).unwrap_or_else(|| {
                        autoeq::refresh(&dir).and_then(|_| autoeq::load(&dir).ok_or_else(|| "the list did not read".into()))
                    }))
                });
            } else {
                ui.label(small(self.shell.heq.status.clone()));
                if ui.button("Try again").clicked() {
                    self.shell.heq.status.clear();
                }
            }
            return;
        };
        let h = &mut self.shell.heq;
        ui.horizontal(|ui| {
            let label = ui.label("Search AutoEq");
            ui.add(egui::TextEdit::singleline(&mut h.query).hint_text("e.g. hd650 or airpods pro 2").desired_width(f32::INFINITY))
                .labelled_by(label.id);
        });
        if h.searched.as_deref() != Some(h.query.as_str()) {
            h.hits = cat.search(&h.query, 40);
            h.searched = Some(h.query.clone());
        }
        egui::ScrollArea::vertical().id_salt("heq_hits").max_height(170.0).auto_shrink([false, true]).show(ui, |ui| {
            for &i in &h.hits {
                let ms = &cat.measurements[i];
                let by = if ms.len() > 1 { format!("{} measurements", ms.len()) } else { ms[0].source.clone() };
                let label = format!("{}   ·  {by}", cat.names[i]);
                if ui.selectable_label(h.pick == Some(i), label).clicked() {
                    h.pick = Some(i);
                    h.meas = 0;
                    h.settings = None;
                }
            }
        });
        if h.query.trim().is_empty() {
            ui.label(small(format!("{} headphones and earphones from AutoEq. Spelling and spaces need not be exact.", cat.names.len())));
        } else if h.hits.is_empty() {
            ui.label(small("Nothing matches. Fewer words, or another spelling?".into()));
        }
        let mut act = None;
        if let Some(i) = h.pick {
            ui.separator();
            ui.label(RichText::new(&cat.names[i]).strong().size(14.0));
            let ms = &cat.measurements[i];
            h.meas = h.meas.min(ms.len() - 1);
            let m = &ms[h.meas];
            let fits = cat.targets_for(m);
            if h.settings.is_none() || !fits.contains(&h.target) {
                h.target = fits[0];
                h.settings = Some(Settings::for_target(&cat.targets[h.target]));
            }
            let mut s = h.settings.clone().expect("set above");
            egui::Grid::new("heq_grid").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                ui.label("Measured by");
                egui::ComboBox::from_id_salt("heq_meas").selected_text(m.label()).width(270.0).show_ui(ui, |ui| {
                    for (k, mm) in ms.iter().enumerate() {
                        if ui.selectable_label(h.meas == k, mm.label()).clicked() {
                            h.meas = k;
                            h.settings = None;
                        }
                    }
                });
                ui.end_row();
                ui.label("Target");
                egui::ComboBox::from_id_salt("heq_target").selected_text(cat.targets[h.target].label.clone()).width(270.0).show_ui(ui, |ui| {
                    for (n, &t) in fits.iter().enumerate() {
                        let rec = n == 0 && cat.targets[t].recommended.iter().any(|f| f.fits(m));
                        let label = if rec { format!("{} (recommended)", cat.targets[t].label) } else { cat.targets[t].label.clone() };
                        if ui.selectable_label(h.target == t, label).clicked() {
                            h.target = t;
                            s = Settings::for_target(&cat.targets[t]);
                        }
                    }
                });
                ui.end_row();
                ui.label("Bass boost");
                ui.add(egui::Slider::new(&mut s.bass.gain, -10.0..=20.0).step_by(0.5).suffix(" dB"));
                ui.end_row();
            });
            ui.collapsing("Advanced", |ui| {
                egui::Grid::new("heq_more").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                    ui.label("Treble boost");
                    ui.add(egui::Slider::new(&mut s.treble, -10.0..=10.0).step_by(0.5).suffix(" dB"));
                    ui.end_row();
                    ui.label("Tilt");
                    ui.add(egui::Slider::new(&mut s.tilt, -1.0..=1.0).step_by(0.1).suffix(" dB/oct"));
                    ui.end_row();
                    ui.label("Max gain");
                    ui.add(egui::Slider::new(&mut s.max_gain, 0.0..=20.0).step_by(0.5).suffix(" dB"));
                    ui.end_row();
                });
                if ui.small_button("autoeq.app's settings").clicked() {
                    s = Settings::for_target(&cat.targets[h.target]);
                }
            });
            if h.settings.is_some() {
                h.settings = Some(s.clone());
            }
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(!busy, egui::Button::new("Use in Moosik")).on_hover_text("Have autoeq.app make it, then put it in Moosik's EQ as a preset").clicked() {
                    act = Some((true, cat.names[i].clone(), m.clone(), s.clone()));
                }
                if ui.add_enabled(!busy, egui::Button::new("Save for other apps"))
                    .on_hover_text("Equalizer APO, Peace, Wavelet, JamesDSP, USB Audio Player PRO, and equalizers with 10 or 31 sliders")
                    .clicked()
                {
                    act = Some((false, cat.names[i].clone(), m.clone(), s.clone()));
                }
                if busy {
                    ui.spinner();
                }
            });
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Open folder").on_hover_text("Where the files for other apps are saved").clicked() {
                let dir = autoeq::dir();
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::process::Command::new("explorer").arg(&dir).spawn();
            }
        });
        if !self.shell.heq.status.is_empty() {
            ui.label(small(self.shell.heq.status.clone()));
        }
        ui.add_space(4.0);
        ui.label(RichText::new("Made by autoeq.app (AutoEq). Only the headphones' name, the measurement and these settings are sent; \
            no sample rate to choose, Moosik fits each song's own.").size(11.0).color(dim));
        if ui.checkbox(&mut self.appearance.autoeq_check, "Check AutoEq's headphone list when Moosik starts").changed() {
            save_appearance(&moosik_dir(), &self.appearance);
        }
        if let Some((moosik, name, m, s)) = act {
            let dir = autoeq::dir();
            self.shell.heq.status = format!("Asking autoeq.app for {name}…");
            if moosik {
                self.shell.heq.asked = Some(self.eq_now());
                self.heq_start(move || use_in_moosik(dir, name, m, s));
            } else {
                self.heq_start(move || save_for_apps(dir, name, m, s));
            }
        }
    }

    /// Settings → Pro tools: the 1.6.2 tools, from one place.
    pub(super) fn pro_tools_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        ui.horizontal_wrapped(|ui| {
            let song = self.current_index.is_some();
            if ui.add_enabled(song, egui::Button::new("Measure this song"))
                .on_hover_text("Loudness, true peak, dynamic range (DR), clipping, the bits it uses and FLAC's check; \
                    then compare it with another master or take a blind test")
                .on_disabled_hover_text("Play a song first")
                .clicked()
            {
                self.measure_current();
            }
            if ui.button("Headphone EQ (AutoEq)…").clicked() {
                self.shell.heq.open = true;
            }
            if ui.button("BPM, beats and sections").on_hover_text("Analyze, with the Harmony panel open").clicked() {
                self.appearance.visual.harmony_panel = true;
                save_appearance(&moosik_dir(), &self.appearance);
                self.set_mode(UiMode::Analyze);
                self.shell.more_open = false;
            }
        });
        ui.label(RichText::new("Crossfeed and room correction are under Headphones below, tempo and key under Practice. \
            The full EQ is in Analyze → Controls.").size(11.0).color(tok.faint));
    }

    /// Open Info and measure the playing song, unless it is measured already.
    fn measure_current(&mut self) {
        let Some(idx) = self.current_index else { return };
        let path = self.playlist[idx].path.clone();
        self.info_open = true;
        let done = self.measure.done.as_ref().is_some_and(|(p, _)| *p == path);
        let running = self.measure.job.as_ref().is_some_and(|j| j.path == path);
        if !done && !running {
            if let Some(j) = self.measure.job.take() {
                j.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.measure.job = Some(crate::measure::Job::start(path));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::Harness;

    /// "Use in Moosik"'s answer goes in as it arrives, with the window closed
    /// too, and not later over the EQ; one that arrives after the EQ was
    /// changed stays out, and the change stays.
    #[test]
    fn an_answer_goes_in_when_it_arrives_unless_the_eq_changed() {
        let mut h = Harness::builder().with_size(egui::vec2(1300.0, 860.0)).build_eframe(|cc| crate::MoosikApp::new(cc));
        h.step();
        let profile = vec![EqBand { gain_db: 3.0, ..EqBand::default_peak(1_000.0) }];
        let asked = |app: &mut MoosikApp| {
            let (tx, rx) = mpsc::channel();
            app.shell.heq.asked = Some(app.eq_now());
            app.shell.heq.rx = Some(rx);
            app.shell.heq.open = false;
            tx
        };
        let tx = asked(h.state_mut());
        tx.send(Done::Used(Ok(("HD 650".into(), -4.5, profile.clone(), String::new())))).unwrap();
        h.step();
        assert!(h.state().eq_now() == (profile.clone(), -4.5, true), "in, with the window closed");
        assert!(h.state().shell.heq.rx.is_none() && h.state().shell.heq.asked.is_none());

        let tx = asked(h.state_mut());
        {
            let mut eq = h.state().spectrum_window.eq_state.lock().unwrap();
            eq.bands[0].gain_db = 1.0;
            eq.bump();
        }
        let mine = h.state().eq_now();
        tx.send(Done::Used(Ok(("HD 600".into(), -2.0, vec![EqBand::default_peak(200.0)], String::new())))).unwrap();
        h.step();
        assert!(h.state().eq_now() == mine, "the change made meanwhile stays");
        assert!(h.state().shell.heq.status.contains("was not put in"), "{}", h.state().shell.heq.status);
    }
}
