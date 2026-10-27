//! Bounded, session-only local draft history. Never changes the lyric sheet
//! until explicit Apply, and never runs or downloads a model while drawing.
use super::*;

#[derive(Default)]
pub(super) struct Workbench {
    pub history: Vec<Preview>,
    pub locked: Vec<bool>,
    pub selected: Vec<bool>,
    pub adopt: Vec<bool>,
    pub undo: Option<(Snapshot, Snapshot)>,
}

impl LyricsWindow {
    pub(super) fn preview_compatible(&self, preview: &Preview) -> bool {
        if self.result_matches(&preview.result) { return true; }
        let Some(old) = preview.result.local.as_ref() else { return false };
        let Some(mut now) = self.local_identity() else { return false };
        now.preferences = old.identity.preferences.clone();
        self.translation.provider == Provider::Local && now == old.identity
    }

    pub(super) fn prepare_workbench_rows(&mut self) {
        let n = self.translation.preview.as_ref().map_or(0, |p| p.result.rows.len());
        let w = &mut self.translation.workbench;
        w.locked.resize(n, false);
        w.selected.resize(n, true);
        w.adopt = vec![true; n];
    }

    pub(super) fn local_revision(&self) -> Option<local_ai::Revision> {
        if self.local_identity()?.route != local_ai::Route::Translate { return None; }
        let Some(p) = self.translation.preview.as_ref() else {
            if self.translation.local.preferences.feedback.trim().is_empty() { return None; }
            let lyrics = self.lyrics.as_ref()?;
            let previous: Vec<_> = lyrics.lines.iter().map(|l| l.translation.clone()).collect();
            if !previous.iter().any(Option::is_some) { return None; }
            let n = previous.len();
            let w = &self.translation.workbench;
            return Some(local_ai::Revision { previous,
                selected: if w.selected.len() == n { w.selected.clone() } else { vec![true;n] },
                locked: if w.locked.len() == n { w.locked.clone() } else { vec![false;n] } });
        };
        if p.result.local.is_none() || !p.snapshot.matches(self) || !self.preview_compatible(p) { return None; }
        Some(local_ai::Revision { previous:p.result.rows.clone(), selected:self.translation.workbench.selected.clone(), locked:self.translation.workbench.locked.clone() })
    }

    pub(super) fn archive_local_preview(&mut self) {
        if let Some(p) = self.translation.preview.take() {
            if p.result.local.is_some() && p.snapshot.matches(self) {
                let history = &mut self.translation.workbench.history;
                history.push(p);
                if history.len() > 8 { history.remove(0); }
            } else {
                self.translation.preview = Some(p);
            }
        }
    }

    pub(super) fn local_creative_settings(&mut self, ui: &mut egui::Ui, idle: bool) -> bool {
        let mut reroll = false;
        let l = &mut self.translation.local;
        egui::CollapsingHeader::new("Custom instructions and variation").id_salt("local_creative").default_open(true).show(ui, |ui| {
            ui.add_enabled_ui(idle, |ui| {
                ui.label("Custom instructions (local only; Save preferences to remember):");
                ui.add(egui::TextEdit::multiline(&mut l.preferences.instructions).desired_rows(2).char_limit(1000)
                    .hint_text("For example: avoid added filler particles and exclamations not present in the original."));
                ui.horizontal_wrapped(|ui| {
                    ui.label("Variation:");
                    for v in [local_ai::Variation::Conservative, local_ai::Variation::Balanced, local_ai::Variation::Free] {
                        ui.selectable_value(&mut l.preferences.variation, v, v.label());
                    }
                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut l.preferences.seed).range(0..=u32::MAX-1));
                });
                ui.label("Same seed is best-effort reproducible on the same runtime/device; a new seed can still produce identical wording. Free variation can reduce accuracy.");
                if ui.button("Save preferences").clicked() {
                    self.status = match l.preferences.save(&l.root) {
                        Ok(()) => "Local translation preferences saved. Feedback and lyric drafts were not saved.".into(),
                        Err(e) => format!("Could not save preferences: {e}"),
                    };
                }
                ui.label("Feedback (session only, reused until cleared; not model training):");
                ui.add(egui::TextEdit::multiline(&mut l.preferences.feedback).desired_rows(2).char_limit(1000)
                    .hint_text("For example: line 3 is sarcastic, not a proposal. Do not invent gender or events."));
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Clear feedback").clicked() { l.preferences.feedback.clear(); }
                    for (label, feedback) in [("Meaning", "Correct mistranslations and omissions; preserve the original meaning."),
                        ("Naturalness", "Make the wording natural, without adding meaning or filler particles."),
                        ("Tone", "Preserve the original tone and emotion; do not invent gender or events."),
                        ("Names", "Keep names and repeated terms consistent with the surrounding lyrics.")] {
                        if ui.button(label).clicked() { l.preferences.feedback = feedback.into(); }
                    }
                });
                reroll = ui.button(crate::shell::plain(self.page_fill, "🎲 Draw another")).on_hover_text("Changes the seed, then explicitly starts a new local request. Locked/unselected rows stay unchanged.").clicked();
                if reroll { l.preferences.reroll(); }
            });
        });
        reroll
    }

    pub(super) fn local_history_ui(&mut self, ui: &mut egui::Ui, idle: bool) {
        if self.translation.provider != Provider::Local { return; }
        let mut select = None;
        ui.add_enabled_ui(idle, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Draft history (last 8; this session only):");
                for (i, p) in self.translation.workbench.history.iter().enumerate() {
                    if !p.snapshot.matches(self) { continue; }
                    let seed = p.result.local.as_ref().map_or(0, |l| l.identity.preferences.seed);
                    if ui.button(format!("Version {} · seed {seed}", i+1)).clicked() { select = Some(i); }
                }
            });
        });
        if let Some(i) = select {
            self.restore_local_draft(i);
        }
        if self.translation.workbench.undo.is_some() && ui.add_enabled(idle, egui::Button::new("Undo last local Apply")).clicked() {
            self.undo_local_adoption();
        }
    }

    pub(super) fn restore_local_draft(&mut self, i: usize) {
        if self.translation.is_busy() || i >= self.translation.workbench.history.len() { return; }
        let chosen = self.translation.workbench.history.remove(i);
            // Only same model/language/device drafts can be restored here.
            if self.preview_compatible(&chosen) && chosen.snapshot.matches(self) {
                self.archive_local_preview();
                self.translation.local.preferences = chosen.result.local.as_ref().unwrap().identity.preferences.clone();
                self.translation.preview = Some(chosen);
                self.prepare_workbench_rows();
            } else {
                self.translation.workbench.history.insert(i, chosen);
                self.status = "Choose the original model/language/device to restore this draft.".into();
            }
    }

    pub(super) fn undo_local_adoption(&mut self) {
        if self.translation.is_busy() { return; }
        if let Some((before, after)) = self.translation.workbench.undo.take() {
            if after.matches(self) {
                self.lyrics = Some(before.lyrics);
                self.translation_changed();
                self.status = "Last local translation adoption undone; Save to keep this change.".into();
            } else {
                self.status = "The sheet changed since Apply; undo refused to overwrite newer edits.".into();
            }
        }
    }
}
