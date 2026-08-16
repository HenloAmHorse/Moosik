//! The tag window: an editor for the common fields, and a viewer for
//! everything else the file carries.

use super::{Edits, EDITABLE};
use crate::pal;
use eframe::egui;
use egui::RichText;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct TagWindow {
    pub open: bool,
    loaded_for: Option<PathBuf>,
    /// Values as they are on disk, for change detection and Revert.
    original: Edits,
    edits: Edits,
    all: Vec<(String, String)>,
    kind: Option<String>,
    /// Set when the file cannot be written; the editor goes read-only and says why.
    locked: Option<String>,
    status: String,
    show_all: bool,
}

impl TagWindow {
    fn load(&mut self, path: &Path) {
        self.original = super::read_edits(path);
        self.edits = self.original.clone();
        self.all = super::read_all(path);
        self.kind = super::tag_kind(path);
        self.locked = super::writable(path).err();
        self.loaded_for = Some(path.to_path_buf());
        self.status.clear();
    }

    fn dirty(&self) -> bool { !self.edits.changed_from(&self.original).is_empty() }

    /// Draw the window. Returns `true` when tags were written, so the caller
    /// can reload the track's metadata.
    pub fn ui(&mut self, ctx: &egui::Context, path: Option<&Path>, title: &str) -> bool {
        if !self.open { return false; }
        let Some(path) = path else { return false };
        if self.loaded_for.as_deref() != Some(path) { self.load(path); }

        let mut saved = false;
        let mut close = false;
        let vp_id = egui::ViewportId::from_hash_of("moosik_tags");
        let vp = egui::ViewportBuilder::default()
            .with_title(format!("Tags — {title}"))
            .with_inner_size([480.0, 560.0])
            .with_resizable(true);

        ctx.show_viewport_immediate(vp_id, vp, |vctx, _class| {
            if vctx.input(|i| i.viewport().close_requested()) { close = true; return; }
            let dark = vctx.style().visuals.dark_mode;

            egui::TopBottomPanel::bottom("tag_bot").show(vctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let can_save = self.locked.is_none() && self.dirty();
                    if ui.add_enabled(can_save, egui::Button::new("💾 Save"))
                        .on_hover_text(
                            "Writes to a copy and renames it over the original, so an \
                             interrupted write cannot damage the file. The result is \
                             read back and checked before it is accepted.")
                        .clicked()
                    {
                        match super::write(path, &self.edits) {
                            Ok(()) => {
                                let n = self.edits.changed_from(&self.original).len();
                                self.load(path);
                                self.status = format!(
                                    "Saved {n} field{}.", if n == 1 { "" } else { "s" });
                                saved = true;
                            }
                            Err(e) => self.status = e,
                        }
                    }
                    if ui.add_enabled(self.dirty(), egui::Button::new("↩ Revert")).clicked() {
                        self.edits = self.original.clone();
                        self.status.clear();
                    }
                    if self.dirty() {
                        ui.label(RichText::new("• unsaved").size(11.5).color(pal::amber(dark)));
                    }
                    if !self.status.is_empty() {
                        ui.label(RichText::new(&self.status).size(11.5)
                                 .color(if self.status.starts_with("Saved") {
                                     pal::ok(dark) } else { pal::warn(dark) }));
                    }
                });
                ui.add_space(4.0);
            });

            egui::CentralPanel::default().show(vctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(why) = &self.locked {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("🔒").size(13.0).color(pal::amber(dark)));
                            ui.label(RichText::new(why).size(12.0).color(pal::amber(dark)));
                        });
                        ui.add_space(6.0);
                    }
                    if let Some(k) = &self.kind {
                        ui.label(RichText::new(format!("{k} tag")).size(11.0)
                                 .color(pal::text_faint(dark)));
                        ui.add_space(4.0);
                    }

                    let editable = self.locked.is_none();
                    egui::Grid::new("tag_edit").num_columns(2).spacing([10.0, 5.0])
                        .striped(true).show(ui, |ui| {
                        for (i, (_, label)) in EDITABLE.iter().enumerate() {
                            let changed = self.edits.values[i] != self.original.values[i];
                            let lbl = RichText::new(*label).size(12.0)
                                .color(if changed { pal::accent(dark) } else { pal::text_dim(dark) });
                            ui.label(lbl);
                            ui.add_enabled(editable, egui::TextEdit::singleline(
                                &mut self.edits.values[i]).desired_width(f32::INFINITY));
                            ui.end_row();
                        }
                    });

                    ui.add_space(8.0);
                    ui.collapsing(RichText::new(format!("All tags ({})", self.all.len()))
                                  .size(12.0), |ui| {
                        ui.label(RichText::new(
                            "Everything the file carries, including keys this editor does \
                             not touch. Saving preserves them.")
                            .size(10.5).color(pal::text_faint(dark)));
                        ui.add_space(4.0);
                        egui::Grid::new("tag_all").num_columns(2).spacing([10.0, 3.0])
                            .striped(true).show(ui, |ui| {
                            for (k, v) in &self.all {
                                ui.label(RichText::new(k).size(11.0).monospace()
                                         .color(pal::text_faint(dark)));
                                // Long values (a whole lyric sheet lives in one)
                                // must not stretch the window.
                                let shown = if v.chars().count() > 300 {
                                    format!("{}…", v.chars().take(300).collect::<String>())
                                } else { v.clone() };
                                ui.label(RichText::new(shown).size(11.0)
                                         .color(pal::text(dark)));
                                ui.end_row();
                            }
                        });
                    });
                    let _ = &mut self.show_all;
                });
            });
        });

        if close { self.open = false; }
        saved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_tracks_the_edit_not_the_load() {
        let mut w = TagWindow {
            original: Edits { values: vec!["a".into(), "b".into()] },
            edits: Edits { values: vec!["a".into(), "b".into()] },
            ..Default::default()
        };
        assert!(!w.dirty());
        w.edits.values[1] = "c".into();
        assert!(w.dirty());
        w.edits = w.original.clone();
        assert!(!w.dirty(), "Revert must clear the unsaved marker");
    }
}
