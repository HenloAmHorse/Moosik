//! Settings → Storage (1.6.2): what Moosik keeps on this computer, how much
//! each part takes, and the parts that can go. Sizes are measured on a
//! thread. A removal goes through the code that owns those files, with its
//! own checks: the local model packs (only Moosik's own folders and files,
//! nothing while a model runs or downloads) and the YouTube cache (never a
//! playlist's song).

use super::*;
use crate::lyrics::local_ai;
use crate::youtube;
use std::path::Path;
use std::sync::mpsc;

/// What a part's button does.
#[derive(Clone, Copy, Debug)]
enum Act {
    None,
    /// Remove an installed translation model.
    Model(&'static local_ai::Pack),
    /// Discard a model's unfinished, or left-behind, download.
    Staged(&'static local_ai::Pack),
    /// Clear the songs and videos played from YouTube, but the playlist's.
    YouTube,
    /// Open the downloads folder.
    Open,
}

impl PartialEq for Act {
    fn eq(&self, o: &Self) -> bool {
        match (self, o) {
            (Act::Model(a), Act::Model(b)) | (Act::Staged(a), Act::Staged(b)) => a.id == b.id,
            _ => std::mem::discriminant(self) == std::mem::discriminant(o),
        }
    }
}

#[derive(Debug)]
struct Part {
    label: String,
    bytes: u64,
    act: Act,
}

/// The measured parts and the drive's free bytes.
type Survey = (Vec<Part>, Option<u64>);

#[derive(Default)]
pub(crate) struct Storage {
    survey: Option<Survey>,
    rx: Option<mpsc::Receiver<Survey>>,
    /// The part whose removal waits for a second click.
    armed: Option<usize>,
    status: String,
}

/// Bytes under `p` (a file, or a folder and everything in it), not
/// following links or junctions.
fn bytes(p: &Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_symlink() => 0,
        Ok(m) if m.is_dir() => std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| bytes(&e.path())).sum(),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

fn size(b: u64) -> String {
    match b {
        b if b >= 1_000_000_000 => format!("{:.1} GB", b as f64 / 1e9),
        b if b >= 1_000_000 => format!("{:.0} MB", b as f64 / 1e6),
        b => format!("{} KB", b.div_ceil(1000)),
    }
}

/// A model's name without its quantisation and runtime, which every pack
/// has one of.
fn short(pack: &local_ai::Pack) -> &str {
    pack.label.split(" (").next().unwrap_or(pack.label)
}

/// Every part of the profile folder `moosik`, the translation models'
/// folder `models` (inside it, but for tests) and the downloads folder.
/// Empty parts are left out.
fn survey(moosik: &Path, models: &Path, downloads: &Path) -> Vec<Part> {
    let chosen = std::fs::read_to_string(models.join("chosen-model.txt")).unwrap_or_default();
    let mut parts = Vec::new();
    let mut packs = 0;
    for &pack in local_ai::PACKS {
        let name = short(pack);
        let n = bytes(&local_ai::pack_dir(models, pack));
        if n > 0 {
            let chosen = if chosen.trim() == pack.id { " · selected" } else { "" };
            parts.push(Part { label: format!("{name}{chosen}"), bytes: n, act: Act::Model(pack) });
        }
        let staged = bytes(&models.join(".staging").join(pack.id));
        if staged > 0 {
            parts.push(Part { label: format!("Unfinished download: {name}"), bytes: staged, act: Act::Staged(pack) });
        }
        packs += n + staged;
    }
    let cache = moosik.join("cache");
    let played = bytes(&youtube::cache_dir(moosik));
    let mut other = 0;
    for e in std::fs::read_dir(moosik).into_iter().flatten().flatten() {
        if !matches!(e.file_name().to_str(), Some("local-ai" | "cache" | "tools" | "models")) {
            other += bytes(&e.path());
        }
    }
    for (label, n, act) in [
        ("Other translation files", bytes(models).saturating_sub(packs), Act::None),
        ("Analyses (spectrum, chords, beats)", bytes(&cache).saturating_sub(played), Act::None),
        ("Played from YouTube", played, Act::YouTube),
        ("YouTube components (yt-dlp, Deno)", bytes(&moosik.join("tools")), Act::None),
        ("Chord model", bytes(&moosik.join("models")), Act::None),
        ("Settings, library and logs", other, Act::None),
        ("Downloaded songs", bytes(downloads), Act::Open),
    ] {
        if n > 0 {
            parts.push(Part { label: label.into(), bytes: n, act });
        }
    }
    parts
}

impl MoosikApp {
    /// Settings → Storage.
    pub(super) fn storage_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        let downloads = self.yt_downloads();
        let fetching = self.youtube_fetching();
        let st = &mut self.shell.storage;
        if let Some(rx) = &st.rx {
            match rx.try_recv() {
                Ok(s) => {
                    st.survey = Some(s);
                    st.rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => ui.ctx().request_repaint_after(std::time::Duration::from_millis(100)),
                Err(mpsc::TryRecvError::Disconnected) => st.rx = None,
            }
        } else if st.survey.is_none() {
            let (tx, rx) = mpsc::channel();
            let (moosik, models, folder) = (moosik_dir(), local_ai::default_root(), downloads.clone());
            let spawned = std::thread::Builder::new().name("storage-survey".into())
                .spawn(move || { let _ = tx.send((survey(&moosik, &models, &folder), local_ai::free_space(&moosik))); });
            if spawned.is_ok() {
                st.rx = Some(rx);
            } else {
                st.status = "Could not start measuring.".into();
            }
        }
        let small = |t: String| RichText::new(t).size(11.5).color(tok.dim);
        let Some((parts, free)) = &st.survey else {
            ui.label(small("Measuring…".into()));
            return;
        };
        let (mut act, mut refresh) = (None, false);
        let total: u64 = parts.iter().map(|p| p.bytes).sum();
        let free = free.map_or(String::new(), |f| format!(" · {} free", size(f)));
        ui.horizontal(|ui| {
            ui.label(small(format!("{} in all{free}", size(total))));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                refresh = ui.add(egui::Button::new("Refresh").small()).clicked();
            });
        });
        ui.add_space(4.0);
        for (i, p) in parts.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let name = match p.act {
                        Act::Model(_) => "Remove",
                        Act::Staged(_) => "Discard",
                        Act::YouTube => "Clear",
                        Act::Open => "Open",
                        Act::None => "",
                    };
                    if !name.is_empty() {
                        let armed = st.armed == Some(i);
                        let text = if armed { RichText::new(format!("{name} {}?", size(p.bytes))).color(ui.visuals().warn_fg_color) } else { RichText::new(name) };
                        let busy = p.act == Act::YouTube && fetching;
                        let b = ui.add_enabled(!busy, egui::Button::new(text).small())
                            .on_disabled_hover_text("Waiting for a YouTube song to finish arriving.");
                        if b.clicked() {
                            if armed || p.act == Act::Open {
                                act = Some(p.act);
                            } else {
                                st.armed = Some(i);
                            }
                        }
                    }
                    ui.label(RichText::new(size(p.bytes)).color(tok.ink));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(egui::Label::new(RichText::new(&p.label).color(tok.ink)).truncate());
                    });
                });
            });
        }
        if !st.status.is_empty() {
            ui.label(small(st.status.clone()));
        }
        ui.label(RichText::new("Remove, Discard and Clear ask for a second click. Analyses have their size limit and \
            clearing in Analyze → Controls → Cache. Downloaded songs are yours: Moosik never deletes them.")
            .size(11.0).color(tok.faint));
        if refresh {
            st.survey = None;
            st.armed = None;
        }
        let Some(act) = act else { return };
        let done = match act {
            Act::Model(pack) => local_ai::remove(&local_ai::default_root(), pack)
                .map(|()| format!("{} removed.", short(pack))),
            Act::Staged(pack) => local_ai::discard_staged(&local_ai::default_root(), pack).map(|()| "Download discarded.".to_string()),
            Act::YouTube => {
                let keep = self.youtube_keep();
                youtube::prune(&youtube::cache_dir(&moosik_dir()), &keep, 0);
                Ok("Cleared; the playlist's songs and videos were kept.".into())
            }
            Act::Open => std::process::Command::new("explorer").arg(&downloads).spawn().map(|_| String::new())
                .map_err(|e| format!("Could not open {}: {e}", downloads.display())),
            Act::None => Ok(String::new()),
        };
        if matches!(act, Act::Model(_) | Act::Staged(_)) {
            self.lyrics_window.local_files_changed();
        }
        let st = &mut self.shell.storage;
        st.armed = None;
        if act != Act::Open {
            st.survey = None;
        }
        st.status = done.unwrap_or_else(|e| e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_survey_names_each_part_and_skips_links() {
        let root = std::env::temp_dir().join(format!("moosik-storage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (moosik, downloads) = (root.join(".moosik"), root.join("Moosik"));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let models = moosik.join("local-ai");
        let pack = local_ai::PACKS[0];
        put(&local_ai::pack_dir(&models, pack).join("m.gguf"), 4000);
        put(&models.join(".staging").join(pack.id).join("m.gguf.part"), 1000);
        std::fs::write(models.join("chosen-model.txt"), pack.id).unwrap();
        put(&models.join(".trash/old/x"), 300);
        put(&moosik.join("cache/v2/a.mix"), 2000);
        put(&youtube::cache_dir(&moosik).join("id.video.webm"), 700);
        put(&moosik.join("tools/youtube/yt-dlp.exe"), 500);
        put(&moosik.join("settings.json"), 10);
        put(&downloads.join("a.opus"), 90);
        let parts = survey(&moosik, &models, &downloads);
        let got: Vec<(String, u64, Act)> = parts.into_iter().map(|p| (p.label, p.bytes, p.act)).collect();
        let name = short(pack);
        let other = pack.id.len() as u64 + 300;
        assert_eq!(got, vec![
            (format!("{name} · selected"), 4000, Act::Model(pack)),
            (format!("Unfinished download: {name}"), 1000, Act::Staged(pack)),
            ("Other translation files".into(), other, Act::None),
            ("Analyses (spectrum, chords, beats)".into(), 2000, Act::None),
            ("Played from YouTube".into(), 700, Act::YouTube),
            ("YouTube components (yt-dlp, Deno)".into(), 500, Act::None),
            ("Settings, library and logs".into(), 10, Act::None),
            ("Downloaded songs".into(), 90, Act::Open),
        ]);
        // A link inside is not followed (a link needs rights a test may lack).
        #[cfg(windows)]
        if std::os::windows::fs::symlink_dir(&root, moosik.join("loop")).is_ok() {
            assert_eq!(bytes(&moosik.join("loop")), 0);
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
