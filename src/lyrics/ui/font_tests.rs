//! Which installed face draws each character of a Chinese translation.
//!
//! egui walks a family's list and takes each glyph from the first face that
//! has it. These tests install the app's fonts (`crate::setup_fonts`) plus one
//! probe family per face, and ask each probe whether its face has a character,
//! so the face that actually supplies it in any chain can be named. They need
//! the Windows faces the app looks for; where those are missing they say so
//! and check nothing.

use super::*;

pub(super) const SIMPLIFIED: &str = "我们说过这些话，长长的门边开着车。爱与梦为什么还在发光？";
pub(super) const TRADITIONAL: &str = "我們說過這些話，長長的門邊開著車。愛與夢為什麼還在發光？";
pub(super) const PUNCTUATION: &str = "，。！？：；「」『』、…—（）";
pub(super) const MIXED: &str = "Love 你 100 次！OK，再見 2024。";

const FACES: [&str; 5] = ["jp", "zh_tw", "zh_cn", "ko", "cjk"];

fn probe_ctx() -> egui::Context {
    // One family per face, from the same files setup_fonts reads.
    let mut defs = egui::FontDefinitions::default();
    for (key, paths) in [
        ("jp", &[r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\yugothic.ttf", r"C:\Windows\Fonts\msgothic.ttc"][..]),
        ("zh_tw", &[r"C:\Windows\Fonts\msjh.ttc"][..]),
        ("zh_cn", &[r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"][..]),
        ("ko", &[r"C:\Windows\Fonts\malgun.ttf"][..]),
    ] {
        if let Some(data) = paths.iter().find_map(|p| std::fs::read(p).ok()) {
            defs.font_data.insert(key.into(), egui::FontData::from_owned(data).into());
            defs.families.insert(egui::FontFamily::Name(format!("probe_{key}").into()), vec![key.into()]);
        }
    }
    let base = egui::FontDefinitions::default().families[&egui::FontFamily::Proportional].clone();
    defs.families.insert(egui::FontFamily::Name("probe_base".into()), base);
    // The app's own chains, as setup_fonts builds them, read back from a
    // context it set up.
    let app = egui::Context::default();
    crate::setup_fonts(&app, None);
    let _ = app.run(egui::RawInput::default(), |_| {});
    for fam in [egui::FontFamily::Proportional, egui::FontFamily::Name(ZH_HANT.into()),
                egui::FontFamily::Name(ZH_HANS.into())] {
        let list = app.fonts(|f| f.lock().fonts.definitions().families.get(&fam).cloned()).unwrap_or_default();
        defs.families.insert(fam, list);
    }
    let ctx = egui::Context::default();
    ctx.set_fonts(defs);
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    ctx
}

/// The face each character of `s` is drawn from in `family`, by name.
pub(super) fn faces(ctx: &egui::Context, family: &egui::FontFamily, s: &str) -> Vec<(char, String)> {
    let list = ctx.fonts(|f| f.lock().fonts.definitions().families.get(family).cloned()).unwrap_or_default();
    let base: Vec<String> = egui::FontDefinitions::default().families[&egui::FontFamily::Proportional].clone();
    s.chars().filter(|c| !c.is_whitespace()).map(|c| {
        let face = list.iter().find(|k| {
            let probe = if base.contains(k) { "probe_base".to_string() } else { format!("probe_{k}") };
            let fam = egui::FontFamily::Name(probe.into());
            ctx.fonts(|f| f.families().contains(&fam) && f.has_glyph(&egui::FontId::new(16.0, fam.clone()), c))
        }).cloned().unwrap_or_else(|| "none".into());
        (c, face)
    }).collect()
}

fn installed() -> bool {
    [r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]
        .iter().all(|p| std::path::Path::new(p).exists())
}

fn han(c: char) -> bool { ('\u{4E00}'..='\u{9FFF}').contains(&c) }

/// The defect, and its repair. In the general chain a Chinese line is split
/// between Meiryo (Japanese forms) and JhengHei; in the translation families
/// no Han character comes from the Japanese face, and each family draws its
/// own script's text from one face.
#[test]
fn chinese_translations_are_drawn_from_one_chinese_face() {
    if !installed() {
        println!("Meiryo / JhengHei / YaHei not installed: nothing to check here");
        return;
    }
    let ctx = probe_ctx();
    let general = egui::FontFamily::Proportional;
    let hant = egui::FontFamily::Name(ZH_HANT.into());
    let hans = egui::FontFamily::Name(ZH_HANS.into());
    let by = |fam: &egui::FontFamily, s: &str| -> std::collections::BTreeSet<String> {
        faces(&ctx, fam, s).into_iter().filter(|(c, _)| han(*c)).map(|(_, f)| f).collect()
    };
    // Setup — the reported defect: in the general chain a Simplified line is
    // split between Meiryo and JhengHei, and a Traditional one is drawn
    // entirely in Meiryo's Japanese forms.
    let mixed = by(&general, SIMPLIFIED);
    assert!(mixed.contains("jp") && mixed.contains("zh_tw"), "setup: the general chain did not mix faces: {mixed:?}");
    assert_eq!(by(&general, TRADITIONAL), ["jp".to_string()].into_iter().collect(), "setup");
    for (name, fam, text, face) in [("Traditional family, Traditional text", &hant, TRADITIONAL, "zh_tw"),
                                    ("Traditional family, Simplified text", &hant, SIMPLIFIED, "zh_tw"),
                                    ("Simplified family, Simplified text", &hans, SIMPLIFIED, "zh_cn"),
                                    ("Simplified family, Traditional text", &hans, TRADITIONAL, "zh_cn")] {
        let got = by(fam, text);
        assert_eq!(got, [face.to_string()].into_iter().collect(), "{name}: drawn from {got:?}");
    }
    for (fam, text) in [(&hant, PUNCTUATION), (&hans, PUNCTUATION), (&hant, MIXED), (&hans, MIXED)] {
        let got = faces(&ctx, fam, text);
        assert!(got.iter().all(|(_, f)| f != "jp" && f != "none"), "{fam:?} {text}: {got:?}");
    }
}

/// The two families are defined whether or not any CJK face was found, so the
/// viewer can always name them; with none found they are egui's own faces.
#[test]
fn the_translation_families_always_exist() {
    let ctx = egui::Context::default();
    crate::setup_fonts(&ctx, Some("No Such Font Family Anywhere"));
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    for fam in [ZH_HANT, ZH_HANS] {
        assert!(ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name(fam.into()))), "{fam} missing");
    }
}

/// Every face named in the translation families is one the general chain
/// already loaded — the data is shared, not loaded twice — and the general
/// chain is unchanged by them.
#[test]
fn the_translation_families_reuse_the_loaded_faces() {
    let app = egui::Context::default();
    crate::setup_fonts(&app, None);
    let _ = app.run(egui::RawInput::default(), |_| {});
    let defs = app.fonts(|f| f.lock().fonts.definitions().clone());
    let general = &defs.families[&egui::FontFamily::Proportional];
    for fam in [ZH_HANT, ZH_HANS] {
        for k in &defs.families[&egui::FontFamily::Name(fam.into())] {
            assert!(defs.font_data.contains_key(k), "{fam} names {k}, which is not loaded");
            assert!(general.contains(k), "{fam} names {k}, which the general chain does not use");
        }
    }
    // The general chain: egui's faces, then the CJK faces found, Japanese first
    // — as before this change.
    let cjk: Vec<&String> = general.iter().filter(|k| FACES.contains(&k.as_str())).collect();
    let mut sorted = cjk.clone();
    sorted.sort_by_key(|k| FACES.iter().position(|f| f == k));
    assert_eq!(cjk, sorted, "the general chain's CJK order changed");
}

/// Renders the representative lines in each family to a PNG, through egui's
/// own text layout and font atlas, so the glyphs can be looked at. Writes to
/// `MOOSIK_FONT_RENDER` (a .png path).
#[test]
#[ignore]
fn render_translation_samples() {
    let out = std::env::var("MOOSIK_FONT_RENDER").expect("MOOSIK_FONT_RENDER");
    let ctx = egui::Context::default();
    crate::setup_fonts(&ctx, None);
    let ppp = 2.0f32;
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(760.0, 560.0))),
        viewports: std::iter::once((egui::ViewportId::ROOT, egui::ViewportInfo {
            native_pixels_per_point: Some(ppp), ..Default::default()
        })).collect(),
        ..Default::default()
    };
    let rows: Vec<(String, egui::FontFamily, &str)> = [
        ("general", egui::FontFamily::Proportional), ("zh_hant", egui::FontFamily::Name(ZH_HANT.into())),
        ("zh_hans", egui::FontFamily::Name(ZH_HANS.into())),
    ].into_iter().flat_map(|(n, f)| [SIMPLIFIED, TRADITIONAL, PUNCTUATION, MIXED]
        .into_iter().map(move |t| (n.to_string(), f.clone(), t))).collect();
    let draw = |c: &egui::Context| {
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(egui::Color32::WHITE)).show(c, |ui| {
            for (n, f, t) in &rows {
                ui.horizontal(|ui| {
                    ui.add_sized([70.0, 20.0], egui::Label::new(egui::RichText::new(n.as_str()).size(11.0).color(egui::Color32::GRAY)));
                    ui.label(egui::RichText::new(*t).size(20.0).family(f.clone()).color(egui::Color32::BLACK));
                });
            }
        });
    };
    let _ = ctx.run(input(), draw);
    let full = ctx.run(input(), draw);
    // The font atlas as it stands after the pass.
    let atlas = ctx.fonts(|f| f.image());
    let (aw, ah) = (atlas.size[0], atlas.size[1]);
    let prims = ctx.tessellate(full.shapes, ppp);
    let (w, h) = ((760.0 * ppp) as u32, (560.0 * ppp) as u32);
    let mut img = image::RgbaImage::from_pixel(w, h, image::Rgba([255, 255, 255, 255]));
    for p in prims {
        let egui::epaint::Primitive::Mesh(m) = p.primitive else { continue };
        if m.texture_id != egui::TextureId::default() { continue; }
        for quad in m.indices.chunks(6) {
            let vs: Vec<_> = quad.iter().map(|&i| m.vertices[i as usize]).collect();
            let (x0, x1) = (vs.iter().map(|v| v.pos.x).fold(f32::MAX, f32::min), vs.iter().map(|v| v.pos.x).fold(f32::MIN, f32::max));
            let (y0, y1) = (vs.iter().map(|v| v.pos.y).fold(f32::MAX, f32::min), vs.iter().map(|v| v.pos.y).fold(f32::MIN, f32::max));
            let (u0, u1) = (vs.iter().map(|v| v.uv.x).fold(f32::MAX, f32::min), vs.iter().map(|v| v.uv.x).fold(f32::MIN, f32::max));
            let (v0, v1) = (vs.iter().map(|v| v.uv.y).fold(f32::MAX, f32::min), vs.iter().map(|v| v.uv.y).fold(f32::MIN, f32::max));
            if (u1 - u0) * aw as f32 <= 1.0 { continue; } // a solid fill, not a glyph
            let col = vs[0].color;
            for py in (y0 * ppp) as u32..((y1 * ppp) as u32).min(h) {
                for px in (x0 * ppp) as u32..((x1 * ppp) as u32).min(w) {
                    let tx = u0 + (u1 - u0) * ((px as f32 / ppp - x0) / (x1 - x0));
                    let ty = v0 + (v1 - v0) * ((py as f32 / ppp - y0) / (y1 - y0));
                    let (ax, ay) = ((tx * aw as f32) as usize, (ty * ah as f32) as usize);
                    let a = atlas.pixels.get(ay.min(ah - 1) * aw + ax.min(aw - 1)).copied().unwrap_or(0.0);
                    let blend = |bg: u8, fg: u8| (bg as f32 * (1.0 - a) + fg as f32 * a) as u8;
                    let d = img.get_pixel_mut(px, py);
                    *d = image::Rgba([blend(d[0], col.r()), blend(d[1], col.g()), blend(d[2], col.b()), 255]);
                }
            }
        }
    }
    img.save(&out).expect("write png");
    println!("wrote {out}");
}

/// Prints the face of every character of each sample in each family.
#[test]
#[ignore]
fn print_faces() {
    let ctx = probe_ctx();
    for (n, fam) in [("general", egui::FontFamily::Proportional), ("zh_hant", egui::FontFamily::Name(ZH_HANT.into())),
                     ("zh_hans", egui::FontFamily::Name(ZH_HANS.into()))] {
        for s in [SIMPLIFIED, TRADITIONAL, PUNCTUATION, MIXED] {
            let f = faces(&ctx, &fam, s);
            println!("{n:8} {}", f.iter().map(|(c, k)| format!("{c}:{k}")).collect::<Vec<_>>().join(" "));
        }
    }
}
