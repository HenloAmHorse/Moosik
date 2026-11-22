//! The EQ as the overlay and "Apply" draw it: kept and worked out again
//! band by band, the same figures as working it all out every frame, each
//! bar at its own frequency, and the overlay the very triangles the old
//! per-frame code drew, tessellated once per change; kept on the graphics
//! card, the very pixels the app's OpenGL painter drew from them, and left
//! to egui where OpenGL lacks vertex array objects.

use super::*;

/// What the overlay and "Apply" took every frame before the gains were kept:
/// each enabled band's coefficients and response, summed in order.
fn per_frame_total(bands: &[EqBand], sr: u32, freq: f32) -> f32 {
    bands.iter()
        .filter(|b| b.enabled)
        .map(|b| { let (b_a, a_a) = eq_biquad_coeffs(b, sr); biquad_response_db(b_a, a_a, freq, sr) })
        .sum()
}

/// Every kind of band, one of them switched off.
fn bands() -> Vec<EqBand> {
    let b = |freq, gain_db, q, kind| EqBand { freq, gain_db, q, kind, enabled: true };
    let mut v = vec![
        b(105.0, 6.0, 0.7, BandKind::LowShelf),
        b(31.0, -2.1, 0.9, BandKind::Peaking),
        b(220.0, -3.4, 1.4, BandKind::Peaking),
        b(1_250.0, 2.2, 2.0, BandKind::Peaking),
        b(3_100.0, -4.8, 3.5, BandKind::Peaking),
        b(6_200.0, 3.9, 5.0, BandKind::Peaking),
        b(10_000.0, -2.0, 0.7, BandKind::HighShelf),
        b(18.0, 0.0, 0.7, BandKind::HighPass),
        b(19_000.0, 0.0, 0.7, BandKind::LowPass),
        b(4_000.0, 0.0, 8.0, BandKind::Notch),
    ];
    v[4].enabled = false;
    v
}

const SCALES: [freq_scale::FreqScale; 4] = [
    freq_scale::FreqScale::Log,
    freq_scale::FreqScale::Erb,
    freq_scale::FreqScale::Blend(0.5),
    freq_scale::FreqScale::Lens { tilt: 0.3, hz: 2_000.0, oct: 2.0, gain: 2.0 },
];

/// Each band's gain and the total at every column, as the figures that were
/// worked out every frame, to the bit.
fn same_as_per_frame(grid: &EqGrid, bands: &[EqBand], (sr, w, lo, hi, scale): EqAt, what: &str) {
    let on: Vec<&EqBand> = bands.iter().filter(|b| b.enabled).collect();
    assert_eq!(grid.each.len(), on.len(), "{what}");
    assert_eq!(grid.total.len(), w + 1, "{what}");
    for px in 0..=w {
        let f = scale.freq_at(px as f32 / w as f32, lo, hi);
        for (k, b) in on.iter().enumerate() {
            let (bv, av) = eq_biquad_coeffs(b, sr);
            assert_eq!(grid.each[k].1[px].to_bits(), biquad_response_db(bv, av, f, sr).to_bits(), "{what}: band {k} px {px}");
        }
        assert_eq!(grid.total[px].to_bits(), per_frame_total(bands, sr, f).to_bits(), "{what}: px {px}");
    }
}

#[test]
fn the_kept_columns_are_the_per_frame_figures_to_the_bit() {
    let bands = bands();
    for scale in SCALES {
        let at = (48_000, 937, 20.0, 20_000.0, scale);
        let mut g = EqGains::default();
        same_as_per_frame(g.columns(&bands, at), &bands, at, &format!("{scale:?}"));
    }
}

#[test]
fn each_bar_takes_the_gain_at_its_own_centre() {
    let (sr, n, lo, hi) = (48_000, 1024, 20.0, 20_000.0);
    let bands = bands();
    for scale in SCALES {
        let mut g = EqGains::default();
        let bars = g.bars(&bands, (sr, n, lo, hi, scale)).to_vec();
        assert_eq!(bars.len(), n);
        for (i, &got) in bars.iter().enumerate() {
            let f = scale.bar_center(i, n, lo, hi);
            assert_eq!(got.to_bits(), per_frame_total(&bands, sr, f).to_bits(), "{scale:?} bar {i}");
        }
    }
    // On the Log axis that is where "Apply" always took it, by the same
    // formula, so nothing there moves.
    let mut g = EqGains::default();
    let bars = g.bars(&bands, (sr, n, lo, hi, freq_scale::FreqScale::Log)).to_vec();
    for (i, &got) in bars.iter().enumerate() {
        let t = (i as f32 + 0.5) / n as f32;
        let f = 10f32.powf(lo.log10() + t * (hi.log10() - lo.log10()));
        assert_eq!(got.to_bits(), per_frame_total(&bands, sr, f).to_bits(), "bar {i}");
    }
}

/// One grid taken through a run of edits, as a drag and the EQ panel make
/// them, is after each the grid worked out from nothing; the bars too.
#[test]
fn the_kept_gains_follow_every_change() {
    let base = bands();
    let at = (48_000, 400, 20.0, 20_000.0, freq_scale::FreqScale::Log);
    let mut g = EqGains::default();
    let mut steps: Vec<(&str, Vec<EqBand>, EqAt)> = Vec::new();
    let mut b = base.clone();
    steps.push(("the start", b.clone(), at));
    for gain in [-6.0, -2.5, 3.0] {
        b[3].gain_db = gain;
        steps.push(("a gain dragged", b.clone(), at));
    }
    b[5].freq = 7_100.0;
    steps.push(("a frequency", b.clone(), at));
    b[2].q = 4.0;
    steps.push(("a Q", b.clone(), at));
    b[6].kind = BandKind::Peaking;
    steps.push(("a kind", b.clone(), at));
    b[2].enabled = false;
    steps.push(("a band switched off", b.clone(), at));
    b[4].enabled = true;
    steps.push(("a band switched on", b.clone(), at));
    b.remove(1);
    steps.push(("a band removed", b.clone(), at));
    b.push(EqBand { gain_db: 4.0, ..EqBand::default_peak(500.0) });
    steps.push(("a band added", b.clone(), at));
    b.clear();
    steps.push(("no bands", b.clone(), at));
    steps.push(("the bands back", base.clone(), at));
    steps.push(("the rate", base.clone(), (44_100, 400, 20.0, 20_000.0, freq_scale::FreqScale::Log)));
    steps.push(("the width", base.clone(), (44_100, 500, 20.0, 20_000.0, freq_scale::FreqScale::Log)));
    steps.push(("the range", base.clone(), (44_100, 500, 20.0, 16_000.0, freq_scale::FreqScale::Log)));
    steps.push(("the axis", base.clone(), (44_100, 500, 20.0, 16_000.0, freq_scale::FreqScale::Erb)));
    steps.push(("and back", base.clone(), at));
    let mut last: Vec<f32> = Vec::new();
    for (what, bands, at) in &steps {
        same_as_per_frame(g.columns(bands, *at), bands, *at, what);
        let bars = g.bars(bands, (at.0, 64, at.2, at.3, at.4)).to_vec();
        assert_eq!(bars, EqGains::default().bars(bands, (at.0, 64, at.2, at.3, at.4)), "{what}: the bars");
        let now = g.columns(bands, *at).total.clone();
        assert!(now != last, "{what} moves the curve");
        last = now;
    }
}

/// The overlay as it was drawn every frame before anything was kept
/// (b481e2f's `draw_eq_overlay`, its total summed by `per_frame_total`, the
/// same sum): each band and the total worked out at every pixel column, a
/// segment per column.
#[allow(clippy::too_many_arguments)]
fn old_draw_eq_overlay(
    painter: &egui::Painter,
    bands: &[EqBand],
    sr: u32,
    preamp_db: f32,
    plot_rect: Rect,
    min_freq: f32,
    max_freq: f32,
    draw_curve: bool,
    hovered: Option<usize>,
    dragging: Option<usize>,
    scale: freq_scale::FreqScale,
) {
    use egui::{Color32, Pos2, Stroke};

    let y_center = plot_rect.center().y;
    let y_per_db = (plot_rect.height() * 0.5) / EQ_DB_RANGE;
    let w = plot_rect.width() as usize;

    if draw_curve && !bands.is_empty() {
        // 0 dB reference line
        painter.line_segment(
            [Pos2::new(plot_rect.left(), y_center), Pos2::new(plot_rect.right(), y_center)],
            Stroke::new(0.5, Color32::from_rgba_unmultiplied(255, 255, 255, 30)),
        );

        // Individual band curves (dimmed)
        for (i, band) in bands.iter().enumerate() {
            if !band.enabled { continue; }
            let col = eq_band_color(i).linear_multiply(0.35);
            let pts: Vec<Pos2> = (0..=w).map(|px| {
                let t = px as f32 / w as f32;
                let freq = scale.freq_at(t, min_freq, max_freq);
                let (bv, av) = eq_biquad_coeffs(band, sr);
                let db = biquad_response_db(bv, av, freq, sr).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
                Pos2::new(plot_rect.left() + t * plot_rect.width(),
                          (y_center - db * y_per_db).clamp(plot_rect.top(), plot_rect.bottom()))
            }).collect();
            for seg in pts.windows(2) {
                painter.line_segment([seg[0], seg[1]], Stroke::new(1.0, col));
            }
        }

        // Combined response curve
        let pts: Vec<Pos2> = (0..=w).map(|px| {
            let t = px as f32 / w as f32;
            let freq = scale.freq_at(t, min_freq, max_freq);
            let db = (per_frame_total(bands, sr, freq) + preamp_db).clamp(-EQ_DB_RANGE, EQ_DB_RANGE);
            Pos2::new(plot_rect.left() + t * plot_rect.width(),
                      (y_center - db * y_per_db).clamp(plot_rect.top(), plot_rect.bottom()))
        }).collect();
        for seg in pts.windows(2) {
            painter.line_segment([seg[0], seg[1]],
                Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 220, 80, 220)));
        }
    }

    // Band nodes
    for (i, band) in bands.iter().enumerate() {
        if !band.enabled { continue; }
        let pos = eq_node_pos(band, plot_rect, min_freq, max_freq, scale);
        let is_active = hovered == Some(i) || dragging == Some(i);
        let r = if is_active { 9.0 } else { 6.0 };
        let col = eq_band_color(i);
        painter.circle_filled(pos, r, col.linear_multiply(if is_active { 1.0 } else { 0.75 }));
        painter.circle_stroke(pos, r, Stroke::new(1.5, Color32::WHITE));
        // Small index label
        painter.text(pos, egui::Align2::CENTER_CENTER,
            format!("{}", i + 1),
            egui::FontId::monospace(8.0), Color32::BLACK);
    }
}

/// One frame of `draw` on a layer of its own, clipped a little outside the
/// plot, and that frame's triangles as the app hands them to the GPU.
fn frame(ctx: &egui::Context, plot: Rect, draw: impl FnMut(&egui::Painter)) -> Vec<egui::ClippedPrimitive> {
    let out = run(ctx, plot, draw);
    ctx.tessellate(out.shapes, out.pixels_per_point)
}

/// One frame of `draw` on a layer of its own, clipped a little outside the
/// plot.
fn run(ctx: &egui::Context, plot: Rect, mut draw: impl FnMut(&egui::Painter)) -> egui::FullOutput {
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("eq_gain_tests"));
    ctx.run(egui::RawInput::default(), |ctx| draw(&ctx.layer_painter(layer).with_clip_rect(plot.expand(4.0))))
}

/// Two frames' triangles are the same to the bit: vertices (position,
/// texture position, colour), indices, texture and clip, in order.
fn same_triangles(a: &[egui::ClippedPrimitive], b: &[egui::ClippedPrimitive], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: pieces");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.clip_rect, y.clip_rect, "{what}: clip");
        let (egui::epaint::Primitive::Mesh(x), egui::epaint::Primitive::Mesh(y)) = (&x.primitive, &y.primitive) else { panic!("{what}: meshes") };
        let bits = |m: &egui::Mesh| m.vertices.iter()
            .map(|v| (v.pos.x.to_bits(), v.pos.y.to_bits(), v.uv.x.to_bits(), v.uv.y.to_bits(), v.color))
            .collect::<Vec<_>>();
        assert!(x.vertices.len() > 10_000, "{what}: {} vertices", x.vertices.len());
        assert_eq!(x.vertices.len(), y.vertices.len(), "{what}: vertex count");
        assert!(bits(x) == bits(y), "{what}: the vertices");
        assert!(x.indices == y.indices, "{what}: the indices");
        assert_eq!(x.texture_id, y.texture_id, "{what}: texture");
    }
}

/// The overlay is, triangle for triangle, what the old per-frame code drew,
/// so the screen shows the same pixels: on every axis, at a fractional and a
/// whole pixel scale, with the plot off the pixel grid, first drawn and then
/// from kept triangles, through a drag, a preamp change, a band removed (the
/// colours after it move) and one switched off. A faded painter (a fading
/// window, a disabled panel) gets the segments themselves, as before.
#[test]
fn the_overlay_is_the_old_overlays_triangles() {
    let ctx = egui::Context::default();
    let plot = Rect::from_min_size(Pos2::new(40.5, 10.25), egui::vec2(800.0, 300.0));
    let (sr, lo, hi) = (48_000, 20.0, 20_000.0);
    for ppp in [1.5, 1.0] {
        ctx.set_pixels_per_point(ppp);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for scale in SCALES {
            let mut g = EqGains::default();
            let (mut bands, mut preamp) = (bands(), -6.3);
            let at = (sr, plot.width() as usize, lo, hi, scale);
            let old = |p: &egui::Painter, bands: &[EqBand], preamp| old_draw_eq_overlay(p, bands, sr, preamp, plot, lo, hi, true, Some(2), None, scale);
            // The labels' glyphs into the font atlas first.
            frame(&ctx, plot, |p| old(p, &bands, preamp));
            for step in ["drawn", "kept", "a drag", "the preamp", "a band removed", "a band switched off"] {
                match step {
                    "a drag" => bands[3].gain_db += 0.5,
                    "the preamp" => preamp -= 1.0,
                    "a band removed" => { bands.remove(4); }
                    "a band switched off" => bands[0].enabled = false,
                    _ => {}
                }
                let what = format!("ppp {ppp}, {scale:?}, {step}");
                let was = frame(&ctx, plot, |p| old(p, &bands, preamp));
                let now = frame(&ctx, plot, |p| draw_eq_overlay(p, &bands, Some((&mut g, true, at, preamp)), plot, lo, hi, Some(2), None, scale));
                same_triangles(&was, &now, &what);
            }
            let faded = |p: &egui::Painter| { let mut p = p.clone(); p.multiply_opacity(0.5); p };
            let was = frame(&ctx, plot, |p| old(&faded(p), &bands, preamp));
            let now = frame(&ctx, plot, |p| draw_eq_overlay(&faded(p), &bands, Some((&mut g, false, at, preamp)), plot, lo, hi, Some(2), None, scale));
            same_triangles(&was, &now, &format!("ppp {ppp}, {scale:?}, faded"));
        }
    }
}

/// Kept triangles are drawn again while nothing changes; a drag tessellates
/// the dragged band's curve and the combined one again, and no other; a new
/// pixel scale tessellates them all again.
#[test]
fn only_a_changed_curve_is_tessellated_again() {
    let ctx = egui::Context::default();
    let plot = Rect::from_min_size(Pos2::new(40.5, 10.25), egui::vec2(800.0, 300.0));
    let (sr, lo, hi, scale) = (48_000, 20.0, 20_000.0, freq_scale::FreqScale::Blend(0.5));
    let at = (sr, plot.width() as usize, lo, hi, scale);
    let mut g = EqGains::default();
    let mut bands = bands();
    let draw = |ctx: &egui::Context, g: &mut EqGains, bands: &[EqBand]| {
        frame(ctx, plot, |p| draw_eq_overlay(p, bands, Some((&mut *g, true, at, -6.3)), plot, lo, hi, None, None, scale));
        g.curves.iter().map(|c| c.mesh.clone()).collect::<Vec<_>>()
    };
    ctx.set_pixels_per_point(1.5);
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    let first = draw(&ctx, &mut g, &bands);
    assert_eq!(first.len(), 10, "nine bands switched on and the combined curve");
    let again = draw(&ctx, &mut g, &bands);
    assert!(first.iter().zip(&again).all(|(a, b)| Arc::ptr_eq(a, b)), "kept while nothing changes");
    bands[3].gain_db += 0.5;
    let dragged = draw(&ctx, &mut g, &bands);
    let made: Vec<usize> = (0..10).filter(|&k| !Arc::ptr_eq(&first[k], &dragged[k])).collect();
    assert_eq!(made, [3, 9], "the dragged band and the combined curve");
    ctx.set_pixels_per_point(2.0);
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    let sharper = draw(&ctx, &mut g, &bands);
    assert!(dragged.iter().zip(&sharper).all(|(a, b)| !Arc::ptr_eq(a, b)), "the pixel scale");
}

/// An OpenGL context on a window that is never shown, and a framebuffer to
/// paint into and read back: the app's painter (egui_glow) on this
/// machine's driver, with nothing on the screen.
#[cfg(windows)]
mod gl {
    use eframe::glow::{self, HasContext};
    use std::ffi::c_void;
    use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
    use windows_sys::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, CS_OWNDC, WNDCLASSW, WS_POPUP};

    /// PIXELFORMATDESCRIPTOR: sizes and flags, the twenty byte-wide fields
    /// (pixel type, colour bits, …, layer type, reserved), three masks.
    #[repr(C)]
    struct PixelFormat {
        size: u16,
        version: u16,
        flags: u32,
        bytes: [u8; 20],
        masks: [u32; 3],
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn ChoosePixelFormat(hdc: HDC, pfd: *const PixelFormat) -> i32;
        fn SetPixelFormat(hdc: HDC, format: i32, pfd: *const PixelFormat) -> i32;
    }
    #[link(name = "opengl32")]
    unsafe extern "system" {
        fn glGetString(name: u32) -> *const u8;
        fn wglCreateContext(hdc: HDC) -> *mut c_void;
        fn wglMakeCurrent(hdc: HDC, rc: *mut c_void) -> i32;
        fn wglDeleteContext(rc: *mut c_void) -> i32;
        fn wglGetProcAddress(name: *const i8) -> *const c_void;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *const c_void;
    }

    pub(super) struct HiddenGl {
        hwnd: *mut c_void,
        hdc: HDC,
        rc: *mut c_void,
        pub(super) gl: std::sync::Arc<glow::Context>,
    }

    /// `glGetString` as an OpenGL 2.1 without vertex array objects answers
    /// it: that version, and the driver's extensions less those.
    extern "system" fn old_get_string(name: u32) -> *const u8 {
        static EXTENSIONS: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
        match name {
            glow::VERSION => c"2.1.0 without vertex array objects".as_ptr().cast(),
            glow::EXTENSIONS => EXTENSIONS.get_or_init(|| {
                let all = unsafe { std::ffi::CStr::from_ptr(glGetString(name).cast()) }.to_str().unwrap();
                std::ffi::CString::new(all.split(' ').filter(|e| !e.contains("vertex_array_object")).collect::<Vec<_>>().join(" ")).unwrap()
            }).as_ptr().cast(),
            _ => unsafe { glGetString(name) },
        }
    }

    impl HiddenGl {
        /// `old`: the context as an OpenGL 2.1 without vertex array objects
        /// shows it, `glGenVertexArrays` and the like not there (glow panics
        /// when asked for one); the rest of the driver as it is.
        pub(super) fn new(old: bool) -> Self {
            unsafe {
                let class: Vec<u16> = "MoosikGlTest\0".encode_utf16().collect();
                let module = GetModuleHandleA(std::ptr::null());
                let wc = WNDCLASSW { style: CS_OWNDC, lpfnWndProc: Some(DefWindowProcW), hInstance: module, lpszClassName: class.as_ptr(), ..std::mem::zeroed() };
                RegisterClassW(&wc); // a second registration fails, and the first serves
                // Not WS_VISIBLE, and never shown.
                let hwnd = CreateWindowExW(0, class.as_ptr(), class.as_ptr(), WS_POPUP, 0, 0, 16, 16, std::ptr::null_mut(), std::ptr::null_mut(), module, std::ptr::null());
                assert!(!hwnd.is_null(), "a hidden window");
                let hdc = GetDC(hwnd);
                // Drawn to a window, OpenGL, double-buffered; RGBA, 32 colour bits, 8 alpha.
                let mut pfd = PixelFormat { size: 40, version: 1, flags: 0x25, bytes: [0; 20], masks: [0; 3] };
                (pfd.bytes[1], pfd.bytes[8]) = (32, 8);
                assert!(SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd) != 0, "a pixel format");
                let rc = wglCreateContext(hdc);
                assert!(!rc.is_null() && wglMakeCurrent(hdc, rc) != 0, "an OpenGL context");
                let lib = GetModuleHandleA(b"opengl32.dll\0".as_ptr());
                let gl = glow::Context::from_loader_function_cstr(|name| {
                    match name.to_str().unwrap_or("") {
                        "glGetString" if old => return old_get_string as *const c_void,
                        n if old && n.contains("VertexArray") => return std::ptr::null(),
                        _ => {}
                    }
                    let p = wglGetProcAddress(name.as_ptr());
                    if matches!(p as isize, -1..=3) { GetProcAddress(lib, name.as_ptr()) } else { p }
                });
                Self { hwnd, hdc, rc, gl: std::sync::Arc::new(gl) }
            }
        }

        /// A framebuffer of `size` pixels, bound for painting.
        pub(super) fn target(&self, [w, h]: [u32; 2]) {
            let gl = &self.gl;
            unsafe {
                let tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, w as i32, h as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                let fbo = gl.create_framebuffer().unwrap();
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
                assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);
            }
        }

        pub(super) fn read(&self, [w, h]: [u32; 2]) -> Vec<u8> {
            let mut px = vec![0u8; (w * h * 4) as usize];
            unsafe { self.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut px))) };
            px
        }
    }

    impl Drop for HiddenGl {
        fn drop(&mut self) {
            unsafe {
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
                wglDeleteContext(self.rc);
                ReleaseDC(self.hwnd, self.hdc);
                DestroyWindow(self.hwnd);
            }
        }
    }
}

/// One frame painted by the app's OpenGL painter into the bound framebuffer,
/// and whether any of it came as a callback.
#[cfg(windows)]
fn paint_gl(painter: &mut eframe::egui_glow::Painter, ctx: &egui::Context, out: egui::FullOutput, size: [u32; 2]) -> bool {
    let called = out.shapes.iter().any(|s| matches!(s.shape, Shape::Callback(_)));
    let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
    painter.clear(size, [0.06, 0.07, 0.08, 1.0]);
    painter.paint_and_update_textures(size, out.pixels_per_point, &prims, &out.textures_delta);
    called
}

/// Kept on the graphics card and drawn from there, the overlay paints the
/// pixels the old per-frame code painted, to the byte: painted by egui_glow,
/// the painter the app uses, on this machine's OpenGL driver; on every axis,
/// at a fractional and a whole pixel scale, first drawn and from kept
/// buffers, through a drag, a preamp change, a band removed and one
/// switched off. A faded painter still gets the segments.
#[cfg(windows)]
#[test]
fn the_overlay_kept_on_the_card_paints_the_old_overlays_pixels() {
    overlay_pixels(&gl::HiddenGl::new(false), true);
}

/// On an OpenGL without vertex array objects (2.1 lacking the extension,
/// where egui_glow emulates them), where glow panics when asked for one,
/// the curves go to egui as meshes, and the pixels are the old ones.
#[cfg(windows)]
#[test]
fn without_vertex_array_objects_the_overlay_goes_to_egui() {
    use eframe::glow::HasContext as _;
    let hidden = gl::HiddenGl::new(true);
    let lacking = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { hidden.gl.create_vertex_array() }));
    assert!(lacking.is_err(), "glow panics when asked for a vertex array object");
    overlay_pixels(&hidden, false);
}

/// The overlay's pixels against the old per-frame code's, through the
/// steps above, drawn from the graphics card if `on_card`.
#[cfg(windows)]
fn overlay_pixels(hidden: &gl::HiddenGl, on_card: bool) {
    let mut painter = eframe::egui_glow::Painter::new(hidden.gl.clone(), "", None, true).expect("egui_glow's painter");
    let ctx = egui::Context::default();
    let plot = Rect::from_min_size(Pos2::new(40.5, 10.25), egui::vec2(800.0, 300.0));
    let (sr, lo, hi) = (48_000, 20.0, 20_000.0);
    let same = |a: &[u8], b: &[u8], w: u32, what: &str| {
        let off: Vec<usize> = (0..a.len() / 4).filter(|&i| a[4 * i..4 * i + 4] != b[4 * i..4 * i + 4]).collect();
        assert!(off.is_empty(), "{what}: {} pixels differ, the first at {:?}", off.len(), off.first().map(|&i| (i % w as usize, i / w as usize)));
    };
    let mut steps = 0;
    for ppp in [1.5, 1.0] {
        ctx.set_pixels_per_point(ppp);
        let size = [((plot.max.x + 8.0) * ppp).ceil() as u32, ((plot.max.y + 8.0) * ppp).ceil() as u32];
        hidden.target(size);
        let mut shot = |out| { let called = paint_gl(&mut painter, &ctx, out, size); (hidden.read(size), called) };
        for scale in SCALES {
            let mut g = EqGains { gl: gl_curves_for(&hidden.gl), ..Default::default() };
            let (mut bands, mut preamp) = (bands(), -6.3);
            let at = (sr, plot.width() as usize, lo, hi, scale);
            let old = |p: &egui::Painter, bands: &[EqBand], preamp, curve| old_draw_eq_overlay(p, bands, sr, preamp, plot, lo, hi, curve, Some(2), None, scale);
            // The new scale's atlas and the labels' glyphs first.
            shot(run(&ctx, plot, |p| old(p, &bands, preamp, true)));
            for step in ["drawn", "kept", "a drag", "the preamp", "a band removed", "a band switched off"] {
                match step {
                    "a drag" => bands[3].gain_db += 0.5,
                    "the preamp" => preamp -= 1.0,
                    "a band removed" => { bands.remove(4); }
                    "a band switched off" => bands[0].enabled = false,
                    _ => {}
                }
                let what = format!("ppp {ppp}, {scale:?}, {step}");
                let (was, _) = shot(run(&ctx, plot, |p| old(p, &bands, preamp, true)));
                let (now, called) = shot(run(&ctx, plot, |p| draw_eq_overlay(p, &bands, Some((&mut g, true, at, preamp)), plot, lo, hi, Some(2), None, scale)));
                assert_eq!(called, on_card, "{what}: drawn from the card");
                same(&was, &now, size[0], &what);
                let (bare, _) = shot(run(&ctx, plot, |p| old(p, &bands, preamp, false)));
                assert!(bare != now, "{what}: the curves are in the picture");
                steps += 1;
            }
            assert!(g.gl.as_ref().is_none_or(|c| !c.lock().unwrap().failed));
            let faded = |p: &egui::Painter| { let mut p = p.clone(); p.multiply_opacity(0.5); p };
            let (was, _) = shot(run(&ctx, plot, |p| old(&faded(p), &bands, preamp, true)));
            let (now, called) = shot(run(&ctx, plot, |p| draw_eq_overlay(&faded(p), &bands, Some((&mut g, false, at, preamp)), plot, lo, hi, Some(2), None, scale)));
            assert!(!called, "faded: the segments themselves");
            same(&was, &now, size[0], &format!("ppp {ppp}, {scale:?}, faded"));
        }
    }
    assert_eq!(steps, 48);
    painter.destroy();
}

/// What a frame of the overlay costs, painted by egui_glow as the app paints
/// it: the curves' triangles handed to egui (copied and sent to the card
/// every frame) against kept on the card. Opt-in, machine-dependent.
#[cfg(windows)]
#[test]
#[ignore]
fn eq_overlay_gl_frame_cost() {
    use eframe::glow::HasContext as _;
    let hidden = gl::HiddenGl::new(false);
    let mut painter = eframe::egui_glow::Painter::new(hidden.gl.clone(), "", None, true).expect("egui_glow's painter");
    let ctx = egui::Context::default();
    let plot = Rect::from_min_size(Pos2::new(40.0, 40.0), egui::vec2(1500.0, 600.0));
    let (sr, lo, hi) = (48_000, 20.0, 20_000.0);
    let many = |k: usize| -> Vec<EqBand> {
        (0..k).map(|i| EqBand { freq: 25.0 * 1.25f32.powi(i as i32), gain_db: if i % 2 == 0 { 3.0 } else { -2.5 }, q: 1.4, kind: BandKind::Peaking, enabled: true }).collect()
    };
    for ppp in [1.0, 1.5] {
        ctx.set_pixels_per_point(ppp);
        let size = [((plot.max.x + 40.0) * ppp).ceil() as u32, ((plot.max.y + 40.0) * ppp).ceil() as u32];
        hidden.target(size);
        for scale in [freq_scale::FreqScale::Log, freq_scale::FreqScale::Blend(0.5)] {
            for (k, drag) in [(0, false), (10, false), (32, false), (32, true)] {
                for kept in [false, true] {
                    let mut g = EqGains { gl: kept.then(Default::default), ..Default::default() };
                    let mut bands = many(k);
                    let at = (sr, plot.width() as usize, lo, hi, scale);
                    let mut spent = std::time::Duration::ZERO;
                    for i in 0..45 {
                        if drag {
                            bands[k / 2].gain_db = 3.0 + (i % 7) as f32 * 0.1;
                        }
                        let start = std::time::Instant::now();
                        let out = run(&ctx, plot, |p| draw_eq_overlay(p, &bands, (k > 0).then_some((&mut g, true, at, -3.0)), plot, lo, hi, None, None, scale));
                        paint_gl(&mut painter, &ctx, out, size);
                        unsafe { hidden.gl.finish() };
                        if i >= 5 {
                            spent += start.elapsed();
                        }
                    }
                    println!("ppp {ppp} {scale:?} {k} bands{}: {} {:.2} ms a frame",
                        if drag { ", dragged" } else { "" }, if kept { "kept on the card" } else { "sent every frame" }, spent.as_secs_f64() * 1e3 / 40.0);
                }
            }
        }
    }
    painter.destroy();
}
