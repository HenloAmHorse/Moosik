//! A music video behind Listen and the spectrum (1.6.1). YouTube delivers
//! the picture as its own stream; Moosik keeps it beside the audio as
//! `<name>.video.webm` (VP9) and Windows decodes it (Media Foundation, so
//! nothing more to install; YouTube's MP4 video comes fragmented, which
//! Media Foundation reads as empty). The audio is still the file the engine plays,
//! bit-perfect or not; the picture is muted and only follows the song's
//! position, on a thread of its own.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The video kept beside `audio`, if there is one: YouTube's
/// `<song>.video.webm` (in, or still arriving), or one of the user's own
/// as `.video.mp4` or `.video.mkv` (Windows plays those too, if not
/// fragmented).
pub fn beside(audio: &Path) -> Option<PathBuf> {
    ["video.webm", "video.mp4", "video.mkv"].iter().map(|e| audio.with_extension(e))
        .find(|v| v.is_file() || crate::youtube::stream::downloading(v))
}

/// A WebM video's picture height, from its header; None for anything else
/// (the user's own MP4 or MKV, say).
pub fn webm_height(path: &Path) -> Option<u32> {
    use std::io::Read;
    let mut head = vec![0u8; 64 << 10];
    let n = std::fs::File::open(path).ok()?.read(&mut head).ok()?;
    pixel_height(&head[..n])
}

/// PixelHeight in a WebM header: into Segment, Tracks, TrackEntry and
/// Video, past everything else.
fn pixel_height(b: &[u8]) -> Option<u32> {
    // An EBML number, its length told by its first byte's leading zeros;
    // ids keep that marker bit, sizes do not.
    let vint = |i: usize, id: bool| -> Option<(u64, usize)> {
        let first = *b.get(i)?;
        let n = first.leading_zeros() as usize + 1;
        if n > 8 {
            return None;
        }
        let mut v = if id { first as u64 } else { first as u64 & (0xFF >> n) };
        for k in 1..n {
            v = (v << 8) | *b.get(i + k)? as u64;
        }
        Some((v, n))
    };
    const INTO: [u64; 4] = [0x1853_8067, 0x1654_AE6B, 0xAE, 0xE0];
    let mut i = 0;
    while i < b.len() {
        let (id, n) = vint(i, true)?;
        let (size, m) = vint(i + n, false)?;
        i += n + m;
        if id == 0xBA {
            let v = b.get(i..i.checked_add(usize::try_from(size).ok()?)?)?;
            return (v.len() <= 4).then(|| v.iter().fold(0, |a, &x| (a << 8) | x as u32));
        }
        if !INTO.contains(&id) {
            i = i.checked_add(usize::try_from(size).ok()?)?;
        }
    }
    None
}

/// One decoded picture, ready for a texture (made on the decoding thread,
/// so the UI only hands it over), for the moment `at_ms`; `content` is the
/// part inside any black bars (left, top, right, bottom as fractions), as
/// found in the first bright enough picture.
pub struct Frame {
    pub image: egui::ColorImage,
    pub content: [f32; 4],
    #[cfg_attr(not(test), allow(dead_code))]
    pub at_ms: u64,
}

struct Shared {
    /// Where the song is, in milliseconds.
    want: AtomicU64,
    /// The largest picture worth handing over (the screen's size): a
    /// bigger video is scaled down on the graphics card.
    max: [u32; 2],
    /// Decode on the processor even where the graphics card could; and
    /// whether the graphics card is decoding.
    software: AtomicBool,
    gpu: AtomicBool,
    /// The video's own picture size (width << 32 | height), once open.
    native: AtomicU64,
    stop: AtomicBool,
    /// The newest picture not yet taken by the UI.
    frame: Mutex<Option<Frame>>,
    failed: Mutex<Option<String>>,
}

/// A video being played along with the song. Dropping it stops the thread.
pub struct Video {
    pub path: PathBuf,
    shared: Arc<Shared>,
}

impl Video {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn open(path: PathBuf) -> Video {
        Self::open_with(path, false, [u32::MAX; 2])
    }

    /// `open`, with `software` keeping the graphics card out of it, and
    /// pictures no larger than `max` (pixels) where the card can scale them.
    pub fn open_with(path: PathBuf, software: bool, max: [u32; 2]) -> Video {
        let shared = Arc::new(Shared {
            want: AtomicU64::new(0),
            max,
            software: AtomicBool::new(software),
            gpu: AtomicBool::new(false),
            native: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            frame: Mutex::new(None),
            failed: Mutex::new(None),
        });
        let s = shared.clone();
        let p = path.clone();
        let _ = std::thread::Builder::new().name("video".into()).spawn(move || {
            let mut r = decode(&p, &s);
            // A graphics card that gives up part way: the processor carries
            // on from where the song is.
            if r.is_err() && s.gpu.swap(false, Ordering::Relaxed) && !s.stop.load(Ordering::Relaxed) {
                crate::mlog!("video   graphics card stopped decoding ({:?}); going on without it", r.as_ref().err());
                s.software.store(true, Ordering::Relaxed);
                r = decode(&p, &s);
            }
            if let Err(e) = r {
                *s.failed.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
            }
        });
        Video { path, shared }
    }

    /// The song is at `pos`.
    pub fn follow(&self, pos: std::time::Duration) {
        self.shared.want.store(pos.as_millis() as u64, Ordering::Relaxed);
    }

    /// The newest picture since the last call.
    pub fn take(&self) -> Option<Frame> {
        self.shared.frame.lock().unwrap_or_else(|p| p.into_inner()).take()
    }

    /// The video's own picture size, once it is open.
    pub fn native_size(&self) -> Option<[u32; 2]> {
        let n = self.shared.native.load(Ordering::Relaxed);
        (n != 0).then_some([(n >> 32) as u32, n as u32])
    }

    /// The graphics card is decoding it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn on_gpu(&self) -> bool {
        self.shared.gpu.load(Ordering::Relaxed)
    }

    /// Why the video cannot be shown, if it cannot.
    pub fn failed(&self) -> Option<String> {
        self.shared.failed.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }
}

/// What the UI draws: the texture, the size of its picture without black
/// bars, and where that part is in the texture.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shown {
    pub tex: egui::TextureId,
    pub size: egui::Vec2,
    pub uv: egui::Rect,
}

impl Shown {
    /// The texture part that fills `target`, cropping what overflows.
    pub fn fill_uv(&self, target: egui::Vec2) -> egui::Rect {
        let k = (target.x / self.size.x.max(1.0)).max(target.y / self.size.y.max(1.0));
        let f = target / (self.size * k);
        egui::Rect::from_center_size(self.uv.center(), self.uv.size() * f)
    }

    /// Where the whole picture fits in `target`, as large as it can (draw
    /// `self.uv` there).
    pub fn fit(&self, target: egui::Rect) -> egui::Rect {
        let k = (target.width() / self.size.x.max(1.0)).min(target.height() / self.size.y.max(1.0));
        egui::Rect::from_center_size(target.center(), self.size * k)
    }
}

/// `size` scaled down to fit `max`, keeping its shape (even sides, as
/// the decoders like); None when it fits already.
fn fit_within([w, h]: [u32; 2], [mw, mh]: [u32; 2]) -> Option<[u32; 2]> {
    if w <= mw && h <= mh || w == 0 || h == 0 {
        return None;
    }
    let k = (mw as f64 / w as f64).min(mh as f64 / h as f64);
    let even = |v: f64| ((v / 2.0).floor() as u32 * 2).max(2);
    Some([even(w as f64 * k), even(h as f64 * k)])
}

/// The picture inside black bars, from the luma plane (`stride` bytes a
/// row): left, top, right, bottom as fractions. None for a picture too dark
/// to tell, or one whose "bars" would leave less than 40% of it.
fn content_box(y: &[u8], w: usize, h: usize, stride: usize) -> Option<[f32; 4]> {
    const BLACK: u32 = 28;
    let px = |x: usize, r: usize| y.get(r * stride + x).copied().unwrap_or(0) as u32;
    let rows: Vec<usize> = (0..32).map(|i| (i * 2 + 1) * h / 64).collect();
    let cols: Vec<usize> = (0..32).map(|i| (i * 2 + 1) * w / 64).collect();
    let col_lit = |x: usize| rows.iter().map(|&r| px(x, r)).sum::<u32>() / rows.len() as u32 > BLACK;
    let row_lit = |r: usize| cols.iter().map(|&x| px(x, r)).sum::<u32>() / cols.len() as u32 > BLACK;
    let mean = rows.iter().flat_map(|&r| cols.iter().map(move |&x| (x, r))).map(|(x, r)| px(x, r)).sum::<u32>() / 1024;
    if mean < 40 {
        return None;
    }
    let (x0, x1) = ((0..w).find(|&x| col_lit(x))?, (0..w).rev().find(|&x| col_lit(x))? + 1);
    let (y0, y1) = ((0..h).find(|&r| row_lit(r))?, (0..h).rev().find(|&r| row_lit(r))? + 1);
    if (x1 - x0) * 10 < w * 4 || (y1 - y0) * 10 < h * 4 {
        return None;
    }
    // A pixel or two in, past the soft edge of the bars.
    let (x0, x1) = (if x0 > 0 { x0 + 2 } else { 0 }, if x1 < w { x1 - 2 } else { w });
    let (y0, y1) = (if y0 > 0 { y0 + 2 } else { 0 }, if y1 < h { y1 - 2 } else { h });
    Some([x0 as f32 / w as f32, y0 as f32 / h as f32, x1 as f32 / w as f32, y1 as f32 / h as f32])
}

#[cfg(not(windows))]
fn decode(_: &Path, _: &Arc<Shared>) -> Result<(), String> {
    Err("videos play on Windows only".into())
}

/// A Direct3D 11 device for Media Foundation's decoders to use.
#[cfg(windows)]
fn gpu_manager() -> windows::core::Result<windows::Win32::Media::MediaFoundation::IMFDXGIDeviceManager> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Media::MediaFoundation::MFCreateDXGIDeviceManager;
    // SAFETY: plain Direct3D and Media Foundation calls on objects made here.
    unsafe {
        let mut dev = None;
        D3D11CreateDevice(None, D3D_DRIVER_TYPE_HARDWARE, None, D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None, D3D11_SDK_VERSION, Some(&mut dev), None, None)?;
        let dev = dev.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
        // Media Foundation uses it from threads of its own.
        if let Ok(mt) = dev.cast::<ID3D11Multithread>() {
            let _ = mt.SetMultithreadProtected(true);
        }
        let (mut token, mut m) = (0u32, None);
        MFCreateDXGIDeviceManager(&mut token, &mut m)?;
        let m = m.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
        m.ResetDevice(&dev, token)?;
        Ok(m)
    }
}

/// Media Foundation is delay-loaded (build.rs), so its first call would end
/// the process where Windows N lacks it; this asks once whether its two DLLs
/// are in System32. They stay loaded.
#[cfg(windows)]
fn media_foundation() -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut std::ffi::c_void, flags: u32) -> *mut std::ffi::c_void;
    }
    const SEARCH_SYSTEM32: u32 = 0x800;
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| ["mfplat.dll", "mfreadwrite.dll"].iter().all(|dll| {
        let name: Vec<u16> = dll.encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated name and no file handle.
        !unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), SEARCH_SYSTEM32) }.is_null()
    }))
}

/// Read pictures in order, handing over the last one at or before the
/// song's position; seek when the song jumps.
#[cfg(windows)]
fn decode(path: &Path, s: &Arc<Shared>) -> Result<(), String> {
    use windows::core::{HSTRING, PROPVARIANT};
    use windows::Win32::Media::MediaFoundation::*;
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
    let e = |e: windows::core::Error| e.message().to_string();
    const VIDEO: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
    if !media_foundation() {
        return Err("Windows Media Foundation is missing (on Windows N, install the Media Feature Pack)".into());
    }
    // SAFETY: COM and Media Foundation calls on this thread's own objects,
    // with the buffer read only while it is locked.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(e)?;
        // The graphics card decodes where it can; else the processor does.
        let open = |gpu: Option<&IMFDXGIDeviceManager>| -> windows::core::Result<IMFSourceReader> {
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 2)?;
            let attrs = attrs.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            if let Some(m) = gpu {
                attrs.SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, m)?;
                attrs.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)?;
                // On the card, its video processor turns the pictures to RGB
                // as well, which leaves the processor only a byte swap.
                attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)?;
            }
            let reader = if crate::youtube::stream::downloading(path) {
                arriving_reader(path, s, &attrs)?
            } else {
                MFCreateSourceReaderFromURL(&HSTRING::from(path.as_os_str()), &attrs)?
            };
            reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
            reader.SetStreamSelection(VIDEO, true)?;
            let t = MFCreateMediaType()?;
            t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            // The decoder's own format: converting every picture to RGB in
            // Media Foundation costs more than decoding; only the shown are.
            t.SetGUID(&MF_MT_SUBTYPE, if gpu.is_some() { &MFVideoFormat_RGB32 } else { &MFVideoFormat_NV12 })?;
            // A 2K or 4K video larger than the screen comes down to its size
            // on the card: every picture is copied and uploaded once more.
            if gpu.is_some()
                && let Ok(size) = reader.GetNativeMediaType(VIDEO, 0).and_then(|n| n.GetUINT64(&MF_MT_FRAME_SIZE))
                && let Some([w, h]) = fit_within([(size >> 32) as u32, size as u32], s.max)
            {
                t.SetUINT64(&MF_MT_FRAME_SIZE, (u64::from(w) << 32) | u64::from(h))?;
            }
            reader.SetCurrentMediaType(VIDEO, None, &t)?;
            Ok(reader)
        };
        let gpu = if s.software.load(Ordering::Relaxed) { None } else { gpu_manager().ok() };
        let reader = match gpu.as_ref().map(|m| open(Some(m))) {
            Some(Ok(r)) => {
                s.gpu.store(true, Ordering::Relaxed);
                r
            }
            _ => open(None).map_err(|e| format!("Windows cannot decode this video ({})", e.message()))?,
        };
        if let Ok(n) = reader.GetNativeMediaType(VIDEO, 0).and_then(|n| n.GetUINT64(&MF_MT_FRAME_SIZE)) {
            s.native.store(n, Ordering::Relaxed);
        }
        let cur = reader.GetCurrentMediaType(VIDEO).map_err(e)?;
        let size = cur.GetUINT64(&MF_MT_FRAME_SIZE).map_err(e)?;
        let (w, h) = ((size >> 32) as usize, (size & 0xffff_ffff) as usize);
        let stride = cur.GetUINT32(&MF_MT_DEFAULT_STRIDE).map_or(w, |s| (s as i32).unsigned_abs() as usize);
        if w == 0 || h == 0 {
            return Err("the video has no picture size".into());
        }
        // The next sample, read but not yet due; the time of the last shown.
        let mut ahead: Option<(u64, IMFSample)> = None;
        let mut shown: Option<u64> = None;
        let mut ended = false;
        // The black bars, once a picture bright enough has shown them.
        let mut content: Option<[f32; 4]> = None;
        // A seek lands on the keyframe before its target, which may be
        // seconds early; decode forward from there rather than seek again.
        let mut sought = 0u64;
        while !s.stop.load(Ordering::Relaxed) {
            let want = s.want.load(Ordering::Relaxed);
            let behind = shown.is_some_and(|t| want + 250 < t);
            let far = ahead.as_ref().is_some_and(|(t, _)| want > t + 3000) && want > sought + 3000;
            if behind || far {
                reader.SetCurrentPosition(&windows::core::GUID::zeroed(), &PROPVARIANT::from(want as i64 * 10_000)).map_err(e)?;
                (ahead, shown, ended, sought) = (None, None, false, want);
            }
            if ahead.is_none() && !ended {
                let (mut flags, mut ts, mut sample) = (0u32, 0i64, None);
                reader.ReadSample(VIDEO, 0, None, Some(&mut flags), Some(&mut ts), Some(&mut sample)).map_err(e)?;
                if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                    ended = true;
                }
                if let Some(sample) = sample {
                    ahead = Some(((ts.max(0) / 10_000) as u64, sample));
                }
                continue;
            }
            match &ahead {
                Some((t, _)) if *t <= want => {
                    let (t, sample) = ahead.take().expect("just matched");
                    // Only the last picture before the song's position is
                    // worth converting; read on while the next is due too.
                    let (mut flags, mut ts, mut next) = (0u32, 0i64, None);
                    reader.ReadSample(VIDEO, 0, None, Some(&mut flags), Some(&mut ts), Some(&mut next)).map_err(e)?;
                    ended = flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0;
                    let next = next.map(|n| ((ts.max(0) / 10_000) as u64, n));
                    // At most 30 pictures a second: a 60 fps video would
                    // otherwise cost twice as much for a backdrop.
                    let soon = shown.is_some_and(|s| t < s + 30);
                    if next.as_ref().is_some_and(|(nt, _)| *nt <= want) || (soon && next.is_some()) {
                        ahead = next;
                        continue;
                    }
                    ahead = next;
                    let buf = sample.ConvertToContiguousBuffer().map_err(e)?;
                    let (mut p, mut len) = (std::ptr::null_mut(), 0u32);
                    buf.Lock(&mut p, None, Some(&mut len)).map_err(e)?;
                    let bytes = std::slice::from_raw_parts(p, len as usize);
                    let rgb = s.gpu.load(Ordering::Relaxed);
                    let pixels = if rgb { bgrx_pixels(bytes, w, h) } else { nv12_pixels(bytes, w, h, stride) };
                    if content.is_none() {
                        content = if rgb {
                            // Green stands in for brightness well enough to find black bars.
                            let g: Vec<u8> = pixels.iter().map(|p| p.g()).collect();
                            content_box(&g, w, h, w)
                        } else {
                            content_box(bytes, w, h, stride)
                        };
                    }
                    let _ = buf.Unlock();
                    *s.frame.lock().unwrap_or_else(|p| p.into_inner()) = Some(Frame { image: egui::ColorImage { size: [w, h], pixels }, content: content.unwrap_or([0.0, 0.0, 1.0, 1.0]), at_ms: t });
                    shown = Some(t);
                }
                _ => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
        let _ = MFShutdown();
    }
    Ok(())
}

/// A reader for a video still arriving: Media Foundation reads it through
/// `Arriving`, waiting at the end of what is in, until the video closes.
#[cfg(windows)]
fn arriving_reader(path: &Path, s: &Arc<Shared>, attrs: &windows::Win32::Media::MediaFoundation::IMFAttributes)
    -> windows::core::Result<windows::Win32::Media::MediaFoundation::IMFSourceReader>
{
    use windows::core::Interface;
    use windows::Win32::Media::MediaFoundation::*;
    let stop = s.clone();
    let src = crate::youtube::stream::open_until(path, move || stop.stop.load(Ordering::Relaxed))
        .map_err(|e| windows::core::Error::new(windows::Win32::Foundation::E_FAIL, e.to_string()))?;
    let len = symphonia::core::io::MediaSource::byte_len(&src).unwrap_or(0);
    let stream: windows::Win32::System::Com::IStream = Arriving { src: Mutex::new(src), len }.into();
    // SAFETY: Media Foundation calls on objects made here.
    unsafe {
        let bytes = MFCreateMFByteStreamOnStream(&stream)?;
        // What it is, said rather than sniffed.
        if let Ok(a) = bytes.cast::<IMFAttributes>() {
            let _ = a.SetString(&MF_BYTESTREAM_CONTENT_TYPE, windows::core::w!("video/webm"));
        }
        MFCreateSourceReaderFromByteStream(&bytes, attrs)
    }
}

/// A video still arriving, as the COM stream Media Foundation reads.
#[cfg(windows)]
#[windows_core::implement(windows::Win32::System::Com::IStream)]
struct Arriving {
    src: Mutex<crate::youtube::stream::Source>,
    len: u64,
}

#[cfg(windows)]
impl windows::Win32::System::Com::ISequentialStream_Impl for Arriving_Impl {
    fn Read(&self, pv: *mut core::ffi::c_void, cb: u32, pcbread: *mut u32) -> windows::core::HRESULT {
        use std::io::Read;
        use windows::Win32::Foundation::{E_FAIL, S_FALSE, S_OK};
        if pv.is_null() {
            return E_FAIL;
        }
        // SAFETY: Media Foundation hands a buffer of `cb` bytes.
        let buf = unsafe { std::slice::from_raw_parts_mut(pv as *mut u8, cb as usize) };
        let mut src = self.src.lock().unwrap_or_else(|p| p.into_inner());
        // Whole reads but at the end: a short one is taken for the end.
        let mut n = 0;
        let mut failed = false;
        while n < buf.len() {
            match src.read(&mut buf[n..]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if !pcbread.is_null() {
            // SAFETY: a place for the count, given with the buffer.
            unsafe { *pcbread = n as u32 };
        }
        if failed { E_FAIL } else if n == buf.len() { S_OK } else { S_FALSE }
    }

    fn Write(&self, _: *const core::ffi::c_void, _: u32, _: *mut u32) -> windows::core::HRESULT {
        windows::Win32::Foundation::STG_E_ACCESSDENIED
    }
}

#[cfg(windows)]
impl windows::Win32::System::Com::IStream_Impl for Arriving_Impl {
    fn Seek(&self, by: i64, from: windows::Win32::System::Com::STREAM_SEEK, at: *mut u64) -> windows::core::Result<()> {
        use std::io::{Seek, SeekFrom};
        use windows::Win32::System::Com::{STREAM_SEEK_CUR, STREAM_SEEK_END, STREAM_SEEK_SET};
        let bad = || windows::core::Error::from(windows::Win32::Foundation::STG_E_INVALIDFUNCTION);
        let to = match from {
            STREAM_SEEK_SET => SeekFrom::Start(u64::try_from(by).map_err(|_| bad())?),
            STREAM_SEEK_CUR => SeekFrom::Current(by),
            STREAM_SEEK_END => SeekFrom::End(by),
            _ => return Err(bad()),
        };
        let p = self.src.lock().unwrap_or_else(|p| p.into_inner()).seek(to).map_err(|_| bad())?;
        if !at.is_null() {
            // SAFETY: a place for the new position, given with the call.
            unsafe { *at = p };
        }
        Ok(())
    }

    fn SetSize(&self, _: u64) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }

    fn CopyTo(&self, _: Option<&windows::Win32::System::Com::IStream>, _: u64, _: *mut u64, _: *mut u64) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }

    fn Commit(&self, _: &windows::Win32::System::Com::STGC) -> windows::core::Result<()> {
        Ok(())
    }

    fn Revert(&self) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }

    fn LockRegion(&self, _: u64, _: u64, _: &windows::Win32::System::Com::LOCKTYPE) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::STG_E_INVALIDFUNCTION.into())
    }

    fn UnlockRegion(&self, _: u64, _: u64, _: u32) -> windows::core::Result<()> {
        Err(windows::Win32::Foundation::STG_E_INVALIDFUNCTION.into())
    }

    fn Stat(&self, st: *mut windows::Win32::System::Com::STATSTG, _: &windows::Win32::System::Com::STATFLAG) -> windows::core::Result<()> {
        if st.is_null() {
            return Err(windows::Win32::Foundation::E_POINTER.into());
        }
        // SAFETY: a place for the answer, given with the call.
        unsafe {
            *st = windows::Win32::System::Com::STATSTG {
                r#type: windows::Win32::System::Com::STGTY_STREAM.0 as u32,
                cbSize: self.len,
                ..Default::default()
            };
        }
        Ok(())
    }

    fn Clone(&self) -> windows::core::Result<windows::Win32::System::Com::IStream> {
        Err(windows::Win32::Foundation::E_NOTIMPL.into())
    }
}

/// Windows' top-down BGRX rows (each `len / h` bytes) as opaque pixels.
fn bgrx_pixels(bgrx: &[u8], w: usize, h: usize) -> Vec<egui::Color32> {
    let stride = (bgrx.len() / h.max(1)).max(w * 4);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let Some(src) = bgrx.get(y * stride..y * stride + w * 4) else { break };
        out.extend(src.chunks_exact(4).map(|b| egui::Color32::from_rgb(b[2], b[1], b[0])));
    }
    out.resize(w * h, egui::Color32::BLACK);
    out
}

/// NV12 (a Y plane, then interleaved U and V at half size; rows `stride`
/// bytes, the planes possibly taller than the picture) to RGBA, BT.709
/// limited range as YouTube encodes.
fn nv12_pixels(buf: &[u8], w: usize, h: usize, stride: usize) -> Vec<egui::Color32> {
    let stride = stride.max(w);
    // The Y plane's height, from the buffer: two thirds of its rows.
    let y_rows = (buf.len() / stride * 2 / 3).max(h);
    let uv = y_rows * stride;
    let mut out = vec![egui::Color32::BLACK; w * h];
    for (y, row) in out.chunks_exact_mut(w).enumerate() {
        let (Some(ys), Some(uvs)) = (buf.get(y * stride..y * stride + w), buf.get(uv + y / 2 * stride..uv + y / 2 * stride + (w + 1) / 2 * 2)) else { break };
        for (x, o) in row.iter_mut().enumerate() {
            let c = 298 * (ys[x] as i32 - 16);
            let (d, e) = (uvs[x / 2 * 2] as i32 - 128, uvs[x / 2 * 2 + 1] as i32 - 128);
            let px = |v: i32| ((v + 128) >> 8).clamp(0, 255) as u8;
            *o = egui::Color32::from_rgb(px(c + 459 * e), px(c - 55 * d - 136 * e), px(c + 541 * d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_video_is_found_beside_its_song() {
        let dir = std::env::temp_dir().join(format!("moosik-beside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let song = dir.join("Artist - Song.flac");
        assert_eq!(beside(&song), None);
        std::fs::write(dir.join("Artist - Song.video.mp4"), b"").unwrap();
        assert_eq!(beside(&song), Some(dir.join("Artist - Song.video.mp4")));
        std::fs::write(dir.join("Artist - Song.video.webm"), b"").unwrap();
        assert_eq!(beside(&song), Some(dir.join("Artist - Song.video.webm")), "YouTube's first");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_webm_says_how_tall_its_picture_is() {
        let mut b = vec![0x1A, 0x45, 0xDF, 0xA3, 0x80]; // EBML header, empty
        b.extend([0x18, 0x53, 0x80, 0x67, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // Segment, size unknown
        b.extend([0x11, 0x4D, 0x9B, 0x74, 0x82, 0xEC, 0x80]); // SeekHead, passed over
        b.extend([0x16, 0x54, 0xAE, 0x6B, 0x8F, 0xAE, 0x8D, 0xD7, 0x81, 0x01]); // Tracks, TrackEntry, TrackNumber
        b.extend([0xE0, 0x88, 0xB0, 0x82, 0x05, 0x00, 0xBA, 0x82, 0x02, 0xD0]); // Video: 1280 × 720
        assert_eq!(pixel_height(&b), Some(720));
        assert_eq!(pixel_height(&b[..20]), None);
        assert_eq!(pixel_height(b"ID3 not a webm"), None);
    }

    #[test]
    fn a_picture_bigger_than_the_screen_is_scaled_to_fit_it() {
        let screen = [2560, 1440];
        assert_eq!(fit_within([3840, 2160], screen), Some([2560, 1440]));
        assert_eq!(fit_within([2560, 1440], screen), None);
        assert_eq!(fit_within([1920, 1080], screen), None);
        // A wide one fits the width, a tall one the height; sides stay even.
        assert_eq!(fit_within([3840, 1600], screen), Some([2560, 1066]));
        assert_eq!(fit_within([2160, 3840], screen), Some([810, 1440]));
        assert_eq!(fit_within([0, 0], screen), None);
    }

    #[test]
    fn black_bars_are_found_and_dark_pictures_left_alone() {
        // 64×36: a 4:3 picture (48 wide) between black bars 8 wide.
        let (w, h) = (64, 36);
        let y: Vec<u8> = (0..w * h).map(|i| if (8..56).contains(&(i % w)) { 120 } else { 16 }).collect();
        let b = content_box(&y, w, h, w).unwrap();
        assert_eq!(b, [10.0 / 64.0, 0.0, 54.0 / 64.0, 1.0]);
        assert_eq!(content_box(&vec![120; w * h], w, h, w), Some([0.0, 0.0, 1.0, 1.0]));
        assert_eq!(content_box(&vec![20; w * h], w, h, w), None, "too dark to tell");
        // Filling a wide target from the 4:3 part crops its top and bottom.
        let s = Shown { tex: egui::TextureId::default(), size: egui::vec2(44.0, 36.0), uv: egui::Rect::from_min_max(egui::pos2(b[0], b[1]), egui::pos2(b[2], b[3])) };
        let uv = s.fill_uv(egui::vec2(88.0, 36.0));
        assert!((uv.width() - s.uv.width()).abs() < 1e-6 && (uv.height() - 0.5).abs() < 1e-6, "{uv:?}");
        assert_eq!(s.fit(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(88.0, 36.0))).width(), 44.0);
    }

    #[test]
    fn nv12_greys_and_colours_come_out_right() {
        // 2×2 picture, rows padded to 4 bytes: black, white; mid grey, then
        // one U/V pair. Neutral chroma gives greys.
        let grey = [16, 235, 0, 0, 126, 126, 0, 0, 128, 128, 0, 0];
        let out = nv12_pixels(&grey, 2, 2, 4);
        assert_eq!(&out[..2], &[egui::Color32::BLACK, egui::Color32::WHITE]);
        assert!([out[2].r(), out[2].g(), out[2].b()].iter().all(|&c| (126..=130).contains(&c)), "{:?}", out[2]);
        // Strong V (red-difference) is red.
        let red = [81, 81, 0, 0, 81, 81, 0, 0, 90, 240, 0, 0];
        let px = nv12_pixels(&red, 2, 2, 4)[0];
        assert!(px.r() > 200 && px.g() < 60 && px.b() < 60, "{px:?}");
    }

    /// A real YouTube video stream, set by MOOSIK_TEST_VIDEO; decodes
    /// pictures for a position and again after a jump back.
    #[test]
    #[ignore]
    fn windows_decodes_a_youtube_video() {
        let Ok(p) = std::env::var("MOOSIK_TEST_VIDEO") else { return };
        let v = Video::open(PathBuf::from(p));
        let wait = |v: &Video| {
            for _ in 0..300 {
                if let Some(f) = v.take() {
                    return Some(f);
                }
                assert_eq!(v.failed(), None);
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            None
        };
        v.follow(std::time::Duration::from_secs(10));
        let f = wait(&v).expect("a picture at 10 s");
        assert!(f.at_ms <= 10_000 && f.at_ms > 9_000, "{}", f.at_ms);
        assert_eq!(f.image.pixels.len(), f.image.size[0] * f.image.size[1]);
        assert!(f.image.pixels.iter().any(|p| p.r() > 40), "not all black");
        v.follow(std::time::Duration::from_secs(2));
        let f = wait(&v).expect("a picture after jumping back");
        assert!(f.at_ms <= 2_000 && f.at_ms > 1_000, "{}", f.at_ms);
    }

    /// A video shows while it is still arriving: written a slice at a time
    /// into its `.part`, as yt-dlp does, and renamed once all in.
    #[test]
    #[ignore]
    fn a_video_still_arriving_shows() {
        let Ok(p) = std::env::var("MOOSIK_TEST_VIDEO") else { return };
        eprintln!("{} pixels tall", webm_height(Path::new(&p)).unwrap());
        let data = std::fs::read(&p).unwrap();
        let dir = std::env::temp_dir().join(format!("moosik-arriving-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (file, part) = (dir.join("x.video.webm"), dir.join("x.video.webm.part"));
        std::fs::write(&part, &data[..4096]).unwrap();
        crate::youtube::stream::begin(&file, part.clone(), data.len() as u64);
        let (f2, d2) = (file.clone(), data.clone());
        let writer = std::thread::spawn(move || {
            use std::io::Write;
            let mut w = std::fs::OpenOptions::new().append(true).open(&part).unwrap();
            // About 2 MB a second.
            for c in d2[4096..].chunks(100_000) {
                std::thread::sleep(std::time::Duration::from_millis(50));
                w.write_all(c).unwrap();
            }
            drop(w);
            std::fs::rename(&part, &f2).unwrap();
            crate::youtube::stream::end(&f2, true);
        });
        let v = Video::open(file.clone());
        v.follow(std::time::Duration::from_secs(2));
        let t = std::time::Instant::now();
        let shown = (0..2000).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            v.take()
        });
        eprintln!("first picture after {:?}, still arriving {}, gpu {}", t.elapsed(), crate::youtube::stream::downloading(&file), v.on_gpu());
        assert!(shown.is_some(), "no picture: {:?}", v.failed());
        assert!(crate::youtube::stream::downloading(&file), "shown before it was all in");
        writer.join().unwrap();
        drop(v);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What following a song costs: ten seconds in real time from 60 s,
    /// as the UI would, reporting CPU time and pictures handed over.
    #[test]
    #[ignore]
    #[cfg(windows)]
    fn following_in_real_time_costs() {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        let Ok(p) = std::env::var("MOOSIK_TEST_VIDEO") else { return };
        let cpu = || {
            let (mut a, mut b, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
            // SAFETY: four valid out-pointers to FILETIME-sized integers.
            unsafe { GetProcessTimes(GetCurrentProcess(), &mut a as *mut _ as _, &mut b as *mut _ as _, &mut k as *mut _ as _, &mut u as *mut _ as _) };
            (k + u) as f64 / 1e7
        };
        // MOOSIK_TEST_MAX=2560x1440: as shown on a screen that size.
        let max = std::env::var("MOOSIK_TEST_MAX").ok()
            .and_then(|m| { let (w, h) = m.split_once('x')?; Some([w.parse().ok()?, h.parse().ok()?]) })
            .unwrap_or([u32::MAX; 2]);
        for software in [false, true] {
            if software && std::env::var("MOOSIK_TEST_GPU_ONLY").is_ok() {
                continue;
            }
            let v = Video::open_with(PathBuf::from(&p), software, max);
            v.follow(std::time::Duration::from_secs(60));
            std::thread::sleep(std::time::Duration::from_millis(800));
            let (c0, t0) = (cpu(), std::time::Instant::now());
            let mut pictures = 0;
            let mut size = [0, 0];
            while t0.elapsed().as_secs_f32() < 10.0 {
                v.follow(std::time::Duration::from_secs(60) + t0.elapsed());
                if let Some(f) = v.take() {
                    pictures += 1;
                    size = f.image.size;
                    if let Ok(png) = std::env::var("MOOSIK_VIDEO_PNG") && pictures == 100 {
                        let png = png.replace(".png", if v.on_gpu() { "-gpu.png" } else { "-cpu.png" });
                        let bytes: Vec<u8> = f.image.pixels.iter().flat_map(|p| p.to_array()).collect();
                        image::RgbaImage::from_raw(f.image.size[0] as u32, f.image.size[1] as u32, bytes).unwrap().save(png).unwrap();
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            let used = cpu() - c0;
            eprintln!("10 s of video ({}, shown {}×{}): {pictures} pictures, {:.2} s CPU ({:.0}% of one core)",
                if v.on_gpu() { "graphics card" } else { "processor" }, size[0], size[1], used, used * 10.0);
            assert!(pictures > 100 && v.failed().is_none(), "{:?}", v.failed());
        }
    }
}
