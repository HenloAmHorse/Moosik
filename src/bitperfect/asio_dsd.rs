// Native DSD output via ASIO — the road to bit-perfect DSD512.
//
// DoP tops out where the PCM carrier does (DSD256 needs 705.6 kHz; DSD512
// would need 1.4112 MHz, beyond almost every DAC's PCM ceiling). WASAPI has
// no DSD transport at all, so on Windows the only native route is the DAC
// vendor's ASIO driver, switched into DSD mode via `ASIOFuture
// (kAsioSetIoFormat)`. In that mode the driver's buffers carry the raw 1-bit
// stream as bytes — exactly what `DsdReader` produces — so the data path here
// is: reader bytes → ring buffer → per-channel copy in the driver callback.
// No packing, no markers, no carrier: the DSD rate is limited only by the
// driver.
//
// ## Why no Steinberg SDK
//
// The ASIO driver interface is a COM object with a fixed vtable. On 32-bit
// x86 its methods use a non-standard thiscall variant that FFI can't express
// without assembly shims (the reason hosts historically needed the SDK's C++
// glue). On x86_64 there is only one calling convention, so the vtable can be
// declared as ordinary `extern "system"` function pointers — no SDK, no
// bindgen, and the whole module cross-compiles for checking. This module is
// therefore **x86_64-only** (enforced below) — which in practice is every
// machine running a modern DAC's ASIO driver.
//
// ## Host model
//
// - All driver calls happen on one dedicated thread (COM apartment), exactly
//   like the WASAPI backend: `AsioDsdStream::open` spawns it, it performs the
//   whole negotiation, reports back over a channel, then parks until drop.
// - The driver delivers audio by calling `bufferSwitch` on ITS thread with no
//   user-data pointer, so the active host state lives in a process-global
//   `AtomicPtr` (classic ASIO host limitation: one stream at a time — which
//   is also all the player ever needs).
// - Track sessions (ring consumer + flags) swap in and out of a Mutex slot
//   with try_lock in the callback, mirroring `render_samples`.

#![cfg(all(windows, feature = "asio-dsd"))]

#[cfg(not(target_arch = "x86_64"))]
compile_error!(
    "the asio-dsd feature requires x86_64: on 32-bit x86 the ASIO vtable uses \
     a thiscall variant these bindings do not (and cannot portably) express"
);

use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::ERROR_SUCCESS;
use windows_sys::Win32::System::Com::{
    CLSIDFromString, CoCreateInstance, CoInitializeEx, CoUninitialize,
    CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED,
};
use windows_sys::Win32::System::Registry::{
    RegCloseKey, RegEnumKeyExW, RegOpenKeyExW, RegQueryValueExW, HKEY,
    HKEY_LOCAL_MACHINE, KEY_READ,
};

use crate::dsd::dop::DSD_SILENCE;

/// The app's real window handle, stashed at startup. Drivers create hidden
/// notification windows parented to init()'s sysHandle — the desktop window
/// is a poor substitute and at least one driver family AVs on it.
static APP_HWND: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

pub fn set_app_hwnd(hwnd: *mut c_void) {
    APP_HWND.store(hwnd as isize, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// ASIO ABI declarations (from the publicly documented driver interface)
// ---------------------------------------------------------------------------

type AsioError = i32;
type AsioBool = i32;

const ASE_OK: AsioError = 0;
/// ASIOFuture's dedicated success code — future() calls return THIS, not
/// ASE_OK, on success (0x3f4847a0; the SMSL C200Pro taught us that one).
const ASE_SUCCESS: AsioError = 0x3f48_47a0;

/// Success test for ASIOFuture selectors: drivers legitimately answer with
/// either code.
fn future_ok(rc: AsioError) -> bool {
    rc == ASE_OK || rc == ASE_SUCCESS
}
const ASIO_TRUE: AsioBool = 1;
const ASIO_FALSE: AsioBool = 0;

// Sample types reported by getChannelInfo while the driver is in DSD mode.
const ASIOST_DSD_INT8_LSB1: i32 = 32; // oldest DSD sample in the LSB
const ASIOST_DSD_INT8_MSB1: i32 = 33; // oldest DSD sample in the MSB (our native order)
const ASIOST_DSD_INT8_NER8: i32 = 40; // 8 samples/byte, no endianness reordering

// ASIOFuture selectors for the DSD io-format switch (ASIO 2.2+).
const K_ASIO_CAN_DO_IO_FORMAT: i32 = 0x23112004;
const K_ASIO_SET_IO_FORMAT: i32 = 0x23111961;

const K_ASIO_FORMAT_DSD: i32 = 1;

// asioMessage selectors we answer.
const K_ASIO_SELECTOR_SUPPORTED: i32 = 1;
const K_ASIO_ENGINE_VERSION: i32 = 2;
const K_ASIO_RESET_REQUEST: i32 = 3;
const K_ASIO_SUPPORTS_TIME_INFO: i32 = 7;

#[repr(C)]
struct AsioIoFormat {
    format_type: i32,
    future: [u8; 508],
}
const _: () = assert!(std::mem::size_of::<AsioIoFormat>() == 512);

#[repr(C)]
struct AsioChannelInfo {
    channel: i32,
    is_input: AsioBool,
    is_active: AsioBool,
    channel_group: i32,
    sample_type: i32,
    name: [u8; 32],
}
const _: () = assert!(std::mem::size_of::<AsioChannelInfo>() == 52);

#[repr(C)]
struct AsioBufferInfo {
    is_input: AsioBool,
    channel_num: i32,
    /// Double buffer: the driver fills/drains buffers[0] and buffers[1]
    /// alternately; bufferSwitch's index says which one is ours to write.
    buffers: [*mut c_void; 2],
}
const _: () = assert!(std::mem::size_of::<AsioBufferInfo>() == 24);

#[repr(C)]
struct AsioCallbacks {
    buffer_switch: unsafe extern "system" fn(index: i32, direct: AsioBool),
    sample_rate_did_change: unsafe extern "system" fn(rate: f64),
    asio_message:
        unsafe extern "system" fn(selector: i32, value: i32, msg: *mut c_void, opt: *mut f64) -> i32,
    buffer_switch_time_info:
        unsafe extern "system" fn(time: *mut c_void, index: i32, direct: AsioBool) -> *mut c_void,
}

/// The IASIO vtable. Order is ABI — do not reorder. The first three entries
/// are IUnknown. On x86_64 every method uses the single Windows calling
/// convention, so `extern "system"` with an explicit `this` is exact.
#[repr(C)]
struct IAsioVtbl {
    query_interface:
        unsafe extern "system" fn(*mut IAsio, *const GUID, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(*mut IAsio) -> u32,
    release: unsafe extern "system" fn(*mut IAsio) -> u32,

    init: unsafe extern "system" fn(*mut IAsio, sys_handle: *mut c_void) -> AsioBool,
    get_driver_name: unsafe extern "system" fn(*mut IAsio, name: *mut u8),
    get_driver_version: unsafe extern "system" fn(*mut IAsio) -> i32,
    get_error_message: unsafe extern "system" fn(*mut IAsio, msg: *mut u8),
    start: unsafe extern "system" fn(*mut IAsio) -> AsioError,
    stop: unsafe extern "system" fn(*mut IAsio) -> AsioError,
    get_channels: unsafe extern "system" fn(*mut IAsio, *mut i32, *mut i32) -> AsioError,
    get_latencies: unsafe extern "system" fn(*mut IAsio, *mut i32, *mut i32) -> AsioError,
    get_buffer_size:
        unsafe extern "system" fn(*mut IAsio, *mut i32, *mut i32, *mut i32, *mut i32) -> AsioError,
    can_sample_rate: unsafe extern "system" fn(*mut IAsio, f64) -> AsioError,
    get_sample_rate: unsafe extern "system" fn(*mut IAsio, *mut f64) -> AsioError,
    set_sample_rate: unsafe extern "system" fn(*mut IAsio, f64) -> AsioError,
    get_clock_sources: unsafe extern "system" fn(*mut IAsio, *mut c_void, *mut i32) -> AsioError,
    set_clock_source: unsafe extern "system" fn(*mut IAsio, i32) -> AsioError,
    get_sample_position:
        unsafe extern "system" fn(*mut IAsio, *mut u64, *mut u64) -> AsioError,
    get_channel_info: unsafe extern "system" fn(*mut IAsio, *mut AsioChannelInfo) -> AsioError,
    create_buffers: unsafe extern "system" fn(
        *mut IAsio,
        *mut AsioBufferInfo,
        i32,
        i32,
        *const AsioCallbacks,
    ) -> AsioError,
    dispose_buffers: unsafe extern "system" fn(*mut IAsio) -> AsioError,
    control_panel: unsafe extern "system" fn(*mut IAsio) -> AsioError,
    future: unsafe extern "system" fn(*mut IAsio, selector: i32, opt: *mut c_void) -> AsioError,
    output_ready: unsafe extern "system" fn(*mut IAsio) -> AsioError,
}

#[repr(C)]
struct IAsio {
    vtbl: *const IAsioVtbl,
}

macro_rules! asio_call {
    ($drv:expr, $method:ident $(, $arg:expr)*) => {
        ((*(*$drv).vtbl).$method)($drv $(, $arg)*)
    };
}

// ---------------------------------------------------------------------------
// Driver discovery (HKLM\SOFTWARE\ASIO)
// ---------------------------------------------------------------------------

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Names of every ASIO driver registered on the system.
pub fn list_asio_drivers() -> Vec<String> {
    let mut out = Vec::new();
    unsafe {
        let mut root: HKEY = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, wide("SOFTWARE\\ASIO").as_ptr(), 0, KEY_READ, &mut root)
            != ERROR_SUCCESS
        {
            return out;
        }
        for i in 0.. {
            let mut name = [0u16; 256];
            let mut len = name.len() as u32;
            if RegEnumKeyExW(root, i, name.as_mut_ptr(), &mut len, null_mut(), null_mut(), null_mut(), null_mut())
                != ERROR_SUCCESS
            {
                break;
            }
            out.push(String::from_utf16_lossy(&name[..len as usize]));
        }
        RegCloseKey(root);
    }
    out
}

/// The CLSID string for a registered driver name.
fn driver_clsid(name: &str) -> Result<GUID, String> {
    unsafe {
        let mut key: HKEY = null_mut();
        let path = wide(&format!("SOFTWARE\\ASIO\\{name}"));
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_READ, &mut key) != ERROR_SUCCESS {
            return Err(format!("ASIO driver \"{name}\" not found in the registry"));
        }
        let mut buf = [0u16; 128];
        let mut len = (buf.len() * 2) as u32;
        let rc = RegQueryValueExW(
            key, wide("CLSID").as_ptr(), null_mut(), null_mut(),
            buf.as_mut_ptr() as *mut u8, &mut len,
        );
        RegCloseKey(key);
        if rc != ERROR_SUCCESS {
            return Err(format!("ASIO driver \"{name}\" has no CLSID value"));
        }
        let mut clsid = std::mem::zeroed::<GUID>();
        if CLSIDFromString(buf.as_ptr(), &mut clsid) != 0 {
            return Err(format!("ASIO driver \"{name}\": malformed CLSID"));
        }
        Ok(clsid)
    }
}

// ---------------------------------------------------------------------------
// Global callback context (ASIO callbacks carry no user data)
// ---------------------------------------------------------------------------

/// One playing track: ring consumer + its decode thread's flags.
struct AsioSession {
    cons: rtrb::Consumer<u8>,
    decode_done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for AsioSession {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

struct AsioShared {
    session: Mutex<Option<AsioSession>>,
    paused: AtomicBool,
    finished: AtomicBool,
    /// Byte-frames (1 byte = 8 DSD samples, per channel) delivered as audio.
    frames_played: AtomicU64,
    /// Driver asked for a reset (device/sample-rate change) — surfaced as
    /// end-of-session; the next play re-negotiates from scratch.
    reset_requested: AtomicBool,
}

/// Everything the realtime callback needs, reachable through one global
/// pointer. Owned by the host thread; published to `ACTIVE` only while the
/// driver is running (set before `start()`, cleared after `stop()`).
struct CallbackCtx {
    shared: Arc<AsioShared>,
    driver: *mut IAsio,
    buffer_infos: Vec<AsioBufferInfo>,
    /// Bytes per channel per half-buffer. ASIO expresses DSD buffer sizes in
    /// 1-bit SAMPLES (8 per byte) — the C200Pro's buffers were 0x2000 apart
    /// for bufferSize=65536, proving bytes = samples/8. Writing `samples`
    /// bytes tramples the driver's heap 8× over.
    buffer_bytes: usize,
    channels: usize,
    /// Reused interleaved-read scratch — no allocation on the audio path.
    scratch: Vec<u8>,
    supports_output_ready: bool,
}

static ACTIVE: AtomicPtr<CallbackCtx> = AtomicPtr::new(null_mut());

/// Fill one half of the double buffer. Realtime path: try_lock only, no
/// allocation, silence is the DSD idle pattern (0x69) so the DAC keeps its
/// DSD lock through pause/underrun instead of thumping on 0x00.
unsafe fn fill_buffers(ctx: &mut CallbackCtx, index: i32) {
    let frames = ctx.buffer_bytes; // byte-frames: 1 byte = 8 DSD samples per channel
    let ch = ctx.channels;
    let want = frames * ch;
    ctx.scratch.clear();

    let sh = &ctx.shared;
    if !sh.paused.load(Ordering::Relaxed)
        && let Ok(mut guard) = sh.session.try_lock()
        && let Some(sess) = guard.as_mut()
    {
        while ctx.scratch.len() < want {
            match sess.cons.pop() {
                Ok(b) => ctx.scratch.push(b),
                Err(_) => break,
            }
        }
        if ctx.scratch.len() < want
            && sess.decode_done.load(Ordering::Acquire)
            && sess.cons.is_empty()
        {
            *guard = None;
            sh.finished.store(true, Ordering::Release);
        }
    }

    // Whole frames only — a torn frame would swap channels.
    let got_frames = ctx.scratch.len() / ch;
    sh.frames_played.fetch_add(got_frames as u64, Ordering::Relaxed);

    for (c, info) in ctx.buffer_infos.iter().enumerate() {
        let dest = info.buffers[index as usize] as *mut u8;
        if dest.is_null() { continue; }
        for f in 0..frames {
            let byte = if f < got_frames { ctx.scratch[f * ch + c] } else { DSD_SILENCE };
            unsafe { *dest.add(f) = byte; }
        }
    }

    if ctx.supports_output_ready {
        let _ = unsafe { asio_call!(ctx.driver, output_ready) };
    }
}

static FIRST_SWITCH_LOGGED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn cb_buffer_switch(index: i32, _direct: AsioBool) {
    if !FIRST_SWITCH_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[asio-dsd] first bufferSwitch (index {index}) — callbacks are flowing");
    }
    let ctx = ACTIVE.load(Ordering::Acquire);
    if !ctx.is_null() {
        unsafe { fill_buffers(&mut *ctx, index & 1) };
    }
}

unsafe extern "system" fn cb_buffer_switch_time_info(
    _time: *mut c_void,
    index: i32,
    _direct: AsioBool,
) -> *mut c_void {
    if !TIME_INFO_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[asio-dsd] driver callback: bufferSwitchTimeInfo (driver ignores our 'no time-info' answer — fine)");
    }
    unsafe { cb_buffer_switch(index, ASIO_FALSE) };
    null_mut()
}

static TIME_INFO_LOGGED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn cb_sample_rate_did_change(rate: f64) {
    eprintln!("[asio-dsd] driver callback: sampleRateDidChange({rate})");
}

unsafe extern "system" fn cb_asio_message(
    selector: i32,
    value: i32,
    _msg: *mut c_void,
    _opt: *mut f64,
) -> i32 {
    // asioMessage traffic is sparse (setup-time negotiation, rare runtime
    // notifications) — log every call while the path is being proven.
    eprintln!("[asio-dsd] driver callback: asioMessage(selector={selector}, value={value})");
    match selector {
        K_ASIO_SELECTOR_SUPPORTED => i32::from(matches!(
            value,
            K_ASIO_ENGINE_VERSION | K_ASIO_RESET_REQUEST
        )),
        K_ASIO_ENGINE_VERSION => 2,
        // Deliberately unsupported: drivers then use the plain bufferSwitch
        // callback, which involves no ASIOTime structures — smallest possible
        // ABI surface while the path is being hardware-proven.
        K_ASIO_SUPPORTS_TIME_INFO => 0,
        K_ASIO_RESET_REQUEST => {
            let ctx = ACTIVE.load(Ordering::Acquire);
            if !ctx.is_null() {
                let shared = unsafe { &(*ctx).shared };
                shared.reset_requested.store(true, Ordering::Release);
                shared.finished.store(true, Ordering::Release);
            }
            1
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// The stream
// ---------------------------------------------------------------------------

pub struct AsioDsdStream {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
    shared: Arc<AsioShared>,
    pub driver_name: String,
    /// The DSD bit rate this stream was negotiated for.
    pub dsd_rate: u32,
    pub channels: u16,
    /// True if the driver wants oldest-sample-in-LSB bytes (we then reverse
    /// each byte at decode time; the reader's native order is MSB-first).
    pub lsb_first: bool,
    /// Negotiated device buffer size (frames = bytes/channel), for a future
    /// latency readout in the device picker.
    #[allow(dead_code)]
    pub buffer_frames: u32,
}

struct SetupOk {
    lsb_first: bool,
    buffer_frames: u32,
}

impl AsioDsdStream {
    /// Open `driver_name` in native-DSD mode at `dsd_rate` (the bit rate,
    /// e.g. 22 579 200 for DSD512) with `channels` outputs. The entire
    /// negotiation runs on a dedicated host thread; errors come back verbatim.
    pub fn open(driver_name: &str, dsd_rate: u32, channels: u16) -> Result<Self, String> {
        if !ACTIVE.load(Ordering::Acquire).is_null() {
            return Err("an ASIO stream is already open".into());
        }
        let clsid = driver_clsid(driver_name)?;
        let shared = Arc::new(AsioShared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            frames_played: AtomicU64::new(0),
            reset_requested: AtomicBool::new(false),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<SetupOk, String>>();

        let t_shared = Arc::clone(&shared);
        let t_stop = Arc::clone(&stop);
        let t_name = driver_name.to_owned();
        let join = std::thread::Builder::new()
            .name("bp-asio-dsd".into())
            .spawn(move || host_thread(t_name, clsid, dsd_rate, channels, t_shared, t_stop, tx))
            .map_err(|e| format!("thread spawn failed: {e}"))?;

        match rx.recv() {
            Ok(Ok(ok)) => Ok(AsioDsdStream {
                stop,
                join: Some(join),
                shared,
                driver_name: driver_name.to_owned(),
                dsd_rate,
                channels,
                lsb_first: ok.lsb_first,
                buffer_frames: ok.buffer_frames,
            }),
            Ok(Err(e)) => { let _ = join.join(); Err(e) }
            Err(_) => { let _ = join.join(); Err("ASIO host thread died during setup".into()) }
        }
    }

    /// Install a new track session (ring consumer + decode-thread flags),
    /// replacing any current one.
    pub fn start_session(
        &self,
        cons: rtrb::Consumer<u8>,
        decode_done: Arc<AtomicBool>,
        session_stop: Arc<AtomicBool>,
    ) {
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
        if let Ok(mut g) = self.shared.session.lock() {
            *g = Some(AsioSession { cons, decode_done, stop: session_stop });
        }
    }

    pub fn stop_session(&self) {
        if let Ok(mut g) = self.shared.session.lock() { *g = None; }
        self.shared.finished.store(false, Ordering::Relaxed);
        self.shared.frames_played.store(0, Ordering::Relaxed);
    }

    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Relaxed); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Relaxed); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    pub fn is_finished(&self) -> bool { self.shared.finished.load(Ordering::Acquire) }
    pub fn reset_requested(&self) -> bool { self.shared.reset_requested.load(Ordering::Acquire) }

    /// Sample-accurate elapsed time: byte-frames delivered × 8 samples ÷ rate.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played.load(Ordering::Relaxed);
        Duration::from_secs_f64(frames as f64 * 8.0 / self.dsd_rate.max(1) as f64)
    }
}

impl Drop for AsioDsdStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(j) = self.join.take() { let _ = j.join(); }
    }
}

// ---------------------------------------------------------------------------
// Decode thread — reader bytes → ring, with lead-in/out DSD silence
// ---------------------------------------------------------------------------

/// Stream a DSD file's bytes into the ring: ~24 ms of DSD silence first (DAC
/// settle), then the audio (bit-reversed per byte if the driver is
/// LSB-first), then a silence tail so the device idles at DSD zero while the
/// ring drains. Mirrors `dop_decode_loop`, minus the packing.
pub fn asio_decode_loop(
    mut reader: crate::dsd::DsdFileReader,
    lsb_first: bool,
    mut prod: rtrb::Producer<u8>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    let info = reader.info().clone();
    let ch = info.channels as usize;
    let lead_frames = (info.sample_rate as usize / 8 * 24 / 1000).max(64);

    let push_all = |bytes: &[u8], prod: &mut rtrb::Producer<u8>, stop: &AtomicBool| -> bool {
        for &b in bytes {
            loop {
                if stop.load(Ordering::Relaxed) { return true; }
                match prod.push(b) {
                    Ok(()) => break,
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        }
        false
    };

    let silence = vec![DSD_SILENCE; lead_frames * ch];
    if push_all(&silence, &mut prod, &stop) { return; }

    let mut buf = vec![0u8; 4096 * ch];
    loop {
        if stop.load(Ordering::Relaxed) { return; }
        let n = match reader.read_frames(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => { eprintln!("[asio-dsd] read error: {e}"); break; }
        };
        let chunk = &mut buf[..n * ch];
        if lsb_first {
            for b in chunk.iter_mut() { *b = b.reverse_bits(); }
        }
        if push_all(chunk, &mut prod, &stop) { return; }
    }

    if !stop.load(Ordering::Relaxed) {
        let _ = push_all(&silence, &mut prod, &stop);
    }
    done.store(true, Ordering::Release);
}

// ---------------------------------------------------------------------------
// Host thread — owns the driver COM object for its whole life
// ---------------------------------------------------------------------------

fn driver_error(driver: *mut IAsio, what: &str) -> String {
    let mut buf = [0u8; 128];
    unsafe { asio_call!(driver, get_error_message, buf.as_mut_ptr()) };
    let msg = buf.iter().position(|&b| b == 0)
        .map(|n| String::from_utf8_lossy(&buf[..n]).into_owned())
        .unwrap_or_default();
    if msg.is_empty() { what.to_string() } else { format!("{what}: {msg}") }
}

fn host_thread(
    name: String,
    clsid: GUID,
    dsd_rate: u32,
    channels: u16,
    shared: Arc<AsioShared>,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<SetupOk, String>>,
) {
    unsafe {
        // ASIO drivers are in-proc COM servers and generally expect an STA.
        let co = CoInitializeEx(null(), COINIT_APARTMENTTHREADED as u32);
        let co_ok = co >= 0;

        let result = host_setup(&name, &clsid, dsd_rate, channels, &shared);
        match result {
            Err(e) => {
                let _ = tx.send(Err(e));
                if co_ok { CoUninitialize(); }
                return;
            }
            Ok((driver, ctx_box, setup)) => {
                // Publish the callback context, then start the engine.
                let ctx_ptr = Box::into_raw(ctx_box);
                ACTIVE.store(ctx_ptr, Ordering::Release);
                let rc = asio_call!(driver, start);
                eprintln!("[asio-dsd] \"{name}\": start rc={rc}");
                if rc != ASE_OK {
                    ACTIVE.store(null_mut(), Ordering::Release);
                    let e = driver_error(driver, "ASIO start failed");
                    asio_call!(driver, dispose_buffers);
                    asio_call!(driver, release);
                    drop(Box::from_raw(ctx_ptr));
                    let _ = tx.send(Err(e));
                    if co_ok { CoUninitialize(); }
                    return;
                }
                let _ = tx.send(Ok(setup));

                // Park until the stream is dropped.
                while !stop.load(Ordering::Acquire) {
                    std::thread::park_timeout(Duration::from_millis(100));
                }

                // Orderly teardown: stop callbacks, unpublish, then release.
                asio_call!(driver, stop);
                ACTIVE.store(null_mut(), Ordering::Release);
                // The driver guarantees no callbacks after stop() returns.
                asio_call!(driver, dispose_buffers);
                asio_call!(driver, release);
                drop(Box::from_raw(ctx_ptr));
            }
        }
        if co_ok { CoUninitialize(); }
    }
}

/// Everything from CoCreateInstance to createBuffers. Returns the driver
/// (with one owned reference), the ready callback context, and the setup
/// summary for the UI.
unsafe fn host_setup(
    name: &str,
    clsid: &GUID,
    dsd_rate: u32,
    channels: u16,
    shared: &Arc<AsioShared>,
) -> Result<(*mut IAsio, Box<CallbackCtx>, SetupOk), String> {
    let mut raw: *mut c_void = null_mut();
    // ASIO quirk: a driver's class IS its interface — the CLSID doubles as
    // the IID passed to CoCreateInstance.
    let hr = unsafe { CoCreateInstance(clsid, null_mut(), CLSCTX_INPROC_SERVER, clsid, &mut raw) };
    if hr != 0 || raw.is_null() {
        return Err(format!(
            "couldn't load ASIO driver \"{name}\" (COM error {hr:#010x}) — is it installed for 64-bit hosts?"
        ));
    }
    let driver = raw as *mut IAsio;
    // Cleanup helper for early exits.
    macro_rules! bail {
        ($e:expr) => {{ unsafe { asio_call!(driver, release) }; return Err($e); }};
    }

    // Drivers want a real window: they parent hidden notification windows to
    // it (subclassing the desktop window is not survivable everywhere). Use
    // the app's main window, falling back to the desktop only if it's not
    // known yet.
    let mut sys_handle = APP_HWND.load(Ordering::Relaxed) as *mut c_void;
    if sys_handle.is_null() {
        sys_handle = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::GetDesktopWindow()
        } as *mut c_void;
    }
    eprintln!("[asio-dsd] \"{name}\": init with hwnd {sys_handle:?}");
    if unsafe { asio_call!(driver, init, sys_handle) } != ASIO_TRUE {
        let e = driver_error(driver, "driver init failed");
        bail!(format!("ASIO \"{name}\": {e}"));
    }
    eprintln!("[asio-dsd] \"{name}\": loaded, init ok (driver version {})",
              unsafe { asio_call!(driver, get_driver_version) });

    // Switch the driver into DSD mode BEFORE querying channel formats or
    // negotiating the rate — everything downstream depends on the mode.
    let mut fmt = AsioIoFormat { format_type: K_ASIO_FORMAT_DSD, future: [0; 508] };
    let can = unsafe { asio_call!(driver, future, K_ASIO_CAN_DO_IO_FORMAT, &mut fmt as *mut _ as *mut c_void) };
    eprintln!("[asio-dsd] \"{name}\": kAsioCanDoIoFormat(DSD) = {can:#x} (ok={})", future_ok(can));
    if !future_ok(can) {
        bail!(format!(
            "ASIO \"{name}\" doesn't support native DSD (kAsioCanDoIoFormat returned {can}) — DoP is this device's ceiling"
        ));
    }
    let mut fmt = AsioIoFormat { format_type: K_ASIO_FORMAT_DSD, future: [0; 508] };
    let set = unsafe { asio_call!(driver, future, K_ASIO_SET_IO_FORMAT, &mut fmt as *mut _ as *mut c_void) };
    eprintln!("[asio-dsd] \"{name}\": kAsioSetIoFormat(DSD) = {set:#x} (ok={})", future_ok(set));
    if !future_ok(set) {
        bail!(format!("ASIO \"{name}\": switching to DSD mode failed ({set})"));
    }

    // Rate: the ASIO DSD convention expresses the rate in *bits per second
    // per channel* (the DSD rate itself); some drivers instead report the
    // byte rate (÷8). Probe both, prefer the spec'd form.
    let mut rate_used = 0f64;
    for cand in [dsd_rate as f64, dsd_rate as f64 / 8.0, dsd_rate as f64 / 16.0] {
        let can = unsafe { asio_call!(driver, can_sample_rate, cand) };
        let set = if can == ASE_OK {
            unsafe { asio_call!(driver, set_sample_rate, cand) }
        } else {
            can
        };
        eprintln!("[asio-dsd] \"{name}\": rate {cand}: canSampleRate={can} setSampleRate={set}");
        if set == ASE_OK {
            rate_used = cand;
            break;
        }
    }
    if rate_used == 0.0 {
        let label = crate::dsd::rate_label(dsd_rate);
        bail!(format!("ASIO \"{name}\": driver rejected {label} ({dsd_rate} Hz) in DSD mode"));
    }

    // Channel sanity + sample type.
    let (mut n_in, mut n_out) = (0i32, 0i32);
    let rc = unsafe { asio_call!(driver, get_channels, &mut n_in, &mut n_out) };
    eprintln!("[asio-dsd] \"{name}\": getChannels rc={rc}, in={n_in}, out={n_out}");
    if rc != ASE_OK || n_out < channels as i32 {
        bail!(format!("ASIO \"{name}\": needs {channels} output channels, driver has {n_out}"));
    }
    let mut info = AsioChannelInfo {
        channel: 0, is_input: ASIO_FALSE, is_active: ASIO_FALSE,
        channel_group: 0, sample_type: 0, name: [0; 32],
    };
    let rc = unsafe { asio_call!(driver, get_channel_info, &mut info) };
    eprintln!("[asio-dsd] \"{name}\": getChannelInfo rc={rc}, sample type {}", info.sample_type);
    if rc != ASE_OK {
        bail!(format!("ASIO \"{name}\": channel info query failed"));
    }
    let lsb_first = match info.sample_type {
        ASIOST_DSD_INT8_MSB1 | ASIOST_DSD_INT8_NER8 => false,
        ASIOST_DSD_INT8_LSB1 => true,
        other => bail!(format!(
            "ASIO \"{name}\": unexpected sample type {other} in DSD mode (expected DSD Int8)"
        )),
    };

    // Buffers at the driver's preferred size.
    let (mut min, mut max, mut preferred, mut gran) = (0i32, 0i32, 0i32, 0i32);
    let rc = unsafe { asio_call!(driver, get_buffer_size, &mut min, &mut max, &mut preferred, &mut gran) };
    eprintln!("[asio-dsd] \"{name}\": getBufferSize rc={rc}, min={min} max={max} preferred={preferred} gran={gran}");
    if rc != ASE_OK || preferred <= 0 {
        bail!(format!("ASIO \"{name}\": buffer size query failed"));
    }
    // In DSD mode the size is in 1-bit samples; the Int8 buffers hold s/8 bytes.
    if preferred % 8 != 0 {
        bail!(format!("ASIO \"{name}\": DSD buffer size {preferred} isn't byte-aligned"));
    }
    let buffer_bytes = (preferred / 8) as usize;
    let mut buffer_infos: Vec<AsioBufferInfo> = (0..channels as i32)
        .map(|c| AsioBufferInfo { is_input: ASIO_FALSE, channel_num: c, buffers: [null_mut(); 2] })
        .collect();
    // The callbacks struct lives in writable memory and is deliberately
    // leaked: the driver holds this pointer until disposeBuffers, and a
    // driver that scribbles on it must not fault on a read-only static.
    // 32 bytes per stream open — negligible.
    let callbacks: *mut AsioCallbacks = Box::into_raw(Box::new(AsioCallbacks {
        buffer_switch: cb_buffer_switch,
        sample_rate_did_change: cb_sample_rate_did_change,
        asio_message: cb_asio_message,
        buffer_switch_time_info: cb_buffer_switch_time_info,
    }));
    eprintln!(
        "[asio-dsd] \"{name}\": calling createBuffers(numChannels={}, bufferSize={preferred})…",
        channels,
    );
    let rc = unsafe { asio_call!(
        driver, create_buffers,
        buffer_infos.as_mut_ptr(), channels as i32, preferred, callbacks
    ) };
    eprintln!(
        "[asio-dsd] \"{name}\": createBuffers rc={rc}, ch0 buffers = {:?}/{:?}",
        buffer_infos[0].buffers[0], buffer_infos[0].buffers[1],
    );
    if rc != ASE_OK {
        let e = driver_error(driver, "createBuffers failed");
        bail!(format!("ASIO \"{name}\": {e}"));
    }

    let supports_output_ready = unsafe { asio_call!(driver, output_ready) } == ASE_OK;
    eprintln!("[asio-dsd] \"{name}\": outputReady supported = {supports_output_ready}");

    let ctx = Box::new(CallbackCtx {
        shared: Arc::clone(shared),
        driver,
        buffer_infos,
        buffer_bytes,
        channels: channels as usize,
        scratch: Vec::with_capacity(buffer_bytes * channels as usize),
        supports_output_ready,
    });
    eprintln!(
        "[asio-dsd] \"{name}\": DSD mode on, rate accepted as {rate_used} \
         ({}), sample type {} ({}), buffer {preferred} samples = {buffer_bytes} bytes/ch",
        crate::dsd::rate_label(dsd_rate),
        info.sample_type,
        if lsb_first { "LSB-first" } else { "MSB-first" },
    );
    Ok((driver, ctx, SetupOk { lsb_first, buffer_frames: preferred as u32 }))
}
