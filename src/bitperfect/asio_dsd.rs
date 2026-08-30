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
use std::time::{Duration, Instant};

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

/// The DSD idle byte for a given bit order.
///
/// `0x69` is the idle pattern with the oldest sample in the most significant
/// bit. An LSB1 driver reads the same byte backwards, so it needs `0x96` — the
/// same pattern, reversed. The feeder reverses audio bytes for LSB1 already;
/// padding and silence used to be left at a fixed `0x69`, so every pause,
/// underrun, lead-in and tail on an LSB1 driver emitted a *different* pattern
/// from the audio around it.
#[inline]
const fn idle_byte(lsb_first: bool) -> u8 {
    if lsb_first { DSD_SILENCE.reverse_bits() } else { DSD_SILENCE }
}

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

/// The driver does not implement this selector.
const ASE_NOT_PRESENT: AsioError = -1000;
// The other documented failures — `ASE_InvalidMode` (-997),
// `ASE_HWMalfunction` (-999) and `ASE_InvalidParameter` (-998) — are not named
// here because nothing needs to distinguish them: every code that is not
// `ASE_NotPresent` is the driver having answered, and the answer is not one to
// send DSD past. The test names them as literals, which is where the numbers
// belong.

/// What a failed `kAsioGetIoFormat` means for the claim.
///
/// Every non-success code was treated as "this driver does not implement the
/// getter" and allowed through as amber Unverified. Two of them are not that
/// at all: `ASE_HWMalfunction` is the driver telling us the hardware is
/// broken, and `ASE_InvalidParameter` is it telling us the call was wrong.
/// Continuing past either and then sending DSD packing to the device is
/// carrying on in the face of the one answer that is evidence — an absent
/// getter says nothing either way, and a malfunction says something.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IoFormatReadback {
    /// The driver answered, and this is what it said.
    Answered,
    /// The driver does not implement the selector, or refused it in a way that
    /// carries no information about the hardware.
    Absent,
    /// The driver reported a real error. Not a missing feature.
    Broken(AsioError),
}

fn classify_io_format(rc: AsioError) -> IoFormatReadback {
    if future_ok(rc) {
        IoFormatReadback::Answered
    } else if rc == ASE_NOT_PRESENT {
        // The one code that means "this driver does not implement the
        // selector", which is the only answer that carries no information
        // about the hardware.
        IoFormatReadback::Absent
    } else {
        // Everything else is the driver saying something, and DSD packing must
        // not be sent past it.
        //
        // `ASE_InvalidMode` was being treated as absence, which it is not: it
        // means the driver understood the call and refused it *in the state it
        // is in* — having just been put into DSD mode, that is a contradiction
        // worth stopping for. And an undocumented code was waved through on
        // the reasoning that drivers return values that appear in no header.
        // They do; but "I do not recognise this answer" is not the same as "no
        // answer was given", and only the second is grounds for carrying on
        // and calling the result unverified.
        IoFormatReadback::Broken(rc)
    }
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
const K_ASIO_GET_IO_FORMAT: i32 = 0x23111983;

const K_ASIO_FORMAT_DSD: i32 = 1;

// asioMessage selectors, numbered exactly as `asio.h` numbers them.
//
// These are an ABI, not a naming convention: the driver sends the integer and
// we answer the integer. Getting one wrong does not fail to compile and does
// not fail to run — it silently answers a different question. The values here
// were previously guessed contiguously from `kAsioLatenciesChanged`, which put
// overload at 10 (`kAsioSupportsInputMonitor`) and buffer-size change at 11
// (`kAsioSupportsInputGain`), so a driver reporting an overload was told it
// had input monitoring and a real overload was never seen.
//
// The full enum, so the gaps below are visibly deliberate rather than
// forgotten:
//
// ```text
//  1 kAsioSelectorSupported      9 kAsioMMCCommand
//  2 kAsioEngineVersion         10 kAsioSupportsInputMonitor
//  3 kAsioResetRequest          11 kAsioSupportsInputGain
//  4 kAsioBufferSizeChange      12 kAsioSupportsInputMeter
//  5 kAsioResyncRequest         13 kAsioSupportsOutputGain
//  6 kAsioLatenciesChanged      14 kAsioSupportsOutputMeter
//  7 kAsioSupportsTimeInfo      15 kAsioOverload
//  8 kAsioSupportsTimeCode
// ```
const K_ASIO_SELECTOR_SUPPORTED: i32 = 1;
const K_ASIO_ENGINE_VERSION: i32 = 2;
/// The driver wants the stream torn down and rebuilt.
const K_ASIO_RESET_REQUEST: i32 = 3;
/// The driver changed its buffer size underneath the running stream.
const K_ASIO_BUFFER_SIZE_CHANGE: i32 = 4;
/// The driver lost sync with its clock source.
const K_ASIO_RESYNC_REQUEST: i32 = 5;
const K_ASIO_LATENCIES_CHANGED: i32 = 6;
const K_ASIO_SUPPORTS_TIME_INFO: i32 = 7;
/// The driver lost realtime: it could not keep up with its own callback.
const K_ASIO_OVERLOAD: i32 = 15;

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
// PCM capability probe
// ---------------------------------------------------------------------------

// ASIO PCM sample types, from `asio.h`. Only the ones that could carry PCM
// exactly are named; anything else the probe reports numerically.
const ASIOST_INT16_MSB: i32 = 0;
const ASIOST_INT24_MSB: i32 = 1;
const ASIOST_INT32_MSB: i32 = 2;
const ASIOST_FLOAT32_MSB: i32 = 3;
const ASIOST_INT16_LSB: i32 = 16;
const ASIOST_INT24_LSB: i32 = 17;
const ASIOST_INT32_LSB: i32 = 18;
const ASIOST_FLOAT32_LSB: i32 = 19;

/// What a driver says it can do with ordinary PCM.
///
/// **A diagnostic, not a route.** Nothing in Moosik plays PCM over ASIO, and
/// this does not make it possible: it opens the driver, asks, and closes it
/// again. It exists so that "can this DAC take PCM over ASIO, and in what
/// format?" has an answer that comes from the driver rather than from
/// guesswork — which is the question that has to be settled before a renderer
/// is worth writing.
///
/// The corresponding renderer does not exist, so there is no ASIO PCM setting,
/// no ASIO PCM transport and no ASIO PCM claim anywhere in the product.
#[derive(Clone, Debug, PartialEq)]
pub struct AsioPcmCapabilities {
    pub driver: String,
    /// Output channels the driver reports.
    pub outputs: i32,
    /// Preferred buffer size in frames, and the range it allows.
    pub min_buffer: i32,
    pub max_buffer: i32,
    pub preferred_buffer: i32,
    /// The sample rate the driver was at when asked, before anything changed.
    pub current_rate: f64,
    /// Standard rates the driver answered `canSampleRate` for.
    pub rates: Vec<u32>,
    /// The sample type of output channel 0, and its human name.
    pub sample_type: i32,
    pub sample_type_name: &'static str,
    /// Whether that sample type is one this codebase could carry exactly, if a
    /// renderer existed. Answering "yes" here is not a claim that it plays.
    pub exact_capable: bool,
}

impl AsioPcmCapabilities {
    pub fn describe(&self) -> String {
        let rates = if self.rates.is_empty() {
            "none of the standard rates".to_string()
        } else {
            self.rates
                .iter()
                .map(|r| super::fmt_khz(*r))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "{}: {} output channel(s), {} ({}), buffers {}..{} (preferred {}), \
             currently {} Hz, accepts {rates}",
            self.driver,
            self.outputs,
            self.sample_type_name,
            if self.exact_capable {
                "could carry PCM exactly"
            } else {
                "not a format this build could carry exactly"
            },
            self.min_buffer,
            self.max_buffer,
            self.preferred_buffer,
            self.current_rate,
        )
    }
}

fn pcm_sample_type_name(t: i32) -> (&'static str, bool) {
    match t {
        ASIOST_INT16_MSB => ("16-bit integer, big-endian", true),
        ASIOST_INT24_MSB => ("24-bit integer, big-endian", true),
        ASIOST_INT32_MSB => ("32-bit integer, big-endian", true),
        ASIOST_FLOAT32_MSB => ("32-bit float, big-endian", true),
        ASIOST_INT16_LSB => ("16-bit integer, little-endian", true),
        ASIOST_INT24_LSB => ("24-bit integer, little-endian", true),
        ASIOST_INT32_LSB => ("32-bit integer, little-endian", true),
        ASIOST_FLOAT32_LSB => ("32-bit float, little-endian", true),
        ASIOST_DSD_INT8_LSB1 | ASIOST_DSD_INT8_MSB1 | ASIOST_DSD_INT8_NER8 => {
            ("a DSD packing, not PCM", false)
        }
        _ => ("an unrecognised sample type", false),
    }
}

/// Ask `driver_name` what it can do with PCM.
///
/// Opens the driver, queries, and closes it. `createBuffers` is never called
/// and `start` is never called, so this takes the hardware for as long as the
/// query and no longer. It runs on its own thread for the same reason the
/// DSD path does: ASIO drivers are in-proc COM servers that expect an STA.
///
/// It refuses to run while an engine is engaged. One `ACTIVE` context exists
/// per process, and probing a second driver while the first is streaming is
/// how a driver ends up calling into a context that has been replaced.
pub fn probe_pcm(driver_name: &str) -> Result<AsioPcmCapabilities, String> {
    let _claim = EngineClaim::take()
        .map_err(|_| "an ASIO stream is open; close it before probing a driver".to_string())?;
    let clsid = driver_clsid(driver_name)?;
    let name = driver_name.to_owned();
    let (tx, rx) = mpsc::channel::<Result<AsioPcmCapabilities, String>>();
    std::thread::Builder::new()
        .name("bp-asio-probe".into())
        .spawn(move || {
            let _ = tx.send(unsafe { probe_pcm_on_thread(&name, &clsid) });
        })
        .map_err(|e| format!("probe thread could not start: {e}"))?
        .join()
        .map_err(|_| "the ASIO probe thread panicked".to_string())?;
    rx.recv()
        .map_err(|_| "the ASIO probe produced no answer".to_string())?
}

/// Standard PCM rates worth asking about.
const PCM_PROBE_RATES: [u32; 9] = [
    44_100, 48_000, 88_200, 96_000, 176_400, 192_000, 352_800, 384_000, 768_000,
];

unsafe fn probe_pcm_on_thread(name: &str, clsid: &GUID) -> Result<AsioPcmCapabilities, String> {
    let co = unsafe { CoInitializeEx(null(), COINIT_APARTMENTTHREADED as u32) };
    let co_ok = co >= 0;
    let result = unsafe { probe_pcm_inner(name, clsid) };
    if co_ok {
        unsafe { CoUninitialize() };
    }
    result
}

unsafe fn probe_pcm_inner(name: &str, clsid: &GUID) -> Result<AsioPcmCapabilities, String> {
    let mut raw: *mut c_void = null_mut();
    let hr = unsafe { CoCreateInstance(clsid, null_mut(), CLSCTX_INPROC_SERVER, clsid, &mut raw) };
    if hr < 0 || raw.is_null() {
        return Err(format!(
            "ASIO \"{name}\": CoCreateInstance failed ({hr:#x})"
        ));
    }
    let driver = raw as *mut IAsio;
    macro_rules! bail {
        ($e:expr) => {{
            unsafe { asio_call!(driver, release) };
            return Err($e);
        }};
    }

    // The app's own window where we have one; drivers parent hidden
    // notification windows to it and do not all survive the desktop window.
    let mut hwnd = APP_HWND.load(Ordering::Relaxed);
    if hwnd == 0 {
        hwnd = unsafe { windows_sys::Win32::UI::WindowsAndMessaging::GetDesktopWindow() } as isize;
    }
    if unsafe { asio_call!(driver, init, hwnd as *mut c_void) } != ASIO_TRUE {
        let e = driver_error(driver, "driver init failed");
        bail!(format!("ASIO \"{name}\": {e}"));
    }

    let (mut n_in, mut n_out) = (0i32, 0i32);
    if unsafe { asio_call!(driver, get_channels, &mut n_in, &mut n_out) } != ASE_OK {
        bail!(format!("ASIO \"{name}\": channel query failed"));
    }
    if n_out < 1 {
        bail!(format!("ASIO \"{name}\": reports no output channels"));
    }

    let (mut min_b, mut max_b, mut pref_b, mut gran) = (0i32, 0i32, 0i32, 0i32);
    if unsafe {
        asio_call!(
            driver,
            get_buffer_size,
            &mut min_b,
            &mut max_b,
            &mut pref_b,
            &mut gran
        )
    } != ASE_OK
    {
        bail!(format!("ASIO \"{name}\": buffer size query failed"));
    }

    let mut current_rate = 0f64;
    let _ = unsafe { asio_call!(driver, get_sample_rate, &mut current_rate) };

    // `canSampleRate` only. The rate is never *set*, so a driver that is
    // currently streaming for another application is not disturbed.
    let mut rates = Vec::new();
    for r in PCM_PROBE_RATES {
        if unsafe { asio_call!(driver, can_sample_rate, r as f64) } == ASE_OK {
            rates.push(r);
        }
    }

    let mut info = AsioChannelInfo {
        channel: 0,
        is_input: ASIO_FALSE,
        is_active: ASIO_FALSE,
        channel_group: 0,
        sample_type: 0,
        name: [0; 32],
    };
    if unsafe { asio_call!(driver, get_channel_info, &mut info) } != ASE_OK {
        bail!(format!("ASIO \"{name}\": channel info query failed"));
    }
    let (sample_type_name, exact_capable) = pcm_sample_type_name(info.sample_type);

    unsafe { asio_call!(driver, release) };
    Ok(AsioPcmCapabilities {
        driver: name.to_string(),
        outputs: n_out,
        min_buffer: min_b,
        max_buffer: max_b,
        preferred_buffer: pref_b,
        current_rate,
        rates,
        sample_type: info.sample_type,
        sample_type_name,
        exact_capable,
    })
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
    /// The generation this session belongs to — see the PCM `Session`. The
    /// callback draws its conclusions from this rather than from whatever the
    /// stream reports at the instant it runs, so a feeder that ended
    /// microseconds before the next `start_session` cannot fail the track that
    /// is starting.
    generation: u64,
    cons: super::frame_ring::FrameConsumer<u8>,
    decode_done: Arc<AtomicBool>,
    /// Set by the feeder only when it reached the end of the file cleanly.
    decode_eof: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Drop for AsioSession {
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

struct AsioShared {
    session: Mutex<Option<AsioSession>>,
    paused: AtomicBool,
    /// Which track is playing and how it ended, in one word — see the PCM
    /// `Shared::play` for why these may not be two.
    ///
    /// Without a generation at all, a fault raised by the session that just
    /// ended was still readable against the one that replaced it: a stream
    /// survives track changes, so "the fault belongs to the stream" was never
    /// the same statement as "the fault belongs to this track". With the two
    /// as separate words, a callback descheduled between reading the clock and
    /// writing the state ended the wrong track instead.
    play: AtomicU64,
    /// True between the ring running dry and the driver having played what it
    /// already holds.
    draining: AtomicBool,
    /// Byte-frames actually handed to the driver since the drain began.
    drain_frames: AtomicU64,
    /// The driver's buffer, in **byte-frames** — the unit the callback deals
    /// in, one byte per channel per eight DSD samples.
    ///
    /// It held `ASIOGetBufferSize`'s answer, which in DSD mode is a count of
    /// 1-bit samples: eight times larger. The drain therefore waited for
    /// sixteen buffers rather than two, which is not a bound so much as a
    /// pause — up to a fifth of a second of the next track's start swallowed
    /// on a large buffer.
    buffer_frames: AtomicU64,
    /// When the drain began, as milliseconds since this stream opened.
    ///
    /// The frame target says how much the driver must play. This says how long
    /// it may take. Without it, a driver that stops calling back at exactly
    /// the wrong moment leaves a track permanently almost-finished: never
    /// advancing, never failing, and never explaining itself.
    /// The track a drain in progress belongs to.
    ///
    /// A drain outlives its session slot — that is what a drain *is*, the
    /// device playing out audio after the ring has been given up — so from
    /// the moment it starts there is nothing left to read a generation from.
    /// Everything the drain then says was said about whatever happened to be
    /// playing at that instant, which after a track change is the wrong
    /// track, and during the gap between sessions is generation zero: a
    /// number no session ever holds, so the evidence went nowhere at all.
    drain_gen: AtomicU64,
    drain_started_ms: AtomicU64,
    /// Callbacks seen since the drain began, so "the driver went quiet" is
    /// distinguishable from "the driver is working through it".
    drain_callbacks: AtomicU64,
    /// This stream's own epoch, for the two fields above.
    opened_at: Instant,
    /// Byte-frames (1 byte = 8 DSD samples, per channel) delivered as audio.
    frames_played: AtomicU64,
    /// Driver asked for a reset (device/sample-rate change) — surfaced as
    /// end-of-session; the next play re-negotiates from scratch.
    reset_requested: AtomicBool,
    /// The first realtime integrity fault of this session, as a `fault::` code.
    ///
    /// The callback publishes a `u8` and nothing else. It cannot format a
    /// string, take the logging mutex or touch the filesystem — an ASIO
    /// bufferSwitch has a hard deadline measured in the buffer it was just
    /// handed, and a driver whose callback blocks is a driver that glitches.
    /// The first *fatal* fault of this session, stamped — the reason it
    /// stopped.
    fault_code: AtomicU64,
    /// The first *recoverable* loss of integrity, stamped. Kept apart from the
    /// reason above so that neither erases the other.
    revoked: AtomicU64,
    /// The window every evidence publication happens inside. See
    /// `bitperfect::PublishWindow`.
    evidence: crate::bitperfect::PublishWindow,
    /// Identifies this `Shared` to `window::fire`. Tests only.
    #[cfg(test)]
    hook_id: u64,
    /// Callbacks that came up short mid-track: silence the DAC played.
    underruns: AtomicU64,
    /// Callbacks that could not reach the session slot in time.
    lock_misses: AtomicU64,
    /// Set when a session is installed, cleared once the ring first keeps up.
    priming: AtomicBool,
    /// Set by the callback on its first run; logged by the host thread, which
    /// is allowed to allocate.
    first_switch: AtomicBool,
    /// Driver events, published as counters because a callback may not format
    /// a message. The host thread turns them into log lines and faults.
    ///
    /// Every one of these means the stream is no longer the stream that was
    /// negotiated, so each ends the session's claim and prevents reuse.
    ev_time_info: AtomicBool,
    ev_rate_changed: AtomicBool,
    ev_reset: AtomicBool,
    ev_resync: AtomicBool,
    ev_overload: AtomicU64,
    ev_buffer_size_changed: AtomicBool,
    /// The last `asioMessage` selector seen, for diagnosis after the fact.
    ev_last_selector: std::sync::atomic::AtomicI32,
}

impl AsioShared {
    #[inline]
    fn generation(&self) -> u64 {
        super::stamped_generation(self.play.load(Ordering::Acquire))
    }

    /// Frames handed to the device under the session that is playing.
    #[inline]
    fn frames_played(&self) -> u64 {
        crate::bitperfect::counted::count(self.frames_played.load(Ordering::Acquire))
    }

    /// Credit `got` frames to `at`, and to `at` alone — see the PCM
    /// `Shared::account_frames`. A callback descheduled past a track change
    /// must not add its frames to the successor's total.
    fn account_frames(&self, at: u64, got: u64) {
        use crate::bitperfect::counted;
        if got == 0 {
            return;
        }
        let mut cur = self.frames_played.load(Ordering::Acquire);
        loop {
            if !counted::belongs_to(cur, at) {
                return;
            }
            let next = counted::pack(at, counted::count(cur).saturating_add(got));
            match self.frames_played.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }


    /// Begin a new track on this stream. Retires the previous one's fault and
    /// terminal state in one step, by making them belong to a generation that
    /// is no longer current.
    fn begin_session(&self) -> u64 {
        self.draining.store(false, Ordering::Relaxed);
        self.drain_frames.store(0, Ordering::Relaxed);
        // From the process allocator, not from the word being replaced: a
        // number derived from the current one is a number some other clock may
        // also be holding, and the stamp checks that guard every callback here
        // are only worth anything while a generation names one session.
        let next = super::gen_alloc::next();
        let want = super::stamped_pack(next, super::done_code::RUNNING);
        let mut cur = self.play.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) >= next {
                return super::stamped_generation(cur);
            }
            match self
                .play
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
        // The frame counter belongs to the new session from here.
        //
        // It was stamped by `start_session` alone, so every other way of
        // opening one left the counter carrying the previous session's tag and
        // `account_frames` refused every frame the new track played — the
        // position simply stopped moving. The same defect the PCM path had,
        // and it belongs where the generation is decided.
        self.frames_played.store(
            crate::bitperfect::counted::pack(next, 0),
            Ordering::Relaxed,
        );
                    return next;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// Move `at` to a terminal state, or refuse — one compare-exchange on the
    /// word that also holds the generation, so a superseded callback cannot
    /// end the track that replaced it however long it was descheduled.
    fn transition(&self, at: u64, code: u8, fatal: bool) {
        let want = super::stamped_pack(at, code);
        let mut cur = self.play.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) != at {
                return;
            }
            let held = super::stamped_code(cur);
            if held != super::done_code::RUNNING
                && !(fatal && held == super::done_code::CLEAN_EOF)
            {
                return;
            }
            match self
                .play
                .compare_exchange_weak(cur, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// The ring has run dry with the feeder finished. The driver still holds
    /// what it was last given, so the track is not over yet.
    ///
    /// `at` is the track that is draining, recorded here because from this
    /// point on there is no session left to ask.
    fn begin_drain(&self, at: u64) {
        self.drain_gen.store(at, Ordering::Release);
        self.drain_frames.store(0, Ordering::Relaxed);
        self.drain_callbacks.store(0, Ordering::Relaxed);
        // One clock read, once per track, on the callback thread. It is a
        // counter read, not a syscall, and it is what makes the wait bounded
        // in time as well as in frames.
        self.drain_started_ms.store(
            self.opened_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        self.draining.store(true, Ordering::Release);
    }

    /// Whether a drain has been open too long to still be a drain.
    ///
    /// Called from the UI thread, which is where a wall clock belongs: the
    /// callback can only count what it is given, and a driver that has stopped
    /// calling back gives it nothing to count.
    fn drain_expired(&self) -> bool {
        if !self.draining.load(Ordering::Acquire) {
            return false;
        }
        let began = self.drain_started_ms.load(Ordering::Relaxed);
        let now = self.opened_at.elapsed().as_millis() as u64;
        now.saturating_sub(began) >= DRAIN_LIMIT.as_millis() as u64
    }

    /// Advance a drain by the frames the driver just asked for, finishing once
    /// it has had enough to have played everything it holds.
    ///
    /// Reporting the track finished the instant the *ring* ran dry ended it
    /// while up to two driver buffers of it were still unplayed — audible on
    /// DSD, where a buffer is milliseconds of a stream that must not break.
    /// Advance a drain by byte-frames the driver has actually been given.
    ///
    /// `submitted` is what was written into the driver's buffers, not what the
    /// driver asked for. A callback that found a null buffer pointer, or that
    /// was counted before it wrote anything, moved the drain along on audio
    /// that never reached the device.
    fn advance_drain(&self, submitted: usize) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        // The track that started the drain, not whatever is playing when the
        // last buffer goes out.
        let at = self.drain_gen.load(Ordering::Acquire);
        self.drain_callbacks.fetch_add(1, Ordering::Relaxed);
        let n = self
            .drain_frames
            .fetch_add(submitted as u64, Ordering::Relaxed)
            + submitted as u64;
        // Twice the driver buffer covers the one being played and the one in
        // flight. A driver that reported nothing gets a small fixed bound
        // rather than an unbounded wait.
        let target = (self.buffer_frames.load(Ordering::Relaxed) * 2).max(2048);
        if n >= target {
            self.draining.store(false, Ordering::Release);
            self.finish_clean(at);
        }
    }

    /// End a drain that has run out of time rather than out of audio.
    ///
    /// Called from the UI thread. A driver that stops calling back mid-drain
    /// has not played what it holds and never will, so this is a failure and
    /// not an ending — the track did not finish, and a playlist that advanced
    /// on it would be advancing on a device that has stopped.
    fn expire_drain(&self) {
        if !self.draining.load(Ordering::Acquire) {
            return;
        }
        self.draining.store(false, Ordering::Release);
        // Against the draining track, not against whatever is playing when
        // the UI notices. A device that stopped mid-drain failed *that*
        // track, and stamping the failure with the current generation put it
        // on a session that had done nothing wrong — or, in the gap between
        // sessions, on generation zero, which no session ever holds.
        self.fail(self.drain_gen.load(Ordering::Acquire), super::fault::BACKEND_DEAD);
    }

    /// Playback ended, cleanly, having played everything. Refused if anything
    /// has already gone wrong: a session that faulted and then drained did not
    /// finish.
    fn finish_clean(&self, at: u64) {
        // An integrity fault costs the badge, not the track — see
        // `bitperfect::fault::is_fatal`. A DSD track that dropped out still
        // ended, and refusing to say so stopped the playlist.
        self.transition(at, super::done_code::CLEAN_EOF, false)
    }

    /// Playback ended because something went wrong. Latching, and it outranks
    /// a clean end. `code` may be `fault::NONE` to keep an already-recorded
    /// fault rather than adding one.
    fn fail(&self, at: u64, code: u8) {
        if code != super::fault::NONE {
            self.raise_fault_in(at, code);
        }
        self.transition(at, super::done_code::FAILED, true);
    }

    /// A loss of integrity that does not end the track, kept apart from the
    /// reason it eventually stops.
    ///
    /// One slot could not hold both. First-wins meant a dropout suppressed the
    /// write error that actually ended the session, so the track reported the
    /// wrong reason for stopping; last-wins would have erased the evidence
    /// that anything was lost before it. Both are true and both are kept.
    #[inline]
    fn revoke(&self, at: u64, code: u8) {
        debug_assert!(!super::fault::is_fatal(code), "revoke is for integrity faults");
        // Opened before anything is read, and closed by the guard on every
        // exit — including the ones that decide not to write at all, which a
        // reader cannot tell apart from a write that has not happened yet.
        let _publishing = self.evidence.publish();
        let want = super::stamped_pack(at, code);
        let mut cur = self.revoked.load(Ordering::Acquire);
        loop {
            if super::stamped_generation(cur) > at {
                return;
            }
            if super::stamped_generation(cur) == at
                && super::stamped_code(cur) != super::fault::NONE
            {
                return;
            }
            match self.revoked.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    #[cfg(test)]
                    crate::bitperfect::window::fire(
                        self.hook_id,
                        crate::bitperfect::window::Site::Publish(1),
                    );
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// One attempt at reading this session's evidence.
    ///
    /// Native sessions run no Q1.31 conversion, so that field is always zero;
    /// the other two are the same two records, with the same hazard between
    /// them.
    fn read_evidence(&self, at: u64) -> crate::bitperfect::Evidence {
        let revoked = self.revoked_in(at);
        #[cfg(test)]
        crate::bitperfect::window::fire(
            self.hook_id,
            crate::bitperfect::window::Site::Evidence(2),
        );
        let fatal = self.fatal_in(at);
        crate::bitperfect::Evidence {
            generation: at,
            off_grid: 0,
            revoked,
            fatal,
            settled: false,
        }
    }

    /// The recoverable integrity loss recorded against `at`, if any. Survives
    /// a later fatal fault.
    #[inline]
    fn revoked_in(&self, at: u64) -> u8 {
        let v = self.revoked.load(Ordering::Acquire);
        if super::stamped_generation(v) == at {
            super::stamped_code(v)
        } else {
            super::fault::NONE
        }
    }

    /// The recoverable loss of the session playing now.
    ///
    /// Tests only. Production reads `evidence_at`, which answers about one
    /// named generation; this answers about "now", which is the question that
    /// cannot be composed with another.
    #[cfg(test)]
    #[inline]
    fn revoked(&self) -> u8 {
        self.revoked_in(self.generation())
    }

    /// How this session ended, or `Running`.
    fn completion(&self) -> super::Completion {
        let v = self.play.load(Ordering::Acquire);
        let generation = super::stamped_generation(v);
        match super::stamped_code(v) {
            super::done_code::CLEAN_EOF => super::Completion::CleanEof,
            super::done_code::FAILED => super::Completion::Failed {
                reason: self.fault_in(generation),
                generation,
            },
            _ => super::Completion::Running,
        }
    }

    /// Publish an integrity fault against `at`, and only `at`. First one wins.
    #[inline]
    fn raise_fault_in(&self, at: u64, code: u8) {
        let _publishing = self.evidence.publish();
        let want = super::stamped_pack(at, code);
        let mut cur = self.fault_code.load(Ordering::Acquire);
        loop {
            // Monotone in the generation: a callback still finishing after its
            // session was replaced must not stamp its own, older generation
            // over a fault the successor legitimately raised.
            if super::stamped_generation(cur) > at {
                return;
            }
            // First one wins *within a session*; a record left by a session
            // that has been replaced is stale, and this one replaces it.
            if super::stamped_generation(cur) == at
                && super::stamped_code(cur) != super::fault::NONE
            {
                return;
            }
            match self.fault_code.compare_exchange_weak(
                cur,
                want,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    #[cfg(test)]
                    crate::bitperfect::window::fire(
                        self.hook_id,
                        crate::bitperfect::window::Site::Publish(2),
                    );
                    return;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// End the session that is playing now. For producers that outlive
    /// sessions — the host thread, the driver's own error notifications.
    #[inline]
    fn fail_now(&self, code: u8) {
        self.fail(self.generation(), code);
    }

    /// This session's fault, or `fault::NONE`. A fault raised by a session
    /// that has since been replaced is not this one's.
    /// The terminal fault of the session playing now — **fatal only**. See
    /// the PCM `Shared::fault` for what the fall-through to recoverable
    /// evidence cost.
    /// The terminal fault recorded against `at`, or `fault::NONE` — fatal
    /// only, and read by the stamp so it composes with the other fields.
    #[inline]
    fn fatal_in(&self, at: u64) -> u8 {
        let v = self.fault_code.load(Ordering::Acquire);
        if super::stamped_generation(v) == at {
            super::stamped_code(v)
        } else {
            super::fault::NONE
        }
    }

    #[inline]
    fn fault(&self) -> u8 {
        self.fatal_in(self.generation())
    }

    /// What this session has to answer for: the reason it stopped if it
    /// stopped, otherwise the claim it lost. A fatal reason outranks a
    /// recoverable one without overwriting it.
    #[inline]
    fn fault_in(&self, at: u64) -> u8 {
        let v = self.fault_code.load(Ordering::Acquire);
        if super::stamped_generation(v) == at
            && super::stamped_code(v) != super::fault::NONE
        {
            return super::stamped_code(v);
        }
        self.revoked_in(at)
    }
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
    /// The idle byte this driver's bit order needs.
    idle: u8,
    /// Interleaved-read scratch, sized once at setup to a whole buffer.
    /// The callback indexes it; it never grows, so nothing allocates there.
    scratch: Vec<u8>,
    supports_output_ready: bool,
}

static ACTIVE: AtomicPtr<CallbackCtx> = AtomicPtr::new(null_mut());

/// Whether an ASIO engine is engaged in this process.
///
/// The callbacks are `extern "system"` functions with no user pointer, so
/// there is exactly one of everything and it lives in `ACTIVE`. Guarding that
/// with `if !ACTIVE.load().is_null() { return Err(..) }` and publishing much
/// later, on another thread, left a window in which two opens could both see
/// null, both proceed, and the second overwrite the first's context — leaving
/// the first driver calling into a `Box` that had been freed.
///
/// The claim is taken here, in one atomic step, and released only when the
/// host thread has finished tearing the engine down.
static ENGAGED: AtomicBool = AtomicBool::new(false);

/// Releases the engine claim however the host thread leaves — including by
/// panic, which would otherwise make every future open fail with "already
/// open" for the life of the process.
struct EngineClaim;

impl EngineClaim {
    /// Take the claim, or report that someone else has it.
    fn take() -> Result<Self, String> {
        ENGAGED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| EngineClaim)
            .map_err(|_| "an ASIO stream is already open".to_string())
    }
}

impl Drop for EngineClaim {
    fn drop(&mut self) {
        // The context is unpublished before the claim is released, so no
        // callback can find a pointer belonging to a session that is gone.
        ACTIVE.store(null_mut(), Ordering::Release);
        ENGAGED.store(false, Ordering::Release);
    }
}
/// How long a drain may take before the driver is presumed to have stopped.
///
/// A driver buffer is milliseconds; this is orders of magnitude longer. It is
/// not a deadline the hardware is expected to meet — it is the point past
/// which "still draining" has stopped being a description of anything.
const DRAIN_LIMIT: Duration = Duration::from_secs(5);

/// How long a driver has to produce its first buffer switch after `ASIOStart`.
///
/// Generous: some drivers take tens of milliseconds to spin up their engine.
/// The point is that "never" is distinguishable from "slow".
const FIRST_CALLBACK_TIMEOUT: Duration = Duration::from_millis(1500);

/// Take up to `frames` byte-frames of the installed session into `scratch`.
///
/// A function of its own so the paused hand-off can be forced without a
/// driver. Everything above it in `fill_buffers` is pointer work a test cannot
/// supply, and everything below it only reads what this returns — so this is
/// the whole of the callback's consumption decision, and the seam the forced
/// schedule stands at.
///
/// Returns the byte-frames taken, the session they belong to (zero if none was
/// reached, and nothing is accounted against zero), and whether this call is
/// the one that started the drain.
fn take_session_frames(sh: &AsioShared, scratch: &mut [u8], frames: usize) -> (usize, u64, bool) {
    let mut got = 0usize;
    // Whether *this* callback is the one that hands the driver the last of the
    // audio. It must not also count that buffer as played — see the caller.
    let mut just_began_drain = false;
    // The session these frames belong to, established while the slot is held.
    // Zero means no session was reached, and nothing is accounted.
    let mut session_at = 0u64;

    if !sh.paused.load(Ordering::Relaxed) {
        #[cfg(test)]
        crate::bitperfect::window::fire(sh.hook_id, crate::bitperfect::window::Site::Render);
        match sh.session.try_lock() {
            Ok(mut guard) => {
                // Asked again, now that the slot is in hand.
                //
                // The outer read is a fast path and nothing more: between it
                // and this lock the UI thread can pause and install a new
                // session, and the answer the fast path gave was about a
                // session this callback is no longer the one to play.
                //
                // `pause` stores with `Release`, and the UI thread then takes
                // this same mutex to install the session. Acquiring it here
                // synchronises with that release, so a session installed after
                // a pause can never be reached with `paused` still reading
                // false. Nothing is consumed before it.
                if !sh.paused.load(Ordering::Acquire)
                    && let Some(sess) = guard.as_mut()
                {
                    // Said about the session in hand, not about whatever the
                    // stream reports at this instant — see the PCM render
                    // path for what the difference cost.
                    let at = sess.generation;
                    session_at = at;
                    // A ring holding a non-multiple of the channel count has
                    // been torn by something outside `frame_ring`; there is no
                    // way to know which channel the stray bytes belong to, so
                    // it ends the track rather than being resynchronised on a
                    // guess — see the PCM render path.
                    if !sess.cons.invariant_holds() {
                        sh.fail(at, super::fault::RING_INVARIANT);
                    }
                    got = sess.cons.pop_frames(scratch, frames);
                    let ended = sess.decode_done.load(Ordering::Acquire) && sess.cons.is_empty();
                    let eof = sess.decode_eof.load(Ordering::Acquire);
                    if got < frames {
                        if ended {
                            *guard = None;
                            // Why the feeder stopped is the feeder's to say.
                            // Drain decides only when.
                            if eof {
                                // The driver still holds what it was last
                                // given; the drain decides when that has been
                                // played.
                                //
                                // Against `at`: the session slot is being
                                // given up in the same breath, so this is the
                                // last moment at which the draining track can
                                // be named at all.
                                sh.begin_drain(at);
                                just_began_drain = true;
                            } else {
                                sh.fail(at, super::fault::SESSION_THREAD_FAILED);
                            }
                        } else if sh.priming.load(Ordering::Relaxed) {
                            // The gap the listener asked for, after a start or
                            // a seek.
                        } else {
                            sh.underruns.fetch_add(1, Ordering::Relaxed);
                            sh.revoke(at, super::fault::UNDERRUN);
                        }
                    } else {
                        sh.priming.store(false, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                // A miss that coincides with a pause is the UI thread holding
                // the slot to install a paused session — the hand-off working,
                // not a dropout.
                if !sh.paused.load(Ordering::Acquire) {
                    sh.lock_misses.fetch_add(1, Ordering::Relaxed);
                    if !sh.priming.load(Ordering::Relaxed) {
                        sh.revoke(sh.generation(), super::fault::CALLBACK_LOCK_MISS);
                    }
                }
            }
        }
    }
    (got, session_at, just_began_drain)
}

/// Fill one channel's device buffer: payload for the byte-frames the session
/// gave up, and this driver's DSD idle pattern for the rest.
///
/// Separate from the pointer arithmetic so a test can see what a paused
/// hand-off puts on the wire — which is idle on every channel, because `got`
/// is zero and there is nothing else this can write.
fn fill_channel(dest: &mut [u8], scratch: &[u8], got: usize, ch: usize, c: usize, idle: u8) {
    for (fr, out) in dest.iter_mut().enumerate() {
        *out = if fr < got { scratch[fr * ch + c] } else { idle };
    }
}

/// Fill one half of the double buffer.
///
/// Strictly realtime: `try_lock` only, no allocation, no formatting, no
/// logging, no filesystem. Silence is the DSD idle pattern (`0x69`) so the DAC
/// keeps its lock through a pause or an underrun instead of thumping on
/// `0x00`, and faults are published as a single relaxed compare-exchange.
///
/// Frame-atomic throughout. The old drain popped bytes one at a time until the
/// ring ran dry, which could stop mid-frame and leave a byte belonging to
/// channel *n* at the head of the ring — where the next callback read it as
/// channel 0, permanently rotating every channel. A pause now consumes nothing
/// at all, and a short read leaves whole frames behind rather than a remainder
/// nobody can attribute.
unsafe fn fill_buffers(ctx: &mut CallbackCtx, index: i32) {
    let frames = ctx.buffer_bytes; // byte-frames: 1 byte = 8 DSD samples per channel
    let ch = ctx.channels;
    // An `Arc` bump, not an allocation — and it lets `scratch` be borrowed
    // mutably below while the shared state is still reachable.
    let sh = Arc::clone(&ctx.shared);
    // One relaxed store; the host thread turns it into a log line, because
    // this function may not.
    sh.first_switch.store(true, Ordering::Relaxed);
    let (got, session_at, just_began_drain) =
        take_session_frames(&sh, &mut ctx.scratch[..frames * ch], frames);
    let mut written = 0usize;
    for (c, info) in ctx.buffer_infos.iter().enumerate() {
        let dest = info.buffers[index as usize] as *mut u8;
        if dest.is_null() { continue; }
        // SAFETY: the driver owns `buffer_bytes` bytes at this pointer for the
        // half it just handed us, which is `frames` — the same bound the
        // byte-at-a-time loop this replaces used.
        let dest = unsafe { std::slice::from_raw_parts_mut(dest, frames) };
        fill_channel(dest, &ctx.scratch, got, ch, c, ctx.idle);
        written += 1;
    }

    // Credited once every selected channel has been written, and not before.
    //
    // The count used to happen above, straight after the pop — so a callback
    // that found a null buffer pointer for one channel, wrote nothing to it,
    // and handed the driver a partial frame still moved the listener's
    // position forward by the whole of it. The position is what the elapsed
    // clock, the seek bar and the boundary arithmetic are all built on; it has
    // to mean "the device has this", not "the ring no longer does".
    if written == ch {
        sh.account_frames(session_at, got as u64);
    }

    if ctx.supports_output_ready {
        let _ = unsafe { asio_call!(ctx.driver, output_ready) };
    }

    // With no session installed and a drain in progress, these callbacks are
    // the driver playing out what it already holds. Counting them is what
    // turns "the ring is empty" into "the listener has heard the whole track"
    // — on DSD especially, where the buffer the driver still holds is a
    // bitstream the DAC is locked to.
    //
    // Counted here, after the write, and only if every channel got one. The
    // count used to happen before the buffers were filled and regardless of
    // whether they could be: a callback that found null pointers advanced the
    // drain on audio the device never received.
    // Not on the callback that started it. That callback has just written the
    // final payload into the driver's buffer; the driver has been *given* it
    // and has not played a sample of it. Counting it here credited the drain
    // with a buffer that was still ahead of the DAC, so the track ended one
    // whole buffer early — on DSD, the point at which the bitstream stops and
    // the next track's begins over the tail of this one.
    if written == ch
        && !just_began_drain
        && sh.session.try_lock().map(|g| g.is_none()).unwrap_or(false)
    {
        sh.advance_drain(frames);
    }
}

unsafe extern "system" fn cb_buffer_switch(index: i32, _direct: AsioBool) {
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
    // One relaxed store. This is a realtime callback like any other: it used to
    // format a string and take the logging mutex on its first call, on the
    // thread with the hardest deadline in the process.
    with_shared(|sh| sh.ev_time_info.store(true, Ordering::Relaxed));
    unsafe { cb_buffer_switch(index, ASIO_FALSE) };
    null_mut()
}

unsafe extern "system" fn cb_sample_rate_did_change(_rate: f64) {
    // The rate changed underneath a stream that was negotiated at a specific
    // one. Whatever is being played is no longer what was agreed, so this is a
    // fault and not a notice.
    with_shared(|sh| {
        sh.ev_rate_changed.store(true, Ordering::Relaxed);
        // The stream was negotiated against a rate that is no longer the
        // driver's. Failing the track alone left the handle looking usable, so
        // the next track was handed to a device that had already said it was
        // running at a different speed.
        sh.reset_requested.store(true, Ordering::Release);
        sh.fail_now(super::fault::BACKEND_RESET);
    });
}

/// Reach the published callback context, if one is live.
///
/// Every callback entry point goes through here, so "callbacks only publish
/// atomics" is enforced by there being nothing else reachable from them.
#[inline]
fn with_shared(f: impl FnOnce(&AsioShared)) {
    let ctx = ACTIVE.load(Ordering::Acquire);
    if !ctx.is_null() {
        // SAFETY: `ACTIVE` is non-null only between publication and the
        // unpublish that follows `ASIOStop`, and the context outlives both.
        f(unsafe { &(*ctx).shared });
    }
}

unsafe extern "system" fn cb_asio_message(
    selector: i32,
    value: i32,
    _msg: *mut c_void,
    _opt: *mut f64,
) -> i32 {
    // No logging here either. `asioMessage` is sparse but it is still a driver
    // callback, and a driver that sends a burst of them during a glitch is
    // exactly when the logging mutex must not be taken.
    with_shared(|sh| sh.ev_last_selector.store(selector, Ordering::Relaxed));
    match selector {
        K_ASIO_SELECTOR_SUPPORTED => i32::from(matches!(
            value,
            K_ASIO_ENGINE_VERSION
                | K_ASIO_RESET_REQUEST
                | K_ASIO_RESYNC_REQUEST
                | K_ASIO_LATENCIES_CHANGED
                | K_ASIO_OVERLOAD
                | K_ASIO_BUFFER_SIZE_CHANGE
        )),
        K_ASIO_ENGINE_VERSION => 2,
        // Deliberately unsupported: drivers then use the plain bufferSwitch
        // callback, which involves no ASIOTime structures — smallest possible
        // ABI surface while the path is being hardware-proven.
        K_ASIO_SUPPORTS_TIME_INFO => 0,
        // Each of these means the stream stopped being the one that was
        // negotiated. They end the session's claim and the host thread reopens
        // rather than carrying on with a stream whose shape it no longer knows.
        K_ASIO_RESET_REQUEST => {
            with_shared(|sh| {
                sh.ev_reset.store(true, Ordering::Relaxed);
                sh.reset_requested.store(true, Ordering::Release);
                sh.fail_now(super::fault::BACKEND_RESET);
            });
            1
        }
        K_ASIO_RESYNC_REQUEST => {
            with_shared(|sh| {
                sh.ev_resync.store(true, Ordering::Relaxed);
                // A resync means the driver's own clock moved under a stream
                // that was negotiated against it. Published as an integrity
                // fault it ended nothing, so the stream stayed reusable and
                // the next track was handed to a device that had already told
                // us it was no longer where we left it.
                sh.reset_requested.store(true, Ordering::Release);
                sh.fail_now(super::fault::BACKEND_RESET);
            });
            1
        }
        K_ASIO_OVERLOAD => {
            with_shared(|sh| {
                sh.ev_overload.fetch_add(1, Ordering::Relaxed);
                // The driver could not keep up with its own callback, so the
                // device played something that was not in the file. Fatal to
                // the claim, and the session ends on it rather than carrying
                // on and then reporting a clean finish.
                sh.fail_now(super::fault::BACKEND_OVERLOAD);
            });
            1
        }
        K_ASIO_BUFFER_SIZE_CHANGE => {
            with_shared(|sh| {
                sh.ev_buffer_size_changed.store(true, Ordering::Relaxed);
                sh.reset_requested.store(true, Ordering::Release);
                sh.fail_now(super::fault::BACKEND_RESET);
            });
            1
        }
        K_ASIO_LATENCIES_CHANGED => 1,
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
    /// Whether the driver confirmed it is in DSD mode when asked.
    ///
    /// `false` means the driver does not implement `kAsioGetIoFormat`, so the
    /// mode was set and never read back. Playback is unaffected; the *claim*
    /// is, because payload-exactness on this route rests on the driver being
    /// in the mode we asked for and nothing here has confirmed that.
    pub io_format_verified: bool,
    /// Whether the sample rate was accepted in the form the ASIO DSD
    /// specification defines, rather than one of the undocumented divided
    /// forms some drivers want.
    ///
    /// A driver that took `rate/8` accepted a number. What it understood by it
    /// is written down nowhere, so the route works and is not proven — and the
    /// badge has to say the second part as well as the first.
    pub rate_spec_form: bool,
}

struct SetupOk {
    lsb_first: bool,
    /// Byte-frames per buffer half — one byte per channel per eight DSD
    /// samples, which is the unit the callback and the drain both use.
    buffer_frames: u32,
    /// Whether the rate was accepted in the form the specification defines.
    rate_spec_form: bool,
    /// Whether the driver confirmed, when asked, that it is in DSD mode.
    ///
    /// `kAsioSetIoFormat` succeeding says the call succeeded, not that the
    /// mode stuck. Drivers that implement `kAsioGetIoFormat` can be asked; on
    /// one that does not, the mode is *unverified* — not wrong, not confirmed.
    /// Playback continues either way, because a driver that accepted the
    /// switch almost certainly made it; what changes is the claim.
    io_format_verified: bool,
}

/// A stream with no device behind it, for the pause-ordering test in `main`.
///
/// `main`'s bring-up adapter is generic over the stream it pauses, and the
/// implementation under test is this type's own `pause`/`resume`. The test
/// needs an `AsioDsdStream` and cannot open a driver, so it gets one whose
/// `shared` is the same device-free fixture the evidence tests use.
#[cfg(test)]
pub(crate) fn for_pause_test() -> AsioDsdStream {
    AsioDsdStream::for_evidence_test(tests::shared())
}

impl AsioDsdStream {
    /// A stream with no device behind it, for testing the evidence adapter.
    ///
    /// The adapter is the thing under test — `evidence_at` on the public type
    /// a caller actually holds — and it touches nothing but `shared`. `join`
    /// is `None`, so `Drop` sets the stop flag and joins nothing.
    #[cfg(test)]
    fn for_evidence_test(shared: Arc<AsioShared>) -> Self {
        AsioDsdStream {
            stop: Arc::new(AtomicBool::new(false)),
            join: None,
            shared,
            driver_name: "test".into(),
            dsd_rate: 2_822_400,
            channels: 2,
            lsb_first: false,
            buffer_frames: 1024,
            io_format_verified: true,
            rate_spec_form: true,
        }
    }

    /// Open `driver_name` in native-DSD mode at `dsd_rate` (the bit rate,
    /// e.g. 22 579 200 for DSD512) with `channels` outputs. The entire
    /// negotiation runs on a dedicated host thread; errors come back verbatim.
    pub fn open(driver_name: &str, dsd_rate: u32, channels: u16) -> Result<Self, String> {
        // Claimed before anything else, in one atomic step, and handed to the
        // host thread so it is released exactly when that thread is done.
        let claim = EngineClaim::take()?;
        let clsid = driver_clsid(driver_name)?;
        let shared = Arc::new(AsioShared {
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            play: AtomicU64::new(super::stamped_pack(0, super::done_code::RUNNING)),
            draining: AtomicBool::new(false),
            drain_frames: AtomicU64::new(0),
            buffer_frames: AtomicU64::new(0),
            drain_gen: AtomicU64::new(0),
            drain_started_ms: AtomicU64::new(0),
            drain_callbacks: AtomicU64::new(0),
            opened_at: Instant::now(),
            frames_played: AtomicU64::new(crate::bitperfect::counted::pack(0, 0)),
            reset_requested: AtomicBool::new(false),
            fault_code: AtomicU64::new(0),
            revoked: AtomicU64::new(0),
            evidence: crate::bitperfect::PublishWindow::new(),
            #[cfg(test)]
            hook_id: crate::bitperfect::window::next_id(),
            underruns: AtomicU64::new(0),
            lock_misses: AtomicU64::new(0),
            priming: AtomicBool::new(false),
            first_switch: AtomicBool::new(false),
            ev_time_info: AtomicBool::new(false),
            ev_rate_changed: AtomicBool::new(false),
            ev_reset: AtomicBool::new(false),
            ev_resync: AtomicBool::new(false),
            ev_overload: AtomicU64::new(0),
            ev_buffer_size_changed: AtomicBool::new(false),
            ev_last_selector: std::sync::atomic::AtomicI32::new(0),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<SetupOk, String>>();

        let t_shared = Arc::clone(&shared);
        let t_stop = Arc::clone(&stop);
        let t_name = driver_name.to_owned();
        let join = std::thread::Builder::new()
            .name("bp-asio-dsd".into())
            .spawn(move || {
                host_thread(t_name, clsid, dsd_rate, channels, t_shared, t_stop, tx, claim)
            })
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
                io_format_verified: ok.io_format_verified,
                rate_spec_form: ok.rate_spec_form,
            }),
            Ok(Err(e)) => { let _ = join.join(); Err(e) }
            Err(_) => { let _ = join.join(); Err("ASIO host thread died during setup".into()) }
        }
    }

    /// Install a new track session (ring consumer + decode-thread flags),
    /// replacing any current one.
    pub fn start_session(
        &self,
        cons: super::frame_ring::FrameConsumer<u8>,
        decode_done: Arc<AtomicBool>,
        decode_eof: Arc<AtomicBool>,
        session_stop: Arc<AtomicBool>,
    ) {
        self.shared
            .buffer_frames
            .store(self.buffer_frames as u64, Ordering::Relaxed);
        self.shared.priming.store(true, Ordering::Relaxed);
        // The clock turns inside the lock that holds the slot.
        //
        // Bumping the generation first and installing afterwards left a window
        // in which the outgoing session sat in the slot with the incoming
        // generation current — and the drain state had already been reset out
        // from under a session that was still using it. They are one change,
        // so they are made as one. The callback only ever `try_lock`s this, so
        // it is never blocked by the swap; it simply finds the old session for
        // an instant longer, which is exactly what it is playing.
        if let Ok(mut g) = self.shared.session.lock() {
            // `begin_session` stamps the frame counter with the generation it
            // installs, so every path that opens a session gets it.
            let generation = self.shared.begin_session();
            *g = Some(AsioSession {
                generation,
                cons,
                decode_done,
                decode_eof,
                stop: session_stop,
            });
        }
    }

    /// Everything about `at`, read by `at`'s own stamps.
    ///
    /// Native sessions have no Q1.31 conversion, so that field is always zero.
    /// See `bitperfect::Evidence` for why the four are read as one thing.
    pub fn evidence_at(&self, at: u64) -> crate::bitperfect::Evidence {
        crate::bitperfect::stable_evidence(&self.shared.evidence, at, || {
            self.shared.read_evidence(at)
        })
    }

    /// The audio generation this stream's evidence is currently about.
    pub fn audio_generation(&self) -> u64 {
        self.shared.generation()
    }

    // `revoked_code` and `fault_code` were here — see the PCM `BpStream` for
    // why the per-field getters are gone rather than merely unused.

    /// Callbacks that came up short mid-track — silence the DAC played.
    pub fn underruns(&self) -> u64 {
        self.shared.underruns.load(Ordering::Relaxed)
    }

    /// `Release`, so that the callback's `Acquire` re-read after it has the
    /// session slot cannot see a stale `false` — see `take_session_frames`.
    pub fn pause(&self)  { self.shared.paused.store(true,  Ordering::Release); }
    pub fn resume(&self) { self.shared.paused.store(false, Ordering::Release); }
    pub fn is_paused(&self) -> bool { self.shared.paused.load(Ordering::Relaxed) }
    /// How this session ended — the only thing that may advance a playlist is
    /// `Completion::CleanEof`.
    ///
    /// Polled from the UI thread, which is why the wall clock is checked here:
    /// a drain is advanced by callbacks, and a driver that has stopped calling
    /// back cannot advance one. Without this the track sits at
    /// almost-finished for as long as the process runs — never advancing,
    /// never failing, and never saying why.
    pub fn completion(&self) -> super::Completion {
        if self.shared.drain_expired() {
            self.shared.expire_drain();
        }
        self.shared.completion()
    }
    pub fn reset_requested(&self) -> bool { self.shared.reset_requested.load(Ordering::Acquire) }

    /// Sample-accurate elapsed time: byte-frames delivered × 8 samples ÷ rate.
    pub fn played(&self) -> Duration {
        let frames = self.shared.frames_played();
        Duration::from_secs_f64(frames as f64 * 8.0 / self.dsd_rate.max(1) as f64)
    }
}

impl Drop for AsioDsdStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(j) = self.join.take() { let _ = j.join(); }
    }
}

// (The decode-thread ring feed lives in `super::native_dsd::feed_loop` — it's
// shared verbatim with the Linux ALSA backend.)

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

#[allow(clippy::too_many_arguments)]
fn host_thread(
    name: String,
    clsid: GUID,
    dsd_rate: u32,
    channels: u16,
    shared: Arc<AsioShared>,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<SetupOk, String>>,
    claim: EngineClaim,
) {
    // Dropped last, on every exit path including a panic: it unpublishes the
    // callback context and then releases the process-wide engine claim. Every
    // early return below used to have to remember to null `ACTIVE`, and a
    // panic remembered nothing.
    let _claim = claim;
    // A host thread that has ended can produce no more audio, whatever the
    // handle looks like. This is the one place that is true on every path.
    let _death = HostDeath(Arc::clone(&shared));
    unsafe {
        // ASIO drivers are in-proc COM servers and generally expect an STA.
        //
        // Declared before anything that owns a driver, so that it is dropped
        // *after* it: locals unwind in reverse, and `StartedDriver` lives in
        // the match arm below, which is an inner scope again. `CoUninitialize`
        // running while a driver object is still alive is undefined, and the
        // manual calls this replaces did exactly that on any panic — there
        // were four of them, one per exit, and none on the unwind path.
        let _com = ComApartment::enter();

        let result = host_setup(&name, &clsid, dsd_rate, channels, &shared);
        match result {
            Err(e) => {
                let _ = tx.send(Err(e));
            }
            Ok((driver, callbacks, ctx_box, setup)) => {
                // Publish the callback context, then start the engine.
                let ctx_ptr = Box::into_raw(ctx_box);
                ACTIVE.store(ctx_ptr, Ordering::Release);
                let rc = asio_call!(driver, start);
                // First statement after the start, and deliberately before the
                // log line.
                //
                // `mlog!` formats a string and takes the logging mutex, either
                // of which can panic — an allocation failure, a mutex poisoned
                // by an earlier panic elsewhere. One line of logging was
                // standing between `ASIOStart` succeeding and anything owning
                // the driver, and a panic in that line left the DAC running
                // with the device held until the process ended.
                let _started = StartedDriver { driver, callbacks, ctx: ctx_ptr };
                crate::mlog!("[asio-dsd] \"{name}\": start rc={rc}");
                if rc != ASE_OK {
                    let e = driver_error(driver, "ASIO start failed");
                    let _ = tx.send(Err(e));
                    return;
                }

                // Success is a callback, not a return code.
                //
                // `ASIOStart` returning `ASE_OK` only means the driver accepted
                // the request. A driver that then never calls back leaves a
                // stream that is open, silent and — before this — reported as
                // playing exactly. Wait a bounded time for the first buffer
                // switch and fail the open if it does not arrive.
                let deadline = Instant::now() + FIRST_CALLBACK_TIMEOUT;
                while !shared.first_switch.load(Ordering::Relaxed) {
                    if Instant::now() >= deadline {
                        let _ = tx.send(Err(format!(
                            "ASIO \"{name}\": the driver started but produced no callback \
                             within {} ms", FIRST_CALLBACK_TIMEOUT.as_millis())));
                        return;
                    }
                    if stop.load(Ordering::Acquire) { break; }
                    std::thread::sleep(Duration::from_millis(2));
                }
                crate::mlog!("[asio-dsd] \"{name}\": first callback observed, streaming");
                let _ = tx.send(Ok(setup));

                // Park until the stream is dropped, reporting what the
                // realtime callback was only allowed to flag. This thread may
                // allocate and take the logging mutex; the callback may not.
                let mut said_switch = false;
                let mut said_fault = false;
                while !stop.load(Ordering::Acquire) {
                    if !said_switch && shared.first_switch.load(Ordering::Relaxed) {
                        said_switch = true;
                        crate::mlog!("[asio-dsd] first bufferSwitch — callbacks are flowing");
                    }
                    let code = shared.fault();
                    if !said_fault && code != super::fault::NONE {
                        said_fault = true;
                        crate::mlog!("[asio-dsd] integrity fault: {}", super::fault::describe(code));
                    }
                    std::thread::park_timeout(Duration::from_millis(100));
                }
                crate::mlog!(
                    "[asio-dsd] \"{name}\": stopping, {} dropout(s), {} lock miss(es)",
                    shared.underruns.load(Ordering::Relaxed),
                    shared.lock_misses.load(Ordering::Relaxed),
                );

                // Torn down by `_started` going out of scope here — one
                // teardown, in one order, on every path including a panic.
            }
        }
    }
}

/// A COM apartment held for the life of a scope.
///
/// The uninitialise was written out at each of the host thread's four exits
/// and at none of its unwind paths, so a panic left the apartment initialised
/// on a thread that was about to end — and, worse, the ordering was by hand:
/// nothing guaranteed that the driver had been released first. As a guard it
/// is dropped after everything declared after it, which is where the drivers
/// live.
struct ComApartment(bool);

impl ComApartment {
    fn enter() -> Self {
        let rc = unsafe { CoInitializeEx(null(), COINIT_APARTMENTTHREADED as u32) };
        ComApartment(rc >= 0)
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.0 {
            unsafe { CoUninitialize() };
        }
    }
}

/// Owns everything an ASIO setup allocates, and frees it in the one order
/// that is safe.
///
/// Three resources with three different lifetimes were being managed by hand:
/// the COM object, the driver's buffers, and the callback table. The table was
/// `Box::into_raw`'d and never freed at all — leaked on every open, by design,
/// with a comment saying so. Worse, every `bail!` after `createBuffers`
/// succeeded called `release` *without* `disposeBuffers`, so a driver that
/// returned a null buffer pointer, or reported a bad rate, was released while
/// it still held buffers it had handed us and a pointer to a table we were
/// about to drop.
///
/// The order matters and is not arbitrary. `disposeBuffers` is what makes the
/// driver let go of the callback table, so the table may only be freed after
/// it; and `release` drops the last COM reference, so it comes after anything
/// that still needs to call the driver.
struct AsioResources {
    driver: *mut IAsio,
    /// Deliberately a raw pointer: the driver holds it until `disposeBuffers`,
    /// and it must be writable memory rather than a read-only static, because
    /// some drivers scribble on it.
    callbacks: *mut AsioCallbacks,
    buffers_created: bool,
    /// Set once ownership has been handed on, so `Drop` does nothing.
    handed_on: bool,
}

impl AsioResources {
    fn new(driver: *mut IAsio) -> Self {
        AsioResources {
            driver,
            callbacks: null_mut(),
            buffers_created: false,
            handed_on: false,
        }
    }

    /// Give up ownership to a caller that will free it. Returns the driver and
    /// the callback table.
    fn hand_on(mut self) -> (*mut IAsio, *mut AsioCallbacks) {
        self.handed_on = true;
        (self.driver, self.callbacks)
    }
}

impl Drop for AsioResources {
    fn drop(&mut self) {
        if self.handed_on {
            return;
        }
        unsafe {
            if self.buffers_created {
                asio_call!(self.driver, dispose_buffers);
            }
            asio_call!(self.driver, release);
            if !self.callbacks.is_null() {
                drop(Box::from_raw(self.callbacks));
            }
        }
    }
}

/// Owns a driver that has been started, so that every way out of the host
/// thread stops it.
///
/// Between `ASIOStart` succeeding and the host thread returning there are four
/// exits, and each one had to remember to call `teardown_started`. A panic
/// remembered nothing: the engine claim's `Drop` unpublished the callback
/// context, so nothing crashed, but the driver was never stopped and never
/// released — the DAC kept running, the COM object leaked, and the device
/// stayed held until the process ended. Everything after a successful start
/// lives inside this.
struct StartedDriver {
    driver: *mut IAsio,
    callbacks: *mut AsioCallbacks,
    ctx: *mut CallbackCtx,
}

impl Drop for StartedDriver {
    fn drop(&mut self) {
        unsafe { teardown_started(self.driver, self.callbacks, self.ctx) };
    }
}

/// Free a driver that has been started, in the order the ASIO contract
/// requires: stop, unpublish, dispose, release, and only then free the memory
/// the callbacks were reading.
///
/// `stop` is what guarantees no further callbacks; unpublishing before the box
/// is freed is what guarantees none of them can reach freed memory even if a
/// driver ignores that guarantee.
unsafe fn teardown_started(
    driver: *mut IAsio,
    callbacks: *mut AsioCallbacks,
    ctx: *mut CallbackCtx,
) {
    unsafe {
        asio_call!(driver, stop);
        ACTIVE.store(null_mut(), Ordering::Release);
        asio_call!(driver, dispose_buffers);
        asio_call!(driver, release);
        if !ctx.is_null() {
            drop(Box::from_raw(ctx));
        }
        if !callbacks.is_null() {
            drop(Box::from_raw(callbacks));
        }
    }
}
/// Publishes the end of the backend however the host thread leaves.
///
/// `finished` releases whoever is waiting for end-of-track; without it an
/// engine that died mid-track left the UI waiting for a buffer that was never
/// coming. `reset_requested` is what `Engine::native_handle_live` reads, so a
/// dead engine also stops being a route.
struct HostDeath(Arc<AsioShared>);

impl Drop for HostDeath {
    fn drop(&mut self) {
        self.0.reset_requested.store(true, Ordering::Release);
        // A **failure**, not an ending. A host thread that died mid-track has
        // not played the track to its end, and publishing it as though it had
        // is what sent the playlist straight into the same dead engine.
        self.0.fail_now(super::fault::BACKEND_DEAD);
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
) -> Result<(*mut IAsio, *mut AsioCallbacks, Box<CallbackCtx>, SetupOk), String> {
    let mut raw: *mut c_void = null_mut();
    // ASIO quirk: a driver's class IS its interface — the CLSID doubles as
    // the IID passed to CoCreateInstance.
    let hr = unsafe { CoCreateInstance(clsid, null_mut(), CLSCTX_INPROC_SERVER, clsid, &mut raw) };
    if hr != 0 || raw.is_null() {
        return Err(format!(
            "couldn't load ASIO driver \"{name}\" (COM error {hr:#010x}) — is it installed for 64-bit hosts?"
        ));
    }
    // From here every exit goes through `owned`, which frees the driver, its
    // buffers and the callback table in the one order that is safe. Early
    // returns used to release the COM object and nothing else.
    let mut owned = AsioResources::new(raw as *mut IAsio);
    let driver = owned.driver;
    macro_rules! bail {
        ($e:expr) => {{ return Err($e); }};
    }

    // Drivers want a real window: they parent hidden notification windows to
    // it (subclassing the desktop window is not survivable everywhere). Use
    // the app's main window, falling back to the desktop only if it's not
    // known yet.
    let mut sys_handle = APP_HWND.load(Ordering::Relaxed) as *mut c_void;
    if sys_handle.is_null() {
        sys_handle = unsafe { windows_sys::Win32::UI::WindowsAndMessaging::GetDesktopWindow() }
            as *mut c_void;
    }
    crate::mlog!("[asio-dsd] \"{name}\": init with hwnd {sys_handle:?}");
    if unsafe { asio_call!(driver, init, sys_handle) } != ASIO_TRUE {
        let e = driver_error(driver, "driver init failed");
        bail!(format!("ASIO \"{name}\": {e}"));
    }
    crate::mlog!(
        "[asio-dsd] \"{name}\": loaded, init ok (driver version {})",
        unsafe { asio_call!(driver, get_driver_version) }
    );

    // Switch the driver into DSD mode BEFORE querying channel formats or
    // negotiating the rate — everything downstream depends on the mode.
    let mut fmt = AsioIoFormat {
        format_type: K_ASIO_FORMAT_DSD,
        future: [0; 508],
    };
    let can = unsafe {
        asio_call!(
            driver,
            future,
            K_ASIO_CAN_DO_IO_FORMAT,
            &mut fmt as *mut _ as *mut c_void
        )
    };
    crate::mlog!(
        "[asio-dsd] \"{name}\": kAsioCanDoIoFormat(DSD) = {can:#x} (ok={})",
        future_ok(can)
    );
    if !future_ok(can) {
        bail!(format!(
            "ASIO \"{name}\" doesn't support native DSD (kAsioCanDoIoFormat returned {can}) — DoP is this device's ceiling"
        ));
    }
    let mut fmt = AsioIoFormat {
        format_type: K_ASIO_FORMAT_DSD,
        future: [0; 508],
    };
    let set = unsafe {
        asio_call!(
            driver,
            future,
            K_ASIO_SET_IO_FORMAT,
            &mut fmt as *mut _ as *mut c_void
        )
    };
    crate::mlog!(
        "[asio-dsd] \"{name}\": kAsioSetIoFormat(DSD) = {set:#x} (ok={})",
        future_ok(set)
    );
    if !future_ok(set) {
        bail!(format!(
            "ASIO \"{name}\": switching to DSD mode failed ({set})"
        ));
    }

    // Rate: the ASIO DSD convention expresses the rate in *bits per second
    // per channel* (the DSD rate itself); some drivers instead report the
    // byte rate (÷8). Probe both, prefer the spec'd form.
    let mut rate_used = 0f64;
    // The first is the ASIO DSD specification's own form. The other two are
    // conventions observed in the field and written down nowhere: a driver
    // accepting `rate/8` tells us it accepted a number, not that it agreed
    // with us about what the number meant.
    for cand in [
        dsd_rate as f64,
        dsd_rate as f64 / 8.0,
        dsd_rate as f64 / 16.0,
    ] {
        let can = unsafe { asio_call!(driver, can_sample_rate, cand) };
        let set = if can == ASE_OK {
            unsafe { asio_call!(driver, set_sample_rate, cand) }
        } else {
            can
        };
        crate::mlog!("[asio-dsd] \"{name}\": rate {cand}: canSampleRate={can} setSampleRate={set}");
        if set == ASE_OK {
            rate_used = cand;
            break;
        }
    }
    // Recorded, because a route negotiated on a guess is not a proven route.
    let rate_spec_form = rate_used == dsd_rate as f64;
    if rate_used == 0.0 {
        let label = crate::dsd::rate_label(dsd_rate);
        bail!(format!(
            "ASIO \"{name}\": driver rejected {label} ({dsd_rate} Hz) in DSD mode"
        ));
    }

    // What the driver *is* set to, not what it accepted being told.
    //
    // `setSampleRate` returning `ASE_OK` means the call succeeded. Drivers do
    // clamp, round to a supported neighbour, or defer to a hardware clock and
    // end up somewhere else — and every one of those cases played the whole
    // track at the wrong rate with the exact claim intact, because nothing
    // ever asked.
    let mut rate_now = 0f64;
    let rc = unsafe { asio_call!(driver, get_sample_rate, &mut rate_now) };
    crate::mlog!("[asio-dsd] \"{name}\": getSampleRate rc={rc}, rate={rate_now}");
    if rc != ASE_OK {
        bail!(format!(
            "ASIO \"{name}\": the driver would not report its sample rate, so the rate it \
             is running at cannot be confirmed"
        ));
    }
    // A tolerance, because the value crosses an `f64` and a clock: a driver
    // reporting 2 822 400.000000001 is at the rate we asked for. Anything
    // beyond a part in a million is a different rate.
    if (rate_now - rate_used).abs() > rate_used * 1e-6 {
        bail!(format!(
            "ASIO \"{name}\": asked for {rate_used} Hz in DSD mode and the driver is running \
             at {rate_now} Hz — refusing rather than sending {rate_used} Hz of data to a \
             {rate_now} Hz clock"
        ));
    }

    // And that it is still in DSD mode. `kAsioSetIoFormat` succeeding is not
    // the same as the mode having stuck: a driver that fell back to PCM would
    // receive raw 1-bit packing as if it were samples.
    let mut mode = AsioIoFormat {
        format_type: -1,
        future: [0; 508],
    };
    let got = unsafe {
        asio_call!(
            driver,
            future,
            K_ASIO_GET_IO_FORMAT,
            &mut mode as *mut _ as *mut c_void
        )
    };
    crate::mlog!(
        "[asio-dsd] \"{name}\": kAsioGetIoFormat = {got:#x} (ok={}), format {}",
        future_ok(got),
        mode.format_type
    );
    if future_ok(got) && mode.format_type != K_ASIO_FORMAT_DSD {
        bail!(format!(
            "ASIO \"{name}\": accepted the DSD mode switch but reports IO format {} — \
             this build will not send DSD packing to a driver that is not in DSD mode",
            mode.format_type
        ));
    }
    // A driver that does not implement the getter is not evidence either way.
    // Playback continues — it accepted the switch, and almost certainly made
    // it — but the mode is *unverified*, and the claim says so rather than
    // asserting a bit-perfect DSD path nobody confirmed.
    //
    // A driver that reported a *malfunction* is a different answer entirely,
    // and it used to reach the same place: every non-success code was read as
    // "not implemented" and waved through. Sending DSD packing to a device
    // that has just said its hardware is broken is not an unverified claim,
    // it is ignoring the only piece of evidence on offer.
    let io_format_verified = match classify_io_format(got) {
        IoFormatReadback::Answered => true,
        IoFormatReadback::Absent => {
            crate::mlog!(
                "[asio-dsd] \"{name}\": driver does not implement kAsioGetIoFormat — the DSD \
                 mode cannot be read back, so this route reports exactness as unverified"
            );
            false
        }
        IoFormatReadback::Broken(rc) => {
            bail!(format!(
                "ASIO \"{name}\": kAsioGetIoFormat reported a driver error ({rc:#x}) rather \
                 than being unimplemented — refusing to send DSD to a device that has just \
                 reported a fault"
            ));
        }
    };

    // Channel sanity + sample type.
    let (mut n_in, mut n_out) = (0i32, 0i32);
    let rc = unsafe { asio_call!(driver, get_channels, &mut n_in, &mut n_out) };
    crate::mlog!("[asio-dsd] \"{name}\": getChannels rc={rc}, in={n_in}, out={n_out}");
    if rc != ASE_OK || n_out < channels as i32 {
        bail!(format!(
            "ASIO \"{name}\": needs {channels} output channels, driver has {n_out}"
        ));
    }
    // Every selected output channel, not only channel zero.
    //
    // The bit order is applied once, to the whole interleaved stream, on the
    // decode thread. That is only correct if every channel wants the same
    // order — and a driver is free to report per-channel types. Querying one
    // channel and applying its answer to all of them is an assumption that
    // fails silently, as one inverted channel, which is not a sound anyone
    // recognises as a bug.
    //
    // `NER8` is refused rather than accepted. It is 8-bit data at one sample
    // per byte — an eighth of the data rate of the 1-bit packing this path
    // produces — and treating it as MSB1, which is what the code used to do,
    // sends a driver eight times the samples it asked for.
    let mut lsb_first: Option<bool> = None;
    for c in 0..channels as i32 {
        let mut info = AsioChannelInfo {
            channel: c,
            is_input: ASIO_FALSE,
            is_active: ASIO_FALSE,
            channel_group: 0,
            sample_type: 0,
            name: [0; 32],
        };
        let rc = unsafe { asio_call!(driver, get_channel_info, &mut info) };
        crate::mlog!(
            "[asio-dsd] \"{name}\": getChannelInfo ch {c} rc={rc}, sample type {}",
            info.sample_type
        );
        if rc != ASE_OK {
            bail!(format!(
                "ASIO \"{name}\": channel info query failed for channel {c}"
            ));
        }
        let want = match info.sample_type {
            ASIOST_DSD_INT8_MSB1 => false,
            ASIOST_DSD_INT8_LSB1 => true,
            ASIOST_DSD_INT8_NER8 => bail!(format!(
                "ASIO \"{name}\": channel {c} reports DSD Int8 NER8 (one sample per byte), \
                 which this build does not produce — only the 1-bit packings MSB1 and LSB1 \
                 are supported"
            )),
            other => bail!(format!(
                "ASIO \"{name}\": channel {c} has unexpected sample type {other} in DSD mode \
                 (expected DSD Int8 MSB1 or LSB1)"
            )),
        };
        match lsb_first {
            None => lsb_first = Some(want),
            Some(first) if first == want => {}
            Some(_) => bail!(format!(
                "ASIO \"{name}\": channels report mixed DSD bit orders — this build applies one \
                 order to the whole interleaved stream and will not guess which channels to \
                 invert"
            )),
        }
    }
    let lsb_first = lsb_first.unwrap_or(false);

    // Buffers at the driver's preferred size.
    let (mut min, mut max, mut preferred, mut gran) = (0i32, 0i32, 0i32, 0i32);
    let rc = unsafe {
        asio_call!(
            driver,
            get_buffer_size,
            &mut min,
            &mut max,
            &mut preferred,
            &mut gran
        )
    };
    crate::mlog!(
        "[asio-dsd] \"{name}\": getBufferSize rc={rc}, min={min} max={max} preferred={preferred} gran={gran}"
    );
    if rc != ASE_OK || preferred <= 0 {
        bail!(format!("ASIO \"{name}\": buffer size query failed"));
    }
    // In DSD mode the size is in 1-bit samples; the Int8 buffers hold s/8 bytes.
    if preferred % 8 != 0 {
        bail!(format!(
            "ASIO \"{name}\": DSD buffer size {preferred} isn't byte-aligned"
        ));
    }
    let buffer_bytes = (preferred / 8) as usize;
    let mut buffer_infos: Vec<AsioBufferInfo> = (0..channels as i32)
        .map(|c| AsioBufferInfo {
            is_input: ASIO_FALSE,
            channel_num: c,
            buffers: [null_mut(); 2],
        })
        .collect();
    // The callbacks struct lives in writable memory and is deliberately
    // leaked: the driver holds this pointer until disposeBuffers, and a
    // driver that scribbles on it must not fault on a read-only static.
    // 32 bytes per stream open — negligible.
    owned.callbacks = Box::into_raw(Box::new(AsioCallbacks {
        buffer_switch: cb_buffer_switch,
        sample_rate_did_change: cb_sample_rate_did_change,
        asio_message: cb_asio_message,
        buffer_switch_time_info: cb_buffer_switch_time_info,
    }));
    let callbacks = owned.callbacks;
    crate::mlog!(
        "[asio-dsd] \"{name}\": calling createBuffers(numChannels={}, bufferSize={preferred})…",
        channels,
    );
    let rc = unsafe {
        asio_call!(
            driver,
            create_buffers,
            buffer_infos.as_mut_ptr(),
            channels as i32,
            preferred,
            callbacks
        )
    };
    crate::mlog!(
        "[asio-dsd] \"{name}\": createBuffers rc={rc}, ch0 buffers = {:?}/{:?}",
        buffer_infos[0].buffers[0],
        buffer_infos[0].buffers[1],
    );
    if rc != ASE_OK {
        let e = driver_error(driver, "createBuffers failed");
        bail!(format!("ASIO \"{name}\": {e}"));
    }
    // The driver now holds our buffers and our callback table. Every exit from
    // here on has to dispose before releasing, which is what the owner does.
    owned.buffers_created = true;

    let supports_output_ready = unsafe { asio_call!(driver, output_ready) } == ASE_OK;
    crate::mlog!("[asio-dsd] \"{name}\": outputReady supported = {supports_output_ready}");

    // Validate every buffer pointer the driver handed back before anything is
    // written through one. A null in the middle of the list used to be skipped
    // in silence, which meant one channel played nothing while the stream
    // reported success.
    for (c, info) in buffer_infos.iter().enumerate() {
        for (half, p) in info.buffers.iter().enumerate() {
            if p.is_null() {
                bail!(format!(
                    "ASIO \"{name}\": driver returned a null buffer for output channel {c}, \
                     half {half}"
                ));
            }
        }
    }

    let ctx = Box::new(CallbackCtx {
        shared: Arc::clone(shared),
        driver,
        buffer_infos,
        buffer_bytes,
        channels: channels as usize,
        idle: idle_byte(lsb_first),
        scratch: vec![0u8; buffer_bytes * channels as usize],
        supports_output_ready,
    });

    // Prefill both halves with correct DSD idle before the engine starts, and
    // without consuming a single source frame.
    //
    // The stream used to start on whatever the driver left in its buffers. A
    // DSD DAC handed uninitialised memory at startup gets a burst of noise
    // before the first audio, which is the click every DSD track began with.
    for info in &ctx.buffer_infos {
        for half in 0..2 {
            let dest = info.buffers[half] as *mut u8;
            // SAFETY: the pointer was validated above, and the driver owns
            // `buffer_bytes` bytes behind it for the life of the buffers.
            unsafe { std::ptr::write_bytes(dest, ctx.idle, buffer_bytes) };
        }
    }
    crate::mlog!(
        "[asio-dsd] \"{name}\": DSD mode on, rate accepted as {rate_used} \
         ({}), bit order {} on all {channels} channel(s),          buffer {preferred} samples = {buffer_bytes} bytes/ch",
        crate::dsd::rate_label(dsd_rate),
        if lsb_first { "LSB-first" } else { "MSB-first" },
    );
    let (driver, callbacks) = owned.hand_on();
    Ok((
        driver,
        callbacks,
        ctx,
        SetupOk {
            lsb_first,
            // Byte-frames, not the 1-bit sample count `ASIOGetBufferSize`
            // reports: the callback fills `buffer_bytes` per channel, and the
            // drain counts what the callback fills. Storing the sample count
            // made the drain wait for sixteen buffers rather than two — up to
            // a fifth of a second of the next track swallowed on a large one.
            buffer_frames: buffer_bytes as u32,
            io_format_verified,
            rate_spec_form,
        },
    ))
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The ASIO callback is exercised against a fake driver context: real
// `AsioBufferInfo` records pointing at owned buffers, a null driver pointer
// (never dereferenced, because `supports_output_ready` is false), and a real
// frame ring. That is the whole realtime path apart from the driver call.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitperfect::fault;
    use crate::bitperfect::frame_ring::channel_ring;

    /// A fake double buffer plus the context that points into it.
    /// Serialises every test that touches `ACTIVE` or `ENGAGED`.
    ///
    /// Both are process-wide, and the test harness runs tests in parallel, so
    /// they were being written by several at once. The failure that made this
    /// necessary: `EngineClaim::drop` nulls `ACTIVE` — correctly, it is what
    /// stops a callback reaching a context belonging to a session that is
    /// gone — so the claim test tore down the pointer another test had just
    /// published, and that test's callback found null and did nothing.
    /// `driver_events_fault_the_session_and_never_log` failed on
    /// `ev_rate_changed` roughly one full-suite run in several hundred, and
    /// passed every time it was run alone.
    ///
    /// Worse than the flake: a `Fake` lives on the stack of the test that
    /// built it, so publishing its address without holding this is publishing
    /// a pointer another thread may dereference after the frame has gone.
    static ASIO_GLOBALS: Mutex<()> = Mutex::new(());

    /// Exclusive use of `ACTIVE` and `ENGAGED` for the life of the value.
    ///
    /// Declare it *after* whatever it will publish, so it is dropped first and
    /// `ACTIVE` is nulled while the target is still alive.
    struct Globals(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl Globals {
        fn take() -> Self {
            // The data is `()`, so poison carries no information; a test that
            // panicked while holding this must not deadlock every later one.
            Globals(ASIO_GLOBALS.lock().unwrap_or_else(|e| e.into_inner()))
        }

        /// Publish a callback context for the life of this guard.
        fn publish(&self, ctx: *mut CallbackCtx) {
            ACTIVE.store(ctx, Ordering::Release);
        }
    }

    impl Drop for Globals {
        fn drop(&mut self) {
            ACTIVE.store(null_mut(), Ordering::Release);
        }
    }

    struct Fake {
        ctx: CallbackCtx,
        /// `[channel][half]` — owned, so the raw pointers in `ctx` stay valid.
        bufs: Vec<[Vec<u8>; 2]>,
    }

    impl Fake {
        /// What the driver would be playing out of one half, for one channel.
        fn half(&self, channel: usize, index: usize) -> &[u8] {
            &self.bufs[channel][index]
        }

        fn new(channels: usize, buffer_bytes: usize, shared: Arc<AsioShared>) -> Box<Self> {
            let mut bufs: Vec<[Vec<u8>; 2]> = (0..channels)
                .map(|_| [vec![0u8; buffer_bytes], vec![0u8; buffer_bytes]])
                .collect();
            let buffer_infos: Vec<AsioBufferInfo> = bufs
                .iter_mut()
                .enumerate()
                .map(|(c, halves)| {
                    let (a, b) = halves.split_at_mut(1);
                    AsioBufferInfo {
                        is_input: ASIO_FALSE,
                        channel_num: c as i32,
                        buffers: [
                            a[0].as_mut_ptr() as *mut c_void,
                            b[0].as_mut_ptr() as *mut c_void,
                        ],
                    }
                })
                .collect();
            Box::new(Fake {
                ctx: CallbackCtx {
                    shared,
                    driver: null_mut(),
                    idle: idle_byte(false),
                    buffer_infos,
                    buffer_bytes,
                    channels,
                    scratch: vec![0u8; buffer_bytes * channels],
                    supports_output_ready: false, // the only use of `driver`
                },
                bufs,
            })
        }

        fn switch(&mut self, half: i32) {
            unsafe { fill_buffers(&mut self.ctx, half) };
        }

        fn out(&self, channel: usize, half: usize) -> &[u8] {
            &self.bufs[channel][half]
        }

        /// Take one channel's output buffer away, as a driver that has begun
        /// tearing down its engine does.
        fn drop_channel(&mut self, channel: usize) {
            self.ctx.buffer_infos[channel].buffers = [std::ptr::null_mut(); 2];
        }
    }

    /// The public adapter answers with both records when both were stored.
    ///
    /// The schedule the old design could not survive, forced through the type
    /// a caller actually holds. A reader is parked between its two loads; a
    /// recoverable record and then a terminal one are stored, each parked
    /// before its publication window closes; the reader resumes and finishes
    /// its loads.
    ///
    /// Under the old protocol both stores were already visible and neither had
    /// moved the counter, so the reader saw the terminal record beside an
    /// empty recoverable slot and called it a snapshot. Under this one the
    /// window is open from before the first store, so the reader knows it is
    /// not looking at a quiet instant and folds what every attempt saw.
    #[test]
    fn asio_evidence_holds_together_across_two_writers() {
        use crate::bitperfect::fault;
        use crate::bitperfect::window::{arm, Site};
        use std::sync::Barrier;

        let sh = shared();
        let at = sh.begin_session();
        let id = sh.hook_id;

        let reader_parked = Arc::new(Barrier::new(2));
        let writers_done = Arc::new(Barrier::new(2));
        {
            let a = Arc::clone(&reader_parked);
            let b = Arc::clone(&writers_done);
            // After the recoverable load and before the terminal one.
            arm(id, Site::Evidence(2), move || {
                a.wait();
                b.wait();
            });
        }

        let for_reader = Arc::clone(&sh);
        let reading = std::thread::spawn(move || {
            AsioDsdStream::for_evidence_test(for_reader).evidence_at(at)
        });

        // The reader is now between its loads, having seen no recoverable
        // record. Both writers store while it waits there.
        reader_parked.wait();
        sh.revoke(at, fault::UNDERRUN);
        sh.fail(at, fault::BACKEND_WRITE);
        writers_done.wait();

        let ev = reading.join().expect("the read returns");
        assert_eq!(ev.generation, at);
        assert_eq!(
            ev.fatal,
            fault::BACKEND_WRITE,
            "the reason the session ended"
        );
        assert_eq!(
            ev.revoked,
            fault::UNDERRUN,
            "and the dropout stored before it — an account with the second and \
             not the first is one that never held"
        );
    }

    /// A record published *after* the fatal one still reaches the reader.
    ///
    /// The previous design's final read took the terminal record first, on the
    /// argument that nothing can be published against a generation after it.
    /// That is not true here: a callback still unwinding, and a decode thread
    /// still finishing, both revoke after the session has been failed.
    #[test]
    fn asio_evidence_published_after_a_fatal_still_arrives() {
        use crate::bitperfect::fault;
        let sh = shared();
        let at = sh.begin_session();

        sh.fail(at, fault::BACKEND_WRITE);
        sh.revoke(at, fault::UNDERRUN);

        let stream = AsioDsdStream::for_evidence_test(Arc::clone(&sh));
        let ev = stream.evidence_at(at);
        assert_eq!(ev.fatal, fault::BACKEND_WRITE);
        assert_eq!(
            ev.revoked,
            fault::UNDERRUN,
            "a session that has already failed can still lose its claim, and \
             the read has to say so"
        );
        assert!(ev.settled, "nothing was in flight, so this is a quiet read");
    }

    /// A callback already past its `paused` check may not play the session a
    /// pause installed behind it.
    ///
    /// The PCM schedule, on this backend: the callback reads `paused = false`,
    /// parks before the session slot, the UI thread pauses and installs a new
    /// session, and only then is the callback released. `take_session_frames`
    /// is the whole of the callback's consumption decision, and `fill_channel`
    /// is the whole of what reaches the driver's buffers — so what the DAC
    /// would have been handed is checked here rather than reasoned about.
    #[test]
    fn a_callback_past_its_check_cannot_play_a_session_paused_behind_it() {
        const CH: usize = 2;
        const FRAMES: usize = 64;
        let idle = idle_byte(false);

        let sh = shared();
        let inside = Arc::new(std::sync::Barrier::new(2));
        let go = Arc::new(std::sync::Barrier::new(2));
        crate::bitperfect::window::arm(
            sh.hook_id,
            crate::bitperfect::window::Site::Render,
            {
                let inside = Arc::clone(&inside);
                let go = Arc::clone(&go);
                move || {
                    inside.wait();
                    go.wait();
                }
            },
        );

        let callback = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || {
                let mut scratch = vec![0u8; FRAMES * CH];
                let out = take_session_frames(&sh, &mut scratch, FRAMES);
                (out, scratch)
            })
        };

        inside.wait();

        // The payload nobody may hear, installed behind the parked callback.
        let payload = frames(FRAMES * 2, CH);
        let (mut prod, cons) = crate::bitperfect::frame_ring::channel_ring::<u8>(CH, 1024);
        prod.push_frames(&payload).unwrap();
        sh.paused.store(true, Ordering::Release);
        install(&sh, cons, false, false);

        go.wait();
        let ((got, session_at, began_drain), scratch) = callback.join().unwrap();

        assert_eq!(got, 0, "a paused session gave up byte-frames");
        assert_eq!(session_at, 0, "and it was not accounted against a session");
        assert!(!began_drain);
        assert!(!sh.draining.load(Ordering::Relaxed), "the drain began");
        assert_eq!(sh.frames_played(), 0, "the position moved");
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.lock_misses.load(Ordering::Relaxed), 0);
        assert_eq!(sh.revoked(), super::super::fault::NONE);
        assert_eq!(sh.fault(), super::super::fault::NONE);

        // What the driver would have been handed, per channel: DSD idle, on
        // every byte of every channel.
        for c in 0..CH {
            let mut dest = vec![0xAAu8; FRAMES];
            fill_channel(&mut dest, &scratch, got, CH, c, idle);
            assert!(
                dest.iter().all(|&b| b == idle),
                "channel {c} was handed payload instead of DSD idle"
            );
        }

        // The ring still holds every byte-frame of it.
        let mut back = vec![0u8; FRAMES * CH];
        let n = sh
            .session
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .cons
            .pop_frames(&mut back, FRAMES);
        assert_eq!(n, FRAMES);
        assert_eq!(back, payload[..FRAMES * CH], "the ring was consumed");
    }

    /// A `try_lock` that missed because of the pause is not a dropout.
    ///
    /// The UI thread is still holding the slot when the callback is released,
    /// which is what installing a session looks like from inside the driver.
    #[test]
    fn an_asio_paused_handoff_is_not_a_lock_miss() {
        let sh = shared();
        let inside = Arc::new(std::sync::Barrier::new(2));
        let go = Arc::new(std::sync::Barrier::new(2));
        crate::bitperfect::window::arm(
            sh.hook_id,
            crate::bitperfect::window::Site::Render,
            {
                let inside = Arc::clone(&inside);
                let go = Arc::clone(&go);
                move || {
                    inside.wait();
                    go.wait();
                }
            },
        );

        let callback = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || {
                let mut scratch = vec![0u8; 128];
                take_session_frames(&sh, &mut scratch, 64)
            })
        };

        inside.wait();
        sh.paused.store(true, Ordering::Release);
        // Held until the call has returned, so the miss is forced rather than
        // raced for.
        let guard = sh.session.lock().unwrap();
        go.wait();
        let (got, _, _) = callback.join().unwrap();
        drop(guard);

        assert_eq!(got, 0);
        assert_eq!(
            sh.lock_misses.load(Ordering::Relaxed),
            0,
            "the UI thread holding the slot to install a paused session is the \
             hand-off working"
        );
        assert_eq!(sh.revoked(), super::super::fault::NONE);
        assert_eq!(sh.fault(), super::super::fault::NONE);
    }

    pub(super) fn shared() -> Arc<AsioShared> {
        shared_opened_ago(Duration::ZERO)
    }

    /// A stream that opened `ago` in the past, so the wall-clock drain bound
    /// can be reached without a test that sleeps for it.
    fn shared_opened_ago(ago: Duration) -> Arc<AsioShared> {
        Arc::new(AsioShared {
            opened_at: Instant::now() - ago,
            session: Mutex::new(None),
            paused: AtomicBool::new(false),
            // Zero, like the production constructor: a test `Shared` that
            // started on generation one could collide with the allocator's
            // first hand-out, and then `begin_session` returned the number the
            // fixture was already using.
            play: AtomicU64::new(crate::bitperfect::stamped_pack(
                0,
                crate::bitperfect::done_code::RUNNING,
            )),
            draining: AtomicBool::new(false),
            drain_frames: AtomicU64::new(0),
            buffer_frames: AtomicU64::new(0),
            drain_gen: AtomicU64::new(0),
            drain_started_ms: AtomicU64::new(0),
            drain_callbacks: AtomicU64::new(0),
            frames_played: AtomicU64::new(crate::bitperfect::counted::pack(0, 0)),
            reset_requested: AtomicBool::new(false),
            fault_code: AtomicU64::new(0),
            revoked: AtomicU64::new(0),
            evidence: crate::bitperfect::PublishWindow::new(),
            #[cfg(test)]
            hook_id: crate::bitperfect::window::next_id(),
            underruns: AtomicU64::new(0),
            lock_misses: AtomicU64::new(0),
            priming: AtomicBool::new(false),
            first_switch: AtomicBool::new(false),
            ev_time_info: AtomicBool::new(false),
            ev_rate_changed: AtomicBool::new(false),
            ev_reset: AtomicBool::new(false),
            ev_resync: AtomicBool::new(false),
            ev_overload: AtomicU64::new(0),
            ev_buffer_size_changed: AtomicBool::new(false),
            ev_last_selector: std::sync::atomic::AtomicI32::new(0),
        })
    }

    /// Install a session. `done` means the feeder has stopped; `eof` means it
    /// stopped by reaching the end of the file, which is the only combination
    /// that may advance a playlist.
    fn install(
        sh: &AsioShared,
        cons: crate::bitperfect::frame_ring::FrameConsumer<u8>,
        done: bool,
        eof: bool,
    ) {
        *sh.session.lock().unwrap() = Some(AsioSession {
            generation: sh.generation(),
            cons,
            decode_done: Arc::new(AtomicBool::new(done)),
            decode_eof: Arc::new(AtomicBool::new(eof)),
            stop: Arc::new(AtomicBool::new(false)),
        });
    }

    /// Interleaved byte-frames where the value identifies both the frame and
    /// the channel it belongs to, so a rotation is visible rather than merely
    /// different.
    fn frames(count: usize, channels: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(count * channels);
        for f in 0..count {
            for c in 0..channels {
                v.push(
                    1u8.wrapping_add((f as u8).wrapping_mul(16))
                        .wrapping_add(c as u8),
                );
            }
        }
        v
    }

    /// Complete frames deinterleave into their own channel buffers, and the
    /// remainder of the callback buffer is DSD silence rather than zeros.
    ///
    /// Would have caught the 1.4.2 drain, which popped bytes until the ring ran
    /// dry — including mid-frame — and left a byte belonging to channel *n* at
    /// the head, where the next callback read it as channel 0.
    #[test]
    fn complete_frames_deinterleave_and_the_rest_is_dsd_silence() {
        for channels in 1..=8usize {
            let buffer_bytes = 16usize;
            let sh = shared();
            let (mut prod, cons) = channel_ring::<u8>(channels, 64);
            let src = frames(10, channels); // fewer frames than the buffer holds
            assert_eq!(prod.push_frames(&src).unwrap(), 10);
            install(&sh, cons, false, false);

            let mut fake = Fake::new(channels, buffer_bytes, Arc::clone(&sh));
            fake.switch(0);

            for c in 0..channels {
                let got = fake.out(c, 0);
                for f in 0..10 {
                    assert_eq!(
                        got[f],
                        src[f * channels + c],
                        "{channels}ch: channel {c} frame {f} came from the wrong lane"
                    );
                }
                for (f, &b) in got.iter().enumerate().skip(10) {
                    assert_eq!(
                        b, DSD_SILENCE,
                        "{channels}ch: padding at {f} must be the DSD idle pattern"
                    );
                }
            }
            assert_eq!(sh.frames_played(), 10);
        }
    }

    /// A short availability never consumes a partial frame: whatever is left
    /// stays whole and arrives, in order, on the next callback.
    #[test]
    fn a_partial_frame_is_never_consumed() {
        for channels in 2..=8usize {
            let buffer_bytes = 4usize;
            let sh = shared();
            let (mut prod, cons) = channel_ring::<u8>(channels, 64);
            let src = frames(10, channels);
            prod.push_frames(&src).unwrap();
            install(&sh, cons, false, false);

            let mut fake = Fake::new(channels, buffer_bytes, Arc::clone(&sh));
            let mut seen: Vec<Vec<u8>> = vec![Vec::new(); channels];
            for i in 0..3usize {
                fake.switch((i & 1) as i32);
                for (c, lane) in seen.iter_mut().enumerate() {
                    lane.extend_from_slice(fake.out(c, i & 1));
                }
            }
            // Three callbacks of four frames each, from ten available: the
            // first eight arrive as audio, the rest is silence, and nothing is
            // rotated between channels.
            for c in 0..channels {
                for f in 0..10 {
                    assert_eq!(
                        seen[c][f],
                        src[f * channels + c],
                        "{channels}ch: channel {c} lost alignment at frame {f}"
                    );
                }
            }
        }
    }

    /// A pause consumes no source at all. The DAC gets DSD silence and the
    /// audio is still there when playback resumes.
    #[test]
    fn a_pause_consumes_no_source_frames() {
        let channels = 2usize;
        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(channels, 64);
        let src = frames(8, channels);
        prod.push_frames(&src).unwrap();
        install(&sh, cons, false, false);

        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));
        sh.paused.store(true, Ordering::Relaxed);
        fake.switch(0);
        for c in 0..channels {
            assert!(
                fake.out(c, 0).iter().all(|&b| b == DSD_SILENCE),
                "a paused callback must emit DSD silence"
            );
        }
        assert_eq!(sh.frames_played(), 0);
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            0,
            "a pause is not a dropout"
        );

        sh.paused.store(false, Ordering::Relaxed);
        fake.switch(1);
        for c in 0..channels {
            for f in 0..8 {
                assert_eq!(
                    fake.out(c, 1)[f],
                    src[f * channels + c],
                    "the pause ate the payload"
                );
            }
        }
    }

    /// Starvation mid-track is a dropout and ends the session's claim; the same
    /// shortfall while priming, or at the end of the track, is not.
    #[test]
    fn starvation_is_a_fault_but_a_track_ending_is_not() {
        let channels = 2usize;

        // Mid-track starvation.
        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(channels, 64);
        prod.push_frames(&frames(2, channels)).unwrap();
        install(&sh, cons, false, false);
        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));
        fake.switch(0);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 1);
        assert_eq!(sh.revoked(), fault::UNDERRUN);

        // The same shortfall while priming after a seek.
        let sh = shared();
        sh.priming.store(true, Ordering::Relaxed);
        let (mut prod, cons) = channel_ring::<u8>(channels, 64);
        prod.push_frames(&frames(2, channels)).unwrap();
        install(&sh, cons, false, false);
        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));
        fake.switch(0);
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            0,
            "a seek gap is not a dropout"
        );
        assert_eq!(sh.fault(), fault::NONE);

        // End of track: the session closes, and that is not a fault either.
        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(channels, 64);
        prod.push_frames(&frames(2, channels)).unwrap();
        install(&sh, cons, true, true);
        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));
        fake.switch(0);
        assert_eq!(sh.underruns.load(Ordering::Relaxed), 0);
        assert_eq!(sh.fault(), fault::NONE);
        assert!(sh.session.lock().unwrap().is_none());

        // ...but it is not *finished* yet. The driver still holds what it was
        // last given, and on DSD that silence is a stream the DAC is locked
        // to: ending the track here starts the next one over the tail of this
        // one.
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "the drain has not run yet"
        );
        assert!(sh.draining.load(Ordering::Acquire));

        // Once the driver has asked for enough to have played it out, it is.
        for _ in 0..300 {
            fake.switch(0);
        }
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::CleanEof,
            "the feeder reached the end and the driver drained"
        );
        assert!(sh.completion().may_advance());
        assert_eq!(sh.fault(), fault::NONE, "a drain is not a dropout");
        assert_eq!(
            sh.underruns.load(Ordering::Relaxed),
            0,
            "the frames a drain asks for are not starvation"
        );
    }

    /// A callback still holding the outgoing session cannot end the incoming
    /// track.
    ///
    /// The schedule: the feeder for track A stops, the callback picks up the
    /// session and reads `decode_done`, and track B is installed before the
    /// callback reaches its conclusion. Every conclusion drawn from "what is
    /// the stream playing now" landed on B — so B failed, or finished, before
    /// its first byte, on the strength of what A's feeder did.
    ///
    /// A session speaks only for itself, which is why it carries its own
    /// generation and why the conclusions below are stamped with that rather
    /// than with the clock.
    #[test]
    fn an_outgoing_session_cannot_end_the_track_that_replaced_it() {
        use std::sync::Barrier;

        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(2, 64);
        prod.push_frames(&frames(1, 2)).unwrap();
        // A feeder that stopped without reaching the end of its file: the one
        // combination that fails rather than draining.
        install(&sh, cons, true, false);
        let outgoing = sh.session.lock().unwrap().as_ref().unwrap().generation;

        let read = Arc::new(Barrier::new(2));
        let conclude = Arc::new(Barrier::new(2));
        let callback = {
            let sh = Arc::clone(&sh);
            let read = Arc::clone(&read);
            let conclude = Arc::clone(&conclude);
            std::thread::spawn(move || {
                // Everything the callback established about the session it was
                // holding, before it was descheduled.
                let at = outgoing;
                read.wait();
                conclude.wait();
                // ...and the conclusion, arrived at late.
                sh.fail(at, fault::SESSION_THREAD_FAILED);
            })
        };

        read.wait();
        // Track B arrives while the callback is mid-thought.
        let (_p, c) = channel_ring::<u8>(2, 64);
        let incoming = {
            let mut g = sh.session.lock().unwrap();
            let generation = sh.begin_session();
            *g = Some(AsioSession {
                generation,
                cons: c,
                decode_done: Arc::new(AtomicBool::new(false)),
                decode_eof: Arc::new(AtomicBool::new(false)),
                stop: Arc::new(AtomicBool::new(false)),
            });
            generation
        };
        conclude.wait();
        callback.join().unwrap();

        assert_ne!(incoming, outgoing);
        assert_eq!(sh.generation(), incoming);
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "track B failed on the strength of track A's feeder"
        );
        assert_eq!(sh.fault(), fault::NONE);

        // B is still a usable track afterwards.
        sh.begin_drain(sh.generation());
        sh.buffer_frames.store(64, Ordering::Relaxed);
        for _ in 0..64 {
            sh.advance_drain(64);
        }
        assert!(sh.completion().may_advance());
    }

    /// The drain waits for the buffer the driver holds, in the driver's units.
    ///
    /// `ASIOGetBufferSize` answers in 1-bit DSD samples; the callback fills
    /// bytes, eight samples to each. Storing the sample count and comparing it
    /// against byte-frames made the drain wait for sixteen buffers instead of
    /// two — on a large exclusive buffer, a fifth of a second of the next
    /// track swallowed at every boundary.
    #[test]
    fn the_drain_waits_in_the_units_the_callback_counts() {
        const BUFFER_BYTES: u64 = 4096;
        let sh = shared();
        sh.buffer_frames.store(BUFFER_BYTES, Ordering::Relaxed);
        let at = sh.begin_session();
        sh.begin_drain(sh.generation());

        // One buffer short of the target: still playing out.
        let mut given = 0u64;
        while given + BUFFER_BYTES < BUFFER_BYTES * 2 {
            sh.advance_drain(BUFFER_BYTES as usize);
            given += BUFFER_BYTES;
        }
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "the driver still holds audio"
        );

        // The buffer that completes it.
        sh.advance_drain(BUFFER_BYTES as usize);
        assert_eq!(sh.completion(), crate::bitperfect::Completion::CleanEof);
        let _ = at;

        // And the bound is a bound, not a suggestion: `.max(2048)` covers a
        // driver that reported nothing at all.
        let sh = shared();
        sh.buffer_frames.store(0, Ordering::Relaxed);
        sh.begin_session();
        sh.begin_drain(sh.generation());
        for _ in 0..8 {
            sh.advance_drain(256);
        }
        assert_eq!(sh.completion(), crate::bitperfect::Completion::CleanEof);
    }

    /// A driver that stops calling back ends the track rather than freezing it.
    ///
    /// A drain is advanced by callbacks. A driver that goes quiet mid-drain
    /// gives it nothing to advance with, so the track sat at almost-finished
    /// for the life of the process: never advancing, never failing, never
    /// explaining itself. The wall clock is checked from the UI thread, which
    /// is the one that is still running.
    #[test]
    fn a_driver_that_goes_quiet_mid_drain_is_a_failure_not_a_freeze() {
        // A stream that has been open for an hour, so the drain can be given a
        // start time in the past without a test that sleeps for five seconds.
        let sh = shared_opened_ago(Duration::from_secs(3600));
        sh.buffer_frames.store(4096, Ordering::Relaxed);
        sh.begin_session();
        sh.begin_drain(sh.generation());
        sh.advance_drain(64); // a token callback, then silence

        assert!(!sh.drain_expired(), "not yet — a drain is allowed to take time");
        assert_eq!(sh.completion(), crate::bitperfect::Completion::Running);

        // The drain began a limit and a second ago; nothing has arrived since.
        let now = sh.opened_at.elapsed().as_millis() as u64;
        sh.drain_started_ms
            .store(now - (DRAIN_LIMIT.as_millis() as u64 + 1_000), Ordering::Relaxed);
        assert!(sh.drain_expired());

        sh.expire_drain();
        assert!(
            !sh.completion().may_advance(),
            "a driver that stopped did not play the track to its end"
        );
        assert_eq!(sh.completion().failure(), Some(fault::BACKEND_DEAD));
    }

    /// The clock and the slot change together.
    ///
    /// The schedule: a callback is inside the session slot when the next track
    /// is installed. Bumping the generation before taking the lock left the
    /// outgoing session sitting there with the incoming generation current,
    /// and reset the drain state out from under a session still using it.
    #[test]
    fn installing_a_session_and_turning_the_clock_are_one_step() {
        use std::sync::Barrier;

        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(2, 64);
        prod.push_frames(&frames(2, 2)).unwrap();
        install(&sh, cons, false, false);
        let first = sh.session.lock().unwrap().as_ref().unwrap().generation;
        assert_eq!(first, sh.generation(), "a session and the clock agree");

        // A reader holding the slot while the installer wants it.
        let holding = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let reader = {
            let sh = Arc::clone(&sh);
            let holding = Arc::clone(&holding);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                let guard = sh.session.lock().unwrap();
                let seen = guard.as_ref().unwrap().generation;
                holding.wait();
                release.wait();
                drop(guard);
                seen
            })
        };

        holding.wait();
        // The installer is about to block on the slot. Whatever it does, the
        // reader must never see a session whose generation is not its own.
        let installer = {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || {
                let (_p, c) = channel_ring::<u8>(2, 64);
                let mut g = sh.session.lock().unwrap();
                let generation = sh.begin_session();
                *g = Some(AsioSession {
                    generation,
                    cons: c,
                    decode_done: Arc::new(AtomicBool::new(false)),
                    decode_eof: Arc::new(AtomicBool::new(false)),
                    stop: Arc::new(AtomicBool::new(false)),
                });
                generation
            })
        };
        release.wait();

        let seen = reader.join().unwrap();
        let second = installer.join().unwrap();
        assert_eq!(seen, first, "the reader saw the session it was holding");
        assert_eq!(second, sh.generation());
        assert_ne!(second, first);
        assert_eq!(
            sh.session.lock().unwrap().as_ref().unwrap().generation,
            sh.generation(),
            "and the slot still agrees with the clock afterwards"
        );
    }

    /// A started driver is stopped and released on every way out.
    ///
    /// Four exits and one unwind path, each of which used to have to remember
    /// the teardown by hand. The fake driver's log is the evidence: `stop`
    /// before `disposeBuffers` before `release`, exactly once each, whatever
    /// ended the scope.
    #[test]
    fn every_exit_from_a_started_driver_tears_it_down_in_order() {
        // `StartedDriver`'s teardown nulls `ACTIVE`, which is asserted below,
        // so this must not run beside a test that has published one.
        let _globals = Globals::take();
        // What happens inside the scope, for each way out of it.
        type Body = Box<dyn Fn()>;
        let cases: Vec<(&str, Body)> = vec![
            ("normal stop", Box::new(|| {})),
            (
                "start failure",
                Box::new(|| { /* the caller returns immediately */ }),
            ),
            (
                "callback timeout",
                Box::new(|| { /* the caller returns after the deadline */ }),
            ),
        ];

        for (what, body) in cases {
            let log = Arc::new(DriverLog::default());
            let (ptr, keep) = fake_driver(&log);
            let callbacks = callbacks_box();
            {
                let _started = StartedDriver { driver: ptr, callbacks, ctx: null_mut() };
                body();
            }
            assert_teardown_order(&log, what);
            assert!(
                ACTIVE.load(Ordering::Acquire).is_null(),
                "{what}: the callback context must be unpublished"
            );
            drop(keep);
        }

        // And the one that no amount of remembering covered.
        let log = Arc::new(DriverLog::default());
        let (ptr, keep) = fake_driver(&log);
        let callbacks = callbacks_box();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _started = StartedDriver { driver: ptr, callbacks, ctx: null_mut() };
            panic!("injected unwind");
        }));
        assert!(outcome.is_err(), "the panic must actually happen");
        assert_teardown_order(&log, "injected unwind");
        drop(keep);
    }

    /// The ASIO contract's order: no further callbacks, then no buffers, then
    /// no driver.
    fn assert_teardown_order(log: &DriverLog, what: &str) {
        let events = log.events();
        assert_eq!(log.count("stop"), 1, "{what}: {events:?}");
        assert_eq!(log.count("dispose_buffers"), 1, "{what}: {events:?}");
        assert_eq!(log.count("release"), 1, "{what}: {events:?}");
        let stop = events.iter().position(|e| *e == "stop").unwrap();
        let dispose = events.iter().position(|e| *e == "dispose_buffers").unwrap();
        let release = events.iter().position(|e| *e == "release").unwrap();
        assert!(
            stop < dispose && dispose < release,
            "{what}: {events:?} is not stop → dispose → release"
        );
    }

    /// The track ends after the last payload has been *played*, and the last
    /// payload is a partial buffer.
    ///
    /// A track does not end on a buffer boundary. The final callback with any
    /// audio in it writes a few frames of music and pads the rest with DSD
    /// idle — and that callback used to count its own buffer as drained,
    /// crediting the drain with a buffer the driver had not begun. On a
    /// double-buffered driver that is one whole buffer early: the bitstream
    /// stops while the DAC is still sounding it.
    ///
    /// The payload here is deliberately smaller than the buffer, and
    /// identifiable, so the assertion is that those exact bytes were written
    /// and *then* that the later switches were required before the track was
    /// called finished.
    #[test]
    fn a_partial_final_payload_must_sound_before_the_track_ends() {
        const CH: usize = 2;
        const BYTES: usize = 2_048;
        const PAYLOAD: usize = 300; // 0 < PAYLOAD < BYTES

        let sh = shared();
        sh.buffer_frames.store(BYTES as u64, Ordering::Relaxed);
        sh.begin_session();

        // Less than a buffer of audio, and a feeder that has finished.
        let (mut prod, cons) = channel_ring::<u8>(CH, BYTES * 4);
        prod.push_frames(&frames(PAYLOAD, CH)).unwrap();
        install(&sh, cons, true, true);
        let mut fake = Fake::new(CH, BYTES, Arc::clone(&sh));

        // Switch 0 takes everything there is: a partial buffer of music
        // followed by idle. This is the callback that starts the drain.
        fake.switch(0);
        let half0 = fake.half(0, 0).to_vec();
        let expected = frames(PAYLOAD, CH);
        for (f, byte) in half0.iter().take(PAYLOAD).enumerate() {
            assert_eq!(
                *byte,
                expected[f * CH],
                "frame {f} of the final payload must be in the driver's buffer"
            );
        }
        assert_eq!(
            half0[PAYLOAD], idle_byte(false),
            "and the rest of that buffer is DSD idle, not stale bytes"
        );
        assert!(
            sh.session.lock().unwrap().is_none(),
            "the ring ran dry, so the session closes here"
        );
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "the driver has been *given* the last of the music and has played none of it"
        );

        // Switch 1: the driver has taken half 1 and is now *playing* half 0,
        // which is the music. Still sounding, so still not finished.
        fake.switch(1);
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "half 0 — the music — is sounding right now"
        );

        // Switch 2: half 0 has been played to its end. The music has been
        // heard, and only now is the track over.
        //
        // Two counted switches, not three: the callback that wrote the payload
        // does not count itself, and the two after it are the buffer being
        // played and the one behind it. That is the whole chronology — a
        // driver holds what it was handed until the switch *after* the one
        // that handed it over.
        fake.switch(0);
        assert_eq!(sh.completion(), crate::bitperfect::Completion::CleanEof);
        assert!(sh.completion().may_advance());
    }

    /// A dropout and then a driver failure: the reason is the failure, the
    /// evidence is the dropout.
    #[test]
    fn an_asio_dropout_survives_the_failure_that_ends_the_track() {
        let sh = shared();
        let at = sh.begin_session();

        sh.revoke(at, fault::UNDERRUN);
        assert_eq!(sh.revoked(), fault::UNDERRUN);
        assert_eq!(
            sh.fault(),
            fault::NONE,
            "a dropout is evidence, not a reason the track stopped"
        );
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "a dropout does not end the track"
        );

        sh.fail_now(fault::BACKEND_WRITE);
        assert_eq!(
            sh.fault(),
            fault::BACKEND_WRITE,
            "what ended the session is what is reported"
        );
        assert_eq!(sh.completion().failure(), Some(fault::BACKEND_WRITE));
        assert_eq!(
            sh.revoked(),
            fault::UNDERRUN,
            "and the dropout is still on the record"
        );

        // A new track starts clean on both.
        sh.begin_session();
        assert_eq!(sh.fault(), fault::NONE);
        assert_eq!(sh.revoked(), fault::NONE);
    }

    /// A driver that reports a fault is not a driver that lacks a feature.
    ///
    /// Every non-success code from `kAsioGetIoFormat` was read as "does not
    /// implement it" and waved through as amber Unverified. Two of them are
    /// the driver saying something: `ASE_HWMalfunction` that the hardware is
    /// broken, `ASE_InvalidParameter` that the call was wrong. Sending DSD
    /// packing to a device that has just reported a fault is not an unverified
    /// claim, it is ignoring the only evidence on offer.
    #[test]
    fn a_malfunctioning_driver_is_not_an_unimplemented_one() {
        // Literals, from `asio.h`, for the same reason the selectors are.
        assert_eq!(classify_io_format(0), IoFormatReadback::Answered);
        assert_eq!(classify_io_format(ASE_SUCCESS), IoFormatReadback::Answered);
        // The only code that means the selector is not implemented.
        assert_eq!(classify_io_format(-1000), IoFormatReadback::Absent); // ASE_NotPresent

        // Everything else is an answer, and DSD must not be sent past one.
        // `ASE_InvalidMode` in particular used to be read as absence: it means
        // the driver understood the call and refused it in the state it is
        // in, which — having just been put into DSD mode — is a contradiction.
        assert_eq!(
            classify_io_format(-997), // ASE_InvalidMode
            IoFormatReadback::Broken(-997)
        );
        assert_eq!(
            classify_io_format(-999), // ASE_HWMalfunction
            IoFormatReadback::Broken(-999)
        );
        assert_eq!(
            classify_io_format(-998), // ASE_InvalidParameter
            IoFormatReadback::Broken(-998)
        );
        // An undocumented code is not absence either. "I do not recognise this
        // answer" and "no answer was given" are different, and only the second
        // is grounds for carrying on and calling the route unverified.
        assert_eq!(
            classify_io_format(-12345),
            IoFormatReadback::Broken(-12345)
        );
    }

    /// Every driver event that invalidates the stream refuses its reuse.
    ///
    /// A rate change, a resync and a reset all mean the device is no longer
    /// where the stream was negotiated. Failing the track alone left the
    /// handle looking usable, so the next track was handed to a device that
    /// had already said so.
    #[test]
    fn an_invalidating_driver_event_refuses_the_next_track_too() {
        for selector in [
            2i32,  // kAsioResetRequest is 3; 2 is kAsioEngineVersion, a control
            3,     // kAsioResetRequest
            5,     // kAsioResyncRequest
            4,     // kAsioBufferSizeChange
        ] {
            let sh = shared();
            let mut fake = Fake::new(2, 8, Arc::clone(&sh));
            let ptr: *mut CallbackCtx = &mut fake.ctx;
            let globals = Globals::take();
            globals.publish(ptr);
            unsafe { cb_asio_message(selector, 0, null_mut(), null_mut()) };
            let invalidating = selector != 2;
            assert_eq!(
                sh.reset_requested.load(Ordering::Acquire),
                invalidating,
                "selector {selector}"
            );
            if invalidating {
                assert!(!sh.completion().may_advance(), "selector {selector}");
            }
        }

        // The rate changing underneath the stream is the same statement made a
        // different way, and it was the one that did not refuse reuse.
        let sh = shared();
        let mut fake = Fake::new(2, 8, Arc::clone(&sh));
        let ptr: *mut CallbackCtx = &mut fake.ctx;
        let globals = Globals::take();
        globals.publish(ptr);
        unsafe { cb_sample_rate_did_change(44_100.0) };
        assert!(sh.ev_rate_changed.load(Ordering::Relaxed));
        assert!(
            sh.reset_requested.load(Ordering::Acquire),
            "a stream negotiated at one rate is not a stream at another"
        );
        assert!(!sh.completion().may_advance());
    }

    /// A new track on the same stream does not inherit the last one's fault.
    ///
    /// A stream outlives its tracks, so "the fault belongs to the stream" was
    /// never the same statement as "the fault belongs to this track".
    #[test]
    fn a_new_native_session_does_not_inherit_the_last_ones_fault() {
        let sh = shared();
        let first = sh.generation();
        sh.fail(first, fault::BACKEND_RESET);
        assert_eq!(sh.fault(), fault::BACKEND_RESET);
        assert!(sh.completion().is_over());

        let second = sh.begin_session();
        assert_eq!(sh.fault(), fault::NONE, "a new track starts clean");
        assert_eq!(sh.completion(), crate::bitperfect::Completion::Running);

        // And the old session cannot reach forward into it: the transition is
        // a compare-exchange on the word that holds the generation, so a
        // superseded caller has nothing it can win.
        sh.fail(first, fault::BACKEND_DEAD);
        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "a dead session may not fail the one that replaced it"
        );
        assert!(!sh.completion().may_advance());
        sh.finish_clean(second);
        assert!(sh.completion().may_advance());
    }

    /// Frames are the listener's position, so they are counted when the driver
    /// has them.
    ///
    /// The count happened straight after the pop, before a single byte had
    /// been written into the driver's buffers — so a callback that found a
    /// null pointer for one channel wrote a partial frame, or none, and still
    /// moved the position forward by the whole buffer. Everything built on the
    /// position moves with it: the elapsed clock, the seek bar, and the frame
    /// arithmetic that decides where a gapless boundary falls.
    #[test]
    fn frames_are_counted_only_when_every_channel_was_written() {
        const CH: usize = 2;
        let sh = shared();
        let g = sh.begin_session();
        let (mut prod, cons) = channel_ring::<u8>(CH, 64);
        assert_eq!(prod.push_frames(&frames(10, CH)).unwrap(), 10);
        *sh.session.lock().unwrap() = Some(AsioSession {
            generation: g,
            cons,
            decode_done: Arc::new(AtomicBool::new(false)),
            decode_eof: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });

        let mut fake = Fake::new(CH, 16, Arc::clone(&sh));
        fake.drop_channel(1);
        fake.switch(0);

        assert_eq!(
            sh.frames_played(),
            0,
            "a frame only one channel of which reached the driver is not a frame played"
        );

        // And with every channel present the same audio is counted, so the
        // assertion above is about the missing channel and not about the
        // fixture being unable to count at all.
        let sh2 = shared();
        let g2 = sh2.begin_session();
        let (mut prod2, cons2) = channel_ring::<u8>(CH, 64);
        assert_eq!(prod2.push_frames(&frames(10, CH)).unwrap(), 10);
        *sh2.session.lock().unwrap() = Some(AsioSession {
            generation: g2,
            cons: cons2,
            decode_done: Arc::new(AtomicBool::new(false)),
            decode_eof: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        });
        let mut whole = Fake::new(CH, 16, Arc::clone(&sh2));
        whole.switch(0);
        assert_eq!(sh2.frames_played(), 10);
    }

    /// A drain names the track it belongs to, and its expiry fails that one.
    ///
    /// A drain outlives its session slot — that is what a drain is — so from
    /// the moment it begins there is nothing left to read a generation from.
    /// Everything it then said was said about whatever the clock happened to
    /// hold, which after a track change is the wrong track.
    ///
    /// **The schedule here is constructed, not observed.** `begin_session`
    /// clears the draining flag, so in production the clock cannot advance
    /// under a live drain by any route I could find; the drain is re-armed
    /// explicitly below to put the two out of step. What the test establishes
    /// is that when they *are* out of step the drain speaks for its own track,
    /// which is what the stamping is for — not that a schedule reaching that
    /// state exists today.
    #[test]
    fn a_drain_that_expires_fails_its_own_track() {
        let sh = shared();
        let first = sh.begin_session();
        sh.begin_drain(first);
        let second = sh.begin_session();
        assert!(second > first, "each session gets its own number");

        // Out of step, deliberately: the drain belongs to `first` and the
        // clock has moved to `second`.
        sh.begin_drain(first);
        sh.expire_drain();

        assert_eq!(
            sh.completion(),
            crate::bitperfect::Completion::Running,
            "a drain that belonged to {first} must not fail {second}"
        );
        assert_eq!(
            sh.fault_in(first),
            super::super::fault::BACKEND_DEAD,
            "and the track it did belong to is the one that failed"
        );
    }

    /// A callback that cannot reach the session is a late callback, and outside
    /// the priming window that means the DAC played silence it should not have.
    #[test]
    fn a_contended_session_slot_is_counted_and_reported() {
        let channels = 2usize;
        let sh = shared();
        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));

        let held = sh.session.lock().unwrap();
        fake.switch(0);
        drop(held);

        assert_eq!(sh.lock_misses.load(Ordering::Relaxed), 1);
        assert_eq!(sh.revoked(), fault::CALLBACK_LOCK_MISS);
        for c in 0..channels {
            assert!(fake.out(c, 0).iter().all(|&b| b == DSD_SILENCE));
        }
    }

    /// Repeated callbacks allocate nothing once the context exists.
    ///
    /// Measured, not asserted from reading the code: a counting allocator armed
    /// on this thread only, so the rest of the suite is unaffected.
    #[test]
    fn a_steady_state_callback_allocates_nothing() {
        let channels = 2usize;
        let sh = shared();
        let (mut prod, cons) = channel_ring::<u8>(channels, 4096);
        prod.push_frames(&frames(2048, channels)).unwrap();
        install(&sh, cons, false, false);
        let mut fake = Fake::new(channels, 8, Arc::clone(&sh));

        // Warm up outside the measurement.
        fake.switch(0);

        // The probe has to be able to see an allocation, or "zero" means
        // nothing. This is the calibration that stops the test being vacuous.
        let before = crate::alloc_probe::arm();
        let canary = vec![0u8; 4096];
        let seen = crate::alloc_probe::disarm(before);
        assert!(
            seen > 0,
            "the allocation probe is not observing allocations"
        );
        drop(canary);

        let before = crate::alloc_probe::arm();
        for i in 0..64i32 {
            fake.switch(i & 1);
        }
        let allocs = crate::alloc_probe::disarm(before);
        assert_eq!(allocs, 0, "the ASIO callback allocated {allocs} time(s)");
    }

    /// LSB1 drivers need the reversed idle byte.
    ///
    /// The feeder reverses audio bytes for an LSB1 driver, but padding, lead-in,
    /// tail, pause and underrun silence were all left at a fixed `0x69` — so on
    /// an LSB1 driver every silent frame carried a different pattern from the
    /// audio around it. `0x69` reversed is `0x96`, and they are not the same
    /// byte.
    #[test]
    fn the_idle_byte_follows_the_drivers_bit_order() {
        assert_eq!(idle_byte(false), 0x69, "MSB1 idle");
        assert_eq!(idle_byte(true), 0x96, "LSB1 idle");
        assert_eq!(idle_byte(true), idle_byte(false).reverse_bits());
        assert_ne!(
            idle_byte(true),
            idle_byte(false),
            "the two orders must not share an idle byte"
        );
    }

    /// Padding uses the idle byte the context was built with, so an LSB1 stream
    /// pads with `0x96` throughout.
    #[test]
    fn padding_uses_the_contexts_idle_byte() {
        for (lsb, want) in [(false, 0x69u8), (true, 0x96u8)] {
            let channels = 2usize;
            let sh = shared();
            let (mut prod, cons) = channel_ring::<u8>(channels, 64);
            let src = frames(3, channels);
            prod.push_frames(&src).unwrap();
            install(&sh, cons, false, false);

            let mut fake = Fake::new(channels, 8, Arc::clone(&sh));
            fake.ctx.idle = idle_byte(lsb);
            fake.switch(0);

            for c in 0..channels {
                let got = fake.out(c, 0);
                for (f, &b) in got.iter().enumerate().skip(3) {
                    assert_eq!(b, want, "lsb={lsb} channel {c} pad byte {f}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Driver ownership, against a fake function table
    // -----------------------------------------------------------------------
    //
    // These build a real `IAsioVtbl` over a counting stub and call the real
    // ownership code through it. Nothing here inspects source text: the
    // assertions are counts of the driver methods that were actually invoked,
    // which is the only way to state "freed exactly once, in this order"
    // without a driver.

    /// What the fake driver was asked to do, in order.
    #[derive(Default)]
    struct DriverLog {
        events: Mutex<Vec<&'static str>>,
    }

    impl DriverLog {
        fn note(&self, what: &'static str) {
            self.events.lock().unwrap().push(what);
        }
        fn events(&self) -> Vec<&'static str> {
            self.events.lock().unwrap().clone()
        }
        fn count(&self, what: &str) -> usize {
            self.events().iter().filter(|e| **e == what).count()
        }
    }

    /// A driver object laid out exactly as the real one: a vtable pointer
    /// first, so `asio_call!` reaches our stubs.
    #[repr(C)]
    struct FakeDriver {
        vtbl: *const IAsioVtbl,
        log: *const DriverLog,
    }

    thread_local! {
        static FAKE_LOG: std::cell::RefCell<Option<Arc<DriverLog>>> =
            const { std::cell::RefCell::new(None) };
    }

    fn note(what: &'static str) {
        FAKE_LOG.with(|l| {
            if let Some(log) = l.borrow().as_ref() {
                log.note(what);
            }
        });
    }

    unsafe extern "system" fn fk_release(_: *mut IAsio) -> u32 {
        note("release");
        0
    }
    unsafe extern "system" fn fk_dispose(_: *mut IAsio) -> AsioError {
        note("dispose_buffers");
        ASE_OK
    }
    unsafe extern "system" fn fk_stop(_: *mut IAsio) -> AsioError {
        note("stop");
        ASE_OK
    }
    unsafe extern "system" fn fk_start(_: *mut IAsio) -> AsioError {
        note("start");
        ASE_OK
    }
    unsafe extern "system" fn fk_qi(_: *mut IAsio, _: *const GUID, _: *mut *mut c_void) -> i32 {
        0
    }
    unsafe extern "system" fn fk_addref(_: *mut IAsio) -> u32 {
        1
    }
    unsafe extern "system" fn fk_init(_: *mut IAsio, _: *mut c_void) -> AsioBool {
        ASIO_TRUE
    }
    unsafe extern "system" fn fk_name(_: *mut IAsio, _: *mut u8) {}
    unsafe extern "system" fn fk_version(_: *mut IAsio) -> i32 {
        2
    }
    unsafe extern "system" fn fk_errmsg(_: *mut IAsio, msg: *mut u8) {
        unsafe { *msg = 0 };
    }
    unsafe extern "system" fn fk_channels(_: *mut IAsio, i: *mut i32, o: *mut i32) -> AsioError {
        unsafe {
            *i = 0;
            *o = 2;
        }
        ASE_OK
    }
    unsafe extern "system" fn fk_lat(_: *mut IAsio, _: *mut i32, _: *mut i32) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_bufsize(
        _: *mut IAsio,
        mn: *mut i32,
        mx: *mut i32,
        pref: *mut i32,
        gran: *mut i32,
    ) -> AsioError {
        unsafe {
            *mn = 64;
            *mx = 4096;
            *pref = 512;
            *gran = 0;
        }
        ASE_OK
    }
    unsafe extern "system" fn fk_can_rate(_: *mut IAsio, _: f64) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_get_rate(_: *mut IAsio, r: *mut f64) -> AsioError {
        unsafe { *r = 2_822_400.0 };
        ASE_OK
    }
    unsafe extern "system" fn fk_set_rate(_: *mut IAsio, _: f64) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_clocks(_: *mut IAsio, _: *mut c_void, _: *mut i32) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_set_clock(_: *mut IAsio, _: i32) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_pos(_: *mut IAsio, _: *mut u64, _: *mut u64) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_chinfo(_: *mut IAsio, i: *mut AsioChannelInfo) -> AsioError {
        unsafe { (*i).sample_type = ASIOST_DSD_INT8_MSB1 };
        ASE_OK
    }
    unsafe extern "system" fn fk_create(
        _: *mut IAsio,
        _: *mut AsioBufferInfo,
        _: i32,
        _: i32,
        _: *const AsioCallbacks,
    ) -> AsioError {
        note("create_buffers");
        ASE_OK
    }
    unsafe extern "system" fn fk_panel(_: *mut IAsio) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_future(_: *mut IAsio, _: i32, _: *mut c_void) -> AsioError {
        ASE_OK
    }
    unsafe extern "system" fn fk_ready(_: *mut IAsio) -> AsioError {
        ASE_OK
    }

    static FAKE_VTBL: IAsioVtbl = IAsioVtbl {
        query_interface: fk_qi,
        add_ref: fk_addref,
        release: fk_release,
        init: fk_init,
        get_driver_name: fk_name,
        get_driver_version: fk_version,
        get_error_message: fk_errmsg,
        start: fk_start,
        stop: fk_stop,
        get_channels: fk_channels,
        get_latencies: fk_lat,
        get_buffer_size: fk_bufsize,
        can_sample_rate: fk_can_rate,
        get_sample_rate: fk_get_rate,
        set_sample_rate: fk_set_rate,
        get_clock_sources: fk_clocks,
        set_clock_source: fk_set_clock,
        get_sample_position: fk_pos,
        get_channel_info: fk_chinfo,
        create_buffers: fk_create,
        dispose_buffers: fk_dispose,
        control_panel: fk_panel,
        future: fk_future,
        output_ready: fk_ready,
    };

    /// A driver behind the fake table, with a log every stub writes to.
    fn fake_driver(log: &Arc<DriverLog>) -> (*mut IAsio, Box<FakeDriver>) {
        FAKE_LOG.with(|l| *l.borrow_mut() = Some(Arc::clone(log)));
        let mut d = Box::new(FakeDriver {
            vtbl: &FAKE_VTBL,
            log: Arc::as_ptr(log),
        });
        let ptr = (&mut *d) as *mut FakeDriver as *mut IAsio;
        (ptr, d)
    }

    fn callbacks_box() -> *mut AsioCallbacks {
        Box::into_raw(Box::new(AsioCallbacks {
            buffer_switch: cb_buffer_switch,
            sample_rate_did_change: cb_sample_rate_did_change,
            asio_message: cb_asio_message,
            buffer_switch_time_info: cb_buffer_switch_time_info,
        }))
    }

    /// A setup that fails *before* `createBuffers` releases the driver and
    /// nothing else — there are no buffers to dispose and no table to free.
    #[test]
    fn a_failure_before_create_buffers_releases_only_the_driver() {
        let log = Arc::new(DriverLog::default());
        let (ptr, _keep) = fake_driver(&log);
        drop(AsioResources::new(ptr));
        assert_eq!(log.events(), vec!["release"]);
    }

    /// A setup that fails *after* `createBuffers` disposes before releasing.
    ///
    /// This is the path that was wrong: every early return after the buffers
    /// existed called `release` alone, leaving the driver holding buffers it
    /// had handed us and a pointer to a callback table about to be freed.
    #[test]
    fn a_failure_after_create_buffers_disposes_before_releasing() {
        let log = Arc::new(DriverLog::default());
        let (ptr, _keep) = fake_driver(&log);
        let mut owned = AsioResources::new(ptr);
        owned.callbacks = callbacks_box();
        owned.buffers_created = true;
        drop(owned);

        assert_eq!(
            log.events(),
            vec!["dispose_buffers", "release"],
            "dispose must come first, and each exactly once"
        );
    }

    /// Handing ownership on frees nothing — otherwise a successful open would
    /// tear down the driver it just returned.
    #[test]
    fn handing_ownership_on_frees_nothing() {
        // `teardown_started` below nulls `ACTIVE`. Auditing the tests for
        // textual `ACTIVE.store` missed this one entirely: it touches the
        // global through a helper, which is the same hazard and harder to see.
        // What matters is what a test *calls*, not what it spells out.
        let _globals = Globals::take();
        let log = Arc::new(DriverLog::default());
        let (ptr, _keep) = fake_driver(&log);
        let mut owned = AsioResources::new(ptr);
        let cb = callbacks_box();
        owned.callbacks = cb;
        owned.buffers_created = true;

        let (driver, callbacks) = owned.hand_on();
        assert_eq!(driver, ptr);
        assert_eq!(callbacks, cb);
        assert!(log.events().is_empty(), "{:?}", log.events());

        // The caller is now responsible, and `teardown_started` is how it
        // discharges that: stop, dispose, release, then free the memory — each
        // exactly once, in that order.
        unsafe { teardown_started(driver, callbacks, null_mut()) };
        assert_eq!(
            log.events(),
            vec!["stop", "dispose_buffers", "release"],
            "a started driver stops before it disposes"
        );
        for what in ["stop", "dispose_buffers", "release"] {
            assert_eq!(log.count(what), 1, "{what} must happen exactly once");
        }
    }

    /// `teardown_started` unpublishes the callback context before the memory
    /// behind it is freed, so no callback can reach a dangling pointer.
    #[test]
    fn teardown_unpublishes_before_it_frees() {
        let log = Arc::new(DriverLog::default());
        let (ptr, _keep) = fake_driver(&log);

        let sh = shared();
        let mut fake = Fake::new(2, 8, Arc::clone(&sh));
        let ctx_ptr: *mut CallbackCtx = &mut fake.ctx;
        let globals = Globals::take();
        globals.publish(ctx_ptr);
        assert!(!ACTIVE.load(Ordering::Acquire).is_null());

        // A null context is passed so the fake's own storage is not freed;
        // what is under test is that `ACTIVE` is cleared by the teardown.
        unsafe { teardown_started(ptr, null_mut(), null_mut()) };
        assert!(
            ACTIVE.load(Ordering::Acquire).is_null(),
            "nothing may reach the context after teardown"
        );
    }

    /// The process-wide engine claim is taken once and released once, and it
    /// survives a panic.
    #[test]
    fn the_engine_claim_is_exclusive_and_panic_safe() {
        // `ENGAGED` is process-wide and this test asserts its exact state, so
        // it cannot run beside another that takes a claim — and
        // `EngineClaim::drop` nulls `ACTIVE`, so it cannot run beside one that
        // has published a context either.
        let _globals = Globals::take();
        let first = EngineClaim::take().expect("no engine is running in this test");
        assert!(
            EngineClaim::take().is_err(),
            "a second engine must not be able to claim"
        );
        drop(first);

        let again = EngineClaim::take().expect("the claim is released on drop");
        drop(again);

        // A claim dropped during an unwind releases too, or every later open
        // in the process would report "already open" forever.
        let r = std::panic::catch_unwind(|| {
            let _claim = EngineClaim::take().expect("claimable");
            panic!("host thread died");
        });
        assert!(r.is_err());
        let after = EngineClaim::take().expect("a panicking host must still release the claim");
        drop(after);
    }

    /// The selector constants are the SDK's numbers, not a contiguous guess.
    ///
    /// Checked as literals against the production constants, because these are
    /// an ABI: a driver sends the integer and we answer the integer, and a
    /// wrong value neither fails to compile nor fails to run — it silently
    /// answers a different question.
    #[test]
    fn the_selector_constants_match_the_sdk() {
        assert_eq!(K_ASIO_SELECTOR_SUPPORTED, 1);
        assert_eq!(K_ASIO_ENGINE_VERSION, 2);
        assert_eq!(K_ASIO_RESET_REQUEST, 3);
        assert_eq!(K_ASIO_BUFFER_SIZE_CHANGE, 4);
        assert_eq!(K_ASIO_RESYNC_REQUEST, 5);
        assert_eq!(K_ASIO_LATENCIES_CHANGED, 6);
        assert_eq!(K_ASIO_SUPPORTS_TIME_INFO, 7);
        assert_eq!(K_ASIO_OVERLOAD, 15);
        // Every one distinct, so no two events can collapse onto one handler.
        let all = [
            K_ASIO_SELECTOR_SUPPORTED,
            K_ASIO_ENGINE_VERSION,
            K_ASIO_RESET_REQUEST,
            K_ASIO_BUFFER_SIZE_CHANGE,
            K_ASIO_RESYNC_REQUEST,
            K_ASIO_LATENCIES_CHANGED,
            K_ASIO_SUPPORTS_TIME_INFO,
            K_ASIO_OVERLOAD,
        ];
        let mut sorted = all.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "selectors must not collide");
    }

    /// The probe names PCM sample types it could carry, and refuses to call a
    /// DSD packing or an unknown type "PCM this build could carry exactly".
    #[test]
    fn the_pcm_probe_only_claims_formats_it_could_carry() {
        // The literal `asio.h` numbers again, for the same reason.
        for (t, want) in [
            (0i32, true), // ASIOSTInt16MSB
            (1, true),    // ASIOSTInt24MSB
            (2, true),    // ASIOSTInt32MSB
            (3, true),    // ASIOSTFloat32MSB
            (16, true),   // ASIOSTInt16LSB
            (17, true),   // ASIOSTInt24LSB
            (18, true),   // ASIOSTInt32LSB
            (19, true),   // ASIOSTFloat32LSB
            (32, false),  // ASIOSTDSDInt8LSB1 — a DSD packing, not PCM
            (33, false),  // ASIOSTDSDInt8MSB1
            (40, false),  // ASIOSTDSDInt8NER8
            (4, false),   // ASIOSTFloat64MSB — no exact route
            (999, false), // and anything we have never heard of
        ] {
            let (name, exact) = pcm_sample_type_name(t);
            assert_eq!(exact, want, "sample type {t} ({name})");
            assert!(!name.is_empty());
        }
    }

    /// Every driver event that means "this is no longer the stream you
    /// negotiated" ends the claim.
    ///
    /// `sampleRateDidChange`, reset, resync, overload and a buffer-size change
    /// all used to be logged and otherwise ignored — from inside the callback,
    /// with a formatting allocation and the logging mutex.
    #[test]
    fn driver_events_fault_the_session_and_never_log() {
        let sh = shared();
        let mut fake = Fake::new(2, 8, Arc::clone(&sh));
        let ptr: *mut CallbackCtx = &mut fake.ctx;
        // This is the test the isolation was written for: it failed on
        // `ev_rate_changed` about one full-suite run in several hundred and
        // passed every time it was run alone, because another test's
        // `EngineClaim::drop` had nulled `ACTIVE` between the store and the
        // callback.
        let globals = Globals::take();
        globals.publish(ptr);

        unsafe { cb_sample_rate_did_change(48_000.0) };
        assert!(sh.ev_rate_changed.load(Ordering::Relaxed));
        assert_eq!(sh.fault(), fault::BACKEND_RESET);
        assert!(sh.completion().is_over(), "the session is over");
        assert!(
            !sh.completion().may_advance(),
            "a rate change underneath the stream is not a track that finished"
        );

        // Each selector is answered and recorded.
        //
        // **Literals, deliberately.** Calling the production constants would
        // make this test agree with whatever they happen to say, which is
        // exactly the bug it exists to catch: the numbers were previously
        // guessed contiguously and overload sat on `kAsioSupportsInputMonitor`.
        // These are the values in `asio.h`.
        let sh2 = shared();
        let mut fake2 = Fake::new(2, 8, Arc::clone(&sh2));
        let ptr2: *mut CallbackCtx = &mut fake2.ctx;
        globals.publish(ptr2);

        // 15 = kAsioOverload.
        assert_eq!(unsafe { cb_asio_message(15, 0, null_mut(), null_mut()) }, 1);
        assert_eq!(sh2.ev_overload.load(Ordering::Relaxed), 1);
        // Its own code, not `UNDERRUN`. A driver telling us it missed its own
        // deadline is not this process failing to keep a ring full, and only
        // one of the two is survivable.
        assert_eq!(sh2.fault(), fault::BACKEND_OVERLOAD);
        assert!(!sh2.completion().may_advance());

        // 3 = kAsioResetRequest.
        assert_eq!(unsafe { cb_asio_message(3, 0, null_mut(), null_mut()) }, 1);
        assert!(sh2.ev_reset.load(Ordering::Relaxed));
        assert!(sh2.reset_requested.load(Ordering::Acquire));

        // 4 = kAsioBufferSizeChange.
        assert_eq!(unsafe { cb_asio_message(4, 0, null_mut(), null_mut()) }, 1);
        assert!(sh2.ev_buffer_size_changed.load(Ordering::Relaxed));

        // 5 = kAsioResyncRequest.
        assert_eq!(unsafe { cb_asio_message(5, 0, null_mut(), null_mut()) }, 1);
        assert!(sh2.ev_resync.load(Ordering::Relaxed));

        // 2 = kAsioEngineVersion; the host expects the ASIO 2 answer.
        assert_eq!(unsafe { cb_asio_message(2, 0, null_mut(), null_mut()) }, 2);
        // 7 = kAsioSupportsTimeInfo, deliberately declined.
        assert_eq!(unsafe { cb_asio_message(7, 0, null_mut(), null_mut()) }, 0);

        // The numbers we must *not* claim. 10 is kAsioSupportsInputMonitor and
        // 11 is kAsioSupportsInputGain — the two the overload and buffer-size
        // handlers used to be wired to. Claiming an input feature this build
        // has no concept of is how the real overload went unseen.
        for sel in [8i32, 9, 10, 11, 12, 13, 14] {
            assert_eq!(
                unsafe { cb_asio_message(1, sel, null_mut(), null_mut()) },
                0,
                "selector {sel} must not be advertised as supported"
            );
        }
        // 1 = kAsioSelectorSupported: the events we do handle.
        for sel in [3i32, 4, 5, 6, 15] {
            assert_eq!(
                unsafe { cb_asio_message(1, sel, null_mut(), null_mut()) },
                1,
                "selector {sel} should be advertised as supported"
            );
        }
    }

    /// No callback entry point may log, allocate or block — not only
    /// `fill_buffers`.
    ///
    /// `bufferSwitchTimeInfo`, `sampleRateDidChange` and `asioMessage` all
    /// called `mlog!` before this release. This reads the source because the
    /// property is about what the functions are permitted to reach, and a
    /// runtime assertion inside a driver callback is not a thing that can be
    /// added safely.
    #[test]
    fn no_asio_callback_entry_point_logs_or_blocks() {
        // Normalised first: this file is stored with CRLF endings, and a
        // delimiter written with a bare newline silently matches nothing
        // against them — which would make the scan cover the rest of the file
        // and fail for the wrong reason, or cover nothing and pass for the
        // wrong reason.
        let src = include_str!("asio_dsd.rs").replace('\r', "");
        for name in [
            "unsafe extern \"system\" fn cb_buffer_switch(",
            "unsafe extern \"system\" fn cb_buffer_switch_time_info(",
            "unsafe extern \"system\" fn cb_sample_rate_did_change(",
            "unsafe extern \"system\" fn cb_asio_message(",
            "unsafe fn fill_buffers(",
        ] {
            let start = src.find(name).unwrap_or_else(|| panic!("{name} moved"));
            let rest = &src[start + name.len()..];
            // Delimit at the next item that starts in column zero.
            let end = rest.find("\n}\n").map(|i| i + 3).unwrap_or(rest.len());
            assert!(
                end < rest.len(),
                "{name}: the delimiter did not match, so this scan proves nothing"
            );
            let body = &rest[..end];
            for forbidden in [
                "mlog!",
                "format!",
                "println!",
                "to_string()",
                "Vec::new",
                "vec![",
                "String::",
            ] {
                assert!(
                    !body.contains(forbidden),
                    "{name} contains `{forbidden}`, which is not realtime-safe"
                );
            }
            assert!(
                !body.replace("try_lock()", "").contains(".lock()"),
                "{name} takes a blocking lock"
            );
        }
    }

    /// The bit-order transformations, as arithmetic rather than as a promise.
    ///
    /// MSB1 is the reader's native order and passes through; LSB1 is the same
    /// bytes with their bits reversed. NER8 is neither — it is 8-bit data at
    /// one sample per byte — and `host_setup` refuses it rather than treating
    /// it as MSB1, which is what produced eight times the intended data rate.
    #[test]
    fn the_supported_bit_orders_are_exactly_msb1_and_lsb1() {
        for b in 0u8..=255 {
            assert_eq!(b.reverse_bits().reverse_bits(), b);
        }
        assert_eq!(0b1000_0000u8.reverse_bits(), 0b0000_0001);
        assert_eq!(0b0110_1001u8.reverse_bits(), 0b1001_0110);
        // The idle pattern reverses to a different byte, so the order genuinely
        // has to be right — silence is not accidentally order-agnostic.
        assert_eq!(DSD_SILENCE.reverse_bits(), 0x96);

        // The three constants are distinct, so the match in `host_setup` cannot
        // collapse two of them by accident.
        let all = [
            ASIOST_DSD_INT8_LSB1,
            ASIOST_DSD_INT8_MSB1,
            ASIOST_DSD_INT8_NER8,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    /// The realtime callback must not log, allocate or block.
    ///
    /// 1.4.1 called `mlog!` from inside a render loop holding a hard deadline —
    /// formatting a `String`, taking a global mutex, and flushing to disk on
    /// the thread it was supposed to be measuring. This reads the source rather
    /// than trusting a comment, because a comment is exactly what was true
    /// before someone added the line back.
    #[test]
    fn the_realtime_callback_does_not_log_or_block() {
        let src = include_str!("asio_dsd.rs");
        let start = src
            .find("unsafe fn fill_buffers(")
            .expect("fill_buffers moved");
        let body = &src[start..];
        let end = body
            .find("\nstatic ACTIVE")
            .or_else(|| body.find("\nunsafe extern"))
            .expect("could not delimit fill_buffers");
        assert!(
            end > 0 && end < body.len(),
            "the delimiter did not land inside the file, so this scan proves nothing"
        );
        let body = &body[..end];

        for forbidden in [
            "mlog!",
            "format!",
            "println!",
            "to_string()",
            "Vec::new",
            "vec![",
            "String::",
            ".lock().",
        ] {
            assert!(
                !body.contains(forbidden),
                "fill_buffers contains `{forbidden}`, which is not realtime-safe"
            );
        }
        // A blocking `lock()` is the one that matters most, and it has to be
        // distinguished from `try_lock()`.
        assert!(
            !body.replace("try_lock()", "").contains("lock()"),
            "fill_buffers takes a blocking lock"
        );
        assert!(
            body.contains("try_lock()"),
            "fill_buffers should use try_lock"
        );
    }
}
