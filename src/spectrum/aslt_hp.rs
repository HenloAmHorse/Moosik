//! The High-accuracy GPU analysis (experimental, opt-in, default off): the
//! Superlet transform computed on an NVIDIA GPU through the driver API and
//! pinned prebuilt PTX, with a different numerical method from Standard — the
//! modulated split (τ = 1e-2), SGEMM V2, DGEMM TN / HANKEL, `acc64` / `cvt32`
//! and an f64 host fold. Its results carry their own identity
//! (`;num=gpu-hp1`, see [`REVISION`]); Standard's are untouched.
//!
//! Ported from the research engine (`claude/v162-mr-research` 7c03b03, rounds
//! 9–11, reviewed by Codex) operation for operation; only the planning, basis,
//! fold and PTX pieces of the research harness it calls come along with it.
//!
//! Admitted where it was measured faster than Standard ([`admits`],
//! [`spread_admits`]): Superlet at 256 to 2048 bars, 100 to 480 frames a
//! second, a window up to 1.2 s, and with two or more wavelets a spread of
//! 0.1 or more (0.3 or more above 48 kHz); stereo at 44.1, 48, 88.2, 96,
//! 176.4 or 192 kHz; at most [`CELL_CAP`] cells. Anything else, and any
//! failure, is the caller's to hand to Standard.
//!
//! Lifecycle (ENGINE_LIFECYCLE_PREDECLARED.md, ENGINE_RESULT_PREDECLARED.md):
//! the engine owns a private context and never touches the primary one; a
//! ledger holds every device allocation until its free succeeds, and every
//! host buffer of a copy until the copy is proven complete or the context is
//! destroyed, whatever the copy's call or wait returned; cleanup reports its
//! status; an admitted pass's result is decided once, after its cleanup.
//! [`analyze_on_this_thread`] makes, uses and closes an engine on the calling
//! thread. A failed teardown is reported as a device failure; that worker
//! must then retire without running another job.
use super::*;
use rayon::prelude::*;
use std::time::Instant;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

type Ptr = *mut c_void;

/// Why a pass produced no codes.
#[derive(Debug, Clone, PartialEq)]
pub enum HpError {
    /// Refused before launch, or a recoverable failure (an allocation) whose
    /// cleanup succeeded: the engine stays usable; the caller analyses on the
    /// Standard route.
    Unavailable(String),
    /// A driver, launch, copy, cleanup or context failure: the engine is
    /// quarantined; the caller analyses on the Standard route.
    Device(String),
    /// Cancelled by the caller: no codes, nothing published, nothing
    /// restarted. Takes precedence over a device failure in the same pass,
    /// which is still recorded as quarantine.
    Cancelled,
}

thread_local! {
    // The dedicated HP worker hands this to its supervisor immediately after
    // the wrapper returns. Cancellation suppresses publication/retry, not a
    // device-health failure from the same job.
    static LAST_DEVICE_FAILURE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Takes this worker's failure notice independently of its caller-facing
/// result, which may be Cancelled. Read before the dedicated worker retires.
pub fn take_device_failure() -> Option<String> {
    LAST_DEVICE_FAILURE.with(|slot| slot.borrow_mut().take())
}

/// Failures to inject, for the lifecycle checks only: the nth (from 1, over
/// the engine's life) module load, allocation, launch, stream query in a GEMM
/// wait, copy call, copy wait, free or context pop fails. A failed `pop` still
/// pops (the error is reported after it); a failed `pop_noop` does not pop,
/// and a failed copy call is not made: the engine cannot tell either way.
#[derive(Default, Clone, Copy)]
pub struct Faults {
    pub module_load: Option<usize>,
    pub alloc: Option<usize>,
    pub launch: Option<usize>,
    pub query: Option<usize>,
    pub copy_call: Option<usize>,
    pub copy_wait: Option<usize>,
    pub free: Option<usize>,
    pub pop: Option<usize>,
    pub pop_noop: Option<usize>,
    /// The pipelined pass's nth helper thread (1 the preparer, 2 the folder)
    /// is refused, as when the system will not start a thread.
    pub helper: Option<usize>,
    /// The preparer stops (panics) at this group; the folder at this group.
    pub prepare_stop: Option<usize>,
    pub fold_stop: Option<usize>,
}

/// How many of each counted operation the engine has issued.
#[derive(Default, Clone, Copy, Debug)]
pub struct Seen {
    pub module_load: usize,
    pub alloc: usize,
    pub launch: usize,
    pub query: usize,
    pub copy: usize,
    pub free: usize,
    pub pop: usize,
}

/// Where a pass first saw its cancel flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seat {
    Start,
    GroupStart,
    /// In a GEMM wait; whether the stream still had work at that moment.
    Wait { stream_busy: bool },
    Fold,
    /// After cleanup, as a successful pass was about to return its codes.
    Boundary,
}

/// The last pass's cancellation and cleanup times.
#[derive(Default, Clone, Copy, Debug)]
pub struct Trace {
    pub observed: Option<(Instant, Seat)>,
    pub drained_s: f64,
    pub stopped: Option<Instant>,
    pub freed_s: f64,
    pub released_s: f64,
    /// Wall time per stage of the pass (`STAGES`), for the log.
    pub stage_s: [f64; STAGES.len()],
}

/// The stages `Trace::stage_s` times, in order. The device work is queued, so
/// its time shows in `wait`, not in `queue`. `pieces`, `fold` and `put` run
/// beside the device in the pipelined pass, so the stages sum to more than
/// the pass; `stall` is the device thread waiting for either helper.
pub const STAGES: [&str; 12] = ["plan", "signal", "block", "pieces", "upload", "queue", "wait", "download", "free", "fold", "put", "stall"];

/// What `close` did: whether the private context was destroyed, the ledger
/// entries its destroy reclaimed (device allocations, held host buffers), the
/// order of events, and every failure.
#[derive(Default, Debug)]
pub struct CloseReport {
    pub context_destroyed: bool,
    pub device_reclaimed: usize,
    pub host_released: usize,
    pub events: Vec<&'static str>,
    pub failures: Vec<String>,
}

/// The longest track of the validated geometry: 310 s at 180 fps.
pub const FRAME_CAP: usize = 55_800;
/// The largest picture admitted, in cells per plane: that track's, whatever
/// the bars and frame rate (more bars or frames a second, a shorter track).
pub const CELL_CAP: usize = FRAME_CAP * N_BARS;
/// Device memory kept free beyond the pass's estimate.
const MARGIN: usize = 256 << 20;
/// CUDA_ERROR_NOT_READY.
const NOT_READY: i32 = 600;
/// The status an injected failure returns.
const INJECTED: i32 = 999;

// The same declarations as the exact GPU lane's (`aslt_exact_gpu.rs`), so
// the two cannot disagree about the signature.
type Handle = *mut c_void;
type FarProc = unsafe extern "system" fn() -> isize;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: Handle, flags: u32) -> Handle;
    fn GetProcAddress(module: Handle, name: *const std::ffi::c_char) -> Option<FarProc>;
    fn FreeLibrary(module: Handle) -> std::ffi::c_int;
}

/// The driver entry points, from an `nvcuda.dll` reference of the holder's own.
struct Drv {
    lib: Handle,
    device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    ctx_create: unsafe extern "C" fn(*mut Ptr, u32, i32) -> i32,
    ctx_destroy: unsafe extern "C" fn(Ptr) -> i32,
    push: unsafe extern "C" fn(Ptr) -> i32,
    pop: unsafe extern "C" fn(*mut Ptr) -> i32,
    current: unsafe extern "C" fn(*mut Ptr) -> i32,
    retain: unsafe extern "C" fn(*mut Ptr, i32) -> i32,
    release: unsafe extern "C" fn(i32) -> i32,
    state: unsafe extern "C" fn(i32, *mut u32, *mut i32) -> i32,
    stream_create: unsafe extern "C" fn(*mut Ptr, u32) -> i32,
    stream_destroy: unsafe extern "C" fn(Ptr) -> i32,
    query: unsafe extern "C" fn(Ptr) -> i32,
    module_load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> i32,
    module_unload: unsafe extern "C" fn(Ptr) -> i32,
    function: unsafe extern "C" fn(*mut Ptr, Ptr, *const u8) -> i32,
    alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    free: unsafe extern "C" fn(u64) -> i32,
    htod: unsafe extern "C" fn(u64, *const c_void, usize, Ptr) -> i32,
    dtoh: unsafe extern "C" fn(*mut c_void, u64, usize, Ptr) -> i32,
    dtoh_sync: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    memset32: unsafe extern "C" fn(u64, u32, usize) -> i32,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    launch: unsafe extern "C" fn(Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut *mut c_void, *mut *mut c_void) -> i32,
}

impl Drv {
    /// `nvcuda.dll` and its symbols, initialised; the library reference is
    /// released again on any failure.
    fn open() -> Result<Drv, HpError> {
        let name: Vec<u16> = "nvcuda.dll".encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated wide name; 0x800 = LOAD_LIBRARY_SEARCH_SYSTEM32.
        let lib = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), 0x800) };
        if lib.is_null() { return Err(HpError::Unavailable("no NVIDIA driver (nvcuda.dll)".into())); }
        let sym = |n: &str| -> Result<*const c_void, HpError> {
            let c: Vec<u8> = n.bytes().chain([0]).collect();
            // SAFETY: a loaded module and a NUL-terminated name.
            let p = unsafe { GetProcAddress(lib, c.as_ptr().cast()) };
            p.map(|f| f as *const c_void).ok_or_else(|| HpError::Unavailable(format!("the driver lacks {n}")))
        };
        let all = (|| -> Result<Drv, HpError> {
            // SAFETY: each symbol is the documented driver entry point with this signature.
            unsafe {
                let init: unsafe extern "C" fn(u32) -> i32 = std::mem::transmute(sym("cuInit")?);
                let rc = init(0);
                if rc != 0 { return Err(HpError::Unavailable(format!("cuInit failed with {rc}"))); }
                Ok(Drv {
                    lib,
                    device_get: std::mem::transmute(sym("cuDeviceGet")?),
                    ctx_create: std::mem::transmute(sym("cuCtxCreate_v2")?),
                    ctx_destroy: std::mem::transmute(sym("cuCtxDestroy_v2")?),
                    push: std::mem::transmute(sym("cuCtxPushCurrent_v2")?),
                    pop: std::mem::transmute(sym("cuCtxPopCurrent_v2")?),
                    current: std::mem::transmute(sym("cuCtxGetCurrent")?),
                    retain: std::mem::transmute(sym("cuDevicePrimaryCtxRetain")?),
                    release: std::mem::transmute(sym("cuDevicePrimaryCtxRelease_v2")?),
                    state: std::mem::transmute(sym("cuDevicePrimaryCtxGetState")?),
                    stream_create: std::mem::transmute(sym("cuStreamCreate")?),
                    stream_destroy: std::mem::transmute(sym("cuStreamDestroy_v2")?),
                    query: std::mem::transmute(sym("cuStreamQuery")?),
                    module_load: std::mem::transmute(sym("cuModuleLoadData")?),
                    module_unload: std::mem::transmute(sym("cuModuleUnload")?),
                    function: std::mem::transmute(sym("cuModuleGetFunction")?),
                    alloc: std::mem::transmute(sym("cuMemAlloc_v2")?),
                    free: std::mem::transmute(sym("cuMemFree_v2")?),
                    htod: std::mem::transmute(sym("cuMemcpyHtoDAsync_v2")?),
                    dtoh: std::mem::transmute(sym("cuMemcpyDtoHAsync_v2")?),
                    dtoh_sync: std::mem::transmute(sym("cuMemcpyDtoH_v2")?),
                    memset32: std::mem::transmute(sym("cuMemsetD32_v2")?),
                    mem_info: std::mem::transmute(sym("cuMemGetInfo_v2")?),
                    launch: std::mem::transmute(sym("cuLaunchKernel")?),
                })
            }
        })();
        if all.is_err() {
            // SAFETY: the reference taken above, released once.
            unsafe { FreeLibrary(lib); }
        }
        all
    }
}

fn arg<T>(x: &mut T) -> *mut c_void { (x as *mut T).cast() }

fn int(x: usize, what: &str) -> Result<i32, HpError> {
    i32::try_from(x).map_err(|_| HpError::Unavailable(format!("{what} = {x} exceeds a launch parameter")))
}

fn mul(a: usize, b: usize) -> Result<usize, HpError> {
    a.checked_mul(b).ok_or_else(|| HpError::Unavailable("a size overflows".into()))
}

/// The most the groups' buffers hold on the device at once, from each group's
/// (pieces and coefficients, results). While group g's kernels run, group
/// g+1's pieces are allocated before group g−1's buffers are freed, and those
/// are all of them, its pieces too: only its results are still needed (they
/// are being downloaded), but one hand-over frees the six together.
fn groups_live(parts: &[(usize, usize)]) -> Result<usize, HpError> {
    let mut most = 0usize;
    for (g, &(pieces, results)) in parts.iter().enumerate() {
        let before = if g > 0 { parts[g - 1] } else { (0, 0) };
        let after = parts.get(g + 1).map_or(0, |n| n.0);
        let live = [results, before.0, before.1, after].iter().try_fold(pieces, |sum, &x| sum.checked_add(x))
            .ok_or_else(|| HpError::Unavailable("a size overflows".into()))?;
        most = most.max(live);
    }
    Ok(most)
}

/// Whether a pass that allocates `need` bytes may start with `free` bytes of
/// device memory.
fn fits(need: usize, free: usize) -> bool {
    need <= free.saturating_sub(MARGIN)
}

/// The engine's custody: every device allocation until its free succeeds, and
/// every host buffer of a copy until the copy is proven complete (or the
/// context that could still be copying is destroyed).
#[derive(Default)]
struct Ledger {
    device: Vec<(u64, usize)>,
    host: Vec<Held>,
}

/// A host buffer in the ledger's custody.
struct Held(Box<dyn Any + Send>);

/// Held host buffers released from custody so far (dropped, or handed back
/// after a proven copy), for the checks, which run one engine at a time.
static RELEASED: AtomicUsize = AtomicUsize::new(0);

impl Drop for Held {
    fn drop(&mut self) { RELEASED.fetch_add(1, Ordering::Relaxed); }
}

/// The engine: reusable across passes until closed or quarantined.
pub struct HpEngine {
    d: Drv,
    /// The engine's private context (null until created, and after its destroy).
    ctx: Ptr,
    stream: Ptr,
    /// The pipelined pass's copies run here, beside the kernels on `stream`.
    copy_stream: Ptr,
    modules: Vec<Ptr>,
    /// acc64, cvt32, sgemm_v2, dgemm_tn, dgemm_hankel, sgemm_v2_ranged.
    k: [Ptr; 6],
    quarantined: Option<String>,
    faults: Faults,
    seen: Seen,
    ledger: Ledger,
    closed: bool,
    /// Checks only: the engine plays the caller and sets the pass's cancel
    /// flag as this group's GEMM wait begins.
    pub cancel_at_wait: Option<usize>,
    pub jit_s: f64,
    pub trace: Trace,
    /// The last pass's failures that its result does not show (round 11).
    pub failures: Vec<String>,
    /// Checks only: run the serial pass.
    #[cfg(test)]
    pub serial: bool,
    /// Checks only: another residual tolerance than [`TOLERANCE`], for the
    /// research report on what a looser one would cost and save.
    #[cfg(test)]
    pub tau: Option<f64>,
    /// Checks only: past the spread rule ([`spread_admits`]), for the
    /// research map of where this route is faster than Standard.
    #[cfg(test)]
    pub any_spread: bool,
}

// SAFETY: the engine's handles belong to its private context, not to a
// thread; every call makes that context current on the calling thread and
// pops it again, and `&mut self` keeps calls exclusive.
unsafe impl Send for HpEngine {}

impl Drop for HpEngine {
    fn drop(&mut self) {
        if !self.closed {
            for f in self.teardown().failures { eprintln!("HpEngine drop: {f}"); }
        }
    }
}

impl HpEngine {
    /// The driver, a private context on device 0, an owned non-blocking stream
    /// and the four pinned modules. Any failure unwinds what was acquired
    /// (the context's destroy) and returns the failure.
    pub fn new(faults: Faults) -> Result<HpEngine, HpError> {
        let d = Drv::open()?;
        let mut e = HpEngine {
            d, ctx: std::ptr::null_mut(), stream: std::ptr::null_mut(), copy_stream: std::ptr::null_mut(), modules: Vec::new(), k: [std::ptr::null_mut(); 6],
            quarantined: None, faults, seen: Seen::default(), ledger: Ledger::default(), closed: false, cancel_at_wait: None,
            jit_s: 0.0, trace: Trace::default(), failures: Vec::new(),
            #[cfg(test)]
            serial: false,
            #[cfg(test)]
            tau: None,
            #[cfg(test)]
            any_spread: false,
        };
        let init = (|| {
            let (mut dev, mut ctx): (i32, Ptr) = (0, std::ptr::null_mut());
            // SAFETY: valid out-pointers; cuCtxCreate makes the new context
            // current on this thread, and the pop below leaves it floating.
            unsafe {
                let rc = (e.d.device_get)(&mut dev, 0);
                if rc != 0 { return Err(HpError::Unavailable(format!("cuDeviceGet failed with {rc}"))); }
                let rc = (e.d.ctx_create)(&mut ctx, 0, dev);
                if rc != 0 { return Err(HpError::Unavailable(format!("cuCtxCreate failed with {rc}"))); }
            }
            e.ctx = ctx;
            e.pop_checked()?;
            let t = Instant::now();
            e.with_ctx(|e| {
                // SAFETY: a valid out-pointer; 1 = CU_STREAM_NON_BLOCKING.
                let rc = unsafe { (e.d.stream_create)(&mut e.stream, 1) };
                if rc != 0 {
                    e.stream = std::ptr::null_mut();
                    return Err(HpError::Unavailable(format!("cuStreamCreate failed with {rc}")));
                }
                // SAFETY: as above.
                let rc = unsafe { (e.d.stream_create)(&mut e.copy_stream, 1) };
                if rc != 0 {
                    e.copy_stream = std::ptr::null_mut();
                    return Err(HpError::Unavailable(format!("cuStreamCreate (copies) failed with {rc}")));
                }
                let acc = e.load(ACC64_PTX)?;
                let sg = e.load(&pinned(SGEMM_PTX))?;
                let dg = e.load(&pinned(DGEMM_PTX))?;
                let rg = e.load(&pinned(RANGED_PTX))?;
                e.k = [e.function(acc, "acc64")?, e.function(acc, "cvt32")?, e.function(sg, "sgemm_v2")?, e.function(dg, "dgemm_tn")?,
                    e.function(dg, "dgemm_hankel")?, e.function(rg, "sgemm_v2_ranged")?];
                Ok(())
            })?;
            e.jit_s = t.elapsed().as_secs_f64();
            Ok(())
        })();
        if let Err(err) = init {
            // Initialization can own a context before it fails. Explicitly
            // inspect its teardown, rather than let Drop hide a cleanup
            // failure behind an otherwise recoverable refusal.
            let had_context = !e.ctx.is_null();
            let rep = e.teardown();
            crate::mlog!("[hp] initialization failed: {err:?}; close: {rep:?}");
            return finish_closed(Err(err), had_context, &rep, &Ok(true), false);
        }
        Ok(e)
    }

    /// The quarantine reason, if the engine has one.
    pub fn quarantined(&self) -> Option<&str> { self.quarantined.as_deref() }

    /// The ledger: device allocations and held host buffers.
    pub fn ledger(&self) -> (usize, usize) { (self.ledger.device.len(), self.ledger.host.len()) }

    /// The counted operations so far.
    pub fn seen(&self) -> Seen { self.seen }

    /// Device-wide free memory, read inside the engine's context.
    pub fn free_memory(&mut self) -> Result<usize, HpError> {
        self.with_ctx(|e| {
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: valid out-pointers, the context current.
            let rc = unsafe { (e.d.mem_info)(&mut free, &mut total) };
            e.check(rc, "cuMemGetInfo")?;
            Ok(free)
        })
    }

    /// Ends the engine. A healthy engine drains, frees, and destroys its
    /// stream and modules; then, healthy or quarantined, its private context is
    /// destroyed, which reclaims whatever the ledger still holds; only after
    /// that are held host buffers released. Every failure is reported.
    pub fn close(mut self) -> CloseReport { self.teardown() }

    fn teardown(&mut self) -> CloseReport {
        let mut rep = CloseReport::default();
        if self.closed { return rep; }
        self.closed = true;
        if !self.ctx.is_null() {
            if self.quarantined.is_none() {
                let tidy = self.with_ctx(|e| {
                    e.wait(None)?;
                    while let Some(&(p, _)) = e.ledger.device.last() { e.free(p)?; }
                    for which in 0..2 {
                        let stream = std::mem::replace(if which == 0 { &mut e.copy_stream } else { &mut e.stream }, std::ptr::null_mut());
                        if !stream.is_null() {
                            // SAFETY: a stream of this engine, idle.
                            let rc = unsafe { (e.d.stream_destroy)(stream) };
                            e.check(rc, "cuStreamDestroy")?;
                        }
                    }
                    for m in std::mem::take(&mut e.modules) {
                        // SAFETY: a module of this engine; nothing queued uses it.
                        let rc = unsafe { (e.d.module_unload)(m) };
                        e.check(rc, "cuModuleUnload")?;
                    }
                    Ok(())
                });
                if let Err(err) = tidy { rep.failures.push(format!("{err:?}")); }
            }
            // SAFETY: the engine's own context; no other user holds it.
            let rc = unsafe { (self.d.ctx_destroy)(self.ctx) };
            rep.events.push("context destroyed");
            if rc == 0 {
                rep.context_destroyed = true;
                rep.device_reclaimed = self.ledger.device.len();
                rep.host_released = self.ledger.host.len();
                self.ledger.device.clear();
                self.ledger.host.clear();
                rep.events.push("held host buffers released");
            } else {
                // Deliberate process-lifetime retention: with the context's
                // fate unknown, the host buffers are forgotten (never freed)
                // and the counts reported; nothing is left to recover them.
                rep.failures.push(format!("cuCtxDestroy failed with {rc}: {} device allocations and {} host buffers are retained for the life of the process",
                    self.ledger.device.len(), self.ledger.host.len()));
                for h in self.ledger.host.drain(..) { std::mem::forget(h); }
            }
            self.ctx = std::ptr::null_mut();
        }
        // SAFETY: the engine's library reference, released once.
        unsafe { FreeLibrary(self.d.lib); }
        rep
    }

    /// Runs `f` with the engine's context current, then pops it: both
    /// statuses are checked, a failed pop quarantines, and the two results go
    /// through `merge`. A panic in `f` (a caller's callback, where panics
    /// unwind) still pops the context and quarantines before it goes on.
    fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R, HpError>) -> Result<R, HpError> {
        if self.ctx.is_null() { return Err(HpError::Unavailable("the engine is closed".into())); }
        // SAFETY: the engine's own context.
        let rc = unsafe { (self.d.push)(self.ctx) };
        if rc != 0 { return Err(self.poison(format!("cuCtxPushCurrent failed with {rc}"))); }
        let r = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self))) {
            Ok(r) => r,
            Err(panic) => {
                let _ = self.pop_checked();
                self.poison("a panic inside the engine's context".into());
                std::panic::resume_unwind(panic)
            }
        };
        match (r, self.pop_checked()) {
            (r, Ok(())) => r,
            (Ok(_), Err(p)) => Err(p),
            (Err(x), Err(p)) => Err(self.merge(x, p)),
        }
    }

    /// Pops the engine's context. A failed pop may or may not have taken
    /// effect, so the failure says whether the context is still current on
    /// this thread, as read back, rather than assuming either.
    fn pop_checked(&mut self) -> Result<(), HpError> {
        self.seen.pop += 1;
        let rc = if self.faults.pop_noop == Some(self.seen.pop) {
            INJECTED
        } else {
            let mut prev: Ptr = std::ptr::null_mut();
            // SAFETY: pops the context this engine made current.
            let rc = unsafe { (self.d.pop)(&mut prev) };
            if self.faults.pop == Some(self.seen.pop) { INJECTED } else { rc }
        };
        if rc == 0 { return Ok(()); }
        let mut cur: Ptr = std::ptr::null_mut();
        // SAFETY: a valid out-pointer.
        let state = match unsafe { (self.d.current)(&mut cur) } {
            0 if cur == self.ctx => "is still current on this thread",
            0 => "is not current on this thread",
            _ => "could not be checked on this thread",
        };
        Err(self.poison(format!("cuCtxPopCurrent failed with {rc}; the engine's context {state}")))
    }

    fn poison(&mut self, why: String) -> HpError {
        self.quarantined.get_or_insert_with(|| why.clone());
        HpError::Device(why)
    }

    /// Which of two failures the caller sees (round 11): a cancellation, then
    /// a device, context or cleanup failure, then a recoverable refusal; the
    /// first on a tie. The other stays in `failures`.
    fn merge(&mut self, a: HpError, b: HpError) -> HpError {
        let rank = |e: &HpError| match e { HpError::Unavailable(_) => 0, HpError::Device(_) => 1, HpError::Cancelled => 2 };
        let (keep, other) = if rank(&b) > rank(&a) { (b, a) } else { (a, b) };
        if other != HpError::Cancelled { self.failures.push(format!("{other:?}")); }
        keep
    }

    /// A driver status: out of memory is recoverable; anything else poisons.
    fn check(&mut self, rc: i32, what: &str) -> Result<(), HpError> {
        match rc {
            0 => Ok(()),
            2 => Err(HpError::Unavailable(format!("{what}: out of device memory"))),
            _ => Err(self.poison(format!("{what} failed with {rc}"))),
        }
    }

    fn observe(&mut self, seat: Seat) { self.trace.observed.get_or_insert((Instant::now(), seat)); }

    fn load(&mut self, ptx: &str) -> Result<Ptr, HpError> {
        self.seen.module_load += 1;
        if self.faults.module_load == Some(self.seen.module_load) {
            return Err(HpError::Unavailable("injected module load failure".into()));
        }
        let image: Vec<u8> = ptx.bytes().chain([0]).collect();
        let mut m: Ptr = std::ptr::null_mut();
        // SAFETY: a NUL-terminated PTX image, the context current.
        let rc = unsafe { (self.d.module_load)(&mut m, image.as_ptr().cast()) };
        if rc != 0 { return Err(HpError::Unavailable(format!("cuModuleLoadData failed with {rc}"))); }
        self.modules.push(m);
        Ok(m)
    }

    fn function(&mut self, module: Ptr, name: &str) -> Result<Ptr, HpError> {
        let c: Vec<u8> = name.bytes().chain([0]).collect();
        let mut f: Ptr = std::ptr::null_mut();
        // SAFETY: a loaded module and a NUL-terminated name.
        let rc = unsafe { (self.d.function)(&mut f, module, c.as_ptr()) };
        if rc != 0 { return Err(HpError::Unavailable(format!("cuModuleGetFunction {name} failed with {rc}"))); }
        Ok(f)
    }

    fn alloc(&mut self, bytes: usize) -> Result<u64, HpError> {
        self.seen.alloc += 1;
        if self.faults.alloc == Some(self.seen.alloc) {
            return Err(HpError::Unavailable("injected allocation failure".into()));
        }
        let mut p = 0u64;
        // SAFETY: a valid out-pointer, the context current.
        let rc = unsafe { (self.d.alloc)(&mut p, bytes.max(1)) };
        self.check(rc, "cuMemAlloc")?;
        self.ledger.device.push((p, bytes));
        Ok(p)
    }

    /// Frees one allocation of the ledger, only while nothing queued uses it.
    /// The entry leaves the ledger only if the free succeeds.
    fn free(&mut self, p: u64) -> Result<(), HpError> {
        self.seen.free += 1;
        let rc = if self.faults.free == Some(self.seen.free) {
            INJECTED
        } else {
            // SAFETY: an allocation of this engine that no queued work uses.
            unsafe { (self.d.free)(p) }
        };
        if rc != 0 { return Err(self.poison(format!("cuMemFree failed with {rc}"))); }
        if let Some(i) = self.ledger.device.iter().position(|&(x, _)| x == p) { self.ledger.device.remove(i); }
        Ok(())
    }

    /// The pass's cleanup: drain, then free everything the ledger holds. A
    /// quarantined engine frees nothing (the context's destroy will) and adds
    /// no failure: its pass already returned the one that quarantined it. A
    /// failed drain or free quarantines and keeps the remaining entries.
    fn finish_pass(&mut self) -> Result<(), HpError> {
        if self.quarantined.is_some() { return Ok(()); }
        self.wait(None)?;
        // Both streams are drained and no copy failed (that would have
        // quarantined): every copy still held was queued ahead by the
        // pipelined pass and is now complete.
        self.ledger.host.clear();
        while let Some(&(p, _)) = self.ledger.device.last() { self.free(p)?; }
        Ok(())
    }

    /// A copy's call, counted; the nth may be an injected failure, not made.
    /// Round 11: a failed status does not prove that nothing was queued (it
    /// may report an earlier asynchronous failure), so the caller's held
    /// buffer stays in the ledger until the context's destroy, and the engine
    /// is quarantined.
    fn copy_call(&mut self, what: &str, call: impl FnOnce(&Drv) -> i32) -> Result<(), HpError> {
        self.seen.copy += 1;
        let rc = if self.faults.copy_call == Some(self.seen.copy) { INJECTED } else { call(&self.d) };
        if rc != 0 { return Err(self.poison(format!("{what} failed with {rc}"))); }
        Ok(())
    }

    /// A copy's wait: the nth copy's may be an injected failure.
    fn copy_wait(&mut self) -> Result<(), HpError> {
        if self.faults.copy_wait == Some(self.seen.copy) {
            return Err(self.poison("injected copy-wait failure".into()));
        }
        self.wait(None).map(|_| ())
    }

    /// Uploads `data`, which the ledger holds until the copy is proven
    /// complete; after a failed call or wait it stays held until the
    /// context's destroy.
    fn upload<T: Copy + Send + 'static>(&mut self, dst: u64, data: Vec<T>) -> Result<(), HpError> {
        let (src, bytes, stream) = (data.as_ptr() as *const c_void, std::mem::size_of_val(data.as_slice()), self.stream);
        self.ledger.host.push(Held(Box::new(data)));
        // SAFETY: dst holds `bytes`; the source is the held buffer, which stays
        // put while held (moving a Vec does not move its storage).
        self.copy_call("cuMemcpyHtoDAsync", |d| unsafe { (d.htod)(dst, src, bytes, stream) })?;
        self.copy_wait()?;
        self.ledger.host.pop();
        Ok(())
    }

    /// Downloads n elements into a buffer the ledger holds until the copy is
    /// proven complete (and, after a failed call or wait, until the context's
    /// destroy).
    fn download<T: Copy + Default + Send + 'static>(&mut self, src: u64, n: usize) -> Result<Vec<T>, HpError> {
        let mut v = vec![T::default(); n];
        let (dst, bytes, stream) = (v.as_mut_ptr() as *mut c_void, std::mem::size_of_val(v.as_slice()), self.stream);
        self.ledger.host.push(Held(Box::new(v)));
        // SAFETY: src holds `bytes`; the destination is the held buffer.
        self.copy_call("cuMemcpyDtoHAsync", |d| unsafe { (d.dtoh)(dst, src, bytes, stream) })?;
        self.copy_wait()?;
        Ok(self.take_download())
    }

    /// The newest held buffer, a download proven complete, out of custody.
    fn take_download<T: 'static>(&mut self) -> Vec<T> {
        let mut held = self.ledger.host.pop().expect("the download's buffer");
        let v = std::mem::replace(&mut held.0, Box::new(()));
        *v.downcast::<Vec<T>>().expect("the download's buffer type")
    }

    /// Queues an upload on the copies' stream and returns: the ledger holds
    /// `data` until `copies_done` proves the copy complete (the caller then
    /// pops it), or, after a failure, until the context's destroy.
    fn upload_ahead<T: Copy + Send + 'static>(&mut self, dst: u64, data: Vec<T>) -> Result<(), HpError> {
        let (src, bytes, stream) = (data.as_ptr() as *const c_void, std::mem::size_of_val(data.as_slice()), self.copy_stream);
        self.ledger.host.push(Held(Box::new(data)));
        // SAFETY: as `upload`.
        self.copy_call("cuMemcpyHtoDAsync", |d| unsafe { (d.htod)(dst, src, bytes, stream) })
    }

    /// Queues a download of n elements on the copies' stream into a buffer
    /// the ledger holds; `take_download` after `copies_done`.
    fn download_ahead<T: Copy + Default + Send + 'static>(&mut self, src: u64, n: usize) -> Result<(), HpError> {
        let mut v = vec![T::default(); n];
        let (dst, bytes, stream) = (v.as_mut_ptr() as *mut c_void, std::mem::size_of_val(v.as_slice()), self.copy_stream);
        self.ledger.host.push(Held(Box::new(v)));
        // SAFETY: as `download`.
        self.copy_call("cuMemcpyDtoHAsync", |d| unsafe { (d.dtoh)(dst, src, bytes, stream) })
    }

    /// Waits for the copies' stream: the last `count` copies are then proven
    /// complete. An injected copy-wait failure of any of them fails here.
    fn copies_done(&mut self, count: usize) -> Result<(), HpError> {
        if self.faults.copy_wait.is_some_and(|n| n <= self.seen.copy && n + count > self.seen.copy) {
            return Err(self.poison("injected copy-wait failure".into()));
        }
        self.wait_stream(self.copy_stream, None).map(|_| ())
    }

    /// A prepared group's kernel pieces and coefficients on their way to the
    /// device (two copies queued; two buffers held).
    fn stage(&mut self, p: Prepared) -> Result<Staged, HpError> {
        let Prepared { group, s_g, q, h, cols, phi, subs, tiles, secs } = p;
        self.trace.stage_s[3] += secs;
        let dh = self.alloc(h.len() * 4)?;
        self.upload_ahead(dh, h)?;
        let dphi = self.alloc(phi.len() * 8)?;
        self.upload_ahead(dphi, phi)?;
        Ok(Staged { group, s_g, q, cols, subs, tiles, dh, dphi })
    }

    fn launch(&mut self, f: Ptr, grid: (u32, u32), block: u32, params: &mut [*mut c_void]) -> Result<(), HpError> {
        self.seen.launch += 1;
        if self.faults.launch == Some(self.seen.launch) {
            return Err(self.poison("injected launch failure".into()));
        }
        // SAFETY: `f` takes exactly `params`; every buffer they name is live.
        let rc = unsafe { (self.d.launch)(f, grid.0, grid.1, 1, block, 1, 1, 0, self.stream, params.as_mut_ptr(), std::ptr::null_mut()) };
        self.check(rc, "cuLaunchKernel")
    }

    /// Waits for the kernels' stream; without a cancel flag (a drain: a
    /// copy's wait, the pass's cleanup, the close) for the copies' stream
    /// first, so that nothing of the engine is still queued anywhere.
    fn wait(&mut self, cancel: Option<&AtomicBool>) -> Result<bool, HpError> {
        if cancel.is_none() && !self.copy_stream.is_null() { self.wait_stream(self.copy_stream, None)?; }
        self.wait_stream(self.stream, cancel)
    }

    /// Waits for a stream by polling it, sleeping 50 µs doubling to 1 ms.
    /// With `cancel` (a GEMM wait), the flag is checked before each poll; once
    /// seen it is recorded, with whether the stream still had work, and the
    /// wait goes on (queued work cannot be pre-empted): Ok(true) = cancelled,
    /// drained. Only GEMM-wait queries count for injection.
    fn wait_stream(&mut self, stream: Ptr, cancel: Option<&AtomicBool>) -> Result<bool, HpError> {
        let mut nap = Duration::from_micros(50);
        let mut seen: Option<Instant> = None;
        loop {
            let flagged = seen.is_none() && cancel.is_some_and(|c| c.load(Ordering::Relaxed));
            let rc = if cancel.is_some() {
                self.seen.query += 1;
                if self.faults.query == Some(self.seen.query) {
                    INJECTED
                } else {
                    // SAFETY: a stream of this engine.
                    unsafe { (self.d.query)(stream) }
                }
            } else {
                // SAFETY: a stream of this engine.
                unsafe { (self.d.query)(stream) }
            };
            if flagged {
                let now = Instant::now();
                seen = Some(now);
                self.trace.observed.get_or_insert((now, Seat::Wait { stream_busy: rc == NOT_READY }));
            }
            match rc {
                0 => {
                    if let Some(t) = seen { self.trace.drained_s = t.elapsed().as_secs_f64(); }
                    return Ok(seen.is_some());
                }
                NOT_READY => {
                    std::thread::sleep(nap);
                    nap = (nap * 2).min(Duration::from_millis(1));
                }
                rc => return Err(self.poison(format!("cuStreamQuery failed with {rc}"))),
            }
        }
    }

    fn elementwise(&mut self, f: Ptr, dst: u64, src: u64, n: usize) -> Result<(), HpError> {
        let (mut d, mut s, mut nn) = (dst, src, u32::try_from(n).map_err(|_| HpError::Unavailable("an element count exceeds u32".into()))?);
        self.launch(f, (nn.div_ceil(256), 1), 256, &mut [arg(&mut d), arg(&mut s), arg(&mut nn)])
    }

    #[allow(clippy::too_many_arguments)]
    fn dgemm(&mut self, hankel: bool, m: usize, n: usize, k: usize, a: u64, lda: usize, nb: usize, lim: usize, b: u64, ldb: usize,
             c: u64, ldc: usize) -> Result<(), HpError> {
        let (mut mm, mut nn, mut kk, mut a, mut lda, mut nb, mut lim, mut b, mut ldb, mut c, mut ldc) = (int(m, "M")?, int(n, "N")?,
            int(k, "K")?, a, int(lda, "lda")?, int(nb, "nb")?, int(lim, "lim")?, b, int(ldb, "ldb")?, c, int(ldc, "ldc")?);
        let mut params: Vec<*mut c_void> = vec![arg(&mut mm), arg(&mut nn), arg(&mut kk), arg(&mut a), arg(&mut lda)];
        if hankel { params.extend([arg(&mut nb), arg(&mut lim)]); }
        params.extend([arg(&mut b), arg(&mut ldb), arg(&mut c), arg(&mut ldc)]);
        let f = self.k[if hankel { 4 } else { 3 }];
        self.launch(f, (int(n.div_ceil(8), "grid")? as u32, int(m.div_ceil(128), "grid")? as u32), 256, &mut params)
    }

    #[allow(clippy::too_many_arguments)]
    fn sgemm_v2(&mut self, m: usize, n: usize, k: usize, q: usize, a: u64, lda: usize, a_step: usize, b: u64, ldb: usize,
                b_step: usize, c: u64) -> Result<(), HpError> {
        let (mut mm, mut nn, mut kk, mut qq, mut a, mut lda, mut a_step, mut b, mut ldb, mut b_step, mut c, mut ldc) = (int(m, "M")?,
            int(n, "N")?, int(k, "K")?, int(q, "q")?, a, int(lda, "lda")?, a_step as i64, b, int(ldb, "ldb")?, b_step as i64, c, int(m, "ldc")?);
        let f = self.k[2];
        self.launch(f, (int(m.div_ceil(256), "grid")? as u32, int(n.div_ceil(72), "grid")? as u32), 288,
            &mut [arg(&mut mm), arg(&mut nn), arg(&mut kk), arg(&mut qq), arg(&mut a), arg(&mut lda), arg(&mut a_step), arg(&mut b),
                arg(&mut ldb), arg(&mut b_step), arg(&mut c), arg(&mut ldc)])
    }

    /// A group's tiles in one launch: grid row y computes the columns 72y..
    /// over the pieces `ranges[y]`, as `sgemm_v2` does for that tile alone.
    /// `a`, `b` and `c` are the group's bases, not moved to a tile or a piece.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_v2_ranged(&mut self, m: usize, n: usize, k: usize, q: usize, a: u64, b: u64, c: u64,
                       ranges: [(usize, usize); 4]) -> Result<(), HpError> {
        let (mut mm, mut nn, mut kk, mut qq, mut a, mut lda, mut a_step, mut b, mut ldb, mut b_step, mut c, mut ldc) = (int(m, "M")?,
            int(n, "N")?, int(k, "K")?, int(q, "q")?, a, int(k, "lda")?, k as i64, b, int(k, "ldb")?, (n * k) as i64, c, int(m, "ldc")?);
        let mut r = [0i32; 8];
        for (i, (lo, hi)) in ranges.into_iter().enumerate() {
            r[2 * i] = int(lo, "lo")?;
            r[2 * i + 1] = int(hi, "hi")?;
        }
        let [mut lo0, mut hi0, mut lo1, mut hi1, mut lo2, mut hi2, mut lo3, mut hi3] = r;
        let f = self.k[5];
        self.launch(f, (int(m.div_ceil(256), "grid")? as u32, int(n.div_ceil(72), "grid")? as u32), 288,
            &mut [arg(&mut mm), arg(&mut nn), arg(&mut kk), arg(&mut qq), arg(&mut a), arg(&mut lda), arg(&mut a_step), arg(&mut b),
                arg(&mut ldb), arg(&mut b_step), arg(&mut c), arg(&mut ldc),
                arg(&mut lo0), arg(&mut hi0), arg(&mut lo1), arg(&mut hi1), arg(&mut lo2), arg(&mut hi2), arg(&mut lo3), arg(&mut hi3)])
    }

    /// One analysis of a stereo pair on the frozen native route: the Mix, L
    /// and R codes (frame-major, N_BARS per frame), or why not. Admission is
    /// decided on the host before any device allocation, and its refusals are
    /// returned as they are; an admitted pass's result is decided once, after
    /// its cleanup (`settle`). `progress` receives the bars folded so far,
    /// never after a cancellation has been seen.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze(&mut self, l: &[f32], r: &[f32], sr: u32, geom: &Geom, cfg: &AsltConfig, pool: &rayon::ThreadPool, cancel: &AtomicBool,
                   progress: &(dyn Fn(usize) + Sync)) -> Result<[Vec<u16>; 3], HpError> {
        self.trace = Trace::default();
        self.failures.clear();
        let t_plan = Instant::now();
        if self.closed || self.ctx.is_null() { return Err(HpError::Unavailable("the engine is closed".into())); }
        if let Some(q) = &self.quarantined { return Err(HpError::Device(format!("quarantined: {q}"))); }
        let refuse = |why: String| Err(HpError::Unavailable(why));
        if !rate_admitted(sr) { return refuse(format!("{sr} Hz is outside the tested envelope")); }
        #[cfg(test)]
        let spread_rule = !self.any_spread;
        #[cfg(not(test))]
        let spread_rule = true;
        if spread_rule && let Err(why) = spread_admits(sr, cfg) { return refuse(why); }
        if l.len() != r.len() { return refuse("the channels differ in length".into()); }
        let n_bars = geom.n_bars;
        let hop = hop_for_fps(sr, geom.fps);
        let frames = frame_count(l.len(), hop);
        if frames == 0 || frames.saturating_mul(n_bars) > CELL_CAP {
            return refuse(format!("{frames} frames of {n_bars} bars is outside the tested length"));
        }
        if cancel.load(Ordering::Relaxed) {
            self.observe(Seat::Start);
            return Err(HpError::Cancelled);
        }
        let plans = plans(sr, geom, cfg);
        if plans.len() != n_bars { return refuse(format!("{} bars below Nyquist, not {n_bars}", plans.len())); }
        // Every group's plan before any device work (fail closed), with its
        // pieces and columns for the estimate.
        let memo = plan_memo(sr, geom, cfg, plans.len().div_ceil(GROUP));
        #[cfg(test)]
        let tau = self.tau.unwrap_or(TOLERANCE);
        #[cfg(not(test))]
        let tau = TOLERANCE;
        let (p0, rows, mod_plans, sizes) = match plan_track(&plans, sr, cfg, hop, frames, pool, &memo, tau) {
            Ok(planned) => planned,
            Err(why) => return refuse(why),
        };
        if cancel.load(Ordering::Relaxed) {
            self.observe(Seat::Start);
            return Err(HpError::Cancelled);
        }
        // Device memory, estimated on the host with checked arithmetic.
        let mut block_max = 0usize;
        for blk in mod_plans.chunks(CBLOCK) {
            let nblock: usize = blk.iter().map(|p| p.4.iter().map(|b| b.len()).sum::<usize>()).sum();
            block_max = block_max.max(mul(nblock, mul(hop, 8)?)? + mul(2, mul(rows, mul(nblock, 8)?)?)?);
        }
        // Per group: its pieces and coefficients (bounded as before), and its
        // results.
        let mut parts: Vec<(usize, usize)> = Vec::with_capacity(sizes.len());
        for (&(q, n_cols), p) in sizes.iter().zip(&mod_plans) {
            let nbmax = p.4.iter().map(|b| b.len()).max().unwrap_or(0);
            parts.push((mul(q, mul(hop, mul(n_cols, 4)?)?)? + mul(nbmax, mul(q, mul(n_cols, 8)?)?)?, mul(2, mul(frames, mul(n_cols, 12)?)?)?));
        }
        let group_max = groups_live(&parts)?;
        // The engine's own allocations only: the signals, a block's basis
        // and products, the groups. Not the driver's or the context's.
        let need = mul(2, mul(rows, mul(hop, 12)?)?)? + block_max + group_max;
        int(rows, "rows")?;
        int(mul(rows, hop)?, "signal length")?;
        let free = self.free_memory()?;
        if !fits(need, free) {
            return refuse(format!("needs {} MB of device memory, {} MB free", need >> 20, free >> 20));
        }
        self.trace.stage_s[0] = t_plan.elapsed().as_secs_f64();
        let out = self.with_ctx(|e| {
            #[cfg(test)]
            let r = if e.serial {
                e.run_serial(l, r, sr, cfg, n_bars, hop, frames, &plans, &mod_plans, p0, rows, pool, cancel, progress)
            } else {
                e.run(l, r, sr, cfg, n_bars, hop, frames, &plans, &mod_plans, p0, rows, pool, cancel, progress)
            };
            #[cfg(not(test))]
            let r = e.run(l, r, sr, cfg, n_bars, hop, frames, &plans, &mod_plans, p0, rows, pool, cancel, progress);
            let t = Instant::now();
            e.trace.stopped = Some(t);
            let fin = e.finish_pass();
            e.trace.freed_s = t.elapsed().as_secs_f64();
            match (r, fin) {
                (r, Ok(())) => r,
                (Ok(_), Err(x)) => Err(x),
                (Err(a), Err(b)) => Err(e.merge(a, b)),
            }
        });
        let t = Instant::now();
        drop((mod_plans, plans));
        self.trace.released_s = t.elapsed().as_secs_f64();
        self.settle(out, cancel)
    }

    /// An admitted pass's result for the caller (round 11), decided once,
    /// after its cleanup and its context's pop, for every outcome: a
    /// cancellation seen in the pass, or flagged now, wins (no codes, nothing
    /// restarted); then a device, context or cleanup failure, which
    /// quarantined the engine; then a recoverable refusal; codes only if none,
    /// and never from a quarantined engine. What the result does not show
    /// stays in `failures`; quarantine is recorded independently.
    fn settle(&mut self, out: Result<[Vec<u16>; 3], HpError>, cancel: &AtomicBool) -> Result<[Vec<u16>; 3], HpError> {
        if self.trace.observed.is_some() || cancel.load(Ordering::Relaxed) {
            if let Err(x) = out { if x != HpError::Cancelled { self.failures.push(format!("{x:?}")); } }
            self.observe(Seat::Boundary);
            return Err(HpError::Cancelled);
        }
        match (out, &self.quarantined) {
            (Ok(_), Some(q)) => Err(HpError::Device(format!("quarantined: {q}"))),
            (o, _) => o,
        }
    }

    /// The pass as rounds 10–11 reviewed it, one group at a time with the host
    /// and the device taking turns. Kept for the checks, which compare the
    /// pipelined pass with it code for code.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn run_serial(&mut self, l: &[f32], r: &[f32], sr: u32, cfg: &AsltConfig, n_bars: usize, hop: usize, frames: usize, plans: &[Plan], mod_plans: &[GroupPlan],
           p0: usize, rows: usize, pool: &rayon::ThreadPool, cancel: &AtomicBool, progress: &(dyn Fn(usize) + Sync))
           -> Result<[Vec<u16>; 3], HpError> {
        let padded_len = rows * hop;
        let mut t = Instant::now();
        macro_rules! lap { ($i:expr) => { self.trace.stage_s[$i] += t.elapsed().as_secs_f64(); t = Instant::now(); } }
        let (mut dx, mut dx64) = ([0u64; 2], [0u64; 2]);
        for (c, sig) in [l, r].into_iter().enumerate() {
            let mut v = vec![0.0f32; padded_len];
            let n = sig.len().min(padded_len - p0);
            v[p0..p0 + n].copy_from_slice(&sig[..n]);
            let v64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            dx[c] = self.alloc(padded_len * 4)?;
            self.upload(dx[c], v)?;
            dx64[c] = self.alloc(padded_len * 8)?;
            self.upload(dx64[c], v64)?;
        }
        lap!(1);
        let mut planes: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0u16; frames * n_bars]);
        let mut blk_col = vec![0usize; mod_plans.len()];
        let (mut dqb, mut dcxb) = (0u64, [0u64; 2]);
        let mut bars = 0usize;
        for (gi, gp) in plans.chunks(GROUP).enumerate() {
            if cancel.load(Ordering::Relaxed) {
                self.observe(Seat::GroupStart);
                return Err(HpError::Cancelled);
            }
            if gi % CBLOCK == 0 {
                let mut qb: Vec<f64> = Vec::new();
                for (k, p) in mod_plans.iter().enumerate().skip(gi).take(CBLOCK) {
                    blk_col[k] = qb.len() / hop;
                    for basis in &p.4 { for b in basis { qb.extend_from_slice(b); } }
                }
                let nblock = qb.len() / hop;
                // The previous block's buffers: the stream is idle here.
                if dqb != 0 { self.free(dqb)?; }
                for p in dcxb { if p != 0 { self.free(p)?; } }
                dqb = self.alloc(qb.len() * 8)?;
                self.upload(dqb, qb)?;
                for c in 0..2 {
                    dcxb[c] = self.alloc(rows * nblock * 8)?;
                    self.dgemm(false, rows, nblock, hop, dx64[c], hop, 1, 0, dqb, hop, dcxb[c], rows)?;
                }
            }
            lap!(2);
            // This group's wavelets (round 10: built here, dropped with the
            // group), the kernel pieces, then each carrier's coefficients (Φ)
            // and residual, exactly as `gpu_codes` builds them under the
            // modulated split.
            let built: Vec<Superlet> = pool.install(|| gp.par_iter().map(|p| Superlet::new(p.freq, p.q, cfg, sr as f32)).collect());
            let group = &built[..];
            let (s_g, q, mut h, _) = gemm_pieces(group, hop);
            let cols: Vec<(usize, usize)> = group.iter().enumerate().flat_map(|(b, sl)| (0..sl.wavelets.len()).map(move |m| (b, m))).collect();
            let n_cols = 2 * cols.len();
            let mut phi: Vec<f64> = Vec::new();
            let mut subs: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
            let (sub, _, _, _, bases) = &mod_plans[gi];
            let mut starts = vec![0usize];
            for sl in group { starts.push(starts.last().unwrap() + 2 * sl.wavelets.len()); }
            let mut ccol = blk_col[gi];
            for (k, (chunk, basis)) in gp.chunks(*sub).zip(bases).enumerate() {
                let (c0, c1, nbk) = (starts[k * sub], starts[k * sub + chunk.len()], basis.len());
                let per_col: Vec<(Vec<f64>, Vec<f32>)> = pool.install(|| (c0..c1).into_par_iter().map(|col| {
                    let (mut coef, mut res) = (vec![0f64; q * nbk], vec![0f32; q * hop]);
                    for p in 0..q {
                        let piece = &h[p * hop * n_cols + col * hop..][..hop];
                        let cm: Vec<f64> = basis.iter().map(|b| b.iter().zip(piece).map(|(a, &v)| a * v as f64).sum()).collect();
                        for (u, out) in res[p * hop..(p + 1) * hop].iter_mut().enumerate() {
                            let fit: f64 = basis.iter().zip(&cm).map(|(b, c)| c * b[u]).sum();
                            *out = (piece[u] as f64 - fit) as f32;
                        }
                        coef[p * nbk..(p + 1) * nbk].copy_from_slice(&cm);
                    }
                    (coef, res)
                }).collect());
                subs.push((c0, c1 - c0, nbk, phi.len(), ccol));
                ccol += nbk;
                for (j, (coef, res)) in per_col.into_iter().enumerate() {
                    phi.extend_from_slice(&coef);
                    for p in 0..q { h[p * hop * n_cols + (c0 + j) * hop..][..hop].copy_from_slice(&res[p * hop..(p + 1) * hop]); }
                }
            }
            lap!(3);
            let dh = self.alloc(h.len() * 4)?;
            self.upload(dh, h)?;
            let dphi = self.alloc(phi.len() * 8)?;
            self.upload(dphi, phi)?;
            let (mut dy, mut dz) = ([0u64; 2], [0u64; 2]);
            for c in 0..2 {
                dy[c] = self.alloc(frames * n_cols * 4)?;
                dz[c] = self.alloc(frames * n_cols * 8)?;
            }
            lap!(4);
            let row0 = (p0 - s_g) / hop;
            let n = frames * n_cols;
            for c in 0..2 {
                for &(c0, nc, nbk, poff, ccol) in &subs {
                    let kq = nbk * q;
                    self.dgemm(true, frames, nc, kq, dcxb[c] + ((ccol * rows + row0) * 8) as u64, rows, nbk, rows - row0,
                        dphi + (poff * 8) as u64, kq, dz[c] + (c0 * frames * 8) as u64, frames)?;
                }
                self.sgemm_v2(frames, n_cols, hop, q, dx[c] + (row0 * hop * 4) as u64, hop, hop, dh, hop, hop * n_cols, dy[c])?;
                let (acc, cvt) = (self.k[0], self.k[1]);
                self.elementwise(acc, dz[c], dy[c], n)?;
                self.elementwise(cvt, dy[c], dz[c], n)?;
            }
            lap!(5);
            if self.cancel_at_wait == Some(gi) { cancel.store(true, Ordering::Relaxed); }
            if self.wait(Some(cancel))? { return Err(HpError::Cancelled); }
            lap!(6);
            let y = [self.download::<f32>(dy[0], n)?, self.download::<f32>(dy[1], n)?];
            lap!(7);
            for p in [dh, dphi, dy[0], dz[0], dy[1], dz[1]] { self.free(p)?; }
            lap!(8);
            if cancel.load(Ordering::Relaxed) {
                self.observe(Seat::Fold);
                return Err(HpError::Cancelled);
            }
            // The host fold in f64 (production expressions), as with
            // MOOSIK_MR_FOLD64=1; each bar polls the flag before its work.
            let folded: Vec<Option<(usize, Cols)>> = pool.install(|| group.par_iter().enumerate().map(|(b, sl)| {
                if cancel.load(Ordering::Relaxed) { return None; }
                let mut acc = vec![[0f64; 3]; frames];
                for (m, (w, weight)) in sl.wavelets.iter().zip(sl.weights.iter()).enumerate() {
                    let ci = cols.iter().position(|&x| x == (b, m)).unwrap();
                    let col = |c: usize| -> Vec<C32> {
                        (0..frames).map(|j| {
                            let centre = (j * hop) as isize;
                            let interior = centre - w.half as isize >= 0 && centre + w.half as isize <= l.len() as isize - 1;
                            if interior {
                                C32 { re: (2.0 * y[c][(2 * ci) * frames + j] as f64) as f32, im: (2.0 * y[c][(2 * ci + 1) * frames + j] as f64) as f32 }
                            } else {
                                w.response_c(if c == 0 { l } else { r }, centre)
                            }
                        }).collect()
                    };
                    let (cl, cr) = (col(0), col(1));
                    for (a, (&lv, &rv)) in acc.iter_mut().zip(cl.iter().zip(&cr)) { fold64_member(a, lv, rv, *weight); }
                }
                let m: Vec<[f32; 3]> = acc.into_iter().map(|a| finish64(a, sl.total_weight)).collect();
                Some((gp[b].bar, Cols { mix: m.iter().map(|v| v[0]).collect(), left: m.iter().map(|v| v[1]).collect(), right: m.iter().map(|v| v[2]).collect() }))
            }).collect());
            if cancel.load(Ordering::Relaxed) || folded.iter().any(Option::is_none) {
                self.observe(Seat::Fold);
                return Err(HpError::Cancelled);
            }
            lap!(9);
            for (bar, c) in folded.iter().flatten() { put(&mut planes, frames, n_bars, *bar, c); }
            lap!(10);
            bars += group.len();
            if cancel.load(Ordering::Relaxed) {
                self.observe(Seat::Fold);
                return Err(HpError::Cancelled);
            }
            progress(bars);
        }
        Ok(planes)
    }

    /// The pass, pipelined (1.6.2 research). Each group's kernels, their
    /// order on the kernels' stream and each bar's arithmetic are the serial
    /// pass's; what moves is everything around them.
    ///
    /// - A preparer builds the next few groups' wavelets, pieces, Φ and
    ///   residuals, and a folder folds finished groups, both on the analysis
    ///   pool, under the same thread budget.
    /// - Copies run on a second stream: while group k's kernels run, group
    ///   k−1's result comes down and group k+1's pieces go up. So up to three
    ///   groups have buffers on the device at once (the estimate counts
    ///   them), and the host holds a bounded few prepared and unfolded groups.
    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, l: &[f32], r: &[f32], sr: u32, cfg: &AsltConfig, n_bars: usize, hop: usize, frames: usize, plans: &[Plan], mod_plans: &[GroupPlan],
           p0: usize, rows: usize, pool: &rayon::ThreadPool, cancel: &AtomicBool, progress: &(dyn Fn(usize) + Sync))
           -> Result<[Vec<u16>; 3], HpError> {
        let padded_len = rows * hop;
        let mut t = Instant::now();
        macro_rules! lap { ($i:expr) => { self.trace.stage_s[$i] += t.elapsed().as_secs_f64(); t = Instant::now(); } }
        let (mut dx, mut dx64) = ([0u64; 2], [0u64; 2]);
        for (c, sig) in [l, r].into_iter().enumerate() {
            let mut v = vec![0.0f32; padded_len];
            let n = sig.len().min(padded_len - p0);
            v[p0..p0 + n].copy_from_slice(&sig[..n]);
            let v64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            dx[c] = self.alloc(padded_len * 4)?;
            self.upload(dx[c], v)?;
            dx64[c] = self.alloc(padded_len * 8)?;
            self.upload(dx64[c], v64)?;
        }
        lap!(1);
        // Each group's first coefficient column within its block: what the
        // serial pass counts as it fills the block's basis.
        let mut blk_col = vec![0usize; mod_plans.len()];
        for blk in (0..mod_plans.len()).step_by(CBLOCK) {
            let mut n = 0usize;
            for (k, p) in mod_plans.iter().enumerate().skip(blk).take(CBLOCK) {
                blk_col[k] = n;
                n += p.4.iter().map(|b| b.len()).sum::<usize>();
            }
        }
        let blk_col = &blk_col;
        let faults = self.faults;
        let (prep_tx, prep_rx) = std::sync::mpsc::sync_channel::<Prepared>(1);
        let (fold_tx, fold_rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Superlet>, Vec<(usize, usize)>, [Vec<f32>; 2])>(1);
        std::thread::scope(|scope| {
            // A helper the system will not start is a refusal like any other
            // (the cores take the track): nothing runs yet, and the buffers
            // above are freed by the pass's cleanup.
            let preparer = helper(scope, "moosik-hp-prepare", faults.helper == Some(1), move || {
                // A few groups at a time, each on the pool (the helper thread
                // itself only waits), handed on in order: the device, which
                // no longer idles through its copies, would otherwise wait
                // for one group's preparation after another.
                let groups: Vec<(usize, &[Plan])> = plans.chunks(GROUP).enumerate().collect();
                'all: for batch in groups.chunks(PREPARE_AHEAD) {
                    if cancel.load(Ordering::Relaxed) { break; }
                    if let Some(&(gi, _)) = batch.iter().find(|g| faults.prepare_stop == Some(g.0)) { panic!("injected: the preparer stops at group {gi}"); }
                    let prepared: Vec<Prepared> = pool.install(|| batch.par_iter()
                        .map(|&(gi, gp)| prepare_group(gp, &mod_plans[gi], blk_col[gi], cfg, sr, hop, pool)).collect());
                    for p in prepared {
                        if prep_tx.send(p).is_err() { break 'all; }
                    }
                }
            })?;
            let folder = helper(scope, "moosik-hp-fold", faults.helper == Some(2), move || {
                let mut planes: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0u16; frames * n_bars]);
                let (mut bars, mut fold_s, mut put_s) = (0usize, 0f64, 0f64);
                for (gi, group, cols, y) in fold_rx {
                    if faults.fold_stop == Some(gi) { panic!("injected: the folder stops at group {gi}"); }
                    let t = Instant::now();
                    let gp = &plans[gi * GROUP..(gi * GROUP + group.len()).min(plans.len())];
                    let folded = fold_group(&group, gp, &cols, &y, l, r, hop, frames, pool, cancel);
                    fold_s += t.elapsed().as_secs_f64();
                    if cancel.load(Ordering::Relaxed) || folded.iter().any(Option::is_none) {
                        return (planes, true, fold_s, put_s);
                    }
                    let t = Instant::now();
                    pool.install(|| {
                        #[cfg(test)]
                        let _at_work = probe::enter();
                        for (bar, codes) in folded.iter().flatten() {
                            for (plane, col) in planes.iter_mut().zip(codes) {
                                for (f, &c) in col.iter().enumerate().take(frames) { plane[f * n_bars + bar] = c; }
                            }
                        }
                    });
                    put_s += t.elapsed().as_secs_f64();
                    bars += group.len();
                    if cancel.load(Ordering::Relaxed) { return (planes, true, fold_s, put_s); }
                    progress(bars);
                }
                (planes, false, fold_s, put_s)
            });
            // The preparer already runs: with its channel closed it ends at
            // its next send, and is joined before the refusal is returned.
            let folder = match folder {
                Ok(folder) => folder,
                Err(refused) => {
                    drop(prep_rx);
                    let _ = preparer.join();
                    return Err(refused);
                }
            };
            let device = (|| -> Result<(), HpError> {
                let (mut dqb, mut dcxb) = (0u64, [0u64; 2]);
                let n_groups = plans.len().div_ceil(GROUP);
                // The group whose pieces are on the device (or on their way),
                // and the group whose kernels ran last and whose result is
                // still there. Copies run on their own stream, so a group's
                // kernels overlap the download before it and the upload after.
                let mut staged: Option<Staged> = None;
                let mut flown: Option<Flown> = None;
                // The preparer stops early for a cancellation; otherwise it
                // has failed, which is not the caller's cancellation.
                macro_rules! prepared {
                    () => {
                        match prep_rx.recv() {
                            Ok(p) => p,
                            Err(_) if !cancel.load(Ordering::Relaxed) => return Err(self.poison("the preparing helper stopped".into())),
                            Err(_) => {
                                self.observe(Seat::GroupStart);
                                return Err(HpError::Cancelled);
                            }
                        }
                    };
                }
                // A downloaded result out of custody, its group's buffers
                // freed, and on to the folder, which stops early for a
                // cancellation; otherwise it has failed.
                macro_rules! hand_over {
                    ($f:expr) => {{
                        let f: Flown = $f;
                        let y1 = self.take_download::<f32>();
                        let y0 = self.take_download::<f32>();
                        lap!(7);
                        for p in f.bufs { self.free(p)?; }
                        lap!(8);
                        let sent = cancel.load(Ordering::Relaxed) || fold_tx.send((f.gi, f.group, f.cols, [y0, y1])).is_ok();
                        if !sent && !cancel.load(Ordering::Relaxed) { return Err(self.poison("the folding helper stopped".into())); }
                        if !sent || cancel.load(Ordering::Relaxed) {
                            self.observe(Seat::Fold);
                            return Err(HpError::Cancelled);
                        }
                        lap!(11);
                    }};
                }
                for gi in 0..n_groups {
                    if cancel.load(Ordering::Relaxed) {
                        self.observe(Seat::GroupStart);
                        return Err(HpError::Cancelled);
                    }
                    if gi % CBLOCK == 0 {
                        let mut qb: Vec<f64> = Vec::new();
                        for p in mod_plans.iter().skip(gi).take(CBLOCK) {
                            for basis in &p.4 { for b in basis { qb.extend_from_slice(b); } }
                        }
                        let nblock = qb.len() / hop;
                        // The previous block's buffers: the kernels' stream is
                        // idle here (the last group's wait is over).
                        if dqb != 0 { self.free(dqb)?; }
                        for p in dcxb { if p != 0 { self.free(p)?; } }
                        dqb = self.alloc(qb.len() * 8)?;
                        self.upload(dqb, qb)?;
                        for c in 0..2 {
                            dcxb[c] = self.alloc(rows * nblock * 8)?;
                            self.dgemm(false, rows, nblock, hop, dx64[c], hop, 1, 0, dqb, hop, dcxb[c], rows)?;
                        }
                    }
                    lap!(2);
                    // Only the first group is staged here; every other was
                    // while the group before it ran.
                    let Staged { group, s_g, q, cols, subs, tiles, dh, dphi } = match staged.take() {
                        Some(s) => s,
                        None => {
                            let p = prepared!();
                            lap!(11);
                            let s = self.stage(p)?;
                            self.copies_done(2)?;
                            self.ledger.host.truncate(self.ledger.host.len() - 2);
                            s
                        }
                    };
                    let n_cols = 2 * cols.len();
                    let (mut dy, mut dz) = ([0u64; 2], [0u64; 2]);
                    for c in 0..2 {
                        dy[c] = self.alloc(frames * n_cols * 4)?;
                        dz[c] = self.alloc(frames * n_cols * 8)?;
                    }
                    lap!(4);
                    let row0 = (p0 - s_g) / hop;
                    let n = frames * n_cols;
                    for c in 0..2 {
                        for &(c0, nc, nbk, poff, ccol) in &subs {
                            let kq = nbk * q;
                            self.dgemm(true, frames, nc, kq, dcxb[c] + ((ccol * rows + row0) * 8) as u64, rows, nbk, rows - row0,
                                dphi + (poff * 8) as u64, kq, dz[c] + (c0 * frames * 8) as u64, frames)?;
                        }
                        if let Some(ranges) = aligned_ranges(n_cols, q, &tiles) {
                            self.sgemm_v2_ranged(frames, n_cols, hop, q, dx[c] + (row0 * hop * 4) as u64, dh, dy[c], ranges)?;
                        } else {
                            for &(c0, nc, lo, hi) in &tiles {
                                self.sgemm_v2(frames, nc, hop, hi - lo, dx[c] + ((row0 + lo) * hop * 4) as u64, hop, hop,
                                    dh + ((lo * n_cols + c0) * hop * 4) as u64, hop, hop * n_cols, dy[c] + (c0 * frames * 4) as u64)?;
                            }
                        }
                        let (acc, cvt) = (self.k[0], self.k[1]);
                        self.elementwise(acc, dz[c], dy[c], n)?;
                        self.elementwise(cvt, dy[c], dz[c], n)?;
                    }
                    lap!(5);
                    // While those run: the next group's pieces up and the
                    // previous group's result down, on the copies' stream.
                    let next = if gi + 1 < n_groups { Some(prepared!()) } else { None };
                    lap!(11);
                    let mut copies = 0;
                    if let Some(f) = &flown {
                        self.download_ahead::<f32>(f.dy[0], f.n)?;
                        self.download_ahead::<f32>(f.dy[1], f.n)?;
                        copies += 2;
                    }
                    if let Some(p) = next {
                        staged = Some(self.stage(p)?);
                        copies += 2;
                    }
                    if copies > 0 { self.copies_done(copies)?; }
                    // The uploads' buffers were held last.
                    if staged.is_some() { self.ledger.host.truncate(self.ledger.host.len() - 2); }
                    lap!(4);
                    if let Some(f) = flown.take() { hand_over!(f); }
                    flown = Some(Flown { gi, group, cols, n, dy, bufs: [dh, dphi, dy[0], dz[0], dy[1], dz[1]] });
                    if self.cancel_at_wait == Some(gi) { cancel.store(true, Ordering::Relaxed); }
                    if self.wait_stream(self.stream, Some(cancel))? { return Err(HpError::Cancelled); }
                    lap!(6);
                }
                // The last group's result.
                if let Some(f) = flown.take() {
                    self.download_ahead::<f32>(f.dy[0], f.n)?;
                    self.download_ahead::<f32>(f.dy[1], f.n)?;
                    self.copies_done(2)?;
                    hand_over!(f);
                }
                Ok(())
            })();
            // Both helpers end once their channels close.
            drop(prep_rx);
            drop(fold_tx);
            let (prepared, folded) = (preparer.join(), folder.join());
            lap!(11);
            device?;
            // A helper that stopped by itself (a panic, in a build that
            // unwinds) is a failure of the pass, after its cleanup, not a
            // result: the engine is not used again. Only when the device
            // loop itself ended well: a cancellation, refusal or device
            // failure it returned comes first (above), and the helper's stop
            // is then not recorded. The release build aborts on a panic, so
            // this is the checks' contract, not a recovery the player has.
            let (Ok(()), Ok((planes, cancelled, fold_s, put_s))) = (prepared, folded) else {
                return Err(self.poison("a helper thread stopped".into()));
            };
            self.trace.stage_s[9] += fold_s;
            self.trace.stage_s[10] += put_s;
            if cancelled || cancel.load(Ordering::Relaxed) {
                self.observe(Seat::Fold);
                return Err(HpError::Cancelled);
            }
            Ok(planes)
        })
    }
}

/// A named helper thread of the pipelined pass, or the refusal when the
/// system will not start one (`refuse`: the checks' stand-in for that).
fn helper<'scope, T: Send + 'scope>(scope: &'scope std::thread::Scope<'scope, '_>, name: &str, refuse: bool,
                                    work: impl FnOnce() -> T + Send + 'scope) -> Result<std::thread::ScopedJoinHandle<'scope, T>, HpError> {
    let started = if refuse { Err(std::io::Error::other("injected")) } else { std::thread::Builder::new().name(name.into()).spawn_scoped(scope, work) };
    started.map_err(|e| HpError::Unavailable(format!("no thread for {name}: {e}")))
}

/// Checks only: how many threads were in the pass's host work at once, and
/// which. A thread that takes up a second piece while its first waits on the
/// pool (a fold stolen inside a preparation) is still one thread at work.
#[cfg(test)]
mod probe {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK: AtomicUsize = AtomicUsize::new(0);
    pub static THREADS: std::sync::Mutex<Vec<(String, usize)>> = std::sync::Mutex::new(Vec::new());
    thread_local! { static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
    pub struct AtWork;
    pub fn enter() -> AtWork {
        if DEPTH.replace(DEPTH.get() + 1) == 0 { PEAK.fetch_max(ACTIVE.fetch_add(1, Relaxed) + 1, Relaxed); }
        THREADS.lock().unwrap().push((std::thread::current().name().unwrap_or("").to_string(), rayon::current_num_threads()));
        AtWork
    }
    impl Drop for AtWork {
        fn drop(&mut self) {
            if DEPTH.replace(DEPTH.get() - 1) == 1 { ACTIVE.fetch_sub(1, Relaxed); }
        }
    }
}

/// A group whose pieces and coefficients are on the device (or queued to be).
struct Staged {
    group: Vec<Superlet>,
    s_g: usize,
    q: usize,
    cols: Vec<(usize, usize)>,
    subs: Vec<(usize, usize, usize, usize, usize)>,
    tiles: Vec<Tile>,
    dh: u64,
    dphi: u64,
}

/// A group whose kernels are queued or done: where its result is, and all
/// its device buffers.
struct Flown {
    gi: usize,
    group: Vec<Superlet>,
    cols: Vec<(usize, usize)>,
    n: usize,
    dy: [u64; 2],
    bufs: [u64; 6],
}

/// One group, ready for the device: its wavelets, the kernel pieces with each
/// carrier's fit taken out (the residual), the fits' coefficients (Φ), and
/// where each sub-group's columns and coefficients sit. Exactly what the
/// serial pass builds in place.
struct Prepared {
    group: Vec<Superlet>,
    s_g: usize,
    q: usize,
    h: Vec<f32>,
    cols: Vec<(usize, usize)>,
    phi: Vec<f64>,
    subs: Vec<(usize, usize, usize, usize, usize)>,
    tiles: Vec<Tile>,
    secs: f64,
}

/// A run of a group's columns the residual SGEMM takes in one launch, and the
/// kernel pieces that launch runs: (first column, columns, first piece, one
/// past the last piece). Outside those pieces every one of its columns is
/// zero.
type Tile = (usize, usize, usize, usize);

/// Column pairs in one SGEMM block (72 columns).
const TILE_PAIRS: usize = 36;

/// The pipelined pass's column layout for a group: which (bar, member) each
/// column pair holds, and the tiles. The kernel's block is 72 columns wide
/// and costs the same however many of them are in use, and a group's short
/// wavelets are zero in most of the pieces its longest one needs. So the
/// pairs are dealt into tiles by wavelet length, the shortest (and any odd
/// remainder) first, and each tile is launched over its own pieces only.
/// Leaving out a piece that is zero for a column leaves that column's sums
/// as they are: a zero piece adds +0 to a sum that is never −0.
///
/// Inside a tile the pairs stay in (bar, member) order, so that the columns
/// of consecutive bars, which share a carrier, are consecutive there too.
/// The fold finds its columns through the returned list, in any order.
fn tile_layout(group: &[Superlet], hop: usize) -> (Vec<(usize, usize)>, Vec<Tile>) {
    let hmax = group.iter().flat_map(|sl| sl.wavelets.iter().map(|w| w.half)).max().unwrap();
    let s_g = hmax.div_ceil(hop) * hop;
    let mut by_len: Vec<(usize, usize)> = group.iter().enumerate().flat_map(|(b, sl)| (0..sl.wavelets.len()).map(move |m| (b, m))).collect();
    by_len.sort_by_key(|&(b, m)| group[b].wavelets[m].half);
    let (mut cols, mut tiles) = (Vec::with_capacity(by_len.len()), Vec::new());
    let mut rest = &by_len[..];
    while !rest.is_empty() {
        let n = if rest.len() % TILE_PAIRS == 0 { TILE_PAIRS } else { rest.len() % TILE_PAIRS };
        let (tile, later) = rest.split_at(n);
        rest = later;
        let mut tile = tile.to_vec();
        tile.sort();
        let span = |&(b, m): &(usize, usize)| { let w = &group[b].wavelets[m]; ((s_g - w.half) / hop, (s_g - w.half + w.re.len() - 1) / hop + 1) };
        let lo = tile.iter().map(|c| span(c).0).min().unwrap();
        let hi = tile.iter().map(|c| span(c).1).max().unwrap();
        tiles.push((2 * cols.len(), 2 * tile.len(), lo, hi));
        cols.extend(tile);
    }
    (cols, tiles)
}

/// The tiles' piece ranges as `sgemm_v2_ranged` takes them, when the tiles
/// are the grid's own 72-column blocks (up to four), each over pieces; or
/// None, and a launch per tile. Nine members fill two blocks; five put a
/// short tile first, which no block lines up with.
fn aligned_ranges(n: usize, q: usize, tiles: &[Tile]) -> Option<[(usize, usize); 4]> {
    if n == 0 || q == 0 || n.div_ceil(72) > 4 || tiles.len() != n.div_ceil(72) { return None; }
    let mut ranges = [(0, q); 4];
    for (y, &(c0, nc, lo, hi)) in tiles.iter().enumerate() {
        if c0 != 72 * y || nc != 72.min(n - c0) || lo >= hi || hi > q { return None; }
        ranges[y] = (lo, hi);
    }
    Some(ranges)
}

fn prepare_group(gp: &[Plan], plan: &GroupPlan, first_col: usize, cfg: &AsltConfig, sr: u32, hop: usize, pool: &rayon::ThreadPool) -> Prepared {
    #[cfg(test)]
    let _at_work = probe::enter();
    let t = Instant::now();
    let group: Vec<Superlet> = pool.install(|| gp.par_iter().map(|p| Superlet::new(p.freq, p.q, cfg, sr as f32)).collect());
    let (cols, tiles) = tile_layout(&group, hop);
    let (s_g, q, mut h, _) = gemm_pieces_at(&group, hop, Some(&cols));
    let n_cols = 2 * cols.len();
    let mut phi: Vec<f64> = Vec::new();
    let mut subs: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    let (sub, _, _, _, bases) = plan;
    // Each carrier's first coefficient column in the block.
    let mut ccols = vec![first_col];
    for basis in bases { ccols.push(ccols.last().unwrap() + basis.len()); }
    // A carrier's columns are consecutive inside a tile: one run per tile
    // and carrier, each with its own coefficients.
    let runs: Vec<(usize, usize, usize)> = tiles.iter().flat_map(|&(t0, tn, _, _)| {
        let cols = &cols;
        (0..bases.len()).filter_map(move |k| {
            let of_k = |pair: usize| cols[pair].0 / *sub == k;
            let first = (t0 / 2..(t0 + tn) / 2).find(|&pair| of_k(pair))?;
            let n = (first..(t0 + tn) / 2).take_while(|&pair| of_k(pair)).count();
            Some((k, 2 * first, 2 * (first + n)))
        })
    }).collect();
    for (k, c0, c1) in runs {
        let (basis, ccol) = (&bases[k], ccols[k]);
        let nbk = basis.len();
        let per_col: Vec<(Vec<f64>, Vec<f32>)> = pool.install(|| (c0..c1).into_par_iter().map(|col| {
            let (mut coef, mut res) = (vec![0f64; q * nbk], vec![0f32; q * hop]);
            for p in 0..q {
                let piece = &h[p * hop * n_cols + col * hop..][..hop];
                let cm: Vec<f64> = basis.iter().map(|b| b.iter().zip(piece).map(|(a, &v)| a * v as f64).sum()).collect();
                for (u, out) in res[p * hop..(p + 1) * hop].iter_mut().enumerate() {
                    let fit: f64 = basis.iter().zip(&cm).map(|(b, c)| c * b[u]).sum();
                    *out = (piece[u] as f64 - fit) as f32;
                }
                coef[p * nbk..(p + 1) * nbk].copy_from_slice(&cm);
            }
            (coef, res)
        }).collect());
        subs.push((c0, c1 - c0, nbk, phi.len(), ccol));
        for (j, (coef, res)) in per_col.into_iter().enumerate() {
            phi.extend_from_slice(&coef);
            for p in 0..q { h[p * hop * n_cols + (c0 + j) * hop..][..hop].copy_from_slice(&res[p * hop..(p + 1) * hop]); }
        }
    }
    Prepared { group, s_g, q, h, cols, phi, subs, tiles, secs: t.elapsed().as_secs_f64() }
}

/// One group's host fold in f64 (production expressions), each bar on the
/// pool: its Mix, L and R codes, or `None` once the flag is seen. The codes
/// are `aslt_code` of the folded magnitudes, as `put` writes them.
#[allow(clippy::too_many_arguments)]
fn fold_group(group: &[Superlet], gp: &[Plan], cols: &[(usize, usize)], y: &[Vec<f32>; 2], l: &[f32], r: &[f32], hop: usize,
              frames: usize, pool: &rayon::ThreadPool, cancel: &AtomicBool) -> Vec<Option<(usize, [Vec<u16>; 3])>> {
    pool.install(|| group.par_iter().enumerate().map(|(b, sl)| {
        if cancel.load(Ordering::Relaxed) { return None; }
        #[cfg(test)]
        let _at_work = probe::enter();
        let mut acc = vec![[0f64; 3]; frames];
        for (m, (w, weight)) in sl.wavelets.iter().zip(sl.weights.iter()).enumerate() {
            let ci = cols.iter().position(|&x| x == (b, m)).unwrap();
            let col = |c: usize| -> Vec<C32> {
                (0..frames).map(|j| {
                    let centre = (j * hop) as isize;
                    let interior = centre - w.half as isize >= 0 && centre + w.half as isize <= l.len() as isize - 1;
                    if interior {
                        C32 { re: (2.0 * y[c][(2 * ci) * frames + j] as f64) as f32, im: (2.0 * y[c][(2 * ci + 1) * frames + j] as f64) as f32 }
                    } else {
                        w.response_c(if c == 0 { l } else { r }, centre)
                    }
                }).collect()
            };
            let (cl, cr) = (col(0), col(1));
            for (a, (&lv, &rv)) in acc.iter_mut().zip(cl.iter().zip(&cr)) { fold64_member(a, lv, rv, *weight); }
        }
        let m: Vec<[f32; 3]> = acc.into_iter().map(|a| finish64(a, sl.total_weight)).collect();
        Some((gp[b].bar, std::array::from_fn(|p| m.iter().map(|v| crate::spectrum::aslt_code(v[p])).collect())))
    }).collect())
}

/// One High-accuracy GPU analysis of a stereo pair, on the calling thread.
///
/// An engine is made here, used once and closed here, so its private context
/// never outlives the call or leaves this thread; afterwards the thread is
/// checked for a current context and the log says if one is left (the worker
/// then ends with its job, and nothing else runs on it). Host work runs on
/// the analysis pool, under the user's thread budget, and records nothing for
/// Standard's calibration. Returns the Mix, L and R codes (frame-major) and
/// the frame count, or why not. `progress` receives the bars folded so far.
#[allow(clippy::too_many_arguments)]
pub fn analyze_on_this_thread(
    l: &[f32],
    r: &[f32],
    sr: u32,
    n_bars: usize,
    min_freq: f32,
    max_freq: f32,
    cfg: &AsltConfig,
    fps: f32,
    cancel: &AtomicBool,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<([Vec<u16>; 3], usize), HpError> {
    let _ = take_device_failure();
    if cancel.load(Ordering::Relaxed) { return Err(HpError::Cancelled); }
    admits(n_bars, min_freq, max_freq, cfg, fps).map_err(HpError::Unavailable)?;
    // Rate, stereo geometry and length are known now. Refuse unsupported
    // input before loading the driver or creating a device context.
    let frames = admitted_frames(l.len(), r.len(), sr, fps, n_bars)?;
    spread_admits(sr, cfg).map_err(HpError::Unavailable)?;
    let geom = Geom { n_bars, lo: min_freq, hi: max_freq, fps };
    let pool = super::super::gpu_calib::pool().ok_or_else(|| HpError::Unavailable("no analysis worker pool".into()))?;
    let t = Instant::now();
    let mut e = match HpEngine::new(faults()) {
        Ok(e) => e,
        Err(err) => {
            // new() already inspected any partial initialization's close.
            // A missing driver is an ordinary refusal; do not reopen it just
            // to query a context the engine never created.
            let got = finish_product_call(Err(err), false, &CloseReport::default(), &Ok(true), cancel.load(Ordering::Relaxed), None);
            crate::mlog!("[hp] initialization result: {got:?}; this worker retires without another job");
            if cancel.load(Ordering::Relaxed) { return Err(HpError::Cancelled); }
            return got;
        }
    };
    let got = e.analyze(l, r, sr, &geom, cfg, &pool, cancel, progress);
    let (jit, seen, failures, quarantine) = (e.jit_s, e.seen(), std::mem::take(&mut e.failures), e.quarantined().map(str::to_owned));
    let stages = e.trace.stage_s;
    let rep = e.close();
    let clean = no_current_context();
    crate::mlog!("[hp] stages: {}", STAGES.iter().zip(stages).map(|(n, s)| format!("{n} {s:.2}s")).collect::<Vec<_>>().join(", "));
    let got = finish_product_call(got, true, &rep, &clean, cancel.load(Ordering::Relaxed), quarantine.as_deref());
    crate::mlog!(
        "[hp] {} after {:.1}s (JIT {:.2}s, {} launches, {} copies); close: {}{}",
        match &got { Ok(_) => "done".to_string(), Err(e) => format!("{e:?}") },
        t.elapsed().as_secs_f64(), jit, seen.launch, seen.copy,
        if rep.context_destroyed { "context destroyed" } else { "context NOT destroyed" },
        if rep.failures.is_empty() { String::new() } else { format!(", failures {:?}", rep.failures) },
    );
    if !failures.is_empty() {
        crate::mlog!("[hp] also failed in that pass: {failures:?}");
    }
    if !matches!(clean, Ok(true)) {
        crate::mlog!("[hp] thread context was not confirmed clear ({clean:?}); this worker retires without another job");
    }
    if cancel.load(Ordering::Relaxed) { return Err(HpError::Cancelled); }
    let planes = got?;
    Ok((planes, frames))
}

/// The worker reports health even when the job's cancelled result hides the
/// same device error. This is thread local, not a process-wide retry flag.
fn finish_product_call<R>(
    got: Result<R, HpError>,
    had_context: bool,
    report: &CloseReport,
    context_clear: &Result<bool, HpError>,
    cancelled: bool,
    quarantine: Option<&str>,
) -> Result<R, HpError> {
    let device_error = quarantine.map(str::to_owned).or_else(|| match &got {
        Err(HpError::Device(reason)) => Some(reason.clone()),
        _ => None,
    });
    let health = finish_closed(device_error.map_or(Ok(()), |reason| Err(HpError::Device(reason))),
                               had_context, report, context_clear, false);
    let mut got = finish_closed(got, had_context, report, context_clear, false);
    if let Err(HpError::Device(reason)) = health {
        LAST_DEVICE_FAILURE.with(|slot| *slot.borrow_mut() = Some(reason.clone()));
        if !matches!(&got, Err(HpError::Cancelled) | Err(HpError::Device(_))) { got = Err(HpError::Device(reason)); }
    }
    if cancelled { Err(HpError::Cancelled) } else { got }
}

/// The product publication boundary includes engine destruction and the
/// calling thread's context check, not just the analysis pass. A cancellation
/// still wins for the caller; its cleanup errors are logged independently.
fn finish_closed<R>(
    got: Result<R, HpError>,
    had_context: bool,
    report: &CloseReport,
    context_clear: &Result<bool, HpError>,
    cancelled: bool,
) -> Result<R, HpError> {
    if cancelled || matches!(&got, Err(HpError::Cancelled)) { return Err(HpError::Cancelled); }
    let mut failed = report.failures.clone();
    if had_context && !report.context_destroyed { failed.push("private context was not destroyed".into()); }
    match context_clear {
        Ok(true) => {},
        Ok(false) => failed.push("this worker still has a current context".into()),
        Err(err) => failed.push(format!("thread context could not be checked: {err:?}")),
    }
    if failed.is_empty() { return got; }
    if let Err(err) = &got { failed.push(format!("analysis also returned {err:?}")); }
    Err(HpError::Device(failed.join("; ")))
}

/// The sample rates the route takes: the two of the 1.6.2 gate, and the
/// high-resolution ones gated the same way since.
fn rate_admitted(sample_rate: u32) -> bool {
    matches!(sample_rate, 44_100 | 48_000 | 88_200 | 96_000 | 176_400 | 192_000)
}

/// The least spread the route was measured faster than Standard with at
/// 44.1 and 48 kHz (q 0.25 to 2). Spread 0 runs a set's shortest members
/// down to 3 cycles, shorter than a hop at most bars; no carrier basis holds
/// them, so about two groups in three go to the f64 products, and the pass
/// was up to 1.9 times slower than Standard. From 0.1 up it was 1.6 to 3.7
/// times faster.
const MIN_SPREAD: f32 = 0.1;

/// The same above 48 kHz, where a hop is longer and an f64 group costs more:
/// at 192 kHz, 0.2 was barely faster at q 0.25 (1.07 times) and spread 0 up to
/// five times slower; from 0.3 up, 1.9 to 8 times faster.
const HI_RES_MIN_SPREAD: f32 = 0.3;

/// Only the spreads the route is faster with ([`MIN_SPREAD`],
/// [`HI_RES_MIN_SPREAD`]). One wavelet has no shorter members, so any spread.
fn spread_admits(sample_rate: u32, cfg: &AsltConfig) -> Result<(), String> {
    let least = if sample_rate > 48_000 { HI_RES_MIN_SPREAD } else { MIN_SPREAD };
    if cfg.n_wavelets > 1 && !(cfg.spread >= least) {
        return Err(format!("at {} kHz it is no faster than Standard with a spread under {least}",
                           sample_rate as f64 / 1000.0));
    }
    Ok(())
}

/// Decoded-input admission, independent of every driver/device operation.
fn admitted_frames(left_len: usize, right_len: usize, sample_rate: u32, fps: f32, n_bars: usize) -> Result<usize, HpError> {
    if !rate_admitted(sample_rate) {
        return Err(HpError::Unavailable(format!("{sample_rate} Hz is outside the tested envelope")));
    }
    if left_len != right_len { return Err(HpError::Unavailable("the channels differ in length".into())); }
    let frames = frame_count(left_len, hop_for_fps(sample_rate, fps));
    if frames == 0 || frames.saturating_mul(n_bars) > CELL_CAP {
        let longest = CELL_CAP / n_bars.max(1) / (fps.max(1.0) as usize).max(1);
        return Err(HpError::Unavailable(format!("at these settings it takes tracks up to {}:{:02}", longest / 60, longest % 60)));
    }
    Ok(frames)
}

/// The faults the next engine injects: none, outside the checks.
#[cfg(not(test))]
fn faults() -> Faults {
    Faults::default()
}

/// The faults the next engine made on this thread injects (checks only).
#[cfg(test)]
fn faults() -> Faults {
    NEXT_FAULTS.with(|f| f.take()).unwrap_or_default()
}

#[cfg(test)]
thread_local! {
    /// Set by a check, on the worker thread that will make the engine.
    pub(crate) static NEXT_FAULTS: std::cell::Cell<Option<Faults>> = const { std::cell::Cell::new(None) };
}

/// Whether the calling thread has no current context (after an engine is closed, and in the checks).
pub fn no_current_context() -> Result<bool, HpError> {
    let d = Drv::open().map_err(|err| HpError::Device(format!("context check could not open the driver: {err:?}")))?;
    let mut c: Ptr = std::ptr::null_mut();
    // SAFETY: a valid out-pointer; the library reference is released once.
    unsafe {
        let rc = (d.current)(&mut c);
        FreeLibrary(d.lib);
        if rc == 0 { Ok(c.is_null()) } else { Err(HpError::Device(format!("cuCtxGetCurrent failed with {rc}"))) }
    }
}


// ---------------------------------------------------------------------------
// The validated envelope
// ---------------------------------------------------------------------------

/// The identity revision of this route's numerics. Bump it whenever they
/// change: τ, the option list, the basis, the fold, the GEMM kernels or their
/// reduction order, or any tiling or batching that changes arithmetic.
pub const REVISION: &str = "gpu-hp1";

/// The display geometry of an analysis: bars, range and frames per second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    pub n_bars: usize,
    pub lo: f32,
    pub hi: f32,
    pub fps: f32,
}

impl Geom {
    /// The geometry rounds 3–11 validated.
    pub const fn validated() -> Geom { Geom { n_bars: N_BARS, lo: LO, hi: HI, fps: FPS } }
}

const N_BARS: usize = 1024;
const LO: f32 = 20.0;
const HI: f32 = 24000.0;
/// The validated frame rate.
const FPS: f32 = 180.0;

/// The validated transform settings.
pub fn validated_cfg() -> AsltConfig {
    AsltConfig {
        q_ratio: 2.0,
        max_window_s: 0.25,
        n_wavelets: 9,
        spread: 0.5,
        scale: crate::spectrum::freq_scale::FreqScale::Blend(0.5),
    }
}

/// Whether the descriptor is the validated one, on the host, before anything
/// is decoded or any device is touched. The rate, the channels and the length
/// are checked again by the engine once the audio is decoded.
/// Whether these settings take the High-accuracy GPU route, or why not.
///
/// The engine itself runs any Superlet settings and decides per group how to
/// stay accurate (a group no carrier basis fits goes through the f64 products
/// whole). What is admitted is the region where the 1.6.2 sweeps found it
/// both within one code of the direct reference and clearly faster than the
/// cores: enough bars and frames a second that its fixed costs pay off, and
/// windows short enough that the kernels stay small.
pub fn admits(n_bars: usize, min_freq: f32, max_freq: f32, cfg: &AsltConfig, fps: f32) -> Result<(), String> {
    if !(MIN_BARS..=2048).contains(&n_bars) {
        return Err(format!("needs {MIN_BARS} bars or more"));
    }
    if !(MIN_FPS..=480.0).contains(&fps) {
        return Err(format!("needs {MIN_FPS} frames per second or more"));
    }
    if !(cfg.max_window_s <= MAX_WINDOW_S) {
        return Err(format!("is no faster with a window over {MAX_WINDOW_S} s"));
    }
    if !(min_freq > 0.0 && max_freq > min_freq) {
        return Err("needs a frequency range".into());
    }
    Ok(())
}

/// The admitted region's edges (see [`admits`]).
pub const MIN_BARS: usize = 256;
pub const MIN_FPS: f32 = 100.0;
pub const MAX_WINDOW_S: f32 = 1.2;

struct Plan { bar: usize, freq: f32, q: f32 }

fn plans(sr: u32, geom: &Geom, cfg: &AsltConfig) -> Vec<Plan> {
    let nyquist = sr as f32 * 0.5;
    // The same bar grid as Standard: the range's top clamped to Nyquist.
    let (n, lo, a_max) = (geom.n_bars, geom.lo, nyquist.min(geom.hi));
    (0..n).filter_map(|bar| {
        let f = cfg.scale.bar_center(bar, n, lo, a_max);
        (f < nyquist).then(|| Plan { bar, freq: f, q: effective_q_at(f, cfg.scale.grid_q_at(bar, n, lo, a_max), cfg) })
    }).collect()
}

#[cfg(test)]
fn put(planes: &mut [Vec<u16>; 3], frames: usize, n_bars: usize, bar: usize, cols: &Cols) {
    for (p, col) in [&cols.mix, &cols.left, &cols.right].into_iter().enumerate() {
        for (f, &m) in col.iter().enumerate().take(frames) {
            planes[p][f * n_bars + bar] = crate::spectrum::aslt_code(m);
        }
    }
}

/// One member into the f64 fold (`MOOSIK_MR_FOLD64`): `mix_norms` and `norms`
/// on the f32 responses evaluated in f64 (squares of f32 values cannot
/// overflow or underflow there), floored, logged and weighted in f64.
fn fold64_member(acc: &mut [f64; 3], l: C32, r: C32, weight: f32) {
    let (lr, li, rr, ri) = (l.re as f64, l.im as f64, r.re as f64, r.im as f64);
    let (mr, mi) = (0.5 * (lr + rr), 0.5 * (li + ri));
    let mags = [(mr * mr + mi * mi).sqrt(), (lr * lr + li * li).sqrt(), (rr * rr + ri * ri).sqrt()];
    for p in 0..3 { acc[p] += weight as f64 * mags[p].max(MAG_FLOOR as f64).ln(); }
}

/// The f64 fold's mean and exponential, rounded once to f32 for the mapper.
fn finish64(acc: [f64; 3], total_weight: f32) -> [f32; 3] {
    acc.map(|a| (a / total_weight as f64).exp() as f32)
}

/// Bars per GEMM group: consecutive bars share one window geometry.
const GROUP: usize = 8;
/// Groups per coefficient DGEMM in the modulated split (a few wide products
/// instead of many thin ones).
const CBLOCK: usize = 8;
/// Groups the pipelined pass prepares at once on the pool. The host holds at
/// most this many prepared groups, and one more in the channel.
const PREPARE_AHEAD: usize = 4;

/// An orthonormal basis of one hop row, f64, one `hop`-long vector per column:
/// s^k for k = 0..=degree, s = (u − c)/(hop/2) with c the row centre, and with
/// a carrier each power times cos and sin of ω(u − c). QR by modified
/// Gram–Schmidt, twice; a column with under 1e-9 of its norm left is dependent
/// (a near-zero carrier) and dropped.
fn row_basis(hop: usize, degree: usize, omega: Option<f64>) -> Vec<Vec<f64>> {
    let c = (hop as f64 - 1.0) / 2.0;
    let mut raw: Vec<Vec<f64>> = Vec::new();
    for k in 0..=degree as i32 {
        let s = |u: usize| ((u as f64 - c) / (hop as f64 / 2.0)).powi(k);
        match omega {
            None => raw.push((0..hop).map(s).collect()),
            Some(w) => {
                raw.push((0..hop).map(|u| s(u) * (w * (u as f64 - c)).cos()).collect());
                raw.push((0..hop).map(|u| s(u) * (w * (u as f64 - c)).sin()).collect());
            }
        }
    }
    let mut q: Vec<Vec<f64>> = Vec::new();
    for mut v in raw {
        let n0 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        for _ in 0..2 {
            for b in &q {
                let d: f64 = v.iter().zip(b).map(|(a, b)| a * b).sum();
                for (a, b) in v.iter_mut().zip(b) { *a -= d * b; }
            }
        }
        let n = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if n > 1e-9 * n0 {
            for a in &mut v { *a /= n; }
            q.push(v);
        }
    }
    q
}

/// A group's carrier: the midpoint of its first and last bar, radians/sample.
fn group_omega(gp: &[Plan], sr: u32) -> f64 {
    std::f64::consts::TAU * 0.5 * (gp[0].freq as f64 + gp[gp.len() - 1].freq as f64) / sr as f64
}

/// A group's kernel in `gpu_codes`' GEMM layout: q blocks of (hop × n_cols),
/// column-major, columns per (bar, member) re then im. Returns the window start
/// (a whole number of hops before the centre), q, H and each bar's first column.
fn gemm_pieces(group: &[Superlet], hop: usize) -> (usize, usize, Vec<f32>, Vec<usize>) {
    gemm_pieces_at(group, hop, None)
}

/// [`gemm_pieces`] with the column pairs where `cols` lists their (bar,
/// member) rather than in (bar, member) order.
fn gemm_pieces_at(group: &[Superlet], hop: usize, cols: Option<&[(usize, usize)]>) -> (usize, usize, Vec<f32>, Vec<usize>) {
    let hmax = group.iter().flat_map(|sl| sl.wavelets.iter().map(|w| w.half)).max().unwrap();
    let s_g = hmax.div_ceil(hop) * hop;
    let q = (s_g + hmax + 1).div_ceil(hop);
    let mut starts = vec![0usize];
    for sl in group { starts.push(starts.last().unwrap() + 2 * sl.wavelets.len()); }
    let n_cols = starts[group.len()];
    let mut h = vec![0.0f32; q * hop * n_cols];
    for (b, sl) in group.iter().enumerate() {
        for (m, w) in sl.wavelets.iter().enumerate() {
            let ci = cols.map_or(starts[b] / 2 + m, |cols| cols.iter().position(|&x| x == (b, m)).unwrap());
            for i in 0..w.re.len() {
                let pos = s_g - w.half + i;
                let (piece, u) = (pos / hop, pos % hop);
                h[piece * hop * n_cols + (2 * ci) * hop + u] = w.re[i];
                h[piece * hop * n_cols + (2 * ci + 1) * hop + u] = w.im[i];
            }
        }
    }
    (s_g, q, h, starts)
}

/// √(residual energy / energy) of one kernel column outside an orthonormal row
/// basis, over all its pieces. `h` is a group's kernel in GEMM layout.
fn column_residual(h: &[f32], col: usize, n_cols: usize, q: usize, hop: usize, basis: &[Vec<f64>]) -> f64 {
    let (mut e_h, mut e_c) = (0f64, 0f64);
    for p in 0..q {
        let piece = &h[p * hop * n_cols + col * hop..][..hop];
        e_h += piece.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
        for b in basis {
            let c: f64 = b.iter().zip(piece).map(|(a, &v)| a * v as f64).sum();
            e_c += c * c;
        }
    }
    if e_h > 0.0 { ((e_h - e_c).max(0.0) / e_h).sqrt() } else { 0.0 }
}

/// The modulated split's choice for one group (`MOOSIK_MR_SPLIT=mod`), from
/// kernel geometry alone: bars per carrier (8, 4, 2 or 1) and degree (0..=6),
/// tried cheapest first by DGEMM work (signal coefficients per carrier plus the
/// projected product). The work depends on `rows` and `frames`, so a plan
/// belongs to a track length as well as to the settings. The first option
/// whose worst column leaves at most `tau` of its norm outside the basis wins.
/// `tau` limits an unrounded L2 kernel residual ratio: a conditioning
/// heuristic, not a bound on any response, code or Diff error.
/// Ok((bars per carrier, degree, worst ratio, options tried)), or Err(the
/// least worst ratio) when no option reaches `tau`: the group has no plan.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn mod_plan<'a>(gp: &[Plan], q: usize, n_cols: usize, hop: usize, sr: u32, rows: usize, frames: usize, tau: f64,
                seen: &Mutex<HashMap<(usize, usize), f64>>, pieces: &dyn Fn() -> &'a Pieces) -> Result<(usize, usize, f64, usize), f64> {
    let mut options: Vec<(usize, usize, usize)> = Vec::new();
    for sub in [8usize, 4, 2, 1] {
        for d in 0..=6usize {
            let nb = 2 * (d + 1);
            options.push((gp.len().div_ceil(sub) * 2 * rows * hop * nb + 2 * frames * nb * q * n_cols, sub, d));
        }
    }
    options.sort();
    let mut least = f64::INFINITY;
    for (tried, &(_, sub, d)) in options.iter().enumerate() {
        // An option's worst residual depends on the group, the rate and the
        // settings, not on the track: computed once, then remembered.
        let known = seen.lock().unwrap().get(&(sub, d)).copied();
        let worst = known.unwrap_or_else(|| {
            let (_, h, starts) = pieces();
            let worst = gp.chunks(sub).enumerate().map(|(k, chunk)| {
                let basis = row_basis(hop, d, Some(group_omega(chunk, sr)));
                (starts[k * sub]..starts[k * sub + chunk.len()]).into_par_iter()
                    .map(|col| column_residual(h, col, n_cols, q, hop, &basis)).reduce(|| 0.0, f64::max)
            }).fold(0.0, f64::max);
            seen.lock().unwrap().insert((sub, d), worst);
            worst
        });
        if worst <= tau { return Ok((sub, d, worst, tried + 1)); }
        least = least.min(worst);
    }
    // No carrier basis reaches the tolerance: the kernel changes too fast
    // within a hop (a wavelet shorter than the hop). Such kernels are short,
    // so the whole group goes through the f64 products instead: the unit
    // basis takes every piece exactly and leaves a zero residual.
    let _ = least;
    Ok((GROUP, EXACT, 0.0, options.len() + 1))
}

/// The modulated split's tolerance τ: the most of a kernel column's norm
/// that may be left outside the carrier basis, for the f32 product. Part of
/// the route's numerics ([`REVISION`]).
const TOLERANCE: f64 = 1e-2;

/// The degree that stands for the unit basis (the all-f64 group).
const EXACT: usize = usize::MAX;

/// A carrier group's basis: modulated polynomials of `degree`, or for
/// [`EXACT`] the unit vectors of the hop.
fn carrier_basis(hop: usize, degree: usize, chunk: &[Plan], sr: u32) -> Vec<Vec<f64>> {
    if degree == EXACT {
        return (0..hop).map(|u| { let mut b = vec![0f64; hop]; b[u] = 1.0; b }).collect();
    }
    row_basis(hop, degree, Some(group_omega(chunk, sr)))
}

/// One group's modulated-split plan: bars per carrier, degree, worst ratio,
/// options tried, and each carrier's basis.
type GroupPlan = (usize, usize, f64, usize, Vec<Vec<Vec<f64>>>);

/// A group's piece count, kernel pieces and column starts (`gemm_pieces`).
type Pieces = (usize, Vec<f32>, Vec<usize>);

/// What planning has worked out for one rate and one group of settings, kept
/// for the session (1.6.2 research): the longest half-window, each group's
/// piece count and columns, and each option's worst residual once it has been
/// evaluated. None of it depends on the track. A track's plan is then chosen
/// exactly as before, by walking its own cost order (which does depend on its
/// length) and taking the first option within tolerance; only the residuals
/// are looked up instead of recomputed.
#[derive(Default)]
struct PlanMemo {
    hmax_all: std::sync::OnceLock<usize>,
    groups: Vec<GroupMemo>,
}

#[derive(Default)]
struct GroupMemo {
    shape: std::sync::OnceLock<(usize, usize)>,
    seen: Mutex<HashMap<(usize, usize), f64>>,
}

fn plan_memo(sr: u32, geom: &Geom, cfg: &AsltConfig, groups: usize) -> Arc<PlanMemo> {
    static MEMO: Mutex<Option<HashMap<String, Arc<PlanMemo>>>> = Mutex::new(None);
    let mut memo = MEMO.lock().unwrap();
    let memo = memo.get_or_insert_with(HashMap::new);
    let key = format!("{sr} {groups} {geom:?} {cfg:?}");
    // ponytail: a session uses few settings; past 16 start over (the next
    // track of each plans from nothing once) rather than track ages.
    if memo.len() >= 16 && !memo.contains_key(&key) { memo.clear(); }
    memo.entry(key)
        .or_insert_with(|| Arc::new(PlanMemo { hmax_all: Default::default(), groups: (0..groups).map(|_| GroupMemo::default()).collect() }))
        .clone()
}

/// A track's geometry and every group's plan: the front padding, the rows,
/// the plans, and each group's piece count and columns. `Err` says how many
/// groups have no admissible plan.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn plan_track(plans: &[Plan], sr: u32, cfg: &AsltConfig, hop: usize, frames: usize, pool: &rayon::ThreadPool, memo: &PlanMemo, tau: f64)
              -> Result<(usize, usize, Vec<GroupPlan>, Vec<(usize, usize)>), String> {
    // Each group's wavelets are built when needed and dropped after (round
    // 10). First the global maximum support, which sets the padding, the rows
    // and the hop origin.
    let build = |gp: &[Plan]| -> Vec<Superlet> { gp.iter().map(|p| Superlet::new(p.freq, p.q, cfg, sr as f32)).collect() };
    let hmax_all = *memo.hmax_all.get_or_init(|| pool.install(|| plans.par_chunks(GROUP)
        .map(|gp| build(gp).iter().flat_map(|sl| sl.wavelets.iter().map(|w| w.half)).max().unwrap_or(0)).max().unwrap_or(0)));
    let (p0, _, rows) = padded_rows(hmax_all, hop, frames);
    let planned: Vec<Result<(GroupPlan, usize, usize), (usize, f64)>> = pool.install(|| plans.par_chunks(GROUP).enumerate().map(|(gi, gp)| {
        let built = std::cell::OnceCell::new();
        let pieces = || -> &Pieces {
            built.get_or_init(|| {
                let (_, q, h, starts) = gemm_pieces(&build(gp), hop);
                (q, h, starts)
            })
        };
        let (q, n_cols) = *memo.groups[gi].shape.get_or_init(|| {
            let (q, _, starts) = pieces();
            (*q, starts[gp.len()])
        });
        mod_plan(gp, q, n_cols, hop, sr, rows, frames, tau, &memo.groups[gi].seen, &pieces)
            .map(|(sub, d, worst, tried)| ((sub, d, worst, tried, gp.chunks(sub).map(|c| carrier_basis(hop, d, c, sr)).collect()),
                q, n_cols))
            .map_err(|least| (gi, least))
    }).collect());
    let refused = planned.iter().filter(|p| p.is_err()).count();
    if refused > 0 { return Err(format!("{refused} groups have no admissible plan")); }
    let (mod_plans, sizes) = planned.into_iter().map(|p| {
        let (plan, q, n_cols) = p.expect("checked above");
        (plan, (q, n_cols))
    }).unzip();
    Ok((p0, rows, mod_plans, sizes))
}

/// The device signal's geometry for `frames` frames: the front padding (whole
/// hops, the longest half-window), the most pieces a window can span, and the
/// rows of whole hops that cover every piece of every frame.
fn padded_rows(hmax_all: usize, hop: usize, frames: usize) -> (usize, usize, usize) {
    let p0 = hmax_all.div_ceil(hop) * hop;
    let qmax = (2 * p0 + 1).div_ceil(hop) + 1;
    (p0, qmax, p0 / hop + frames + qmax)
}

// ---------------------------------------------------------------------------
// The pinned PTX
// ---------------------------------------------------------------------------

/// The pinned, prebuilt PTX the native runner loads (NVRTC 12.8, SGEMM_OPTS,
/// from SGEMM_CU and DGEMM_CU; `mr_ptx_pin` recompiles and compares).
const SGEMM_PTX: &str = include_str!("aslt_hp_sgemm.ptx");
const DGEMM_PTX: &str = include_str!("aslt_hp_dgemm.ptx");
/// `sgemm_v2_ranged`: round 7's SGEMM source unchanged, plus an entry that
/// gives each 72-column block of the grid its own piece range; the same
/// NVRTC 12.8 and options. Its FMA chain is `sgemm_v2`'s. Built in the
/// algorithm lab of 2026-11-15 (`codex/v162-algorithm-lab`, `ranged_sgemm.cu`).
const RANGED_PTX: &str = include_str!("aslt_hp_sgemm_ranged.ptx");
/// A pinned file's text as NVRTC wrote it (a checkout may add CRs).
fn pinned(ptx: &str) -> String { ptx.replace("\r\n", "\n") }
/// `acc64`: `dst[i] += (f64) src[i]`, the hop pieces summed in f64 rather
/// than f32. `cvt32`: `dst[i] = (f32) src[i]`. `shift`: the shifted copy
/// `dst[r + rows·(p·nb + k)] = src[r + p + rows·k]` (0 past the last row) of a
/// rows × nb column-major matrix, for every piece p, so one GEMM sees every
/// piece's rows side by side.
const ACC64_PTX: &str = "
.version 6.4
.target sm_75
.address_size 64
.visible .entry acc64(.param .u64 p_dst, .param .u64 p_src, .param .u32 p_n)
{
.reg .pred %p<2>;
.reg .b32 %r<6>;
.reg .b64 %rd<8>;
.reg .f32 %f<2>;
.reg .f64 %fd<4>;
ld.param.u64 %rd1, [p_dst];
ld.param.u64 %rd2, [p_src];
ld.param.u32 %r1, [p_n];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.s32 %r5, %r2, %r3, %r4;
setp.ge.u32 %p1, %r5, %r1;
@%p1 bra DONE;
cvta.to.global.u64 %rd3, %rd1;
cvta.to.global.u64 %rd4, %rd2;
mul.wide.u32 %rd5, %r5, 8;
add.s64 %rd6, %rd3, %rd5;
mul.wide.u32 %rd7, %r5, 4;
add.s64 %rd7, %rd4, %rd7;
ld.global.f32 %f1, [%rd7];
cvt.f64.f32 %fd1, %f1;
ld.global.f64 %fd2, [%rd6];
add.f64 %fd3, %fd2, %fd1;
st.global.f64 [%rd6], %fd3;
DONE:
ret;
}
.visible .entry cvt32(.param .u64 p_dst, .param .u64 p_src, .param .u32 p_n)
{
.reg .pred %p<2>;
.reg .b32 %r<6>;
.reg .b64 %rd<8>;
.reg .f32 %f<2>;
.reg .f64 %fd<2>;
ld.param.u64 %rd1, [p_dst];
ld.param.u64 %rd2, [p_src];
ld.param.u32 %r1, [p_n];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.s32 %r5, %r2, %r3, %r4;
setp.ge.u32 %p1, %r5, %r1;
@%p1 bra DONE2;
cvta.to.global.u64 %rd3, %rd1;
cvta.to.global.u64 %rd4, %rd2;
mul.wide.u32 %rd5, %r5, 4;
add.s64 %rd6, %rd3, %rd5;
mul.wide.u32 %rd7, %r5, 8;
add.s64 %rd7, %rd4, %rd7;
ld.global.f64 %fd1, [%rd7];
cvt.rn.f32.f64 %f1, %fd1;
st.global.f32 [%rd6], %f1;
DONE2:
ret;
}
.visible .entry shift(.param .u64 p_dst, .param .u64 p_src, .param .u32 p_rows, .param .u32 p_nb, .param .u32 p_n)
{
.reg .pred %p<3>;
.reg .b32 %r<14>;
.reg .b64 %rd<9>;
.reg .f64 %fd<2>;
ld.param.u64 %rd1, [p_dst];
ld.param.u64 %rd2, [p_src];
ld.param.u32 %r1, [p_rows];
ld.param.u32 %r2, [p_nb];
ld.param.u32 %r3, [p_n];
mov.u32 %r4, %ctaid.x;
mov.u32 %r5, %ntid.x;
mov.u32 %r6, %tid.x;
mad.lo.s32 %r7, %r4, %r5, %r6;
setp.ge.u32 %p1, %r7, %r3;
@%p1 bra DONE3;
rem.u32 %r8, %r7, %r1;
div.u32 %r9, %r7, %r1;
div.u32 %r10, %r9, %r2;
rem.u32 %r11, %r9, %r2;
add.s32 %r12, %r8, %r10;
mov.f64 %fd1, 0d0000000000000000;
setp.ge.u32 %p2, %r12, %r1;
@%p2 bra STORE3;
mad.lo.s32 %r13, %r11, %r1, %r12;
cvta.to.global.u64 %rd3, %rd2;
mul.wide.u32 %rd4, %r13, 8;
add.s64 %rd5, %rd3, %rd4;
ld.global.f64 %fd1, [%rd5];
STORE3:
cvta.to.global.u64 %rd6, %rd1;
mul.wide.u32 %rd7, %r7, 8;
add.s64 %rd8, %rd6, %rd7;
st.global.f64 [%rd8], %fd1;
DONE3:
ret;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_context() -> CloseReport {
        CloseReport { context_destroyed: true, ..CloseReport::default() }
    }

    /// The groups' peak on the device, by replaying the pipelined pass's
    /// allocations and frees in its order rather than by the estimate's
    /// formula: the first group is staged; then per group its results are
    /// allocated, the next group is staged, and only then the group before
    /// is handed over (all its buffers freed); the last is handed over after
    /// the loop.
    fn traced_peak(parts: &[(usize, usize)]) -> usize {
        let (mut live, mut peak, mut flown) = (0usize, 0usize, None);
        let mut alloc = |live: &mut usize, n: usize| { *live += n; peak = peak.max(*live); };
        if let Some(first) = parts.first() { alloc(&mut live, first.0); }
        for (g, &(pieces, results)) in parts.iter().enumerate() {
            alloc(&mut live, results);
            if let Some(next) = parts.get(g + 1) { alloc(&mut live, next.0); }
            if let Some((p, r)) = flown.replace((pieces, results)) { live -= p + r; }
        }
        if let Some((p, r)) = flown { live -= p + r; }
        assert_eq!(live, 0, "everything is freed");
        peak
    }

    /// The estimate's bound on the groups' buffers is the peak of the
    /// pass's own order of allocations and frees: one group, two, three
    /// uniform ones (3 pieces + 2 results, the previous group's pieces
    /// included), unequal ones with the peak at the first, a middle and the
    /// last group. And a device with less than that free is refused.
    #[test]
    fn the_estimate_counts_every_group_buffer_live_at_once() {
        let cases: [&[(usize, usize)]; 8] = [
            &[],
            &[(7, 11)],
            &[(7, 11), (5, 3)],
            &[(10, 100), (10, 100), (10, 100)],
            &[(900, 50), (800, 40), (1, 1), (1, 1)],
            &[(1, 1), (700, 5), (600, 900), (500, 5), (1, 1)],
            &[(1, 1), (1, 1), (300, 20), (400, 900)],
            &[(64, 8), (32, 512), (128, 4), (16, 256), (256, 2), (8, 1024)],
        ];
        for parts in cases {
            assert_eq!(groups_live(parts).unwrap(), traced_peak(parts), "{parts:?}");
        }
        assert_eq!(groups_live(cases[3]).unwrap(), 3 * 10 + 2 * 100);
        // A device with room for everything but the previous group's pieces.
        let need = groups_live(cases[3]).unwrap();
        assert!(fits(need, need + MARGIN));
        assert!(!fits(need, need + MARGIN - 1));
        assert!(!fits(need, need - 10 + MARGIN), "the bound without the previous pieces would have fitted");
        assert!(groups_live(&[(usize::MAX, 1), (1, 1)]).is_err());
    }

    /// A finished transform is not a publishable result if destruction or
    /// restoration of the worker's thread state could not be confirmed.
    #[test]
    fn completed_codes_require_a_successful_close_and_context_check() {
        assert_eq!(finish_closed(Ok(17usize), true, &closed_context(), &Ok(true), false), Ok(17));
        let cleanup_error = CloseReport {
            context_destroyed: true,
            failures: vec!["stream cleanup failed".into()],
            ..CloseReport::default()
        };
        for (report, clear) in [
            (CloseReport::default(), Ok(true)),
            (cleanup_error, Ok(true)),
            (closed_context(), Ok(false)),
            (closed_context(), Err(HpError::Device("context query failed".into()))),
        ] {
            assert!(matches!(finish_closed(Ok(17usize), true, &report, &clear, false), Err(HpError::Device(_))));
        }
    }

    /// Cancellation may arrive while close() runs after the engine produced
    /// codes. It still prohibits publication and a Standard restart.
    #[test]
    fn cancellation_during_close_wins_over_codes_and_cleanup_failure() {
        let report = CloseReport {
            failures: vec!["context destruction failed".into()],
            ..CloseReport::default()
        };
        let unknown = Err(HpError::Device("context state unknown".into()));
        for result in [Ok(17usize), Err(HpError::Unavailable("allocation refused".into())),
                       Err(HpError::Device("device failure".into()))] {
            assert_eq!(finish_closed(result, true, &report, &unknown, true), Err(HpError::Cancelled));
        }
        assert_eq!(finish_closed::<usize>(Err(HpError::Cancelled), true, &report, &unknown, false), Err(HpError::Cancelled));
        // The borrowed report remains available to the caller's diagnostic log.
        assert_eq!(report.failures, ["context destruction failed"]);
    }

    /// A cancelled job must not retry, while its failed device is still
    /// disabled by the worker's supervisor for later jobs.
    #[test]
    fn cancelled_product_call_keeps_a_separate_device_failure_notice() {
        let _ = take_device_failure();
        let result = finish_product_call::<usize>(Err(HpError::Cancelled), true, &closed_context(), &Ok(true), false,
                                                  Some("query failed during cancellation"));
        assert_eq!(result, Err(HpError::Cancelled));
        assert_eq!(take_device_failure().as_deref(), Some("query failed during cancellation"));
        assert_eq!(take_device_failure(), None);

        let failed_close = CloseReport { failures: vec!["context destruction failed".into()], ..CloseReport::default() };
        assert_eq!(finish_product_call(Ok(17usize), true, &failed_close, &Ok(false), true, None), Err(HpError::Cancelled));
        assert!(take_device_failure().is_some_and(|reason| reason.contains("context destruction failed")));

        assert_eq!(finish_product_call(Ok(17usize), true, &closed_context(), &Ok(true), false, None), Ok(17));
        assert_eq!(take_device_failure(), None);
    }

    /// A constructor can fail after creating a context. Its original
    /// refusal must not hide an unsuccessful teardown of that context.
    #[test]
    fn partial_initialization_cleanup_can_upgrade_a_refusal_to_device_failure() {
        let refused = || Err::<usize, _>(HpError::Unavailable("module unavailable".into()));
        assert!(matches!(finish_closed(refused(), false, &CloseReport::default(), &Ok(true), false), Err(HpError::Unavailable(_))));
        assert!(matches!(finish_closed(refused(), true, &closed_context(), &Ok(true), false), Err(HpError::Unavailable(_))));
        let result = finish_closed(refused(), true, &CloseReport::default(), &Ok(true), false);
        assert!(matches!(result, Err(HpError::Device(reason)) if reason.contains("not destroyed") && reason.contains("module unavailable")));
    }

    /// This host-only gate runs before driver loading/context creation.
    #[test]
    fn decoded_input_admission_rejects_unsupported_rates_and_geometry() {
        for sr in [44_100, 48_000, 88_200, 96_000, 176_400, 192_000] {
            let hop = hop_for_fps(sr, FPS);
            assert_eq!(admitted_frames(hop * FRAME_CAP, hop * FRAME_CAP, sr, FPS, N_BARS), Ok(FRAME_CAP));
            assert!(admitted_frames(hop * (FRAME_CAP + 1), hop * (FRAME_CAP + 1), sr, FPS, N_BARS).is_err());
            assert!(admitted_frames(0, 0, sr, FPS, N_BARS).is_err());
            assert!(admitted_frames(hop, hop + 1, sr, FPS, N_BARS).is_err());
            assert!(admitted_frames(usize::MAX, usize::MAX, sr, FPS, N_BARS).is_err());
        }
        for sr in [0, 22_050, 32_000, 352_800, 384_000] {
            assert!(admitted_frames(48_000, 48_000, sr, FPS, N_BARS).is_err());
        }
    }

    /// Only the spreads measured faster: 0.1 or more at 44.1 and 48 kHz, 0.3
    /// or more above; any spread with one wavelet; a NaN never.
    #[test]
    fn only_the_spreads_measured_faster_are_taken() {
        let at = |spread: f32| AsltConfig { spread, ..validated_cfg() };
        for sr in [44_100, 48_000] {
            for spread in [0.1, 0.3, 0.5, 0.95] { assert_eq!(spread_admits(sr, &at(spread)), Ok(()), "{sr} Hz, spread {spread}"); }
            for spread in [0.0, 0.09, f32::NAN] { assert!(spread_admits(sr, &at(spread)).is_err(), "{sr} Hz, spread {spread}"); }
        }
        for sr in [88_200, 96_000, 176_400, 192_000] {
            for spread in [0.3, 0.5, 0.95] { assert_eq!(spread_admits(sr, &at(spread)), Ok(()), "{sr} Hz, spread {spread}"); }
            for spread in [0.0, 0.1, 0.29, f32::NAN] { assert!(spread_admits(sr, &at(spread)).is_err(), "{sr} Hz, spread {spread}"); }
        }
        for sr in [44_100, 192_000] {
            for n_wavelets in [2, 9] { assert!(spread_admits(sr, &AsltConfig { n_wavelets, ..at(0.0) }).is_err()); }
            assert_eq!(spread_admits(sr, &AsltConfig { n_wavelets: 1, ..at(0.0) }), Ok(()), "{sr} Hz, one wavelet");
        }
        assert_eq!(spread_admits(48_000, &at(0.0)), Err("at 48 kHz it is no faster than Standard with a spread under 0.1".into()));
        assert_eq!(spread_admits(192_000, &at(0.0)), Err("at 192 kHz it is no faster than Standard with a spread under 0.3".into()));
        assert_eq!(spread_admits(88_200, &at(0.2)), Err("at 88.2 kHz it is no faster than Standard with a spread under 0.3".into()));
    }

    // ── 1.6.2 research: the pipelined pass against the serial pass ──────────

    /// A deterministic stereo test signal: a rising tone, a fixed tone and
    /// noise, different in each channel, with a quiet stretch and a click.
    fn stereo(sr: u32, secs: f32) -> (Vec<f32>, Vec<f32>) {
        let n = (sr as f32 * secs) as usize;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut noise = move || {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let sweep = (std::f32::consts::TAU * (40.0 * t + 400.0 * t * t)).sin();
            let tone = (std::f32::consts::TAU * 3_000.0 * t).sin();
            let quiet = if (0.4..0.5).contains(&(t / secs)) { 0.001 } else { 1.0 };
            let click = if i == n / 3 { 0.9 } else { 0.0 };
            l.push(quiet * (0.4 * sweep + 0.1 * noise()) + click);
            r.push(quiet * (0.3 * tone + 0.2 * sweep + 0.1 * noise()));
        }
        (l, r)
    }

    fn pool6() -> rayon::ThreadPool { rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap() }

    /// The process's peak commit so far, in MB.
    fn peak_commit_mb() -> usize {
        #[repr(C)]
        #[derive(Default)]
        struct Pmc { cb: u32, faults: u32, peak_ws: usize, ws: usize, a: usize, b: usize, c: usize, d: usize, commit: usize, peak_commit: usize, private: usize }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn K32GetProcessMemoryInfo(process: isize, counters: *mut Pmc, cb: u32) -> i32;
        }
        let mut m = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
        // SAFETY: -1 is the current process; `m` is a valid, sized out-buffer.
        unsafe { K32GetProcessMemoryInfo(-1, &mut m, m.cb) };
        m.peak_commit >> 20
    }

    struct Pass { out: Result<[Vec<u16>; 3], HpError>, trace: Trace, failures: Vec<String>, close: CloseReport, wall: f64 }

    /// One pass on a new engine, closed afterwards: serial or pipelined, with
    /// an optional cancellation set as a group's wait begins or once the
    /// progress reaches a bar count.
    fn pass(serial: bool, geom: &Geom, cfg: &AsltConfig, l: &[f32], r: &[f32], sr: u32, at_wait: Option<usize>, at_bars: Option<usize>) -> Pass {
        pass_on(&pool6(), Faults::default(), serial, geom, cfg, l, r, sr, at_wait, at_bars)
    }

    /// The same, on a given pool and with failures to inject.
    #[allow(clippy::too_many_arguments)]
    fn pass_on(pool: &rayon::ThreadPool, faults: Faults, serial: bool, geom: &Geom, cfg: &AsltConfig, l: &[f32], r: &[f32], sr: u32,
               at_wait: Option<usize>, at_bars: Option<usize>) -> Pass {
        let cancel = AtomicBool::new(false);
        let mut e = HpEngine::new(faults).expect("the engine");
        e.serial = serial;
        e.cancel_at_wait = at_wait;
        // Research only: the gate sweep at another tolerance.
        e.tau = std::env::var("MOOSIK_HP_TAU").ok().and_then(|t| t.parse().ok());
        let progress = |bars: usize| if at_bars.is_some_and(|b| bars >= b) { cancel.store(true, Ordering::Relaxed) };
        let t = Instant::now();
        let out = e.analyze(l, r, sr, geom, cfg, pool, &cancel, &progress);
        let wall = t.elapsed().as_secs_f64();
        let (trace, failures) = (e.trace, std::mem::take(&mut e.failures));
        Pass { out, trace, failures, close: e.close(), wall }
    }

    fn stages(t: &Trace) -> String {
        STAGES.iter().zip(t.stage_s).map(|(n, s)| format!("{n} {s:.2}")).collect::<Vec<_>>().join(", ")
    }

    /// The pipelined pass returns the serial pass's codes, every one, at the
    /// validated settings and at others (the bars, range and frame rate are
    /// still the engine's own), at both rates. Real device; by hand.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn the_pipelined_pass_returns_the_serial_codes() {
        let v = validated_cfg();
        let cases: [(&str, AsltConfig, u32); 7] = [
            ("validated 48k", v.clone(), 48_000),
            ("validated 44.1k", v.clone(), 44_100),
            ("validated 96k", v.clone(), 96_000),
            ("validated 192k", v.clone(), 192_000),
            ("5 wavelets, spread 0.3", AsltConfig { n_wavelets: 5, spread: 0.3, ..v.clone() }, 48_000),
            ("13 wavelets, q 3", AsltConfig { n_wavelets: 13, q_ratio: 3.0, ..v.clone() }, 44_100),
            ("window 0.5 s, log scale", AsltConfig { max_window_s: 0.5, scale: crate::spectrum::freq_scale::FreqScale::Blend(0.0), ..v.clone() }, 48_000),
        ];
        for (what, cfg, sr) in cases {
            let (l, r) = stereo(sr, 20.0);
            let a = pass(true, &Geom::validated(), &cfg, &l, &r, sr, None, None);
            let peak_a = peak_commit_mb();
            let b = pass(false, &Geom::validated(), &cfg, &l, &r, sr, None, None);
            let peak_b = peak_commit_mb();
            eprintln!("{what}: serial {:.2}s, pipelined {:.2}s, peak commit {peak_a} -> {peak_b} MB", a.wall, b.wall);
            eprintln!("  serial    {}", stages(&a.trace));
            eprintln!("  pipelined {}", stages(&b.trace));
            match (&a.out, &b.out) {
                (Ok(x), Ok(y)) => {
                    for (p, name) in ["Mix", "Left", "Right"].iter().enumerate() {
                        let differ = x[p].iter().zip(&y[p]).filter(|(a, b)| a != b).count();
                        assert_eq!(differ, 0, "{what}: {name} differs in {differ} codes");
                        assert_eq!(x[p].len(), y[p].len());
                    }
                    assert!(x[0].iter().any(|&c| c > 0), "{what}: an empty picture proves nothing");
                }
                // Settings the planner refuses are refused alike, and say so.
                (Err(HpError::Unavailable(x)), Err(HpError::Unavailable(y))) => {
                    assert_eq!(x, y);
                    eprintln!("  refused alike: {x}");
                }
                (x, y) => panic!("{what}: serial {:?}, pipelined {:?}", x.as_ref().map(|_| "codes"), y.as_ref().map(|_| "codes")),
            }
            for q in [&a, &b] {
                assert!(q.failures.is_empty() && q.close.failures.is_empty() && q.close.context_destroyed, "{what}: {:?} {:?}", q.failures, q.close);
            }
            assert_eq!((a.close.device_reclaimed, a.close.host_released), (b.close.device_reclaimed, b.close.host_released), "{what}: left for close");
        }
    }

    /// A cancellation ends the pipelined pass as it ends the serial one:
    /// cancelled, nothing returned, the pass's own cleanup done (close finds
    /// what the serial pass leaves it, no more), no failure, and both helper
    /// threads gone before the call returns. Seen in a wait, and from the
    /// progress callback while later groups are in flight. Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_cancelled_pipelined_pass_ends_like_the_serial_one() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        for (what, at_wait, at_bars) in [("in group 40's wait", Some(40), None), ("at the first group's wait", Some(0), None),
                                         ("from progress at 256 bars", None, Some(256)), ("from progress at the last bars", None, Some(1016))] {
            let a = pass(true, &Geom::validated(), &cfg, &l, &r, 48_000, at_wait, at_bars);
            let b = pass(false, &Geom::validated(), &cfg, &l, &r, 48_000, at_wait, at_bars);
            eprintln!("{what}: serial {:.2}s {:?}, pipelined {:.2}s {:?}; close {:?} / {:?}", a.wall, a.trace.observed.map(|o| o.1),
                      b.wall, b.trace.observed.map(|o| o.1), (a.close.device_reclaimed, a.close.host_released), (b.close.device_reclaimed, b.close.host_released));
            for q in [&a, &b] {
                assert_eq!(q.out.as_ref().err(), Some(&HpError::Cancelled), "{what}");
                assert!(q.trace.observed.is_some(), "{what}: no seat recorded");
                assert!(q.failures.is_empty() && q.close.failures.is_empty() && q.close.context_destroyed, "{what}: {:?} {:?}", q.failures, q.close);
            }
            assert_eq!((a.close.device_reclaimed, a.close.host_released), (b.close.device_reclaimed, b.close.host_released), "{what}: left for close");
        }
        // And the device is whole afterwards: a full pass still gives codes.
        assert!(pass(false, &Geom::validated(), &cfg, &l, &r, 48_000, None, None).out.is_ok());
    }

    /// A remembered plan is the plan: for tracks of several lengths, planning
    /// with what earlier tracks left behind gives exactly what planning from
    /// nothing gives (choice, degree, worst residual, options tried, bases,
    /// geometry), at two rates and two groups of settings. Host only.
    #[test]
    fn a_remembered_plan_is_the_plan() {
        let pool = pool6();
        for (cfg, sr) in [(validated_cfg(), 48_000u32), (validated_cfg(), 44_100), (AsltConfig { n_wavelets: 5, spread: 0.3, ..validated_cfg() }, 48_000)] {
            let hop = hop_for_fps(sr, 180.0);
            let plans = plans(sr, &Geom::validated(), &cfg);
            let groups = plans.len().div_ceil(GROUP);
            let fresh = || PlanMemo { hmax_all: Default::default(), groups: (0..groups).map(|_| GroupMemo::default()).collect() };
            let warm = fresh();
            // Short tracks first, so that longer ones meet a partly filled memo.
            for frames in [1usize, 900, 3_600, 38_896, FRAME_CAP] {
                let cold = plan_track(&plans, sr, &cfg, hop, frames, &pool, &fresh(), TOLERANCE).expect("a plan");
                let again = plan_track(&plans, sr, &cfg, hop, frames, &pool, &warm, TOLERANCE).expect("a plan");
                assert!(cold == again, "{sr} Hz, {} wavelets, {frames} frames: the remembered plan differs", cfg.n_wavelets);
            }
            let evaluated: usize = warm.groups.iter().map(|g| g.seen.lock().unwrap().len()).sum();
            eprintln!("{sr} Hz, {} wavelets: {evaluated} option residuals remembered for {groups} groups", cfg.n_wavelets);
        }
    }

    /// The tiled layout holds every column pair once, at most a block's worth
    /// per tile and in (bar, member) order inside it; a tile's pieces cover
    /// everything that is not zero in its columns (the kernel itself, laid
    /// out that way, is searched); and laid out that way a column is the
    /// column it was. With 9 members a group is two full tiles, with 5 a
    /// short first one.
    #[test]
    fn a_tile_runs_every_piece_its_columns_have() {
        for (cfg, sr) in [(validated_cfg(), 48_000u32), (validated_cfg(), 44_100), (AsltConfig { n_wavelets: 5, ..validated_cfg() }, 48_000)] {
            let hop = hop_for_fps(sr, 180.0);
            let plans = plans(sr, &Geom::validated(), &cfg);
            for gp in [&plans[..GROUP], &plans[504..504 + GROUP], &plans[plans.len() - 3..]] {
                let group: Vec<Superlet> = gp.iter().map(|p| Superlet::new(p.freq, p.q, &cfg, sr as f32)).collect();
                let (cols, tiles) = tile_layout(&group, hop);
                let (s_g, q, h, starts) = gemm_pieces(&group, hop);
                let (s_t, q_t, h_t, _) = gemm_pieces_at(&group, hop, Some(&cols));
                let n_cols = 2 * cols.len();
                assert_eq!((s_g, q, n_cols), (s_t, q_t, starts[group.len()]));
                let mut seen = cols.clone();
                seen.sort();
                assert_eq!(seen, group.iter().enumerate().flat_map(|(b, sl)| (0..sl.wavelets.len()).map(move |m| (b, m))).collect::<Vec<_>>());
                let mut next = 0;
                for (t, &(c0, nc, lo, hi)) in tiles.iter().enumerate() {
                    assert_eq!(c0, next, "tiles follow one another");
                    assert!(nc <= 2 * TILE_PAIRS && nc > 0 && (t == 0 || nc == 2 * TILE_PAIRS) && lo < hi && hi <= q);
                    assert!(cols[c0 / 2..(c0 + nc) / 2].windows(2).all(|w| w[0] < w[1]), "(bar, member) order inside a tile");
                    for col in c0..c0 + nc {
                        let (b, m) = cols[col / 2];
                        let old = starts[b] + 2 * m + col % 2;
                        for p in 0..q {
                            let piece = &h_t[p * hop * n_cols + col * hop..][..hop];
                            assert!(piece == &h[p * hop * n_cols + old * hop..][..hop], "a column is the column it was");
                            assert!((lo..hi).contains(&p) || piece.iter().all(|&v| v == 0.0), "tile {t}, column {col}: piece {p} of {lo}..{hi} is not zero");
                        }
                    }
                    next = c0 + nc;
                }
                assert_eq!(next, n_cols);
                // The longest tile runs every piece.
                assert_eq!((tiles.last().unwrap().2, tiles.last().unwrap().3), (0, q));
            }
        }
    }

    /// Research note, host only: how much of the residual SGEMM's work is
    /// whole zero pieces. Per group the launch runs every column over all `q`
    /// pieces; a column's wavelet may span fewer. Prints the launched piece
    /// count, the count if each 72-column tile ran only its own span (columns
    /// as they are, and sorted by span), and the exact per-column count.
    #[test]
    #[ignore = "a report, not a check"]
    fn zero_pieces_report() {
        for (what, cfg, sr) in [("validated 48k", validated_cfg(), 48_000u32), ("validated 44.1k", validated_cfg(), 44_100),
                                ("5 wavelets", AsltConfig { n_wavelets: 5, ..validated_cfg() }, 48_000)] {
            let hop = hop_for_fps(sr, 180.0);
            let plans = plans(sr, &Geom::validated(), &cfg);
            let (mut launched, mut tiles, mut sorted, mut exact, mut laid) = (0usize, 0usize, 0usize, 0usize, 0usize);
            for gp in plans.chunks(GROUP) {
                let group: Vec<Superlet> = gp.iter().map(|p| Superlet::new(p.freq, p.q, &cfg, sr as f32)).collect();
                let (_, q, h, starts) = gemm_pieces(&group, hop);
                let n_cols = starts[group.len()];
                let mut span: Vec<(usize, usize)> = (0..n_cols).map(|col| {
                    let live: Vec<usize> = (0..q).filter(|p| h[p * hop * n_cols + col * hop..][..hop].iter().any(|&v| v != 0.0)).collect();
                    (*live.first().unwrap_or(&0), live.last().map_or(0, |l| l + 1))
                }).collect();
                let cover = |cols: &[(usize, usize)]| cols.chunks(72).map(|t| {
                    let (lo, hi) = (t.iter().map(|s| s.0).min().unwrap(), t.iter().map(|s| s.1).max().unwrap());
                    72 * (hi - lo)
                }).sum::<usize>();
                launched += n_cols.div_ceil(72) * 72 * q;
                laid += tile_layout(&group, hop).1.iter().map(|t| 72 * (t.3 - t.2)).sum::<usize>();
                tiles += cover(&span);
                exact += span.iter().map(|s| s.1 - s.0).sum::<usize>();
                span.sort_by_key(|s| s.1 - s.0);
                sorted += cover(&span);
            }
            eprintln!("{what}: launched {launched} column-pieces; per tile {tiles} ({:.0}%); tiles sorted by span {sorted} ({:.0}%); exact {exact} ({:.0}%); the pass's tiles {laid} ({:.0}%)",
                      100.0 * tiles as f64 / launched as f64, 100.0 * sorted as f64 / launched as f64, 100.0 * exact as f64 / launched as f64, 100.0 * laid as f64 / launched as f64);
        }
    }

    /// Research report, on the device: what a looser residual tolerance τ
    /// would save and change, on a whole track (`MOOSIK_HP_INPUTS`,
    /// `MOOSIK_HP_TRACK`, default song1_48k). Per τ: the plan's mix of
    /// options, two passes' times (the second with the plan remembered), and
    /// against today's τ the cells whose code differs in Mix, L, R and Diff
    /// with the largest difference. Accuracy against the reference is the
    /// gate sweep's job (`MOOSIK_HP_TAU`).
    #[test]
    #[ignore = "a report, not a check"]
    fn tolerance_tradeoff_report() {
        let dir = std::path::PathBuf::from(std::env::var("MOOSIK_HP_INPUTS").expect("MOOSIK_HP_INPUTS"));
        let name = std::env::var("MOOSIK_HP_TRACK").unwrap_or_else(|_| "song1_48k".into());
        let sr = input_rate(&name);
        let (l, r) = excerpt(&dir.join(format!("{name}.f32")), sr, 0.0, 100_000.0);
        let (geom, cfg, pool) = (Geom::validated(), validated_cfg(), pool6());
        eprintln!("{name}: {:.1} s", l.len() as f64 / sr as f64);
        let mut base: Option<[Vec<u16>; 3]> = None;
        for tau in [TOLERANCE, 3e-2, 1e-1, 3e-1, 1.0] {
            let mut e = HpEngine::new(Faults::default()).expect("the engine");
            e.tau = Some(tau);
            let cancel = AtomicBool::new(false);
            let mut walls = Vec::new();
            let mut planes = None;
            for _ in 0..2 {
                let t = Instant::now();
                planes = Some(e.analyze(&l, &r, sr, &geom, &cfg, &pool, &cancel, &|_| {}).expect("codes"));
                walls.push(t.elapsed().as_secs_f64());
            }
            let trace = e.trace;
            assert!(e.close().failures.is_empty());
            let planes = planes.unwrap();
            let hop = hop_for_fps(sr, geom.fps);
            let memo = plan_memo(sr, &geom, &cfg, 128);
            let (_, _, mod_plans, _) = plan_track(&plans(sr, &geom, &cfg), sr, &cfg, hop, frame_count(l.len(), hop), &pool, &memo, tau).expect("a plan");
            let mut mix: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
            for p in &mod_plans { *mix.entry((p.0, if p.1 == EXACT { 99 } else { p.1 })).or_default() += 1; }
            let basis: usize = mod_plans.iter().map(|p| p.4.iter().map(|b| b.len()).sum::<usize>()).sum();
            eprintln!("tau {tau}: passes {:.2}s, {:.2}s | {} | plans (bars per carrier, degree) x groups: {mix:?}, {basis} basis vectors", walls[0], walls[1], stages(&trace));
            match &base {
                None => base = Some(planes),
                Some(b) => {
                    let (mut differ, mut max) = ([0usize; 4], [0i32; 4]);
                    for i in 0..b[0].len() {
                        let d: [i32; 3] = std::array::from_fn(|k| planes[k][i] as i32 - b[k][i] as i32);
                        for (k, v) in [d[0], d[1], d[2], d[2] - d[1]].into_iter().enumerate() {
                            differ[k] += (v != 0) as usize;
                            max[k] = max[k].max(v.abs());
                        }
                    }
                    eprintln!("  against tau {TOLERANCE}: cells differing in Mix, L, R, Diff {differ:?} of {}, largest {max:?}", b[0].len());
                }
            }
        }
    }

    /// Research map, on the device: this route against Standard on one
    /// excerpt (`MOOSIK_HP_INPUTS`, `MOOSIK_HP_TRACK`, `MOOSIK_HP_SECS` from
    /// 0:20, default 20 s) at the validated settings with each `q,spread` of
    /// `MOOSIK_HP_MAP` (`;`-separated): the pass on a fresh engine (its plan
    /// included), Standard on the same pool, and the plan's options (bars per
    /// carrier, degree; 99 = all-f64) by groups. The high-rate spread rule is
    /// skipped so the map reaches below it.
    #[test]
    #[ignore = "a report, not a check"]
    fn spread_speed_map() {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let dir = std::path::PathBuf::from(var("MOOSIK_HP_INPUTS").expect("MOOSIK_HP_INPUTS"));
        let name = var("MOOSIK_HP_TRACK").unwrap_or_else(|| "song1_48k".into());
        let secs: f32 = var("MOOSIK_HP_SECS").and_then(|s| s.parse().ok()).unwrap_or(20.0);
        let sr = input_rate(&name);
        let (l, r) = excerpt(&dir.join(format!("{name}.f32")), sr, 20.0, secs);
        let (geom, pool) = (Geom::validated(), pool6());
        let (hop, frames) = (hop_for_fps(sr, geom.fps), frame_count(l.len(), hop_for_fps(sr, geom.fps)));
        for pair in var("MOOSIK_HP_MAP").expect("MOOSIK_HP_MAP").split(';') {
            let (q, spread) = pair.split_once(',').expect("q,spread");
            let cfg = AsltConfig { q_ratio: q.trim().parse().unwrap(), spread: spread.trim().parse().unwrap(), ..validated_cfg() };
            let now = admits(geom.n_bars, geom.lo, geom.hi, &cfg, geom.fps).is_ok() && spread_admits(sr, &cfg).is_ok();
            let cancel = AtomicBool::new(false);
            let mut e = HpEngine::new(Faults::default()).expect("the engine");
            e.any_spread = true;
            let t = Instant::now();
            e.analyze(&l, &r, sr, &geom, &cfg, &pool, &cancel, &|_| {}).expect("codes");
            let hp = t.elapsed().as_secs_f64();
            let trace = e.trace;
            assert!(e.close().failures.is_empty());
            let memo = plan_memo(sr, &geom, &cfg, 128);
            let (_, _, mod_plans, _) = plan_track(&plans(sr, &geom, &cfg), sr, &cfg, hop, frames, &pool, &memo, TOLERANCE).expect("a plan");
            let mut mix: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
            for p in &mod_plans { *mix.entry((p.0, if p.1 == EXACT { 99 } else { p.1 })).or_default() += 1; }
            let t = Instant::now();
            let _ = pool.install(|| analyze_input_codes(Input::Joint { left: &l, right: &r }, sr, geom.n_bars, geom.lo,
                (sr as f32 / 2.0).min(geom.hi), hop, &cfg, &|| true, &|_| {}, &|_, _| {}, &|m| crate::spectrum::aslt_code(m)));
            let std_s = t.elapsed().as_secs_f64();
            eprintln!("MAP {name} q {} spread {} | admitted now {now} | HP {hp:.1}s | Standard {std_s:.1}s | {:.2}x | {} | plans {mix:?}",
                      cfg.q_ratio, cfg.spread, std_s / hp, stages(&trace));
        }
    }

    // ── 1.6.2 review round: the helpers' failures and the thread budget ─────

    /// A helper thread the system will not start is a refusal, whichever of
    /// the two it is: no codes, the cores' route for the caller, the engine
    /// not quarantined, nothing left in custody after the pass's own cleanup
    /// (the signal's device buffers were live), the helper that did start
    /// joined before the call returns, and a clean close. The same engine
    /// then analyses normally. Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_helper_that_will_not_start_is_a_refusal() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        let whole = pass(false, &Geom::validated(), &cfg, &l, &r, 48_000, None, None).out.expect("codes");
        for nth in [1usize, 2] {
            let (pool, cancel) = (pool6(), AtomicBool::new(false));
            let mut e = HpEngine::new(Faults { helper: Some(nth), ..Default::default() }).expect("the engine");
            let t = Instant::now();
            let out = e.analyze(&l, &r, 48_000, &Geom::validated(), &cfg, &pool, &cancel, &|_| {});
            let wall = t.elapsed().as_secs_f64();
            eprintln!("helper {nth} refused: {:?} in {wall:.2}s, ledger {:?}", out.as_ref().err(), e.ledger());
            assert!(matches!(&out, Err(HpError::Unavailable(why)) if why.contains("no thread")), "helper {nth}: {:?}", out.as_ref().map(|_| "codes"));
            assert!(wall < 5.0, "helper {nth}: {wall:.1}s to retire");
            assert_eq!((e.quarantined(), e.ledger(), e.failures.len()), (None, (0, 0), 0), "helper {nth}");
            // The refusal is not a property of the engine: with the fault
            // gone it gives the codes.
            e.faults = Faults::default();
            let again = e.analyze(&l, &r, 48_000, &Geom::validated(), &cfg, &pool, &cancel, &|_| {}).expect("codes after the refusal");
            assert!(again == whole, "helper {nth}: other codes after the refusal");
            let close = e.close();
            assert!(close.failures.is_empty() && close.context_destroyed, "helper {nth}: {close:?}");
        }
    }

    /// A helper that stops by itself mid-pass, with the device's buffers
    /// live, fails the pass after its cleanup: a device failure for the
    /// caller (the cores' route), never codes and never a cancellation; both
    /// helpers joined; the engine quarantined and closed cleanly. Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_helper_that_stops_fails_the_pass_after_its_cleanup() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        for (what, faults) in [("the preparer at group 40", Faults { prepare_stop: Some(40), ..Default::default() }),
                               ("the folder at group 40", Faults { fold_stop: Some(40), ..Default::default() }),
                               ("the folder at the last group", Faults { fold_stop: Some(127), ..Default::default() })] {
            let t = Instant::now();
            let p = pass_on(&pool6(), faults, false, &Geom::validated(), &cfg, &l, &r, 48_000, None, None);
            eprintln!("{what}: {:?} in {:.2}s; failures {:?}; close {:?}", p.out.as_ref().err(), t.elapsed().as_secs_f64(), p.failures, p.close);
            assert!(matches!(&p.out, Err(HpError::Device(why)) if why.contains("helper")), "{what}: {:?}", p.out.as_ref().map(|_| "codes"));
            assert!(p.trace.observed.is_none(), "{what}: recorded as a cancellation");
            assert!(p.close.failures.is_empty() && p.close.context_destroyed, "{what}: {:?}", p.close);
            assert!(t.elapsed().as_secs_f64() < 30.0, "{what}: slow to retire");
        }
    }

    /// A cancellation and a failure in the same pass, each made to happen:
    /// the cancellation wins, no codes are returned, and what the failure
    /// leaves is kept.
    ///
    /// - A device failure after the cancellation: the pass is cancelled as
    ///   group 40's wait begins, and the last free of its cleanup fails (the
    ///   free's number is read from the same pass run without the fault).
    ///   Cancelled for the caller; the free's failure kept in `failures`;
    ///   the engine quarantined; the allocation still in the ledger, and
    ///   reclaimed by the close. Serial and pipelined alike.
    /// - A helper stopping after the cancellation: the progress callback,
    ///   which runs on the folding helper, sets the flag and then panics.
    ///   Cancelled for the caller, both helpers joined, a clean close. The
    ///   helper's stop is not recorded in this combination (see `run`).
    ///
    /// Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_cancellation_wins_over_a_failure_that_also_happened() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        let pool = pool6();
        for serial in [true, false] {
            let run = |faults: Faults| {
                let cancel = AtomicBool::new(false);
                let mut e = HpEngine::new(faults).expect("the engine");
                e.serial = serial;
                e.cancel_at_wait = Some(40);
                let out = e.analyze(&l, &r, 48_000, &Geom::validated(), &cfg, &pool, &cancel, &|_| {});
                (out, cancel.load(Ordering::Relaxed), e)
            };
            // Without the fault: how many frees the cancelled pass makes.
            let (out, flagged, e) = run(Faults::default());
            let frees = e.seen().free;
            assert!(out == Err(HpError::Cancelled) && flagged && frees > 0 && e.quarantined().is_none() && e.ledger() == (0, 0), "serial {serial}: setup");
            drop(e);
            // With it: the last of those frees, which is after the flag.
            let (out, flagged, mut e) = run(Faults { free: Some(frees), ..Default::default() });
            eprintln!("serial {serial}: free {frees} of {frees} failed after the cancellation: {:?}; failures {:?}; quarantined {:?}; ledger {:?}",
                      out.as_ref().err(), e.failures, e.quarantined(), e.ledger());
            assert!(flagged, "serial {serial}: the cancellation was not set");
            assert_eq!(e.seen().free, frees, "serial {serial}: the failing free was not reached");
            let fired = |f: &String| f.contains("cuMemFree failed");
            assert!(e.failures.iter().any(fired) || matches!(&out, Err(HpError::Device(why)) if fired(why)), "serial {serial}: the fault did not fire: {:?}", e.failures);
            assert_eq!(out.as_ref().err(), Some(&HpError::Cancelled), "serial {serial}: the cancellation did not win");
            assert!(e.failures.iter().any(|f| f.contains("cuMemFree failed")), "serial {serial}: the failure was not kept: {:?}", e.failures);
            assert!(e.quarantined().is_some_and(|q| q.contains("cuMemFree")), "serial {serial}: not quarantined");
            assert!(e.ledger().0 >= 1, "serial {serial}: the allocation left the ledger");
            let kept = e.ledger().0;
            e.failures.clear();
            let close = e.close();
            assert!(close.context_destroyed && close.failures.is_empty() && close.device_reclaimed >= kept, "serial {serial}: {close:?}");
        }
        // The folding helper stops by itself once it has set the flag.
        let (cancel, fired) = (AtomicBool::new(false), AtomicBool::new(false));
        let mut e = HpEngine::new(Faults::default()).expect("the engine");
        let progress = |bars: usize| if bars >= 320 {
            cancel.store(true, Ordering::Relaxed);
            fired.store(true, Ordering::Relaxed);
            panic!("injected: the folder stops after setting the cancellation");
        };
        let t = Instant::now();
        let out = e.analyze(&l, &r, 48_000, &Geom::validated(), &cfg, &pool, &cancel, &progress);
        eprintln!("the folder stopped after cancelling: {:?} in {:.2}s; failures {:?}", out.as_ref().err(), t.elapsed().as_secs_f64(), e.failures);
        assert!(cancel.load(Ordering::Relaxed) && fired.load(Ordering::Relaxed), "the helper's stop did not fire");
        assert_eq!(out.as_ref().err(), Some(&HpError::Cancelled), "the cancellation did not win");
        let close = e.close();
        assert!(close.context_destroyed && close.failures.is_empty(), "{close:?}");
    }

    /// A device failure while the helpers are busy ends the pipelined pass as
    /// it ends the serial one: the same error for the caller, no codes, a
    /// clean close that reclaims what was live. A failed launch (the
    /// engine is quarantined), a failed allocation (it is not) and a failed
    /// copy wait, each well into the pass. Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_device_failure_mid_pass_ends_like_the_serial_one() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        for (what, faults) in [("launch 300", Faults { launch: Some(300), ..Default::default() }),
                               ("allocation 300", Faults { alloc: Some(300), ..Default::default() }),
                               ("copy wait 300", Faults { copy_wait: Some(300), ..Default::default() })] {
            let a = pass_on(&pool6(), faults, true, &Geom::validated(), &cfg, &l, &r, 48_000, None, None);
            let t = Instant::now();
            let b = pass_on(&pool6(), faults, false, &Geom::validated(), &cfg, &l, &r, 48_000, None, None);
            eprintln!("{what}: serial {:?}, pipelined {:?} in {:.2}s; close {:?} / {:?}", a.out.as_ref().err(), b.out.as_ref().err(), t.elapsed().as_secs_f64(),
                      (a.close.device_reclaimed, a.close.host_released), (b.close.device_reclaimed, b.close.host_released));
            assert!(a.out.is_err() && a.out.as_ref().err() == b.out.as_ref().err(), "{what}");
            assert!(matches!(&a.out, Err(HpError::Device(why) | HpError::Unavailable(why)) if why.contains("injected")), "{what}: the fault did not fire: {:?}", a.out.as_ref().err());
            assert_eq!(a.failures, b.failures, "{what}: failures kept");
            for q in [&a, &b] { assert!(q.close.failures.is_empty() && q.close.context_destroyed, "{what}: {:?}", q.close); }
            // What is left for close: nothing after a recoverable failure;
            // after a quarantine, whatever was live, which the context's
            // destroy reclaims. The pipelined pass has up to three groups'
            // buffers live where the serial one has one, so the counts differ.
            let left = |q: &Pass| (q.close.device_reclaimed, q.close.host_released);
            if matches!(a.out, Err(HpError::Unavailable(_))) { assert_eq!((left(&a), left(&b)), ((0, 0), (0, 0)), "{what}: left for close"); }
            else { assert!(left(&a).0 > 0 && left(&b).0 > 0, "{what}: left for close {:?} / {:?}", left(&a), left(&b)); }
        }
    }

    /// The pass keeps to the pool it is given: at budgets of 1, 2 and 6 every
    /// piece of its host work that is more than a copy (a group's
    /// preparation, each bar's fold, the planes' filling) runs on that pool's
    /// threads, no more pieces at once than the budget, and the codes are the
    /// same at every budget. The device thread is left with uploads,
    /// downloads and waits. Real device.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn the_pass_keeps_to_the_thread_budget() {
        let cfg = validated_cfg();
        let (l, r) = stereo(48_000, 20.0);
        let mut codes: Vec<[Vec<u16>; 3]> = Vec::new();
        for budget in [1usize, 2, 6] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(budget).thread_name(|i| format!("hp-budget-{i}")).build().unwrap();
            probe::PEAK.store(0, Ordering::Relaxed);
            probe::THREADS.lock().unwrap().clear();
            let p = pass_on(&pool, Faults::default(), false, &Geom::validated(), &cfg, &l, &r, 48_000, None, None);
            let (peak, seen) = (probe::PEAK.load(Ordering::Relaxed), std::mem::take(&mut *probe::THREADS.lock().unwrap()));
            eprintln!("budget {budget}: {:.2}s, {} pieces of work, at most {peak} at once; {}", p.wall, seen.len(), stages(&p.trace));
            assert!(seen.len() > 1024, "budget {budget}: setup: {} pieces seen", seen.len());
            for (name, n) in &seen {
                assert!(name.starts_with("hp-budget-"), "budget {budget}: work ran on {name:?}");
                assert_eq!(*n, budget, "budget {budget}: a pool of {n} threads ran it");
            }
            assert!(peak <= budget, "budget {budget}: {peak} pieces at once");
            assert!(budget == 1 || peak > 1, "budget {budget}: setup: nothing overlapped");
            codes.push(p.out.expect("codes"));
        }
        assert!(codes.windows(2).all(|w| w[0] == w[1]), "the codes depend on the budget");
        let serial = pass(true, &Geom::validated(), &cfg, &l, &r, 48_000, None, None).out.expect("codes");
        assert!(codes[0] == serial, "not the serial pass's codes");
    }

    // ── 1.6.2 research: accuracy at other settings, against a stored-kernel reference ──
    //
    // What is compared, each kept apart:
    //   1. The finite stored-kernel response (the definition): for one member
    //      and channel, the sum over the kernel's taps of sample × tap, the
    //      samples and the stored `re`/`im` taps taken as the exact f32 values
    //      they are. Interior frames (the whole kernel inside the signal) are
    //      not divided; edge frames are divided by the sum of the stored
    //      envelope taps that survive, as production does. Not a continuous
    //      wavelet, and no wider support.
    //   2. Its numerical value here: both operands promoted to f64 before the
    //      product (exact: 24 + 24 bits), summed with the rounding errors
    //      carried (`Sum2`). `certify_witnesses.py` recomputes chosen cells in
    //      exact rational arithmetic from the dumped bits, without this code.
    //   3. The fold: member = 2·sum/divisor; Mix = (L + R)/2 per member;
    //      magnitude; the weighted mean of ln(max(magnitude, floor)) over the
    //      members with the stored weights; exp. All in f64. (The engine and
    //      Standard round each member to f32 before this; the reference does
    //      not.)
    //   4. The code: `shipped` rounds the folded magnitude to f32 and applies
    //      the shipped f32 mapper (`aslt_code`); `wide` maps in f64. Cells
    //      where the two differ are counted, and the engine is held to
    //      `shipped`.
    //   5. Earlier codes: the validated settings' codes are unchanged (the
    //      app-seam check against `refs_r8t_N`), whatever this reference says.

    /// A sum of f64 terms that carries its rounding errors (TwoSum per term,
    /// errors summed apart): off by about n·2⁻¹⁰⁶·Σ|term| plus one rounding.
    #[derive(Default, Clone, Copy)]
    struct Sum2 { hi: f64, lo: f64 }

    impl Sum2 {
        fn add(&mut self, x: f64) {
            let s = self.hi + x;
            let v = s - self.hi;
            self.lo += (self.hi - (s - v)) + (x - v);
            self.hi = s;
        }
        fn get(self) -> f64 { self.hi + self.lo }
    }

    /// One member's stored-kernel sums for one channel at `centre` and the
    /// divisor: 1 when the whole kernel is inside the signal, the surviving
    /// envelope's sum at an edge, 0 when nothing survives.
    fn member_exact(w: &Morlet, x: &[f32], centre: isize) -> (f64, f64, f64) {
        let base = centre - w.half as isize;
        let (lo, hi) = (base.max(0), (base + w.re.len() as isize).min(x.len() as isize));
        if hi <= lo { return (0.0, 0.0, 0.0); }
        let (k0, k1) = ((lo - base) as usize, (hi - base) as usize);
        let whole = k0 == 0 && k1 == w.re.len();
        let (mut re, mut im, mut env) = (Sum2::default(), Sum2::default(), Sum2::default());
        for (k, &s) in (k0..k1).zip(&x[lo as usize..hi as usize]) {
            re.add(s as f64 * w.re[k] as f64);
            im.add(s as f64 * w.im[k] as f64);
            if !whole { env.add(w.env[k] as f64); }
        }
        (re.get(), im.get(), if whole { 1.0 } else { env.get() })
    }

    /// The reference for one cell (Mix, L, R): the code by the shipped f32
    /// mapper, the code mapped in f64, and the f64 code position unrounded.
    struct Cell { shipped: [u16; 3], wide: [u16; 3], pos: [f64; 3] }

    fn exact_cell(sl: &Superlet, l: &[f32], r: &[f32], centre: isize) -> Cell {
        let mut acc = [0f64; 3];
        let part = |v: f64, d: f64| if d > 0.0 { 2.0 * v / d } else { 0.0 };
        for (w, weight) in sl.wavelets.iter().zip(sl.weights.iter()) {
            let ((lr, li, ld), (rr, ri, rd)) = (member_exact(w, l, centre), member_exact(w, r, centre));
            let (lr, li, rr, ri) = (part(lr, ld), part(li, ld), part(rr, rd), part(ri, rd));
            let (mr, mi) = (0.5 * (lr + rr), 0.5 * (li + ri));
            let mags = [(mr * mr + mi * mi).sqrt(), (lr * lr + li * li).sqrt(), (rr * rr + ri * ri).sqrt()];
            for p in 0..3 { acc[p] += *weight as f64 * mags[p].max(MAG_FLOOR as f64).ln(); }
        }
        let mags = acc.map(|a| (a / sl.total_weight as f64).exp());
        let max = crate::spectrum::cache::V4_MAX_CODE as f64;
        let pos = mags.map(|m| (((20.0 * (m * crate::spectrum::ASLT_DB_REF as f64).log10()).max(-80.0) + 80.0) / 80.0).clamp(0.0, 1.0) * max);
        Cell { shipped: mags.map(|m| crate::spectrum::aslt_code(m as f32)), wide: pos.map(|p| p.round() as u16), pos }
    }

    /// The first screen's reference (2026-11-12), kept to count what its f32
    /// products changed: `response_parts` multiplies in f32 before it sums in
    /// f64. Interior frames only.
    fn f32_product_codes(sl: &Superlet, l: &[f32], r: &[f32], centre: isize) -> [u16; 3] {
        let mut acc = [0f64; 3];
        for (w, weight) in sl.wavelets.iter().zip(sl.weights.iter()) {
            let ((lr, li, ld), (rr, ri, rd)) = (w.response_parts(l, centre), w.response_parts(r, centre));
            let (lr, li, rr, ri) = (2.0 * lr / ld, 2.0 * li / ld, 2.0 * rr / rd, 2.0 * ri / rd);
            let (mr, mi) = (0.5 * (lr + rr), 0.5 * (li + ri));
            let mags = [(mr * mr + mi * mi).sqrt(), (lr * lr + li * li).sqrt(), (rr * rr + ri * ri).sqrt()];
            for p in 0..3 { acc[p] += *weight as f64 * mags[p].max(MAG_FLOOR as f64).ln(); }
        }
        finish64(acc, sl.total_weight).map(crate::spectrum::aslt_code)
    }

    /// A research input's rate, from its name's suffix (`song1_48k`, `x_192k`).
    fn input_rate(name: &str) -> u32 {
        let k = name.rsplit('_').next().and_then(|s| s.strip_suffix('k')).expect("a rate suffix such as _48k");
        match k { "44" => 44_100, "48" => 48_000, "88" => 88_200, "96" => 96_000, "176" => 176_400, "192" => 192_000,
                  "352" => 352_800, o => panic!("unknown rate suffix _{o}k") }
    }

    /// One whole track through the engine at the validated descriptor, timed
    /// (`MOOSIK_HP_INPUTS`, `MOOSIK_HP_TRACK`): the high-resolution check.
    #[test]
    #[ignore = "needs the NVIDIA device and an input; run by hand"]
    fn hp_track_once() {
        let dir = std::path::PathBuf::from(std::env::var("MOOSIK_HP_INPUTS").expect("MOOSIK_HP_INPUTS"));
        let name = std::env::var("MOOSIK_HP_TRACK").expect("MOOSIK_HP_TRACK");
        let sr = input_rate(&name);
        let (l, r) = excerpt(&dir.join(format!("{name}.f32")), sr, 0.0, 100_000.0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        let mut e = HpEngine::new(Faults::default()).expect("the NVIDIA device");
        let t = Instant::now();
        let got = e.analyze(&l, &r, sr, &Geom::validated(), &validated_cfg(), &pool, &AtomicBool::new(false), &|_| {});
        let wall = t.elapsed().as_secs_f64();
        let (launches, stages) = (e.seen.launch, e.trace.stage_s);
        let close = e.close();
        eprintln!("HP track {name}: {:.1} s at {sr} Hz in {wall:.1} s, {launches} launches, stages {stages:.2?}; {}; context destroyed {}",
                  l.len() as f64 / sr as f64, match &got { Ok(p) => format!("{} cells a plane", p[0].len()), Err(e) => format!("{e:?}") },
                  close.context_destroyed);
        let planes = got.expect("codes");
        if let Ok(out) = std::env::var("MOOSIK_HP_WRITE") {
            let out = std::path::PathBuf::from(out);
            std::fs::create_dir_all(&out).unwrap();
            for (plane, which) in planes.iter().zip(["mix", "left", "right"]) {
                std::fs::write(out.join(format!("G_{which}.u16")), plane.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
        }
    }

    fn excerpt(path: &std::path::Path, sr: u32, from_s: f32, secs: f32) -> (Vec<f32>, Vec<f32>) {
        let bytes = std::fs::read(path).expect("input");
        let s: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let (a, n) = ((from_s * sr as f32) as usize * 2, (secs * sr as f32) as usize * 2);
        let s = &s[a.min(s.len())..(a + n).min(s.len())];
        (s.iter().step_by(2).copied().collect(), s.iter().skip(1).step_by(2).copied().collect())
    }

    /// One cell's inputs as bits, for `certify_witnesses.py`: per member its
    /// weight, the surviving taps and the samples under them.
    fn dump_witness(path: &std::path::Path, head: &str, sl: &Superlet, l: &[f32], r: &[f32], centre: isize) {
        use std::fmt::Write;
        let hex = |v: &[f32]| v.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>().join(" ");
        let mut s = format!("{head}\ntotal_weight {:08x}\n", sl.total_weight.to_bits());
        for (w, weight) in sl.wavelets.iter().zip(sl.weights.iter()) {
            let base = centre - w.half as isize;
            let (lo, hi) = (base.max(0), (base + w.re.len() as isize).min(l.len() as isize));
            let (k0, k1) = ((lo - base).max(0) as usize, (hi - base).max(0) as usize);
            let (a, b) = (lo.max(0) as usize, hi.max(lo).max(0) as usize);
            writeln!(s, "member weight {:08x} whole {}", weight.to_bits(), (k0 == 0 && k1 == w.re.len()) as u8).unwrap();
            for (name, v) in [("re", &w.re[k0..k1.max(k0)]), ("im", &w.im[k0..k1.max(k0)]), ("env", &w.env[k0..k1.max(k0)]), ("l", &l[a..b]), ("r", &r[a..b])] {
                writeln!(s, "{name} {}", hex(v)).unwrap();
            }
        }
        std::fs::write(path, s).expect("the witness file");
    }

    /// What one row comes to in the gate: the word its report carries, and
    /// why it fails the gate if it does. `outcome` is the row's worst code
    /// difference, or why the engine gave no codes. An admitted setting must
    /// give codes and be within one; a setting the route does not admit is
    /// reported and fails nothing.
    fn gate_row(admitted: bool, outcome: &Result<i32, String>) -> (&'static str, Option<String>) {
        match (admitted, outcome) {
            (_, Ok(worst)) if *worst <= 1 => ("WITHIN 1", None),
            (true, Ok(worst)) => ("OVER", Some(format!("{worst} codes off"))),
            (false, Ok(_)) => ("OVER", None),
            (true, Err(why)) => ("NO CODES", Some(format!("no codes: {why}"))),
            (false, Err(_)) => ("NO CODES", None),
        }
    }

    /// The gate's decision, without a device: an admitted setting passes
    /// within one code and fails beyond it or without codes; a setting the
    /// route does not admit is reported either way and never fails the gate.
    #[test]
    fn the_gate_fails_an_admitted_setting_that_is_off_or_gives_no_codes() {
        let cfg = validated_cfg();
        let admitted = admits(1024, 20.0, 24_000.0, &cfg, 180.0).is_ok();
        let unsupported = admits(1024, 20.0, 24_000.0, &cfg, 60.0).is_ok();
        assert!(admitted && !unsupported, "setup");
        let refused: Result<i32, String> = Err("Unavailable(\"refused\")".into());
        assert_eq!(gate_row(admitted, &Ok(0)), ("WITHIN 1", None));
        assert_eq!(gate_row(admitted, &Ok(1)), ("WITHIN 1", None));
        assert_eq!(gate_row(admitted, &Ok(2)), ("OVER", Some("2 codes off".into())));
        assert_eq!(gate_row(admitted, &refused), ("NO CODES", Some("no codes: Unavailable(\"refused\")".into())));
        assert_eq!(gate_row(unsupported, &Ok(1)), ("WITHIN 1", None));
        assert_eq!(gate_row(unsupported, &Ok(5)), ("OVER", None));
        assert_eq!(gate_row(unsupported, &refused), ("NO CODES", None));
    }

    /// The accuracy gate and its report, by hand, on the device: the engine's
    /// codes against the stored-kernel reference (above) at settings across
    /// the controls' ranges. Fails if a setting the route admits gives no
    /// codes, or has a cell more than one code off in Mix, L, R or the Diff
    /// view (R − L), interior or edge (`gate_row`).
    ///
    /// - `MOOSIK_HP_INPUTS`: the research inputs' folder (interleaved f32
    ///   stereo, `<name>_44k.f32` or `_48k.f32`); `MOOSIK_HP_SWEEP_INPUTS`
    ///   which of them; `MOOSIK_HP_SWEEP_SECS` the excerpt's length.
    /// - `MOOSIK_HP_SWEEP`: `designed`, or a seed for sixteen draws;
    ///   `MOOSIK_HP_SWEEP_ONLY`: only the cases so named (`;` between).
    /// - `MOOSIK_HP_SWEEP_DENSE`: bars (`,` between) around which, 24 either
    ///   side, every frame is compared; elsewhere about 240 interior frames
    ///   and 48 edge frames a side per bar.
    /// - `MOOSIK_HP_SWEEP_STD`: Standard on the same excerpt, as well.
    /// - `MOOSIK_HP_WITNESS_DIR`: where cells are dumped for the independent
    ///   certification: every cell two or more off, the cells named in
    ///   `MOOSIK_HP_WITNESS` (`bar,frame;…`), and per row the cell whose Mix
    ///   is furthest below its channels.
    #[test]
    #[ignore = "the accuracy gate, on the NVIDIA device"]
    fn sweep_against_the_direct_reference() {
        use crate::spectrum::freq_scale::FreqScale;
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let dir = std::path::PathBuf::from(var("MOOSIK_HP_INPUTS").expect("MOOSIK_HP_INPUTS"));
        let which = var("MOOSIK_HP_SWEEP").unwrap_or_else(|| "designed".into());
        let secs: f32 = var("MOOSIK_HP_SWEEP_SECS").and_then(|s| s.parse().ok()).unwrap_or(30.0);
        let only: Option<Vec<String>> = var("MOOSIK_HP_SWEEP_ONLY").map(|s| s.split(';').map(|n| n.trim().to_string()).collect());
        let dense: Vec<usize> = var("MOOSIK_HP_SWEEP_DENSE").map(|s| s.split(',').map(|b| b.trim().parse().expect("a bar")).collect()).unwrap_or_default();
        let named: Vec<(usize, usize)> = var("MOOSIK_HP_WITNESS").map(|s| s.split(';').map(|c| {
            let (b, f) = c.split_once(',').expect("bar,frame");
            (b.trim().parse().expect("a bar"), f.trim().parse().expect("a frame"))
        }).collect()).unwrap_or_default();
        let witness_dir = var("MOOSIK_HP_WITNESS_DIR").map(std::path::PathBuf::from);
        let with_std = var("MOOSIK_HP_SWEEP_STD").is_some();
        let v = validated_cfg();
        let vg = Geom::validated();
        let cfg = |q: f32, win: f32, n: usize, spread: f32, scale: FreqScale| AsltConfig { q_ratio: q, max_window_s: win, n_wavelets: n, spread, scale };
        let mut cases: Vec<(String, Geom, AsltConfig)> = Vec::new();
        if which == "designed" {
            cases.push(("validated".into(), vg, v.clone()));
            for (name, p) in [("Fast", AsltPreset::Fast), ("Standard", AsltPreset::Standard), ("High", AsltPreset::High),
                              ("Ultra", AsltPreset::Ultra), ("Extreme", AsltPreset::Extreme)] {
                cases.push((format!("preset {name}"), vg, p.config()));
            }
            cases.push(("validated, ERB".into(), vg, AsltConfig { scale: FreqScale::Erb, ..v.clone() }));
            cases.push(("validated, Log".into(), vg, AsltConfig { scale: FreqScale::Log, ..v.clone() }));
            cases.push(("1 wavelet".into(), vg, cfg(1.0, 0.25, 1, 0.5, FreqScale::Blend(0.5))));
            cases.push(("q 0.25, spread 0".into(), vg, cfg(0.25, 0.25, 9, 0.0, FreqScale::Blend(0.5))));
            cases.push(("q 0.25, spread 0.5".into(), vg, cfg(0.25, 0.25, 9, 0.5, FreqScale::Blend(0.5))));
            cases.push(("q 2, spread 0".into(), vg, cfg(2.0, 0.25, 9, 0.0, FreqScale::Blend(0.5))));
            // The spread rule's edges (`spread_admits`): 0.1 at 44.1 and 48 kHz, 0.3 above.
            cases.push(("q 0.25, spread 0.1".into(), vg, cfg(0.25, 0.25, 9, 0.1, FreqScale::Blend(0.5))));
            cases.push(("q 2, spread 0.1".into(), vg, cfg(2.0, 0.25, 9, 0.1, FreqScale::Blend(0.5))));
            cases.push(("q 0.25, spread 0.3".into(), vg, cfg(0.25, 0.25, 9, 0.3, FreqScale::Blend(0.5))));
            cases.push(("q 2, spread 0.3".into(), vg, cfg(2.0, 0.25, 9, 0.3, FreqScale::Blend(0.5))));
            cases.push(("q 2, spread 0.95, window 0.1".into(), vg, cfg(2.0, 0.1, 9, 0.95, FreqScale::Blend(0.5))));
            cases.push(("window 8 s".into(), vg, cfg(1.0, 8.0, 5, 0.5, FreqScale::Log)));
            for (name, g) in [("512 bars, 60 fps, to 20 kHz", Geom { n_bars: 512, lo: 20.0, hi: 20_000.0, fps: 60.0 }),
                              ("2048 bars, 120 fps", Geom { n_bars: 2048, lo: 20.0, hi: 24_000.0, fps: 120.0 }),
                              ("256 bars, 240 fps, 30 Hz-16 kHz", Geom { n_bars: 256, lo: 30.0, hi: 16_000.0, fps: 240.0 }),
                              ("1024 bars, 60 fps", Geom { n_bars: 1024, lo: 20.0, hi: 24_000.0, fps: 60.0 }),
                              ("100 bars, 24 fps", Geom { n_bars: 100, lo: 20.0, hi: 24_000.0, fps: 24.0 }),
                              ("1024 bars, 480 fps", Geom { n_bars: 1024, lo: 20.0, hi: 24_000.0, fps: 480.0 })] {
                cases.push((format!("validated settings, {name}"), g, v.clone()));
            }
        } else {
            // Seeded draws over the controls' ranges.
            let mut seed: u64 = which.parse().expect("MOOSIK_HP_SWEEP: `designed` or a seed");
            let mut unit = move || {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            for i in 0..16 {
                let q = (0.25 + 1.75 * unit()) as f32;
                let win = (0.1 * 80f64.powf(unit())) as f32;
                let n = 1 + (unit() * 9.0) as usize;
                let spread = (0.95 * unit()) as f32;
                let scale = match (unit() * 4.0) as usize { 0 => FreqScale::Log, 1 => FreqScale::Erb, _ => FreqScale::Blend((-0.5 + 2.0 * unit()) as f32) };
                let g = Geom { n_bars: [256, 512, 1024, 1024, 2048][(unit() * 5.0) as usize], lo: [20.0, 20.0, 30.0, 50.0][(unit() * 4.0) as usize],
                               hi: [24_000.0, 22_000.0, 20_000.0, 16_000.0][(unit() * 4.0) as usize], fps: [60.0, 120.0, 180.0, 180.0, 240.0][(unit() * 5.0) as usize] };
                cases.push((format!("draw {i}"), g, cfg(q, win.min(8.0), n.min(9), spread, scale)));
            }
        }
        if let Some(only) = &only { cases.retain(|c| only.contains(&c.0)); }
        assert!(!cases.is_empty(), "no case is named so");
        let inputs: Vec<(String, u32)> = var("MOOSIK_HP_SWEEP_INPUTS").unwrap_or_else(|| "song1_48k,song2_44k".into())
            .split(',').map(|n| (n.to_string(), input_rate(n))).collect();
        let pool = pool6();
        // What one screen of three planes found: [interior, edge] × (largest
        // difference in Mix, L, R, Diff; cells two or more off; cells), the
        // cells whose f64-mapped code is not the shipped mapper's, and the
        // cells the first screen's f32-product reference coded otherwise.
        #[derive(Default, Clone, Copy)]
        struct Tally { max: [[i32; 4]; 2], over: [[usize; 4]; 2], cells: [usize; 2], mapper: usize, f32_ref: usize }
        let mut gate: Vec<String> = Vec::new();
        for (name, geom, cfg) in &cases {
            for (input, sr) in &inputs {
                let (l, r) = excerpt(&dir.join(format!("{input}.f32")), *sr, 20.0, secs);
                // An input shorter than that start is taken from its beginning.
                let (l, r) = if l.len() < *sr as usize { excerpt(&dir.join(format!("{input}.f32")), *sr, 0.0, secs) } else { (l, r) };
                let admitted = admits(geom.n_bars, geom.lo, geom.hi, cfg, geom.fps).is_ok() && spread_admits(*sr, cfg).is_ok();
                let t = Instant::now();
                let got = pass(false, geom, cfg, &l, &r, *sr, None, None);
                let wall = t.elapsed().as_secs_f64();
                let head = format!("{name} | {geom:?} | {cfg:?} | {input} | {}", if admitted { "admitted" } else { "not admitted" });
                let planes = match got.out {
                    Ok(p) => p,
                    Err(e) => {
                        let (word, fails) = gate_row(admitted, &Err(format!("{e:?}")));
                        eprintln!("ROW {head} | {word} {e:?}");
                        gate.extend(fails.map(|why| format!("{name} | {input}: {why}")));
                        continue;
                    }
                };
                assert!(got.failures.is_empty() && got.close.failures.is_empty(), "{head}: {:?} {:?}", got.failures, got.close);
                let hop = hop_for_fps(*sr, geom.fps);
                let frames = frame_count(l.len(), hop);
                let plans = plans(*sr, geom, cfg);
                let standard = with_std.then(|| {
                    let t = Instant::now();
                    let a = pool.install(|| analyze_input_codes(Input::Joint { left: &l, right: &r }, *sr, geom.n_bars, geom.lo,
                        (*sr as f32 / 2.0).min(geom.hi), hop, cfg, &|| true, &|_| {}, &|_, _| {}, &|m| crate::spectrum::aslt_code(m)));
                    (a.codes.map(|c| c.0), t.elapsed().as_secs_f64())
                });
                let std_planes = standard.as_ref().and_then(|s| s.0.as_ref());
                let tag = format!("{}_{input}", name.replace([' ', ','], "_"));
                let t = Instant::now();
                // Per bar: its interior frames and its edge frames, sampled
                // (every frame near a `dense` bar).
                let screen = |planes: &[Vec<u16>; 3], what: &str, dump: bool| -> Tally {
                    let rows: Vec<(Tally, Option<(i32, usize)>)> = pool.install(|| plans.par_iter().map(|p| {
                        let sl = Superlet::new(p.freq, p.q, cfg, *sr as f32);
                        let half = sl.wavelets.iter().map(|w| w.half).max().unwrap();
                        // Frames first..=last have every member's kernel inside the signal.
                        let (first, last) = (half.div_ceil(hop), (l.len().saturating_sub(1 + half)) / hop);
                        let every = dense.iter().any(|&d| p.bar.abs_diff(d) <= 24);
                        let has_interior = first <= last && first < frames;
                        let last = last.min(frames - 1);
                        let step = (last.saturating_sub(first) / 240).max(1);
                        let mut js: Vec<usize> = if every {
                            (0..frames).collect()
                        } else if !has_interior {
                            (0..frames).step_by((frames / 96).max(1)).collect()
                        } else {
                            (0..first).step_by((first / 48).max(1)).chain((first + p.bar % step..=last).step_by(step))
                                .chain((last + 1..frames).step_by(((frames - last) / 48).max(1))).collect()
                        };
                        js.extend(named.iter().filter(|n| n.0 == p.bar && n.1 < frames).map(|n| n.1));
                        js.sort_unstable();
                        js.dedup();
                        let mut tally = Tally::default();
                        let mut cancelled: Option<(i32, usize)> = None;
                        for j in js {
                            let interior = has_interior && j >= first && j <= last;
                            let k = !interior as usize;
                            let centre = (j * hop) as isize;
                            let cell = exact_cell(&sl, &l, &r, centre);
                            let d = cell.shipped.map(|c| c as i32);
                            let g: [i32; 3] = std::array::from_fn(|q| planes[q][j * geom.n_bars + p.bar] as i32);
                            let e = [g[0] - d[0], g[1] - d[1], g[2] - d[2], (g[2] - g[1]) - (d[2] - d[1])];
                            for q in 0..4 {
                                tally.max[k][q] = tally.max[k][q].max(e[q].abs());
                                tally.over[k][q] += (e[q].abs() >= 2) as usize;
                            }
                            tally.cells[k] += 1;
                            tally.mapper += (cell.shipped != cell.wide) as usize;
                            tally.f32_ref += (interior && f32_product_codes(&sl, &l, &r, centre) != cell.shipped) as usize;
                            let two_off = e.iter().any(|v| v.abs() >= 2);
                            if two_off {
                                eprintln!("  two off ({what}, {}): bar {} ({:.1} Hz) frame {j}: codes {g:?}, reference {d:?}, f64-mapped {:?} at {:.4?}",
                                          if interior { "interior" } else { "edge" }, p.bar, p.freq, cell.wide, cell.pos);
                            }
                            if dump && (two_off || named.contains(&(p.bar, j))) && let Some(dir) = &witness_dir {
                                let s = std_planes.map(|s| format!("{:?}", [0, 1, 2].map(|q| s[q][j * geom.n_bars + p.bar])));
                                dump_witness(&dir.join(format!("{tag}_bar{}_frame{j}.txt", p.bar)),
                                    &format!("cell {name} | {input} | bar {} | frame {j} | {} | engine {g:?} | standard {} | shipped {d:?} | wide {:?}",
                                             p.bar, if interior { "interior" } else { "edge" }, s.as_deref().unwrap_or("-"), cell.wide), &sl, &l, &r, centre);
                            }
                            // How far Mix sits below the quieter channel, where it still shows.
                            if interior && d[0] > 0 && cancelled.is_none_or(|c| d[1].min(d[2]) - d[0] > c.0) { cancelled = Some((d[1].min(d[2]) - d[0], j)); }
                        }
                        (tally, cancelled.map(|c| (c.0, p.bar * frames + c.1)))
                    }).collect());
                    let mut all = Tally::default();
                    for (t, _) in &rows {
                        for k in 0..2 {
                            for q in 0..4 { all.max[k][q] = all.max[k][q].max(t.max[k][q]); all.over[k][q] += t.over[k][q]; }
                            all.cells[k] += t.cells[k];
                        }
                        all.mapper += t.mapper;
                        all.f32_ref += t.f32_ref;
                    }
                    if dump && let (Some(dir), Some((depth, at))) = (&witness_dir, rows.iter().filter_map(|r| r.1).max()) {
                        let (bar, j) = (at / frames, at % frames);
                        let p = &plans[bar];
                        let sl = Superlet::new(p.freq, p.q, cfg, *sr as f32);
                        let g = [0, 1, 2].map(|q| planes[q][j * geom.n_bars + bar]);
                        let cell = exact_cell(&sl, &l, &r, (j * hop) as isize);
                        dump_witness(&dir.join(format!("{tag}_cancel_bar{bar}_frame{j}.txt")),
                            &format!("cell {name} | {input} | bar {bar} | frame {j} | interior, Mix {depth} codes under the quieter channel | engine {g:?} | standard - | shipped {:?} | wide {:?}",
                                     cell.shipped, cell.wide), &sl, &l, &r, (j * hop) as isize);
                    }
                    all
                };
                let line = |t: &Tally| format!("interior max {:?} >=2 {:?} of {} | edge max {:?} >=2 {:?} of {} | f64 mapper differs in {} | f32-product reference differs in {}",
                                               t.max[0], t.over[0], t.cells[0], t.max[1], t.over[1], t.cells[1], t.mapper, t.f32_ref);
                let mine = screen(&planes, "engine", true);
                let worst = mine.max.iter().flatten().copied().max().unwrap();
                let std_line = match &standard {
                    Some((Some(p), s)) => format!(" | Standard {s:.1}s: {}", line(&screen(p, "Standard", false))),
                    Some((None, s)) => format!(" | Standard {s:.1}s (no codes)"),
                    None => String::new(),
                };
                let (word, fails) = gate_row(admitted, &Ok(worst));
                eprintln!("ROW {head} | {word} | engine {} | pass {wall:.1}s ref {:.1}s{std_line}", line(&mine), t.elapsed().as_secs_f64());
                gate.extend(fails.map(|why| format!("{name} | {input}: {why}")));
            }
        }
        assert!(gate.is_empty(), "admitted settings that fail the gate: {gate:#?}");
    }

    fn sha256(b: &[u8]) -> String {
        ring::digest::digest(&ring::digest::SHA256, b).as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The kernels are the reviewed ones, byte for byte: the pins the research
    /// runs (rounds 8–11) recorded, whatever line endings a checkout gave them.
    #[test]
    fn the_ptx_is_the_pinned_ptx() {
        assert_eq!(sha256(pinned(SGEMM_PTX).as_bytes()), "b682d1565eb9ce3ee779e909b01069abd4d8629591535732bb12906ebad72dca");
        assert_eq!(sha256(pinned(DGEMM_PTX).as_bytes()), "147a978fd5294f2f3ad9207396c9107cf7af1eab313b966e9c4a5fc857654627");
        assert_eq!(sha256(pinned(RANGED_PTX).as_bytes()), "db07b5e9a05dd5ad35ce77d31339e7e75fe49f49e6a22e7cd94789ebf55547ab");
        assert!(RANGED_PTX.contains(".entry sgemm_v2_ranged("), "sgemm_v2_ranged missing");
        for entry in ["acc64", "cvt32"] {
            assert!(ACC64_PTX.contains(&format!(".visible .entry {entry}(")), "{entry} missing");
        }
    }

    /// A group's tiles share a launch only when they are the grid's own
    /// blocks, each over pieces; anything else keeps a launch per tile.
    #[test]
    fn only_aligned_tiles_share_a_launch() {
        assert!(aligned_ranges(144, 45, &[(0, 72, 10, 34), (72, 72, 0, 45)]).is_some());
        assert!(aligned_ranges(80, 7, &[(0, 72, 1, 6), (72, 8, 0, 7)]).is_some());
        for (n, q, tiles) in [
            (0, 1, vec![]), (72, 0, vec![(0, 72, 0, 1)]),
            (360, 7, (0..5).map(|i| (72 * i, 72, 0, 7)).collect()),
            (144, 7, vec![(0, 72, 0, 7)]),
            (144, 7, vec![(0, 72, 0, 7), (71, 72, 0, 7)]),
            (144, 7, vec![(0, 71, 0, 7), (72, 72, 0, 7)]),
            (72, 7, vec![(0, 72, 3, 3)]), (72, 7, vec![(0, 72, 4, 3)]),
            (72, 7, vec![(0, 72, 0, 8)]),
        ] { assert!(aligned_ranges(n, q, &tiles).is_none(), "accepted {n} columns, q {q}, {tiles:?}"); }
        // The production layout: nine members fill two blocks; five put
        // their short remainder first, and keep a launch per tile.
        for members in [5usize, 9] {
            let cfg = AsltConfig { n_wavelets: members, ..validated_cfg() };
            let group: Vec<_> = plans(48_000, &Geom::validated(), &cfg).into_iter().skip(880).take(8)
                .map(|p| Superlet::new(p.freq, p.q, &cfg, 48_000.0)).collect();
            let (cols, tiles) = tile_layout(&group, 267);
            let (_, q, _, _) = gemm_pieces_at(&group, 267, Some(&cols));
            assert_eq!(aligned_ranges(2 * cols.len(), q, &tiles).is_some(), members == 9, "{members} members: {tiles:?}");
        }
    }

    /// One ranged launch writes what a launch per tile writes, bit for bit:
    /// partial blocks, one to four grid rows, short and full piece ranges,
    /// signed zeros. Outputs start as NaN, so a column left unwritten shows.
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn the_ranged_sgemm_writes_what_a_launch_per_tile_writes() {
        fn sample(i: usize, salt: u64) -> f32 {
            let mut z = (i as u64).wrapping_add(salt).wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            match i % 31 {
                0 => 0.0, 1 => -0.0, 2 => 0.5, 3 => -0.5,
                _ => ((z >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.5,
            }
        }
        let cases: Vec<(usize, usize, usize, usize, Vec<Tile>, bool)> = vec![
            (1, 72, 17, 1, vec![(0, 72, 0, 1)], true),
            (257, 144, 245, 7, vec![(0, 72, 1, 6), (72, 72, 0, 7)], false),
            (257, 80, 17, 7, vec![(0, 72, 1, 6), (72, 8, 0, 7)], false),
            (513, 216, 267, 45, vec![(0, 72, 10, 34), (72, 72, 4, 41), (144, 72, 0, 45)], false),
            (257, 288, 17, 7, vec![(0, 72, 1, 5), (72, 72, 0, 7), (144, 72, 2, 6), (216, 72, 3, 7)], false),
            (513, 144, 267, 45, vec![(0, 72, 10, 34), (72, 72, 0, 45)], false),
            (1, 288, 245, 1, vec![(0, 72, 0, 1), (72, 72, 0, 1), (144, 72, 0, 1), (216, 72, 0, 1)], false),
            (257, 72, 267, 7, vec![(0, 72, 2, 7)], false),
            (513, 216, 17, 1, vec![(0, 72, 0, 1), (72, 72, 0, 1), (144, 72, 0, 1)], false),
        ];
        for (m, n, k, q, tiles, zeros) in cases {
            let ranges = aligned_ranges(n, q, &tiles).expect("aligned");
            let mut e = HpEngine::new(Faults::default()).expect("the NVIDIA device");
            let out = e.with_ctx(|e| {
                let got = (|| {
                    let a: Vec<f32> = (0..(m + q - 1) * k).map(|i| if zeros { if i % 2 == 0 { -0.0 } else { 0.0 } } else { sample(i, 17) }).collect();
                    // Pieces outside a tile's range are zero, as the pass's are.
                    let mut b = vec![0.0f32; q * n * k];
                    for p in 0..q {
                        for col in 0..n {
                            let (lo, hi) = ranges[col / 72];
                            if (lo..hi).contains(&p) {
                                for u in 0..k { let i = (p * n + col) * k + u; b[i] = sample(i, 0x1234_5678); }
                            }
                        }
                    }
                    let (da, db) = (e.alloc(a.len() * 4)?, e.alloc(b.len() * 4)?);
                    e.upload(da, a)?;
                    e.upload(db, b)?;
                    let (old, new) = (e.alloc(m * n * 4)?, e.alloc(m * n * 4)?);
                    let poison = vec![f32::from_bits(0x7fc0_0162); m * n];
                    e.upload(old, poison.clone())?;
                    e.upload(new, poison)?;
                    let before = e.seen.launch;
                    for &(c0, nc, lo, hi) in &tiles {
                        e.sgemm_v2(m, nc, k, hi - lo, da + (lo * k * 4) as u64, k, k, db + ((lo * n + c0) * k * 4) as u64, k, n * k,
                            old + (c0 * m * 4) as u64)?;
                    }
                    let mid = e.seen.launch;
                    e.sgemm_v2_ranged(m, n, k, q, da, db, new, ranges)?;
                    let launches = (mid - before, e.seen.launch - mid);
                    e.wait(None)?;
                    Ok((e.download::<f32>(old, m * n)?, e.download::<f32>(new, m * n)?, launches))
                })();
                let cleanup = e.finish_pass();
                match (got, cleanup) {
                    (got, Ok(())) => got,
                    (Ok(_), Err(err)) => Err(err),
                    (Err(a), Err(b)) => Err(e.merge(a, b)),
                }
            });
            let ledger = e.ledger();
            let close = e.close();
            assert!(close.context_destroyed && close.failures.is_empty(), "{close:?}");
            let (old, new, launches) = out.expect("the pass");
            assert_eq!(ledger, (0, 0), "allocations survived the pass");
            assert_eq!(launches, (tiles.len(), 1));
            assert!(old.iter().all(|v| v.is_finite()), "M {m} N {n}: a column was not written");
            assert!(old.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits()), "M {m} N {n} K {k} q {q}: the launches differ");
            eprintln!("ranged SGEMM M {m} N {n} K {k} q {q}: {} launches -> 1, every f32 equal", tiles.len());
        }
    }

    /// Each of the four modules, the ranged SGEMM's last, failing to load
    /// is a refusal with the context destroyed (a cleanup failure would make
    /// it a device error instead).
    #[test]
    #[ignore = "needs the NVIDIA device"]
    fn a_module_that_will_not_load_is_a_clean_refusal() {
        for at in 1..=4 {
            let got = HpEngine::new(Faults { module_load: Some(at), ..Faults::default() });
            assert!(matches!(&got, Err(HpError::Unavailable(why)) if why == "injected module load failure"), "module {at}: {:?}", got.err());
        }
        let e = HpEngine::new(Faults::default()).expect("the NVIDIA device");
        assert_eq!(e.seen.module_load, 4);
        let close = e.close();
        assert!(close.context_destroyed && close.failures.is_empty(), "{close:?}");
    }

    /// The admitted region: the validated settings and the presets with short
    /// windows; not few bars, low frame rates or long windows, where the
    /// sweeps found the route no faster than the cores.
    #[test]
    fn the_swept_region_is_admitted_and_no_more() {
        let cfg = validated_cfg();
        assert_eq!(admits(1024, 20.0, 24_000.0, &cfg, 180.0), Ok(()));
        for preset in [AsltPreset::Fast, AsltPreset::Standard, AsltPreset::High] {
            assert_eq!(admits(1024, 20.0, 24_000.0, &preset.config(), 180.0), Ok(()), "{preset:?}");
        }
        for preset in [AsltPreset::Ultra, AsltPreset::Extreme] {
            assert!(admits(1024, 20.0, 24_000.0, &preset.config(), 180.0).is_err(), "{preset:?}");
        }
        assert_eq!(admits(2048, 30.0, 16_000.0, &AsltConfig { n_wavelets: 1, q_ratio: 0.25, spread: 0.0, ..cfg.clone() }, 480.0), Ok(()));
        assert_eq!(admits(256, 20.0, 24_000.0, &cfg, 100.0), Ok(()));
        assert!(admits(255, 20.0, 24_000.0, &cfg, 180.0).is_err());
        assert!(admits(2049, 20.0, 24_000.0, &cfg, 180.0).is_err());
        assert!(admits(1024, 20.0, 24_000.0, &cfg, 60.0).is_err());
        assert!(admits(1024, 20.0, 24_000.0, &cfg, f32::NAN).is_err());
        assert!(admits(1024, 20.0, 24_000.0, &AsltConfig { max_window_s: 1.3, ..cfg.clone() }, 180.0).is_err());
        assert!(admits(1024, 20.0, 24_000.0, &AsltConfig { max_window_s: f32::NAN, ..cfg.clone() }, 180.0).is_err());
        assert!(admits(1024, 20.0, 20.0, &cfg, 180.0).is_err());
    }
}
