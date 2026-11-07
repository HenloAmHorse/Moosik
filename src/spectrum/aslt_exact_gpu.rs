//! The exact GPU lane (1.6.1, experimental, off unless the analysis settings
//! turn it on): the 32768-point shared-block inverses of a superlet
//! analysis's 2^15 pass, for 32 kernels of one length and both channels at a
//! time, on an NVIDIA GPU through the CUDA driver.
//!
//! It is the same arithmetic, not an approximation of it. The shader
//! (`aslt_exact_gpu.ptx`, derived by `tools/aslt_exact_gpu/derive_ptx.py` from
//! the 2026-11-03 exact-ASLT research) evaluates rustfft's own AVX graph for
//! this size — `MixedRadix8xn(MixedRadix8xn(Butterfly512))`, the spectral
//! product fused into the first pass, only the requested outputs of the
//! 512-point leaves — with that plan's stored coefficients
//! (`rustfft::ExactGpuWords32768`), in the same order, IEEE round-to-nearest,
//! no fused multiply-add and no flush-to-zero. What it returns are the
//! unscaled inverse outputs; the scaling `2·(1/n)·y` is applied here, on the
//! CPU, by the expression `InvWork::outputs` uses. Every analysis holds its
//! first device block to the CPU's bits before trusting the device; a lane
//! that does not match, fails, or is not admitted leaves everything to the
//! unchanged CPU route.
//!
//! All CUDA state is created, used and destroyed on the lane's own thread
//! (raw driver handles are not `Send`); the analysis talks to it through a
//! bounded channel, one bank at a time.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::{SignalBlocks, C32};

/// The transform size the lane computes.
pub(super) const N: usize = 32_768;
/// Kernels in a bank. With the two channels, the shader's 64 pairs.
pub(super) const BANK: usize = 32;
const PAIRS: usize = 2 * BANK;
/// Requested outputs of one block the lane takes. The layout holds up to
/// `MAX_M` (101 covers every 2^15 kernel at 44.1 and 48 kHz with the 180 fps
/// hop); blocks with fewer than `MIN_M` (in practice the partial last ones)
/// stay on the CPU, which also keeps every leaf path of the shader in use.
pub(super) const MAX_M: usize = 101;
pub(super) const MIN_M: usize = 64;
const MAX_TASKS: usize = PAIRS * MAX_M;
// The research layout's formulas (integratable_161 model.rs) at MAX_M = 101.
const PD: usize = 32 + 2 * 512;
const OD: usize = PD + 8 * MAX_TASKS;
const HEADER_BYTES: usize = (OD + 4 * MAX_TASKS) * 4;
const META_OFFSET: usize = OD * 4;
/// Metadata words: first output, output count, three a rank, then the hop.
const HOP_WORD: usize = 2 + 3 * MAX_M;
const META_WORDS: usize = HOP_WORD + 1;
const _: () = assert!(META_OFFSET + META_WORDS * 4 <= HEADER_BYTES);
const ROOT_TW: usize = 32;
const INNER_TW: usize = ROOT_TW + 2 * 4096 * 7;
const SOURCE_DESC: usize = INNER_TW + 2 * 512 * 7;
const PREFIX_WORDS: usize = SOURCE_DESC + 4 * PAIRS;
const S_OFFSET: usize = PREFIX_WORDS * 4;
const H_OFFSET: usize = S_OFFSET + 2 * N * 8;
const INPUT_BYTES: usize = H_OFFSET + BANK * N * 8;
/// The five device buffers: inputs (prefix, two signal blocks, 32 kernels),
/// the root pass's output, the leaf header and inner pass's output, the leaf
/// scratch, and the requested outputs.
const SIZES: [usize; 5] =
    [INPUT_BYTES, PAIRS * N * 8, HEADER_BYTES + PAIRS * N * 8, MAX_TASKS * 32 * 8, MAX_TASKS * 8];
/// Device memory the lane allocates, all five buffers (the driver's own
/// context and module memory are on top).
pub(super) const DEVICE_BYTES: usize = SIZES[0] + SIZES[1] + SIZES[2] + SIZES[3] + SIZES[4];

static PTX: &str = include_str!("aslt_exact_gpu.ptx");

/// The shader with this layout's integer addresses filled in.
fn ptx() -> String {
    let s = PTX
        .replace("@@META_OFFSET@@", &META_OFFSET.to_string())
        .replace("@@HEADER_BYTES@@", &HEADER_BYTES.to_string())
        .replace("@@HOP_WORD@@", &(HOP_WORD * 4).to_string());
    debug_assert!(!s.contains("@@"));
    s
}

/// Test builds only: faults a test asks the next banks to suffer, as the
/// device would deliver them. `FAIL_AT` = the device block (counted over the
/// process) at which a bank fails; `CORRUPT` = flip the lowest bit of the
/// first output of the next bank's first block, which only the check can see.
#[cfg(test)]
pub(super) mod fault {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    pub static FAIL_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
    pub static BLOCKS: AtomicUsize = AtomicUsize::new(0);
    pub static CORRUPT: AtomicBool = AtomicBool::new(false);
    /// Hold the device before the block at which `BLOCKS` equals this, until
    /// `RELEASE` — or a minute, so a test that fails cannot strand the lane.
    pub static HOLD_AT: AtomicUsize = AtomicUsize::new(usize::MAX);
    pub static HOLDING: AtomicBool = AtomicBool::new(false);
    pub static RELEASE: AtomicBool = AtomicBool::new(false);
    /// Set when a wait for the device has seen the analysis stop.
    pub static STOP_SEEN: AtomicBool = AtomicBool::new(false);
    /// Report the release of the first buffer as failed (it is still released).
    pub static FAIL_CLEANUP: AtomicBool = AtomicBool::new(false);
    /// Panic on the lane's thread: 1 during setup, 2 at the next bank.
    pub static PANIC: AtomicUsize = AtomicUsize::new(0);
    /// Cleanup steps attempted: the synchronise, each free, the unload, the destroy.
    pub static CLEANUP_STEPS: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn hold() {
        if BLOCKS.load(Ordering::SeqCst) != HOLD_AT.load(Ordering::SeqCst) {
            return;
        }
        HOLD_AT.store(usize::MAX, Ordering::SeqCst);
        HOLDING.store(true, Ordering::SeqCst);
        let t = std::time::Instant::now();
        while !RELEASE.load(Ordering::SeqCst) && t.elapsed() < std::time::Duration::from_secs(60) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        HOLDING.store(false, Ordering::SeqCst);
    }

    pub(super) fn panic_if(stage: usize) {
        if PANIC.compare_exchange(stage, 0, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            panic!("forced lane panic (test)");
        }
    }
}

/// Whether the analysis settings have turned the lane on. Off by default.
static ENABLED: AtomicBool = AtomicBool::new(false);
pub(super) fn set_enabled(on: bool) { ENABLED.store(on, Ordering::Relaxed) }
pub(super) fn enabled() -> bool { ENABLED.load(Ordering::Relaxed) }

/// One overlap-save block a bank sends to the device: the block, the first
/// frame it owns (`lo`), that frame's position in the block's inverse output
/// (`first`), and how many hop-spaced frames it owns (`m`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub index: usize,
    pub lo: usize,
    pub first: usize,
    pub m: usize,
}

/// A bank's work and the columns it fills. `h` holds 32 kernel spectra of
/// one tap length; `cols[k] = [left, right]` are kernel `k`'s complex columns,
/// into which the device blocks' scaled outputs are written in block order —
/// so where two blocks own a frame the later one's value stands, as on the
/// CPU route. Everything is moved in and handed back.
pub(super) struct Bank {
    pub h: Vec<Vec<C32>>,
    pub sets: [Arc<SignalBlocks>; 2],
    pub hop: usize,
    pub blocks: Vec<Block>,
    pub cols: Vec<[Vec<C32>; 2]>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct RunStats {
    pub blocks: usize,
    pub launches: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    /// The first block's scaled outputs, pair-major (`pair · m + j`, pair =
    /// channel · 32 + kernel), when the job asked for them.
    pub check: Option<Vec<C32>>,
}

/// How a bank ended. Only `Done` hands the bank back: anything else leaves
/// its columns partly written, and the caller recomputes those members on
/// the CPU from their descriptors, never from these.
pub(super) enum Reply {
    Done(Bank, RunStats),
    /// Stopped between blocks.
    Cancelled,
    /// The lane is not to be used again.
    Failed(String),
}

/// The metadata the two meta leaf entries read for one block: its first
/// requested position, the count, per output rank `[jstart, nj, ix]` (the
/// run of ranks sharing `ix = position / 1024`), and the hop.
fn metadata(first: usize, m: usize, hop: usize, md: &mut [u32]) -> Result<(), String> {
    if !(1..=MAX_M).contains(&m) || hop == 0 || md.len() != META_WORDS {
        return Err(format!("block metadata bounds (m {m}, hop {hop})"));
    }
    let last = first.checked_add(hop * (m - 1)).ok_or("block position overflow")?;
    if last >= N {
        return Err(format!("requested position {last} outside the block"));
    }
    md.fill(0);
    md[0] = first as u32;
    md[1] = m as u32;
    md[HOP_WORD] = hop as u32;
    let at = |j: usize| first + hop * j;
    let mut jstart = 0;
    while jstart < m {
        let ix = at(jstart) / 1024;
        let mut end = jstart + 1;
        while end < m && at(end) / 1024 == ix {
            end += 1;
        }
        for rank in jstart..end {
            md[2 + 3 * rank..5 + 3 * rank].copy_from_slice(&[jstart as u32, (end - jstart) as u32, ix as u32]);
        }
        jstart = end;
    }
    Ok(())
}

/// Where the shader's output `t` belongs: `(pair, j)`, the inverse of its
/// phase-2 task mapping over the metadata ranks.
fn caller(t: usize, md: &[u32]) -> (usize, usize) {
    let rank = t / PAIRS;
    let (jstart, nj) = (md[2 + 3 * rank] as usize, md[3 + 3 * rank] as usize);
    let z = t - PAIRS * jstart;
    (z / nj, jstart + z % nj)
}

// ---------------------------------------------------------------------------
// The CUDA driver, loaded at run time
// ---------------------------------------------------------------------------

type Handle = *mut c_void;
type ResultCode = c_int;
type Device = c_int;
type DevicePtr = u64;
type FarProc = unsafe extern "system" fn() -> isize;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: Handle, flags: u32) -> Handle;
    fn GetProcAddress(module: Handle, name: *const c_char) -> Option<FarProc>;
    fn FreeLibrary(module: Handle) -> c_int;
}

fn checked(operation: &str, status: ResultCode) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{operation}: CUresult {status}")) }
}

/// `nvcuda.dll`, from System32 only. Absent on a machine without an NVIDIA
/// driver, which is the ordinary case and simply means no lane.
struct Library(Handle);

impl Library {
    fn load() -> Result<Self, String> {
        let name: Vec<u16> = "nvcuda.dll\0".encode_utf16().collect();
        // SAFETY: a NUL-terminated wide string; LOAD_LIBRARY_SEARCH_SYSTEM32.
        let h = unsafe { LoadLibraryExW(name.as_ptr(), ptr::null_mut(), 0x800) };
        if h.is_null() { Err("no NVIDIA driver (nvcuda.dll)".into()) } else { Ok(Self(h)) }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle LoadLibraryExW returned, released once.
        unsafe { FreeLibrary(self.0) };
    }
}

#[allow(clippy::type_complexity)]
struct Api {
    init: unsafe extern "system" fn(c_uint) -> ResultCode,
    count: unsafe extern "system" fn(*mut c_int) -> ResultCode,
    device: unsafe extern "system" fn(*mut Device, c_int) -> ResultCode,
    name: unsafe extern "system" fn(*mut c_char, c_int, Device) -> ResultCode,
    capability: unsafe extern "system" fn(*mut c_int, *mut c_int, Device) -> ResultCode,
    create: unsafe extern "system" fn(*mut Handle, c_uint, Device) -> ResultCode,
    destroy: unsafe extern "system" fn(Handle) -> ResultCode,
    sync: unsafe extern "system" fn() -> ResultCode,
    load: unsafe extern "system" fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> ResultCode,
    unload: unsafe extern "system" fn(Handle) -> ResultCode,
    function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> ResultCode,
    alloc: unsafe extern "system" fn(*mut DevicePtr, usize) -> ResultCode,
    free: unsafe extern "system" fn(DevicePtr) -> ResultCode,
    htod: unsafe extern "system" fn(DevicePtr, *const c_void, usize) -> ResultCode,
    dtoh: unsafe extern "system" fn(*mut c_void, DevicePtr, usize) -> ResultCode,
    launch: unsafe extern "system" fn(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void,
    ) -> ResultCode,
}

impl Api {
    fn load(lib: &Library) -> Result<Self, String> {
        macro_rules! sym {
            ($name:literal) => {{
                // SAFETY: a NUL-terminated export name of a loaded module.
                let f = unsafe { GetProcAddress(lib.0, concat!($name, "\0").as_ptr().cast()) }
                    .ok_or(concat!("the driver has no ", $name))?;
                // SAFETY: the documented signature of this driver export.
                unsafe { std::mem::transmute::<FarProc, _>(f) }
            }};
        }
        Ok(Self {
            init: sym!("cuInit"),
            count: sym!("cuDeviceGetCount"),
            device: sym!("cuDeviceGet"),
            name: sym!("cuDeviceGetName"),
            capability: sym!("cuDeviceComputeCapability"),
            create: sym!("cuCtxCreate_v2"),
            destroy: sym!("cuCtxDestroy_v2"),
            sync: sym!("cuCtxSynchronize"),
            load: sym!("cuModuleLoadDataEx"),
            unload: sym!("cuModuleUnload"),
            function: sym!("cuModuleGetFunction"),
            alloc: sym!("cuMemAlloc_v2"),
            free: sym!("cuMemFree_v2"),
            htod: sym!("cuMemcpyHtoD_v2"),
            dtoh: sym!("cuMemcpyDtoH_v2"),
            launch: sym!("cuLaunchKernel"),
        })
    }
}

/// A context, the module, the five buffers. Created, used and destroyed on
/// the lane's thread; [`Session::release`] synchronises, frees, unloads and
/// destroys in that order, and dropping it does the same unchecked.
struct Session {
    api: Api,
    context: Handle,
    module: Handle,
    functions: [Handle; 4],
    memory: [DevicePtr; 5],
    name: String,
    _lib: Library,
}

impl Session {
    fn open() -> Result<Self, String> {
        let lib = Library::load()?;
        let api = Api::load(&lib)?;
        // SAFETY (this block): driver calls with valid out-pointers, in the
        // documented order.
        unsafe {
            checked("cuInit", (api.init)(0))?;
            let mut count = 0;
            checked("cuDeviceGetCount", (api.count)(&mut count))?;
            if count < 1 { return Err("no CUDA device".into()); }
            let mut device = -1;
            checked("cuDeviceGet", (api.device)(&mut device, 0))?;
            let mut name = [0u8; 256];
            checked("cuDeviceGetName", (api.name)(name.as_mut_ptr().cast(), name.len() as c_int, device))?;
            let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            let name = String::from_utf8_lossy(&name[..end]).into_owned();
            let (mut major, mut minor) = (0, 0);
            checked("cuDeviceComputeCapability", (api.capability)(&mut major, &mut minor, device))?;
            // The device class the exact lane has been held to the CPU on
            // (Turing, sm_75). Elsewhere, the CPU route.
            if (major, minor) != (7, 5) {
                return Err(format!("{name}: compute capability {major}.{minor} is not one the exact lane is validated on"));
            }
            let mut context = ptr::null_mut();
            // CU_CTX_SCHED_BLOCKING_SYNC: the waiting thread sleeps.
            checked("cuCtxCreate", (api.create)(&mut context, 4, device))?;
            if context.is_null() { return Err("cuCtxCreate returned no context".into()); }
            let mut s = Self { api, context, module: ptr::null_mut(), functions: [ptr::null_mut(); 4], memory: [0; 5], name, _lib: lib };
            let ready = s.load_module().and_then(|()| {
                for (slot, &bytes) in s.memory.iter_mut().zip(&SIZES) {
                    checked("cuMemAlloc", (s.api.alloc)(slot, bytes))?;
                    if *slot == 0 { return Err("cuMemAlloc returned null".to_string()); }
                }
                Ok(())
            });
            match ready {
                Ok(()) => Ok(s),
                Err(e) => Err(and_cleanup(e, s.release())),
            }
        }
    }

    /// Synchronises, frees the buffers, unloads the module and destroys the
    /// context. Every step is attempted whatever the others return; the
    /// failures are reported together. Once only: what it released is
    /// forgotten, so dropping the session afterwards does nothing.
    fn release(&mut self) -> Result<(), String> {
        if self.context.is_null() {
            return Ok(());
        }
        let mut failed = Vec::new();
        let mut step = |what: &str, status: ResultCode| {
            #[cfg(test)]
            fault::CLEANUP_STEPS.fetch_add(1, Ordering::SeqCst);
            if let Err(e) = checked(what, status) {
                failed.push(e);
            }
        };
        // SAFETY: handles of this context, each released once, the context last.
        unsafe {
            step("cuCtxSynchronize", (self.api.sync)());
            for m in self.memory.iter_mut().rev().filter(|m| **m != 0) {
                let status = (self.api.free)(*m);
                #[cfg(test)]
                let status = if fault::FAIL_CLEANUP.swap(false, Ordering::SeqCst) { 1 } else { status };
                step("cuMemFree", status);
                *m = 0;
            }
            if !self.module.is_null() {
                step("cuModuleUnload", (self.api.unload)(self.module));
                self.module = ptr::null_mut();
            }
            step("cuCtxDestroy", (self.api.destroy)(self.context));
            self.context = ptr::null_mut();
        }
        if failed.is_empty() { Ok(()) } else { Err(failed.join("; ")) }
    }

    fn load_module(&mut self) -> Result<(), String> {
        let source = CString::new(ptx()).map_err(|_| "NUL in the shader")?;
        let mut errors = [0u8; 4096];
        // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES (a value
        // in the pointer slot).
        let mut options = [5i32, 6];
        let mut values = [errors.as_mut_ptr().cast::<c_void>(), errors.len() as *mut c_void];
        let mut module = ptr::null_mut();
        // SAFETY: a NUL-terminated PTX image and option arrays of equal length.
        let status = unsafe {
            (self.api.load)(&mut module, source.as_ptr().cast(), 2, options.as_mut_ptr(), values.as_mut_ptr())
        };
        if status != 0 {
            let end = errors.iter().position(|&b| b == 0).unwrap_or(errors.len());
            return Err(format!("cuModuleLoadDataEx: CUresult {status}: {}", String::from_utf8_lossy(&errors[..end])));
        }
        self.module = module;
        for (slot, name) in self.functions.iter_mut().zip([c"root8", c"inner8", c"whole_phase1_meta", c"whole_phase2_meta"]) {
            // SAFETY: a loaded module and a NUL-terminated entry name.
            checked("cuModuleGetFunction", unsafe { (self.api.function)(slot, module, name.as_ptr()) })?;
            if slot.is_null() { return Err("cuModuleGetFunction returned null".into()); }
        }
        Ok(())
    }

    fn htod(&self, buffer: usize, offset: usize, bytes: &[u8]) -> Result<(), String> {
        if offset.checked_add(bytes.len()).is_none_or(|end| end > SIZES[buffer]) {
            return Err("upload outside its buffer".into());
        }
        // SAFETY: inside an allocation of this context, from a live slice.
        checked("cuMemcpyHtoD", unsafe {
            (self.api.htod)(self.memory[buffer] + offset as u64, bytes.as_ptr().cast(), bytes.len())
        })
    }

    fn htod_c(&self, buffer: usize, offset: usize, v: &[C32]) -> Result<(), String> {
        // SAFETY: Complex<f32> is two f32, plain data.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
        self.htod(buffer, offset, bytes)
    }

    fn htod_u32(&self, buffer: usize, offset: usize, v: &[u32]) -> Result<(), String> {
        // SAFETY: plain words.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
        self.htod(buffer, offset, bytes)
    }

    fn launch(&self, entry: usize, grid: (u32, u32), args: [u64; 3], count: u32) -> Result<(), String> {
        let (mut a, mut b, mut c, mut n) = (args[0], args[1], args[2], count);
        let mut params: [*mut c_void; 4] = [
            ptr::addr_of_mut!(a).cast(), ptr::addr_of_mut!(b).cast(), ptr::addr_of_mut!(c).cast(), ptr::addr_of_mut!(n).cast(),
        ];
        // SAFETY: a function of this module, four parameters of the entry's
        // declared types (three device pointers and a u32).
        checked("cuLaunchKernel", unsafe {
            (self.api.launch)(self.functions[entry], grid.0, grid.1, 1, 128, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut())
        })
    }
}

impl Drop for Session {
    /// Best effort, for a session no one retired (an unwinding lane thread):
    /// the same steps, their result unseen.
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// The prefix of the input buffer: layout words, the root and inner
/// twiddles, the rotation mask and half-root-2, and each pair's signal and
/// kernel offsets (signal = pair / 32, kernel = pair % 32).
fn prefix(w: &rustfft::ExactGpuWords32768) -> Result<Vec<u32>, String> {
    if w.root.len() != 4096 * 7 || w.inner.len() != 512 * 7 || w.leaf.outer.len() != 512 {
        return Err("the 32768-point plan's coefficients have another shape".into());
    }
    let mut p = vec![0u32; PREFIX_WORDS];
    p[..6].copy_from_slice(&[
        PAIRS as u32, MAX_M as u32, HEADER_BYTES as u32, (ROOT_TW * 4) as u32, (INNER_TW * 4) as u32, (SOURCE_DESC * 4) as u32,
    ]);
    p[8..11].copy_from_slice(&[w.rotation[0], w.rotation[1], w.leaf.phase2[14]]);
    for (i, z) in w.root.iter().enumerate() {
        p[ROOT_TW + 2 * i..ROOT_TW + 2 * i + 2].copy_from_slice(z);
    }
    for (i, z) in w.inner.iter().enumerate() {
        p[INNER_TW + 2 * i..INNER_TW + 2 * i + 2].copy_from_slice(z);
    }
    for pair in 0..PAIRS {
        let s = (S_OFFSET + (pair / BANK) * N * 8) as u32;
        let h = (H_OFFSET + (pair % BANK) * N * 8) as u32;
        p[SOURCE_DESC + 4 * pair..SOURCE_DESC + 4 * pair + 4].copy_from_slice(&[s, h, 0, 0]);
    }
    Ok(p)
}

/// The leaf header: the 512-point leaf's constants. (The research layout also
/// kept per-cohort task counts and the legacy descriptor tables here; the
/// meta entries read neither.)
fn header(w: &rustfft::ExactGpuWords32768) -> Vec<u32> {
    let mut h = vec![0u32; PD];
    h[..16].copy_from_slice(&w.leaf.phase2);
    h[16..20].copy_from_slice(&w.leaf.tw16);
    h[20..26].copy_from_slice(&[512, 16, 0, 0, (PD * 4) as u32, (OD * 4) as u32]);
    for (i, z) in w.leaf.outer.iter().enumerate() {
        h[32 + 2 * i..34 + 2 * i].copy_from_slice(z);
    }
    h
}

impl Session {
    /// The constants, once per session.
    fn upload_constants(&self) -> Result<(), String> {
        let w = rustfft::ExactGpuWords32768::new(rustfft::FftDirection::Inverse)
            .ok_or("this CPU has no AVX/FMA plan for the exact lane")?;
        self.htod_u32(0, 0, &prefix(&w)?)?;
        self.htod_u32(2, 0, &header(&w))
    }

    /// One bank: kernels uploaded once, then per block both channels'
    /// spectra and metadata, the four entries, and the outputs scaled into
    /// the columns. `None` when cancelled between blocks.
    fn run(&self, bank: &mut Bank, cancel: &AtomicBool, check: bool, out: &mut [[u32; 2]], md: &mut [u32])
        -> Result<Option<RunStats>, String>
    {
        #[cfg(test)]
        fault::panic_if(2);
        let [left, right] = &bank.sets;
        if bank.h.len() != BANK || bank.h.iter().any(|h| h.len() != N) || bank.cols.len() != BANK
            || left.n != N || right.n != N || left.stride != right.stride || left.blocks() != right.blocks()
            || bank.blocks.windows(2).any(|w| w[0].index >= w[1].index)
        {
            return Err("bank shape".into());
        }
        let frames = bank.cols[0][0].len();
        if bank.cols.iter().any(|c| c[0].len() != frames || c[1].len() != frames)
            || bank.blocks.iter().any(|b| b.index >= left.blocks() || !(MIN_M..=MAX_M).contains(&b.m) || b.lo + b.m > frames)
        {
            return Err("bank columns or blocks".into());
        }
        let mut stats = RunStats::default();
        for (k, h) in bank.h.iter().enumerate() {
            self.htod_c(0, H_OFFSET + k * N * 8, h)?;
        }
        stats.h2d_bytes += (BANK * N * 8) as u64;
        let scale = 1.0 / N as f32;
        let [m0, m1, m2, m3, m4] = self.memory;
        for (bi, b) in bank.blocks.iter().enumerate() {
            #[cfg(test)]
            fault::hold();
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let tasks = PAIRS * b.m;
            #[cfg(test)]
            if fault::BLOCKS.fetch_add(1, Ordering::SeqCst) == fault::FAIL_AT.load(Ordering::SeqCst) {
                return Err("forced device failure (test)".into());
            }
            self.htod_c(0, S_OFFSET, left.block(b.index))?;
            self.htod_c(0, S_OFFSET + N * 8, right.block(b.index))?;
            metadata(b.first, b.m, bank.hop, md)?;
            self.htod_u32(2, META_OFFSET, md)?;
            stats.h2d_bytes += (2 * N * 8 + META_WORDS * 4) as u64;
            self.launch(0, ((PAIRS * 4096 / 128) as u32, 1), [m0, m1, m2], (PAIRS * 4096) as u32)?;
            self.launch(1, ((PAIRS * 4096 / 128) as u32, 1), [m0, m1, m2], (PAIRS * 4096) as u32)?;
            self.launch(2, (b.m as u32, 16), [m2, m3, m4], (tasks * 32) as u32)?;
            self.launch(3, (tasks.div_ceil(128) as u32, 1), [m2, m3, m4], tasks as u32)?;
            stats.launches += 4;
            // SAFETY: a context call on this thread.
            checked("cuCtxSynchronize", unsafe { (self.api.sync)() })?;
            // SAFETY: `tasks` outputs of the output buffer into a live slice.
            checked("cuMemcpyDtoH", unsafe { (self.api.dtoh)(out.as_mut_ptr().cast(), m4, tasks * 8) })?;
            stats.d2h_bytes += (tasks * 8) as u64;
            #[cfg(test)]
            if bi == 0 && fault::CORRUPT.swap(false, Ordering::SeqCst) {
                out[0][0] ^= 1;
            }
            let mut first_block = (check && bi == 0).then(|| vec![C32 { re: 0.0, im: 0.0 }; tasks]);
            for (t, z) in out[..tasks].iter().enumerate() {
                let (pair, j) = caller(t, md);
                let (re, im) = (f32::from_bits(z[0]), f32::from_bits(z[1]));
                let v = C32 { re: 2.0 * scale * re, im: 2.0 * scale * im };
                bank.cols[pair % BANK][pair / BANK][b.lo + j] = v;
                if let Some(c) = first_block.as_mut() {
                    c[pair * b.m + j] = v;
                }
            }
            if first_block.is_some() {
                stats.check = first_block;
            }
            stats.blocks += 1;
        }
        Ok(Some(stats))
    }
}

// ---------------------------------------------------------------------------
// The lane: one thread owning one session
// ---------------------------------------------------------------------------

struct Job {
    bank: Bank,
    cancel: Arc<AtomicBool>,
    check: bool,
}

/// How often a wait for the device looks at the analysis. Once the cores have
/// run out of work nothing else does, and a bank takes about half a second.
const POLL: std::time::Duration = std::time::Duration::from_millis(20);

/// A primary error and what its cleanup reported, both kept.
fn and_cleanup(e: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => e,
        Err(c) => format!("{e}; cleanup: {c}"),
    }
}

fn panicked(p: Box<dyn std::any::Any + Send>) -> String {
    let what = p.downcast_ref::<&str>().map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into());
    format!("the lane thread panicked: {what}")
}

/// The device, for the length of one analysis pass.
pub(super) struct Lane {
    tx: Option<std::sync::mpsc::SyncSender<Job>>,
    /// Behind a lock only so a `&Lane` can be shared with the pool's scope;
    /// one bank is in flight at a time.
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<Reply>>,
    /// The thread's result is its session's release.
    join: Option<std::thread::JoinHandle<Result<(), String>>>,
    pub device: String,
    /// Driver load, context, module (compiled on first use, then the
    /// driver's cache), buffers and constants.
    pub setup: std::time::Duration,
}

impl Lane {
    /// Starts the lane's thread and waits until its session is ready, or
    /// says why there is none (with whatever its cleanup or thread reported).
    pub(super) fn start() -> Result<Self, String> {
        let t0 = std::time::Instant::now();
        let (tx, jobs) = std::sync::mpsc::sync_channel::<Job>(1);
        let (replies, rx) = std::sync::mpsc::sync_channel::<Reply>(1);
        let (ready_tx, ready) = std::sync::mpsc::sync_channel::<Result<String, String>>(1);
        let join = std::thread::Builder::new()
            .name("aslt-exact-gpu".into())
            .spawn(move || {
                #[cfg(test)]
                fault::panic_if(1);
                let session = Session::open().and_then(|mut s| match s.upload_constants() {
                    Ok(()) => Ok(s),
                    Err(e) => Err(and_cleanup(e, s.release())),
                });
                let mut session = match session {
                    Ok(s) => {
                        let _ = ready_tx.send(Ok(s.name.clone()));
                        s
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return Ok(());
                    }
                };
                let mut out = vec![[0u32; 2]; MAX_TASKS];
                let mut md = vec![0u32; META_WORDS];
                while let Ok(Job { mut bank, cancel, check }) = jobs.recv() {
                    let reply = match session.run(&mut bank, &cancel, check, &mut out, &mut md) {
                        Ok(Some(stats)) => Reply::Done(bank, stats),
                        Ok(None) => Reply::Cancelled,
                        Err(e) => Reply::Failed(e),
                    };
                    let failed = matches!(reply, Reply::Failed(..));
                    if replies.send(reply).is_err() || failed {
                        break;
                    }
                }
                session.release()
            })
            .map_err(|e| format!("lane thread: {e}"))?;
        match ready.recv() {
            Ok(Ok(device)) => Ok(Self { tx: Some(tx), rx: std::sync::Mutex::new(rx), join: Some(join), device, setup: t0.elapsed() }),
            Ok(Err(e)) => Err(match join.join() {
                Ok(cleanup) => and_cleanup(e, cleanup),
                Err(p) => format!("{e}; {}", panicked(p)),
            }),
            Err(_) => Err(match join.join() {
                Ok(cleanup) => and_cleanup("the lane thread ended during setup".into(), cleanup),
                Err(p) => panicked(p),
            }),
        }
    }

    /// Runs one bank to its reply. The wait polls `stop`, the analysis's
    /// cancellation, every [`POLL`]; once it says stop, `cancel` is set and the
    /// device stops between blocks. The reply is still awaited, so the lane is
    /// idle whatever this returns.
    pub(super) fn run(&self, bank: Bank, cancel: &Arc<AtomicBool>, check: bool, stop: &dyn Fn() -> bool) -> Reply {
        use std::sync::mpsc::RecvTimeoutError;
        let job = Job { bank, cancel: Arc::clone(cancel), check };
        let Some(tx) = self.tx.as_ref() else { return Reply::Failed("the lane is closed".into()) };
        let rx = self.rx.lock().unwrap_or_else(|p| p.into_inner());
        if tx.send(job).is_err() {
            return Reply::Failed("the lane thread has ended".into());
        }
        loop {
            match rx.recv_timeout(POLL) {
                Ok(reply) => return reply,
                Err(RecvTimeoutError::Timeout) => {
                    if !cancel.load(Ordering::Relaxed) && stop() {
                        cancel.store(true, Ordering::Relaxed);
                        #[cfg(test)]
                        fault::STOP_SEEN.store(true, Ordering::SeqCst);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Reply::Failed("the lane thread ended without replying".into());
                }
            }
        }
    }

    /// Closes the lane: its thread releases the session, every step attempted,
    /// and is joined. A failed step or a panic on the thread is the error.
    pub(super) fn retire(mut self) -> Result<(), String> {
        self.close()
    }

    fn close(&mut self) -> Result<(), String> {
        drop(self.tx.take());
        match self.join.take().map(|j| j.join()) {
            None | Some(Ok(Ok(()))) => Ok(()),
            Some(Ok(Err(cleanup))) => Err(format!("cleanup: {cleanup}")),
            Some(Err(p)) => Err(panicked(p)),
        }
    }
}

impl Drop for Lane {
    /// Best effort, for a lane no one retired (an unwinding analysis).
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_the_researchs_at_its_own_capacity() {
        // The research's figures at 78 outputs, from the same formulas.
        let at = |m: usize| {
            let t = PAIRS * m;
            let od = PD + 8 * t;
            ((od + 4 * t) * 4, od * 4)
        };
        assert_eq!(at(78), (243_840, 163_968));
        assert_eq!((HEADER_BYTES, META_OFFSET), at(MAX_M));
        assert!(ptx().contains(&format!("add.u64 %descriptor,%input,{META_OFFSET};")));
        assert!(!ptx().contains("@@"));
    }

    #[test]
    fn metadata_ranks_cover_every_output_once() {
        let mut md = vec![0u32; META_WORDS];
        for (first, m, hop) in [(12_141, 78, 267), (12_044, 77, 267), (4_136, 92, 267), (4_000, 100, 245), (5_000, 64, 245)] {
            metadata(first, m, hop, &mut md).unwrap();
            let mut seen = vec![false; PAIRS * m];
            for t in 0..PAIRS * m {
                let (pair, j) = caller(t, &md);
                assert!(pair < PAIRS && j < m);
                assert!(!std::mem::replace(&mut seen[pair * m + j], true), "output {t} maps twice");
            }
            assert!(seen.iter().all(|&s| s));
        }
        assert!(metadata(30_000, 101, 267, &mut md).is_err(), "past the block");
        assert!(metadata(0, MAX_M + 1, 245, &mut md).is_err());
    }
}
