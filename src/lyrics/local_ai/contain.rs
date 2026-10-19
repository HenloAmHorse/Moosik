//! Every runtime child (tokenizer and inference) is created inside a Windows
//! job object that Moosik alone holds, with "kill on job close": when Moosik
//! exits for any reason, a crash included, Windows closes the handle and ends
//! every process still in the job.
//!
//! The job is named at creation (`PROC_THREAD_ATTRIBUTE_JOB_LIST` in an
//! extended startup info, Windows 10+): if `CreateProcessW` succeeds, the
//! process already belongs to the job; if the association fails, so does the
//! creation, and no process exists. There is no moment in which a child
//! exists outside the job — not even suspended — so a parent that dies right
//! after creation leaves nothing behind. (std's `Command` cannot pass this
//! attribute on stable Rust, which is why this module calls `CreateProcessW`
//! itself.)
//!
//! Only the three standard handles are inherited
//! (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`): the job handle is created
//! uninheritable and is not on that list, so a child can never keep its own
//! job alive. Children start at below-normal priority with no window.
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

/// What to run: program, arguments, environment (`None`: inherit Moosik's)
/// and working directory. Built from a `Command` so callers keep using it.
#[derive(Clone, Debug)]
pub struct Spec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Option<Vec<(OsString, OsString)>>,
    pub cwd: Option<PathBuf>,
}

impl Spec {
    /// `clear_env`: the command's environment is exactly what it set (as
    /// after `env_clear`); otherwise Moosik's, with what it set on top.
    pub fn from_command(c: &Command, clear_env: bool) -> Spec {
        let set: Vec<(OsString, Option<OsString>)> =
            c.get_envs().map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string()))).collect();
        let env = if clear_env {
            Some(set.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect())
        } else if set.is_empty() {
            None
        } else {
            let mut all: Vec<(OsString, OsString)> = std::env::vars_os().collect();
            for (k, v) in set {
                let up = k.to_ascii_uppercase();
                all.retain(|(a, _)| a.to_ascii_uppercase() != up);
                if let Some(v) = v {
                    all.push((k, v));
                }
            }
            Some(all)
        };
        Spec {
            program: PathBuf::from(c.get_program()),
            args: c.get_args().map(|a| a.to_os_string()).collect(),
            env,
            cwd: c.get_current_dir().map(PathBuf::from),
        }
    }
}

#[cfg(windows)]
mod sys {
    #![allow(non_snake_case)]
    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    pub const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    pub const EXTENDED_STARTUPINFO_PRESENT: u32 = 0x0008_0000;
    pub const CREATE_UNICODE_ENVIRONMENT: u32 = 0x0000_0400;
    pub const STARTF_USESTDHANDLES: u32 = 0x0000_0100;
    pub const PROC_THREAD_ATTRIBUTE_HANDLE_LIST: usize = 0x0002_0002;
    pub const PROC_THREAD_ATTRIBUTE_JOB_LIST: usize = 0x0002_000D;
    pub const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
    pub const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    pub const HANDLE_FLAG_INHERIT: u32 = 1;
    pub const WAIT_OBJECT_0: u32 = 0;
    pub const WAIT_TIMEOUT: u32 = 0x102;

    #[repr(C)]
    #[derive(Default)]
    pub struct BasicLimit {
        pub per_process_user_time: i64,
        pub per_job_user_time: i64,
        pub flags: u32,
        pub min_ws: usize,
        pub max_ws: usize,
        pub active_processes: u32,
        pub affinity: usize,
        pub priority_class: u32,
        pub scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct ExtendedLimit {
        pub basic: BasicLimit,
        pub io: [u64; 6],
        pub process_memory: usize,
        pub job_memory: usize,
        pub peak_process_memory: usize,
        pub peak_job_memory: usize,
    }

    #[repr(C)]
    pub struct SecurityAttributes {
        pub length: u32,
        pub descriptor: *mut core::ffi::c_void,
        pub inherit: i32,
    }

    #[repr(C)]
    pub struct StartupInfoW {
        pub cb: u32,
        pub reserved: *mut u16,
        pub desktop: *mut u16,
        pub title: *mut u16,
        pub x: u32,
        pub y: u32,
        pub x_size: u32,
        pub y_size: u32,
        pub x_chars: u32,
        pub y_chars: u32,
        pub fill: u32,
        pub flags: u32,
        pub show_window: u16,
        pub reserved2_len: u16,
        pub reserved2: *mut u8,
        pub std_input: isize,
        pub std_output: isize,
        pub std_error: isize,
    }

    #[repr(C)]
    pub struct StartupInfoExW {
        pub info: StartupInfoW,
        pub attributes: *mut core::ffi::c_void,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct ProcessInformation {
        pub process: isize,
        pub thread: isize,
        pub pid: u32,
        pub tid: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateJobObjectW(attrs: *const core::ffi::c_void, name: *const u16) -> isize;
        pub fn SetInformationJobObject(job: isize, class: i32, info: *const core::ffi::c_void, len: u32) -> i32;
        #[cfg(test)]
        pub fn IsProcessInJob(process: isize, job: isize, result: *mut i32) -> i32;
        pub fn InitializeProcThreadAttributeList(list: *mut core::ffi::c_void, count: u32, flags: u32, size: *mut usize) -> i32;
        pub fn UpdateProcThreadAttribute(
            list: *mut core::ffi::c_void, flags: u32, attribute: usize, value: *const core::ffi::c_void,
            size: usize, previous: *mut core::ffi::c_void, returned: *mut usize,
        ) -> i32;
        pub fn DeleteProcThreadAttributeList(list: *mut core::ffi::c_void);
        pub fn CreateProcessW(
            app: *const u16, cmdline: *mut u16, pattrs: *const core::ffi::c_void, tattrs: *const core::ffi::c_void,
            inherit: i32, flags: u32, env: *const core::ffi::c_void, cwd: *const u16,
            si: *const StartupInfoExW, pi: *mut ProcessInformation,
        ) -> i32;
        pub fn CreatePipe(read: *mut isize, write: *mut isize, attrs: *const SecurityAttributes, size: u32) -> i32;
        pub fn SetHandleInformation(h: isize, mask: u32, flags: u32) -> i32;
        pub fn CreateFileW(name: *const u16, access: u32, share: u32, attrs: *const SecurityAttributes,
                           disposition: u32, flags: u32, template: isize) -> isize;
        pub fn TerminateProcess(h: isize, code: u32) -> i32;
        pub fn WaitForSingleObject(h: isize, ms: u32) -> u32;
        pub fn GetExitCodeProcess(h: isize, code: *mut u32) -> i32;
        pub fn CloseHandle(h: isize) -> i32;
        pub fn GetLastError() -> u32;
    }
}

/// A job object with kill-on-close. Closing it (dropping this, or Moosik
/// exiting) ends every process in it.
pub struct Job(#[cfg_attr(not(windows), allow(dead_code))] isize);

// Used only through kernel calls, which are safe from any thread.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    #[cfg(windows)]
    pub fn new() -> Result<Job, String> {
        use sys::*;
        let h = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if h == 0 {
            return Err(format!("could not create a job object (error {})", unsafe { GetLastError() }));
        }
        let job = Job(h);
        let mut info = ExtendedLimit::default();
        info.basic.flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = unsafe {
            SetInformationJobObject(h, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, &info as *const _ as *const _,
                                    std::mem::size_of::<ExtendedLimit>() as u32)
        };
        if ok == 0 {
            return Err(format!("could not set the job object's limits (error {})", unsafe { GetLastError() }));
        }
        Ok(job)
    }

    #[cfg(not(windows))]
    pub fn new() -> Result<Job, String> {
        Err("process containment is only implemented on Windows".into())
    }

    /// Whether `p` is in this job.
    #[cfg(all(test, windows))]
    pub fn holds(&self, p: &Proc) -> bool {
        self.holds_raw(p.raw())
    }

    #[cfg(all(test, windows))]
    pub fn holds_raw(&self, process: isize) -> bool {
        let mut r = 0i32;
        unsafe { sys::IsProcessInJob(process, self.0, &mut r) != 0 && r != 0 }
    }

    /// A job whose handle is not a job, so association (and with it creation)
    /// fails. Test seam for the refusal path.
    #[cfg(test)]
    pub fn broken() -> Job {
        Job(-2) // the current-thread pseudo-handle: not a job object
    }
}

#[cfg(windows)]
impl Drop for Job {
    fn drop(&mut self) {
        if self.0 > 0 {
            unsafe { sys::CloseHandle(self.0) };
        }
    }
}

/// The job every runtime child of this process runs in, created on first use
/// and never closed: the operating system closes it when Moosik exits.
pub fn process_job() -> Result<&'static Job, String> {
    static JOB: std::sync::OnceLock<Result<Job, String>> = std::sync::OnceLock::new();
    JOB.get_or_init(Job::new).as_ref().map_err(Clone::clone)
}

/// Held across every window in which this process has an inheritable handle
/// that a child it is creating must not leak to another child.
///
/// `spawn_in` makes the runtime's pipe writers inheritable, creates the child,
/// and closes its copies — all under this lock. Rust's own `Command::spawn`
/// inherits every inheritable handle in the process (it takes std's private
/// spawn lock, which `spawn_in` cannot), so a program launched meanwhile by
/// `Command` would inherit those writers and keep the runtime's output open
/// after the runtime exits. Every other place Moosik launches a program on
/// Windows therefore goes through [`ordinary_spawn`], which takes this lock
/// for the creation only: the Log button's Explorer window and the pack
/// install's `tar.exe`. (Found by `grep`: these are the only `Command`
/// launches outside tests.)
static LAUNCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn launch_lock() -> std::sync::MutexGuard<'static, ()> {
    LAUNCH.lock().unwrap_or_else(|e| e.into_inner())
}

/// `Command::spawn` for any program that is not a runtime child, never at a
/// moment when a runtime child's inheritable pipe ends exist. Only the
/// creation is serialised; the program then runs on its own, outside the job.
pub fn ordinary_spawn(cmd: &mut Command) -> std::io::Result<std::process::Child> {
    let _guard = launch_lock();
    cmd.spawn()
}

/// An owned child process: its handle (kept until this drops, so its exit
/// status and identity stay readable) and its output pipes.
pub struct Proc {
    #[cfg_attr(not(windows), allow(dead_code))]
    handle: isize,
    pid: u32,
    pub stdout: Option<std::fs::File>,
    pub stderr: Option<std::fs::File>,
}

unsafe impl Send for Proc {}

impl std::fmt::Debug for Proc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Proc {{ pid: {} }}", self.pid)
    }
}

impl Proc {
    pub fn id(&self) -> u32 {
        self.pid
    }

    #[cfg(windows)]
    #[cfg_attr(not(test), allow(dead_code))] // read by tests and the test-only peak record
    pub fn raw(&self) -> isize {
        self.handle
    }

    /// Ask Windows to end it, through its own handle. Harmless if it already
    /// exited; what counts is the confirmation `try_wait` gives.
    #[cfg(windows)]
    pub fn kill(&mut self) -> std::io::Result<()> {
        if unsafe { sys::TerminateProcess(self.handle, 1) } != 0 || matches!(self.try_wait(), Ok(Some(_))) {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }

    #[cfg(windows)]
    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        use std::os::windows::process::ExitStatusExt;
        match unsafe { sys::WaitForSingleObject(self.handle, 0) } {
            sys::WAIT_OBJECT_0 => {
                let mut code = 0u32;
                if unsafe { sys::GetExitCodeProcess(self.handle, &mut code) } == 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(Some(ExitStatus::from_raw(code)))
            }
            sys::WAIT_TIMEOUT => Ok(None),
            _ => Err(std::io::Error::last_os_error()),
        }
    }

    #[cfg(not(windows))]
    pub fn kill(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::other("unsupported"))
    }

    #[cfg(not(windows))]
    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        Err(std::io::Error::other("unsupported"))
    }
}

#[cfg(windows)]
impl Drop for Proc {
    fn drop(&mut self) {
        unsafe { sys::CloseHandle(self.handle) };
    }
}

/// Windows command-line quoting for one argument (the rules
/// `CommandLineToArgvW` and the C runtime parse).
#[cfg(windows)]
fn quote(arg: &std::ffi::OsStr, out: &mut Vec<u16>) {
    use std::os::windows::ffi::OsStrExt;
    let w: Vec<u16> = arg.encode_wide().collect();
    let needs = w.is_empty() || w.iter().any(|&c| c == b' ' as u16 || c == b'\t' as u16 || c == b'"' as u16);
    if !needs {
        out.extend(w);
        return;
    }
    out.push(b'"' as u16);
    let mut slashes = 0;
    for &c in &w {
        if c == b'\\' as u16 {
            slashes += 1;
        } else {
            if c == b'"' as u16 {
                out.extend(std::iter::repeat_n(b'\\' as u16, slashes + 1));
            }
            slashes = 0;
        }
        out.push(c);
    }
    out.extend(std::iter::repeat_n(b'\\' as u16, slashes));
    out.push(b'"' as u16);
}

#[cfg(windows)]
fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain(Some(0)).collect()
}

/// Create `spec` inside `job`, stdin from NUL and stdout/stderr piped, no
/// window, below-normal priority. On failure no process exists.
#[cfg(windows)]
pub fn spawn_in(spec: &Spec, job: &Job) -> Result<Proc, String> {
    use std::os::windows::io::FromRawHandle;
    use sys::*;
    let refuse = |what: &str, e: u32| {
        Err(format!("The local runtime was not started: {what} (error {e}). No process was created and nothing ran."))
    };
    // From the first inheritable handle to the last one closed: no other
    // launch in this process may happen meanwhile (see `LAUNCH`).
    let _launch = launch_lock();
    let inherit = SecurityAttributes { length: std::mem::size_of::<SecurityAttributes>() as u32, descriptor: std::ptr::null_mut(), inherit: 1 };
    // Handles to close on every way out.
    struct Owned(Vec<isize>);
    impl Drop for Owned {
        fn drop(&mut self) {
            for &h in &self.0 {
                if h != 0 && h != -1 {
                    unsafe { CloseHandle(h) };
                }
            }
        }
    }
    let (mut out_r, mut out_w, mut err_r, mut err_w) = (0isize, 0isize, 0isize, 0isize);
    unsafe {
        if CreatePipe(&mut out_r, &mut out_w, &inherit, 0) == 0 {
            return refuse("its output pipe could not be created", GetLastError());
        }
    }
    let mut parent_ends = Owned(vec![out_r]);
    let mut child_ends = Owned(vec![out_w]);
    unsafe {
        if CreatePipe(&mut err_r, &mut err_w, &inherit, 0) == 0 {
            return refuse("its log pipe could not be created", GetLastError());
        }
    }
    parent_ends.0.push(err_r);
    child_ends.0.push(err_w);
    let nul = unsafe {
        CreateFileW(wide(std::ffi::OsStr::new("NUL")).as_ptr(), 0x8000_0000, 0x3, &inherit, 3, 0, 0)
    };
    if nul == -1 {
        return refuse("NUL could not be opened for its input", unsafe { GetLastError() });
    }
    child_ends.0.push(nul);
    unsafe {
        // Moosik's ends never go to a child.
        SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);
    }

    // The attribute list: the job, and exactly the three handles to inherit.
    let mut size = 0usize;
    unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 2, 0, &mut size) };
    let mut list = vec![0u8; size];
    let lp = list.as_mut_ptr() as *mut core::ffi::c_void;
    if unsafe { InitializeProcThreadAttributeList(lp, 2, 0, &mut size) } == 0 {
        return refuse("its attribute list could not be prepared", unsafe { GetLastError() });
    }
    struct List(*mut core::ffi::c_void);
    impl Drop for List {
        fn drop(&mut self) {
            unsafe { DeleteProcThreadAttributeList(self.0) };
        }
    }
    let _list = List(lp);
    let jobs = [job.0];
    let handles = [nul, out_w, err_w];
    unsafe {
        if UpdateProcThreadAttribute(lp, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST, jobs.as_ptr() as *const _,
                                     std::mem::size_of_val(&jobs), std::ptr::null_mut(), std::ptr::null_mut()) == 0 {
            return refuse("Moosik's job object could not be named for it", GetLastError());
        }
        if UpdateProcThreadAttribute(lp, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles.as_ptr() as *const _,
                                     std::mem::size_of_val(&handles), std::ptr::null_mut(), std::ptr::null_mut()) == 0 {
            return refuse("its inherited handles could not be limited", GetLastError());
        }
    }

    let app = wide(spec.program.as_os_str());
    let mut cmdline = Vec::new();
    quote(spec.program.as_os_str(), &mut cmdline);
    for a in &spec.args {
        cmdline.push(b' ' as u16);
        quote(a, &mut cmdline);
    }
    cmdline.push(0);
    let env: Option<Vec<u16>> = spec.env.as_ref().map(|vars| {
        use std::os::windows::ffi::OsStrExt;
        let mut vars = vars.clone();
        vars.sort_by_key(|(k, _)| k.to_ascii_uppercase());
        let mut b = Vec::new();
        for (k, v) in vars {
            b.extend(k.encode_wide());
            b.push(b'=' as u16);
            b.extend(v.encode_wide());
            b.push(0);
        }
        if b.is_empty() {
            b.push(0);
        }
        b.push(0);
        b
    });
    let cwd = spec.cwd.as_ref().map(|c| wide(c.as_os_str()));
    let si = StartupInfoExW {
        info: StartupInfoW {
            cb: std::mem::size_of::<StartupInfoExW>() as u32,
            reserved: std::ptr::null_mut(), desktop: std::ptr::null_mut(), title: std::ptr::null_mut(),
            x: 0, y: 0, x_size: 0, y_size: 0, x_chars: 0, y_chars: 0, fill: 0,
            flags: STARTF_USESTDHANDLES, show_window: 0, reserved2_len: 0, reserved2: std::ptr::null_mut(),
            std_input: nul, std_output: out_w, std_error: err_w,
        },
        attributes: lp,
    };
    #[cfg(test)]
    seam::inheritable(spec);
    let mut pi = ProcessInformation::default();
    let ok = unsafe {
        CreateProcessW(
            app.as_ptr(), cmdline.as_mut_ptr(), std::ptr::null(), std::ptr::null(), 1,
            CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
            env.as_ref().map_or(std::ptr::null(), |e| e.as_ptr() as *const _),
            cwd.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            &si, &mut pi,
        )
    };
    if ok == 0 {
        return refuse("Windows refused to create it inside Moosik's job object", unsafe { GetLastError() });
    }
    unsafe { CloseHandle(pi.thread) };
    // The child has its copies; Moosik closes its own of the child's ends so
    // the pipes close when the child does.
    drop(child_ends);
    parent_ends.0.clear();
    #[cfg(test)]
    seam::created(pi.pid, pi.process);
    Ok(Proc {
        handle: pi.process,
        pid: pi.pid,
        stdout: Some(unsafe { std::fs::File::from_raw_handle(out_r as _) }),
        stderr: Some(unsafe { std::fs::File::from_raw_handle(err_r as _) }),
    })
}

#[cfg(not(windows))]
pub fn spawn_in(_: &Spec, _: &Job) -> Result<Proc, String> {
    Err("The local runtime runs only on 64-bit Windows.".into())
}

/// Create a runtime child in this process's job.
pub fn spawn(spec: &Spec) -> Result<Proc, String> {
    let job = process_job().map_err(|e| format!("The local runtime was not started: {e}. No process was created."))?;
    spawn_in(spec, job)
}

/// Test seam: runs right after `CreateProcessW` succeeded, with the child's
/// pid and handle — the moment a parent could die before doing anything else.
#[cfg(test)]
pub(crate) mod seam {
    use std::cell::RefCell;
    pub type OnCreated = Option<Box<dyn FnMut(u32, isize)>>;
    thread_local! {
        pub static CREATED: RefCell<OnCreated> = const { RefCell::new(None) };
    }
    /// Pause a `spawn_in` whose arguments contain `marker`, at the moment its
    /// pipe writers are inheritable: `reached` is set, then it waits for `go`.
    /// (marker, reached, go)
    pub type Hold = Option<(String, std::sync::Arc<std::sync::atomic::AtomicBool>, std::sync::Arc<std::sync::atomic::AtomicBool>)>;
    pub static HOLD: std::sync::Mutex<Hold> = std::sync::Mutex::new(None);
    pub(super) fn inheritable(spec: &super::Spec) {
        use std::sync::atomic::Ordering;
        let hold = HOLD.lock().unwrap().clone();
        if let Some((marker, reached, go)) = hold
            && spec.args.iter().any(|a| a.to_string_lossy().contains(&marker))
        {
            reached.store(true, Ordering::SeqCst);
            while !go.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
    pub(super) fn created(pid: u32, handle: isize) {
        CREATED.with(|h| {
            if let Some(f) = h.borrow_mut().as_mut() {
                f(pid, handle);
            }
        });
    }
}
