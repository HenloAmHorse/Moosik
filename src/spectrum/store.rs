//! The analysis cache store (cache namespace v2): immutable generations, one
//! head per analysis, pins, and conservative collection.
//!
//! # Owner policy this implements (decisions of 2026-10-13)
//!
//! * **D1 = L1 + L1a.** Files written by earlier versions stay where they are,
//!   untouched. They are never exact hits; the display may offer them only as
//!   clearly labelled manual alternatives (see `alt_cache`). Nothing is
//!   imported, rewritten, moved or deleted from the legacy namespace.
//! * **D2.** Other programs may still write the source while it is analysed.
//!   Publication re-checks the source on the job's own open handle, under the
//!   coordination lock, immediately before the head commit. A write landing
//!   after that check and before the commit is **not** detected: the known
//!   check-to-commit window. This is not atomic freshness.
//! * **D3.** Source identity is path + size + last-write time + a sampled
//!   content digest ([`SAMPLE_POLICY`]). A tag edit changes the time, so it
//!   can cause a re-analysis. An edit confined to unsampled bytes that keeps
//!   the size and the time is **not** detected.
//! * **D4.** A separate namespace (`<cache>/v2/`) and versioned envelopes
//!   ([`ENVELOPE_VERSION`]): older builds never read these files, and this
//!   build never reads theirs as exact hits.
//! * **D5.** Per-analysis coordination files (`S.coord`, `S.pub`) are stable:
//!   the protocol never deletes them.
//! * **D6.** Deletion of any kind runs only where the filesystem scope is
//!   validated (Windows, local fixed disk, NTFS) **and** an in-process lock
//!   probe passed in the directory. Anywhere else nothing is deleted and the
//!   bytes are reported.
//! * **D7.** Process-crash coherence only: a reader sees the old pair or the
//!   complete new pair, never a hybrid, across a crash of the publisher. No
//!   power-loss freshness is claimed: after power loss an older valid
//!   generation may be what the head names.
//! * **D8.** Pinned (displayed or in-use) generations are never deleted; the
//!   bytes they hold are reported, so a budget can be exceeded and is said to
//!   be.
//!
//! # Files, per analysis stem `S`
//!
//! | file | role | lifetime |
//! |---|---|---|
//! | `S.coord` | coordination: exclusive lock held briefly around every head read + pin, head write and deletion decision | stable (D5) |
//! | `S.pub` | publisher authority: one publisher per stem for a whole publication | stable (D5) |
//! | `S.head` | names the committed generation; replaced by an atomic rename under `S.coord` | removed only by whole-stem eviction |
//! | `S.g{G}.pin` | lease file: a shared lock on it is a pin | deleted with its generation |
//! | `S.g{G}.mix`, `S.g{G}.lr` | immutable enveloped payloads, written once (temp, `sync_all`, rename) | deleted only under `S.coord` with an exclusive lock on the pin |
//!
//! A reader takes its pin while holding `S.coord`, after reading the head
//! under it; a collector re-reads the head and tries the exclusive lock under
//! `S.coord`. So a generation is either pinned before the collector decides,
//! or not named by the head and unpinned when it decides — never deleted
//! between a reader's head read and its pin.
//!
//! # Identity
//!
//! * [`SourceSnapshot`]: path, size, last-write time, sampled digest — always
//!   taken from one open handle.
//! * [`Descriptor`]: the settings record (every result-affecting setting, the
//!   effective rate and the hop; `identity::AnalysisSettings::record`) plus
//!   the source snapshot. Its canonical text is deterministic and its digest
//!   is SHA-256. Exact lookup and alternate discovery compare descriptors and
//!   records through this one type.
//! * [`Envelope`]: what publication recorded — generation, kind, descriptor
//!   digest, decoded-audio digest, the Mix payload digest the pair is bound
//!   to, and the payload digest. Never compared at lookup except as binding.
//! * [`Installed`]: an installed generation and the pin that keeps it alive.
//!
//! Persistent digests are SHA-256 (FIPS 180-4) from `ring`, which the
//! application already links through its TLS stack. Nothing persistent uses
//! Rust's `DefaultHasher`.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The envelope format this build writes and accepts.
pub const ENVELOPE_VERSION: u16 = 1;
const MAGIC: &[u8; 8] = b"MOOSKGEN";
const HEAD_MAGIC: &str = "MOOSKHEAD1 ";
const HEADER: usize = 168;
/// Longest descriptor an envelope may carry.
pub const MAX_DESCRIPTOR: usize = 16 << 10;
/// Largest generation file read into memory: the cache payload limit plus the
/// envelope.
pub const MAX_GENERATION_FILE: usize = (512 << 20) + HEADER + MAX_DESCRIPTOR;

/// The sampled-content policy, stated in every descriptor so a change of
/// policy is a change of identity.
///
/// A file of at most `SAMPLE_WINDOWS × SAMPLE_WINDOW` bytes is digested whole.
/// A larger one is digested over `SAMPLE_WINDOWS` windows of `SAMPLE_WINDOW`
/// bytes, the first at offset 0, the last ending at the end of the file, the
/// others evenly spaced between (offset `i × (size − W) / (K − 1)`, integer
/// division). Each window is fed as its offset (u64 LE) then its bytes, after
/// the domain string and the size. At most 1 MiB is read.
pub const SAMPLE_POLICY: &str = "w65536x16";
const SAMPLE_WINDOW: u64 = 64 << 10;
const SAMPLE_WINDOWS: u64 = 16;

pub type Digest = [u8; 32];

pub fn sha256(parts: &[&[u8]]) -> Digest {
    let mut c = ring::digest::Context::new(&ring::digest::SHA256);
    for p in parts {
        c.update(p);
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(c.finish().as_ref());
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ── source ──────────────────────────────────────────────────────────────────

/// What a source file is, read from one open handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSnapshot {
    pub path: PathBuf,
    pub size: u64,
    pub mtime_ns: u128,
    pub sample: Digest,
}

impl SourceSnapshot {
    /// A fresh snapshot of what `path` names now, through a handle opened for
    /// it. Used at lookup and at installation, never per frame.
    pub fn of_path(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        Self::of_handle(&mut f, path)
    }

    fn of_handle(f: &mut File, path: &Path) -> io::Result<Self> {
        let m = f.metadata()?;
        let size = m.len();
        let mtime_ns = m
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let sample = sample_digest(f, size)?;
        Ok(Self { path: path.to_path_buf(), size, mtime_ns, sample })
    }
}

fn sample_digest(f: &mut File, size: u64) -> io::Result<Digest> {
    let mut c = ring::digest::Context::new(&ring::digest::SHA256);
    c.update(b"moosik-sample/1\0");
    c.update(SAMPLE_POLICY.as_bytes());
    c.update(&size.to_le_bytes());
    let whole = size <= SAMPLE_WINDOW * SAMPLE_WINDOWS;
    let offsets: Vec<u64> = if whole {
        vec![0]
    } else {
        (0..SAMPLE_WINDOWS).map(|i| i * (size - SAMPLE_WINDOW) / (SAMPLE_WINDOWS - 1)).collect()
    };
    let len = if whole { size } else { SAMPLE_WINDOW };
    let mut buf = vec![0u8; len as usize];
    for off in offsets {
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut buf)?;
        c.update(&off.to_le_bytes());
        c.update(&buf);
    }
    f.seek(SeekFrom::Start(0))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(c.finish().as_ref());
    Ok(out)
}

/// An analysis's open source. The snapshot, the decoder and the publication
/// checkpoint all use this one handle (the decoder reads a duplicate of it,
/// which refers to the same open file).
pub struct OpenSource {
    file: File,
    pub snapshot: SourceSnapshot,
}

impl OpenSource {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let snapshot = SourceSnapshot::of_handle(&mut file, path)?;
        Ok(Self { file, snapshot })
    }

    /// A duplicate handle for a decoder. It refers to the same open file, so
    /// a later replacement of the pathname does not change what it reads.
    pub fn reader(&self) -> io::Result<File> {
        let mut f = self.file.try_clone()?;
        f.seek(SeekFrom::Start(0))?;
        Ok(f)
    }

    /// Whether this handle's own content (size, time, sampled digest) is what
    /// it was at `open`. Says nothing about what the pathname names now.
    pub fn unchanged(&mut self) -> io::Result<bool> {
        let now = SourceSnapshot::of_handle(&mut self.file, &self.snapshot.path.clone())?;
        Ok(now == self.snapshot)
    }
}

/// The decoded audio's digest, accumulated during the existing decode pass.
pub struct DecodedDigest {
    ctx: ring::digest::Context,
    buf: Vec<u8>,
}

impl DecodedDigest {
    /// `kind` names the sample representation ("pcm-i16", "dsd-f32").
    pub fn new(kind: &str, channels: usize, rate: u32) -> Self {
        let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
        ctx.update(b"moosik-decoded/1\0");
        ctx.update(kind.as_bytes());
        ctx.update(&(channels as u64).to_le_bytes());
        ctx.update(&rate.to_le_bytes());
        Self { ctx, buf: Vec::with_capacity(1 << 16) }
    }

    #[inline]
    pub fn push_i16(&mut self, s: i16) {
        self.buf.extend_from_slice(&s.to_le_bytes());
        if self.buf.len() >= 1 << 16 {
            self.flush();
        }
    }

    #[inline]
    pub fn push_f32(&mut self, s: f32) {
        self.buf.extend_from_slice(&s.to_bits().to_le_bytes());
        if self.buf.len() >= 1 << 16 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        self.ctx.update(&self.buf);
        self.buf.clear();
    }

    pub fn finish(mut self) -> Digest {
        self.flush();
        let mut out = [0u8; 32];
        out.copy_from_slice(self.ctx.finish().as_ref());
        out
    }
}

// ── descriptor ──────────────────────────────────────────────────────────────

/// Everything an exact hit must match: the settings record (settings, rate,
/// hop) and the source snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub record: String,
    pub source: SourceSnapshot,
}

fn path_bytes(p: &Path) -> &[u8] {
    p.as_os_str().as_encoded_bytes()
}

impl Descriptor {
    /// Deterministic text: one `key=value` per line in a fixed order, the path
    /// as hex of its bytes, the record as written (ASCII, no newlines).
    pub fn canonical(&self) -> String {
        format!(
            "moosik-store-descriptor/1\nrecord={}\npath={}\nsize={}\nmtime_ns={}\nsample_policy={SAMPLE_POLICY}\nsample={}\n",
            self.record.replace(['\n', '\r'], "_"),
            hex(path_bytes(&self.source.path)),
            self.source.size,
            self.source.mtime_ns,
            hex(&self.source.sample),
        )
    }

    pub fn digest(&self) -> Digest {
        sha256(&[self.canonical().as_bytes()])
    }

    /// The settings record a canonical descriptor states. Matching a found
    /// descriptor is then done by rebuilding the expected canonical text from
    /// this record and the current source snapshot and comparing the whole
    /// text, so the path is never reconstructed from stored bytes.
    pub fn record_in(text: &str) -> Option<&str> {
        let mut lines = text.split('\n');
        (lines.next()? == "moosik-store-descriptor/1").then_some(())?;
        lines.next()?.strip_prefix("record=")
    }
}

/// The stem of an analysis: the path and the settings, never the content, so
/// a changed source keeps its stem (and its stable coordination files) and
/// misses on the descriptor. The path half comes first so one track's stems
/// share a prefix ([`path_prefix`]).
pub fn stem(track: &Path, settings_canonical: &str) -> String {
    format!(
        "{}_{}",
        path_prefix(track),
        hex(&sha256(&[b"moosik-store/settings\0", settings_canonical.as_bytes()])[..8])
    )
}

pub fn path_prefix(track: &Path) -> String {
    hex(&sha256(&[b"moosik-store/path\0", path_bytes(track)])[..8])
}

// ── envelope ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mix = 1,
    Lr = 2,
}

/// What publication recorded about one payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub kind: Kind,
    pub generation: u128,
    pub descriptor: Digest,
    pub decoded: Digest,
    /// The Mix payload's digest: the Mix's own, or the one an L/R pair is
    /// bound to.
    pub mix: Digest,
    pub payload: Digest,
}

fn encode_envelope(e: &Envelope, descriptor: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + descriptor.len() + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&ENVELOPE_VERSION.to_le_bytes());
    out.push(e.kind as u8);
    out.push(0);
    out.extend_from_slice(&(descriptor.len() as u32).to_le_bytes());
    out.extend_from_slice(&e.generation.to_le_bytes());
    out.extend_from_slice(&e.descriptor);
    out.extend_from_slice(&e.decoded);
    out.extend_from_slice(&e.mix);
    out.extend_from_slice(&e.payload);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    debug_assert_eq!(out.len(), HEADER);
    out.extend_from_slice(descriptor.as_bytes());
    out.extend_from_slice(payload);
    out
}

/// Why a generation file was refused. Every one means "not this", never "part
/// of this".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvelopeError {
    TooShort,
    Magic,
    Version(u16),
    Kind(u8),
    Reserved,
    DescriptorLength(usize),
    Length { declared: Option<usize>, actual: usize },
    DescriptorDigest,
    DescriptorText,
    PayloadDigest,
    MixBinding,
}

fn digest_at(raw: &[u8], at: usize) -> Digest {
    raw[at..at + 32].try_into().unwrap()
}

/// The envelope's header and descriptor, validated, without the payload.
/// `raw` may be just the first `HEADER + descriptor` bytes.
fn parse_head(raw: &[u8]) -> Result<(Envelope, String, usize), EnvelopeError> {
    if raw.len() < HEADER {
        return Err(EnvelopeError::TooShort);
    }
    if &raw[..8] != MAGIC {
        return Err(EnvelopeError::Magic);
    }
    let version = u16::from_le_bytes([raw[8], raw[9]]);
    if version != ENVELOPE_VERSION {
        return Err(EnvelopeError::Version(version));
    }
    let kind = match raw[10] {
        1 => Kind::Mix,
        2 => Kind::Lr,
        k => return Err(EnvelopeError::Kind(k)),
    };
    if raw[11] != 0 {
        return Err(EnvelopeError::Reserved);
    }
    let dlen = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    if dlen == 0 || dlen > MAX_DESCRIPTOR {
        return Err(EnvelopeError::DescriptorLength(dlen));
    }
    let generation = u128::from_le_bytes(raw[16..32].try_into().unwrap());
    let e = Envelope {
        kind,
        generation,
        descriptor: digest_at(raw, 32),
        decoded: digest_at(raw, 64),
        mix: digest_at(raw, 96),
        payload: digest_at(raw, 128),
    };
    let plen = u64::from_le_bytes(raw[160..168].try_into().unwrap());
    let plen = usize::try_from(plen).map_err(|_| EnvelopeError::Length { declared: None, actual: raw.len() })?;
    let desc = raw.get(HEADER..HEADER + dlen).ok_or(EnvelopeError::TooShort)?;
    if sha256(&[desc]) != e.descriptor {
        return Err(EnvelopeError::DescriptorDigest);
    }
    let text = std::str::from_utf8(desc).map_err(|_| EnvelopeError::DescriptorText)?.to_string();
    Ok((e, text, plen))
}

/// A whole generation file, validated: header, exact length, descriptor and
/// payload digests, and for a Mix that its binding digest is its own payload.
#[cfg(test)]
pub fn decode_envelope(raw: &[u8]) -> Result<(Envelope, String, &[u8]), EnvelopeError> {
    let (e, text, at) = check_envelope(raw)?;
    let payload = &raw[at..];
    verified(&e, payload, |_| ())?;
    Ok((e, text, payload))
}

/// A whole generation file's structure: header, exact length and descriptor
/// digest, and where its payload starts. Not its payload digest.
fn check_envelope(raw: &[u8]) -> Result<(Envelope, String, usize), EnvelopeError> {
    let (e, text, plen) = parse_head(raw)?;
    let total = HEADER
        .checked_add(text.len())
        .and_then(|n| n.checked_add(plen))
        .ok_or(EnvelopeError::Length { declared: None, actual: raw.len() })?;
    if total != raw.len() {
        return Err(EnvelopeError::Length { declared: Some(total), actual: raw.len() });
    }
    let at = HEADER + text.len();
    Ok((e, text, at))
}

/// `decode(payload)`, returned only if the payload's digest matches its
/// envelope (and a Mix is bound to its own payload); dropped otherwise.
///
/// The digest is computed on a second thread while `decode` runs here on the
/// same bytes, so the two passes overlap instead of adding up. That is sound
/// only because every decoder passed in is bounded and treats its input as
/// untrusted: they are the decoders that read unverified 1.5.2.1 caches. What
/// they produce stays private to this call until the digest has matched, and
/// both have finished before it returns. If the second thread cannot be
/// started, the digest is computed here first, as before.
fn verified<T>(e: &Envelope, payload: &[u8], decode: impl FnOnce(&[u8]) -> T) -> Result<T, EnvelopeError> {
    let digest = || {
        let _p = prof::span("env.payload_sha");
        sha256(&[payload]) == e.payload
    };
    let (matched, t) = std::thread::scope(|s| match std::thread::Builder::new().spawn_scoped(s, digest) {
        Ok(h) => {
            let t = decode(payload);
            // A digest thread that panicked is not a match.
            (h.join().unwrap_or(false), Some(t))
        }
        Err(_) => {
            let ok = digest();
            (ok, ok.then(|| decode(payload)))
        }
    });
    if !matched {
        return Err(EnvelopeError::PayloadDigest);
    }
    if e.kind == Kind::Mix && e.mix != e.payload {
        return Err(EnvelopeError::MixBinding);
    }
    t.ok_or(EnvelopeError::PayloadDigest)
}

/// A validated generation file's payload, borrowed from the buffer that was
/// read rather than copied out of it. The payload is the file's tail, from
/// `at`; only the store hands out a `Payload` of a file, after its checks.
pub struct Payload {
    raw: Vec<u8>,
    at: usize,
}

impl std::ops::Deref for Payload {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.raw[self.at..]
    }
}

/// Bytes that are a payload in full (a freshly encoded Mix).
impl From<Vec<u8>> for Payload {
    fn from(raw: Vec<u8>) -> Self {
        Self { raw, at: 0 }
    }
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Payload({} bytes)", self.len())
    }
}

// ── load phase probes ───────────────────────────────────────────────────────

/// Load phase timing for measurement builds: test-only, and off unless
/// `MOOSIK_LOAD_PROFILE=1`. Each span records its start and end (seconds from
/// the first span) and its thread, so overlapping work stays visible and is
/// never summed by accident. Outside tests a span is a zero-sized no-op.
#[cfg(test)]
pub mod prof {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    pub type Rec = (&'static str, f64, f64, String);
    static ON: OnceLock<bool> = OnceLock::new();
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    static SPANS: Mutex<Vec<Rec>> = Mutex::new(Vec::new());

    pub struct Span(Option<(&'static str, Instant)>);

    pub fn on() -> bool {
        *ON.get_or_init(|| std::env::var("MOOSIK_LOAD_PROFILE").as_deref() == Ok("1"))
    }

    pub fn span(name: &'static str) -> Span {
        Span(on().then(|| {
            EPOCH.get_or_init(Instant::now);
            (name, Instant::now())
        }))
    }

    impl Drop for Span {
        fn drop(&mut self) {
            if let Some((name, t0)) = self.0.take() {
                let t1 = Instant::now();
                let e = *EPOCH.get().unwrap();
                let rec = (name, (t0 - e).as_secs_f64(), (t1 - e).as_secs_f64(), format!("{:?}", std::thread::current().id()));
                SPANS.lock().unwrap_or_else(|e| e.into_inner()).push(rec);
            }
        }
    }

    /// Used by the measurement harness only.
    #[allow(dead_code)]
    pub fn take() -> Vec<Rec> {
        std::mem::take(&mut *SPANS.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

#[cfg(not(test))]
pub mod prof {
    pub struct Span;
    impl Drop for Span {
        #[inline(always)]
        fn drop(&mut self) {}
    }
    #[inline(always)]
    pub fn span(_: &'static str) -> Span {
        Span
    }
}

// ── test seams ──────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod seam {
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    pub type Hook = Arc<dyn Fn(&str, &str) + Send + Sync>;
    pub type Fault = Arc<dyn Fn(&str, &Path) -> Option<io::ErrorKind> + Send + Sync>;
    static HOOKS: Mutex<Vec<(PathBuf, u64, Hook)>> = Mutex::new(Vec::new());
    static FAULTS: Mutex<Vec<(PathBuf, u64, Fault)>> = Mutex::new(Vec::new());
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

    /// Removes its hook or fault when dropped.
    pub struct Armed(u64);
    impl Drop for Armed {
        fn drop(&mut self) {
            HOOKS.lock().unwrap_or_else(|e| e.into_inner()).retain(|(_, id, _)| *id != self.0);
            FAULTS.lock().unwrap_or_else(|e| e.into_inner()).retain(|(_, id, _)| *id != self.0);
        }
    }

    /// The canonical directory, created if needed: a seam armed before the
    /// store first opens its directory must still match it.
    fn key(dir: &Path) -> PathBuf {
        let _ = std::fs::create_dir_all(dir);
        std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
    }

    /// Call `f(point, stem)` at every named point of the store in `dir`.
    pub fn hook(dir: &Path, f: impl Fn(&str, &str) + Send + Sync + 'static) -> Armed {
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HOOKS.lock().unwrap().push((key(dir), id, Arc::new(f)));
        Armed(id)
    }

    /// Fail named operations of the store in `dir`.
    pub fn fault(dir: &Path, f: impl Fn(&str, &Path) -> Option<io::ErrorKind> + Send + Sync + 'static) -> Armed {
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        FAULTS.lock().unwrap().push((key(dir), id, Arc::new(f)));
        Armed(id)
    }

    pub(super) fn at(dir: &Path, point: &str, stem: &str) {
        let hooks: Vec<Hook> =
            HOOKS.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|(d, ..)| d == dir).map(|(_, _, h)| h.clone()).collect();
        for h in hooks {
            h(point, stem);
        }
    }

    pub(super) fn fail(dir: &Path, point: &str, p: &Path) -> io::Result<()> {
        let faults: Vec<Fault> =
            FAULTS.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|(d, ..)| d == dir).map(|(_, _, f)| f.clone()).collect();
        for f in faults {
            if let Some(k) = f(point, p) {
                return Err(io::Error::new(k, format!("injected at {point}")));
            }
        }
        Ok(())
    }
}

#[cfg(not(test))]
mod seam {
    use std::io;
    use std::path::Path;
    #[inline(always)]
    pub(super) fn at(_: &Path, _: &str, _: &str) {}
    #[inline(always)]
    pub(super) fn fail(_: &Path, _: &str, _: &Path) -> io::Result<()> {
        Ok(())
    }
}

// ── locks ───────────────────────────────────────────────────────────────────

fn open_lock_file(p: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).create(true).truncate(false).open(p)
}

/// A shared lock on a generation's pin file. Dropping it releases the pin.
#[derive(Debug)]
pub struct PinGuard {
    _file: File,
    pub generation: u128,
}

struct StemLock(File);
impl Drop for StemLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn random_u128() -> u128 {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut v = 0u128;
    for lane in 0..2u64 {
        // RandomState is seeded per process from OS randomness; the time,
        // process and counter make two draws in one process differ. A
        // generation ID only has to be unique in its directory, and a
        // collision is refused by `create_new`, never overwritten.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(t);
        h.write_u32(std::process::id());
        h.write_u64(n);
        h.write_u64(lane);
        v = (v << 64) | h.finish() as u128;
    }
    v
}

// ── scope ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Capability {
    /// The validated scope AND a passing lock probe: deletion is admitted.
    Supported(String),
    /// No deletion of any kind (D6).
    Unsupported(String),
}

impl Capability {
    /// Whether this scope admits deletion of any kind, including a store's
    /// own uncommitted files (D6).
    pub fn deletes(&self) -> bool {
        matches!(self, Capability::Supported(_))
    }
}

/// Windows, local fixed disk, NTFS: where cross-process lock behaviour was
/// validated. Everything else is unvalidated.
pub fn classify_scope(remote: bool, drive_type: u32, fs_name: &str) -> Result<String, String> {
    const DRIVE_FIXED: u32 = 3;
    if remote {
        return Err("network (UNC) path".into());
    }
    if drive_type != DRIVE_FIXED {
        return Err(format!("drive type {drive_type} is not a local fixed disk"));
    }
    if fs_name != "NTFS" {
        return Err(format!("filesystem {fs_name:?} is not validated"));
    }
    Ok("windows/local-fixed/NTFS".into())
}

#[cfg(windows)]
fn scope_of_resolved(canon: &Path) -> Result<String, String> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    unsafe extern "system" {
        fn GetVolumePathNameW(path: *const u16, out: *mut u16, len: u32) -> i32;
        fn GetDriveTypeW(root: *const u16) -> u32;
        fn GetVolumeInformationW(
            root: *const u16,
            name: *mut u16,
            name_len: u32,
            serial: *mut u32,
            max_comp: *mut u32,
            flags: *mut u32,
            fs: *mut u16,
            fs_len: u32,
        ) -> i32;
    }
    let s = canon.to_string_lossy().to_string();
    let (remote, plain) = if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        (true, format!(r"\\{rest}"))
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        (false, rest.to_string())
    } else {
        (s.starts_with(r"\\"), s.clone())
    };
    if remote {
        return classify_scope(true, 0, "");
    }
    let wide: Vec<u16> = std::ffi::OsStr::new(&plain).encode_wide().chain([0]).collect();
    let mut root = vec![0u16; 1024];
    // SAFETY: valid NUL-terminated input; the output buffer length is passed.
    if unsafe { GetVolumePathNameW(wide.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
        return Err(format!("GetVolumePathNameW failed: {}", io::Error::last_os_error()));
    }
    // SAFETY: `root` is NUL-terminated by the call above.
    let drive_type = unsafe { GetDriveTypeW(root.as_ptr()) };
    let mut fsn = vec![0u16; 64];
    // SAFETY: optional outputs are null; the name buffer length is passed.
    let ok = unsafe {
        GetVolumeInformationW(
            root.as_ptr(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            fsn.as_mut_ptr(),
            fsn.len() as u32,
        )
    };
    if ok == 0 {
        return Err(format!("GetVolumeInformationW failed: {}", io::Error::last_os_error()));
    }
    let fs_name = std::ffi::OsString::from_wide(&fsn[..fsn.iter().position(|&c| c == 0).unwrap_or(fsn.len())])
        .to_string_lossy()
        .to_string();
    classify_scope(false, drive_type, &fs_name)
}

#[cfg(not(windows))]
fn scope_of_resolved(_dir: &Path) -> Result<String, String> {
    Err("platform not validated".into())
}

/// In this directory: a shared lock on one handle excludes an exclusive lock
/// on another, and release lets it through. One admission condition, never
/// sufficient alone.
///
/// Run only where the scope is already validated, so a probe never leaves a
/// file on an unvalidated filesystem. Its file is removed only when the probe
/// passed: a failed probe is not deletion permission (D6), so its file is
/// left and named in the error.
fn probe_locks(dir: &Path) -> Result<(), String> {
    let p = dir.join(format!(".probe{:08x}", random_u128() as u32));
    seam::at(dir, "probe", "");
    let r = (|| -> Result<(), String> {
        let a = open_lock_file(&p).map_err(|e| e.to_string())?;
        let b = OpenOptions::new().read(true).write(true).open(&p).map_err(|e| e.to_string())?;
        seam::fail(dir, "probe.lock", &p).map_err(|e| e.to_string())?;
        a.lock_shared().map_err(|e| e.to_string())?;
        match b.try_lock() {
            Err(TryLockError::WouldBlock) => {}
            Ok(()) => return Err("exclusive lock granted over a shared one".into()),
            Err(TryLockError::Error(e)) => return Err(e.to_string()),
        }
        a.unlock().map_err(|e| e.to_string())?;
        b.try_lock().map_err(|_| "exclusive lock refused after release".to_string())?;
        b.unlock().map_err(|e| e.to_string())?;
        Ok(())
    })();
    match r {
        Ok(()) => {
            let _ = fs::remove_file(&p);
            Ok(())
        }
        Err(e) if p.exists() => Err(format!("{e}; probe file {} left", p.display())),
        Err(e) => Err(e),
    }
}

// ── store ───────────────────────────────────────────────────────────────────

/// One cache namespace directory.
pub struct Store {
    dir: PathBuf,
    pub capability: Capability,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Store({}, {:?})", self.dir.display(), self.capability)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PublishError {
    /// Another publisher holds this stem's authority. Nothing was written.
    Busy,
    /// Cancelled before the commit decision. Nothing was committed.
    Cancelled { cleanup_pending: bool },
    /// The job's own open source changed (or could not be checked) at the
    /// checkpoint. Nothing was committed; the old head and pair are untouched.
    SourceChanged { cleanup_pending: bool },
    SourceCheck { error: String, cleanup_pending: bool },
    /// A written payload did not read back identically.
    Verify { cleanup_pending: bool },
    Io(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Miss {
    NoHead,
    /// The head names a generation whose pin or payload is missing or damaged.
    Damaged(String),
    /// A valid generation for another source or other settings.
    DescriptorMismatch,
    Io(String),
}

/// A committed generation on its way to installation, with the publisher's
/// pin. Carried through the completion message so the generation stays
/// pinned from commit to installation.
#[derive(Debug)]
pub struct Published {
    pub stem: String,
    pub generation: u128,
    pub envelope: Envelope,
    pub has_lr: bool,
    /// Always `Some` from `publish`. Optional only so the completion path can
    /// state what it does without one (re-pin, which can fail).
    pub pin: Option<PinGuard>,
}

/// An installed generation: what is displayed and what a lazy L/R read
/// belongs to. Every in-process user shares the pin.
#[derive(Debug)]
pub struct Installed {
    pub store: Arc<Store>,
    pub stem: String,
    pub generation: u128,
    pub envelope: Envelope,
    pub has_lr: bool,
    /// Held for its lock: while any holder of this `Installed` lives, the
    /// generation is pinned and no collection can delete it.
    #[allow(dead_code)]
    pub pin: PinGuard,
}

impl Installed {
    #[cfg(test)]
    pub fn mix_path(&self) -> PathBuf {
        self.store.gpath(&self.stem, self.generation, "mix")
    }
    pub fn lr_path(&self) -> PathBuf {
        self.store.gpath(&self.stem, self.generation, "lr")
    }
}

/// What a deletion pass did. `deleted_bytes` counts only payload files whose
/// removal succeeded. `partial`: a payload removal failed (pin kept, retried
/// later). `pin_pending`: payloads gone, pin file left (cleanup pending).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CollectReport {
    pub deleted: Vec<u128>,
    pub skipped_pinned: Vec<u128>,
    pub partial: Vec<u128>,
    pub pin_pending: Vec<u128>,
    pub pinned_bytes: u64,
    pub deleted_bytes: u64,
    pub unsupported: Option<String>,
    pub heads_removed: usize,
    pub errors: Vec<String>,
}

impl CollectReport {
    /// Add another pass's outcome to this one.
    pub fn merge(&mut self, b: CollectReport) {
        self.deleted.extend(b.deleted);
        self.skipped_pinned.extend(b.skipped_pinned);
        self.partial.extend(b.partial);
        self.pin_pending.extend(b.pin_pending);
        self.pinned_bytes += b.pinned_bytes;
        self.deleted_bytes += b.deleted_bytes;
        if self.unsupported.is_none() {
            self.unsupported = b.unsupported;
        }
        self.heads_removed += b.heads_removed;
        self.errors.extend(b.errors);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadState {
    Absent,
    Valid(u128),
    Unreadable(String),
}

enum PinCheck {
    Free(Option<File>),
    Pinned,
    Error(String),
}

/// A head the discovery found: its stem, generation and Mix descriptor.
#[derive(Clone, Debug)]
pub struct Found {
    pub stem: String,
    /// The Mix envelope's canonical descriptor text.
    pub text: String,
    pub mix_bytes: u64,
    pub lr_bytes: Option<u64>,
}

impl Store {
    /// Open (creating) the namespace directory. It is resolved once to an
    /// absolute canonical path, and that path is classified, probed and used
    /// for everything after. Replacing the directory or an ancestor after this
    /// is not detected and is unsupported.
    pub fn open(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let dir = fs::canonicalize(dir)?;
        let scope = scope_of_resolved(&dir);
        #[cfg(test)]
        let scope = if test_scope::forced_unvalidated(&dir) { Err("unvalidated (test)".to_string()) } else { scope };
        // The scope first: an unvalidated filesystem gets no probe file.
        let capability = match scope {
            Err(s) => Capability::Unsupported(format!("filesystem scope not validated: {s}")),
            Ok(s) => match probe_locks(&dir) {
                Ok(()) => Capability::Supported(s),
                Err(e) => Capability::Unsupported(format!("lock probe failed: {e}")),
            },
        };
        Ok(Self { dir, capability })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn at(&self, point: &str, stem: &str) {
        seam::at(&self.dir, point, stem);
    }

    fn fail(&self, point: &str, p: &Path) -> io::Result<()> {
        seam::fail(&self.dir, point, p)
    }

    fn path(&self, stem: &str, suffix: &str) -> PathBuf {
        self.dir.join(format!("{stem}.{suffix}"))
    }

    fn gpath(&self, stem: &str, g: u128, ext: &str) -> PathBuf {
        self.dir.join(format!("{stem}.g{g:032x}.{ext}"))
    }

    fn stem_lock(&self, stem: &str) -> io::Result<StemLock> {
        let f = open_lock_file(&self.path(stem, "coord"))?;
        f.lock()?;
        Ok(StemLock(f))
    }

    pub fn head_state(&self, stem: &str) -> HeadState {
        let p = self.path(stem, "head");
        match self.fail("head.read", &p).and_then(|_| fs::read(&p)) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => HeadState::Absent,
            Err(e) => HeadState::Unreadable(e.to_string()),
            Ok(b) => match std::str::from_utf8(&b)
                .ok()
                .and_then(|t| t.strip_prefix(HEAD_MAGIC))
                .and_then(|t| t.strip_suffix('\n'))
                .filter(|t| t.len() == 32)
                .and_then(|t| u128::from_str_radix(t, 16).ok())
            {
                Some(g) => HeadState::Valid(g),
                None => HeadState::Unreadable(format!("malformed head ({} bytes)", b.len())),
            },
        }
    }

    /// Write `bytes` to a new temporary file, sync it and rename it over
    /// `final_path`. On failure the temporary file is removed only where this
    /// scope admits deletion; elsewhere it is left and the error says so.
    fn write_atomic(&self, final_path: &Path, bytes: &[u8]) -> io::Result<()> {
        let tmp = final_path.with_extension(format!(
            "{}.tmp{:08x}",
            final_path.extension().and_then(|e| e.to_str()).unwrap_or(""),
            random_u128() as u32
        ));
        let mut created = false;
        let r = (|| {
            let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
            created = true;
            f.write_all(bytes)?;
            f.sync_all()?;
            drop(f);
            self.fail("rename", final_path)?;
            fs::rename(&tmp, final_path)
        })();
        match r {
            // Nothing of ours was created (a name that was already taken is
            // never ours to remove).
            Err(e) if !created => Err(e),
            // D6: permission comes from the scope alone. Whether the entry can
            // still be seen only changes the wording; an existence check that
            // fails (metadata error, dangling link) is not absence.
            Err(e) if !self.capability.deletes() => {
                let what = if fs::symlink_metadata(&tmp).is_ok() { "left" } else { "may remain" };
                Err(io::Error::new(e.kind(), format!("{e}; temporary file {what} (cleanup pending)")))
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                Err(e)
            }
            ok => ok,
        }
    }

    // ── publication ──

    /// Publish one generation (Mix, and L/R when given) for `desc`, guarded
    /// by the job's open source. Order: publisher authority; pin created under
    /// `S.coord` before any payload; L/R then Mix written and read back; then,
    /// under `S.coord`, the cancellation decision and the source checkpoint,
    /// and only then the head commit. On any refusal the coordination lock is
    /// released before the uncommitted generation is abandoned (which takes
    /// the lock itself), and nothing is committed.
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &self,
        desc: &Descriptor,
        settings_canonical: &str,
        decoded: Digest,
        mix: &[u8],
        lr: Option<&[u8]>,
        src: &mut OpenSource,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Published, PublishError> {
        let io = |e: io::Error| PublishError::Io(e.to_string());
        let stem = stem(&desc.source.path, settings_canonical);
        if cancelled() {
            return Err(PublishError::Cancelled { cleanup_pending: false });
        }
        let auth = open_lock_file(&self.path(&stem, "pub")).map_err(io)?;
        match auth.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(PublishError::Busy),
            Err(TryLockError::Error(e)) => return Err(io(e)),
        }
        self.at("pub.authority", &stem);
        // A new generation: its pin is created new under the stem lock before
        // any payload exists. An ID that is taken is never reused.
        let mut pin = None;
        for _ in 0..8 {
            let g = random_u128();
            let _s = self.stem_lock(&stem).map_err(io)?;
            match OpenOptions::new().read(true).write(true).create_new(true).open(self.gpath(&stem, g, "pin")) {
                Ok(f) => {
                    if self.gpath(&stem, g, "mix").exists() || self.gpath(&stem, g, "lr").exists() {
                        drop(f);
                        // The empty pin is this publisher's own; elsewhere
                        // than a validated scope it is left (D6).
                        if self.capability.deletes() {
                            let _ = fs::remove_file(self.gpath(&stem, g, "pin"));
                        }
                        continue;
                    }
                    f.lock_shared().map_err(io)?;
                    pin = Some(PinGuard { _file: f, generation: g });
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io(e)),
            }
        }
        let pin = pin.ok_or_else(|| PublishError::Io("no unused generation ID".into()))?;
        let g = pin.generation;
        self.at("pub.created", &stem);

        let canonical = desc.canonical();
        let mix_digest = sha256(&[mix]);
        let env = |kind, payload: &[u8]| Envelope {
            kind,
            generation: g,
            descriptor: desc.digest(),
            decoded,
            mix: mix_digest,
            payload: sha256(&[payload]),
        };
        let mix_env = env(Kind::Mix, mix);
        let mix_bytes = encode_envelope(&mix_env, &canonical, mix);
        let lr_bytes = lr.map(|lr| encode_envelope(&env(Kind::Lr, lr), &canonical, lr));
        let written = (|| -> io::Result<()> {
            if let Some(b) = &lr_bytes {
                self.write_atomic(&self.gpath(&stem, g, "lr"), b)?;
                self.at("pub.lr_written", &stem);
            }
            self.write_atomic(&self.gpath(&stem, g, "mix"), &mix_bytes)?;
            self.at("pub.mix_written", &stem);
            Ok(())
        })();
        let verified = written.is_ok()
            && lr_bytes.as_ref().is_none_or(|b| fs::read(self.gpath(&stem, g, "lr")).ok().as_deref() == Some(&b[..]))
            && fs::read(self.gpath(&stem, g, "mix")).ok().as_deref() == Some(&mix_bytes[..]);
        if !verified {
            let complete = self.abandon(&stem, pin);
            return Err(match written {
                Err(e) => PublishError::Io(format!("{e}{}", if complete { "" } else { "; cleanup pending" })),
                Ok(()) => PublishError::Verify { cleanup_pending: !complete },
            });
        }
        self.at("pub.before_head", &stem);
        {
            let s = self.stem_lock(&stem).map_err(io)?;
            // The commit decision: cancellation first, then the source
            // checkpoint on the job's own handle.
            let refusal = if cancelled() {
                Some(Refusal::Cancelled)
            } else {
                match self.fail("source.check", &src.snapshot.path).and_then(|_| src.unchanged()) {
                    Ok(true) => None,
                    Ok(false) => Some(Refusal::Changed),
                    Err(e) => Some(Refusal::Check(e.to_string())),
                }
            };
            if let Some(r) = refusal {
                // Lock order: release S.coord BEFORE abandon re-acquires it.
                drop(s);
                self.at("pub.guard_rejected", &stem);
                let complete = self.abandon(&stem, pin);
                drop(auth);
                return Err(match r {
                    Refusal::Cancelled => PublishError::Cancelled { cleanup_pending: !complete },
                    Refusal::Changed => PublishError::SourceChanged { cleanup_pending: !complete },
                    Refusal::Check(error) => PublishError::SourceCheck { error, cleanup_pending: !complete },
                });
            }
            self.at("pub.after_check", &stem);
            if let Err(e) = self.write_atomic(&self.path(&stem, "head"), format!("{HEAD_MAGIC}{g:032x}\n").as_bytes()) {
                drop(s);
                let complete = self.abandon(&stem, pin);
                return Err(PublishError::Io(format!("head commit failed: {e}{}", if complete { "" } else { "; cleanup pending" })));
            }
        }
        self.at("pub.after_head", &stem);
        drop(auth);
        Ok(Published { stem, generation: g, envelope: mix_env, has_lr: lr_bytes.is_some(), pin: Some(pin) })
    }

    /// Remove this publisher's own uncommitted generation. True only when its
    /// payloads and its pin file are confirmed gone.
    ///
    /// Outside a validated scope nothing is deleted (D6): the pin handle is
    /// released, the files stay, and the answer is false (cleanup pending).
    fn abandon(&self, stem: &str, pin: PinGuard) -> bool {
        if !self.capability.deletes() {
            drop(pin);
            return false;
        }
        let g = pin.generation;
        let Ok(_s) = self.stem_lock(stem) else { return false };
        let (_, failed) = self.remove_gen_data(stem, g);
        drop(pin);
        if !failed.is_empty() {
            return false; // the pin file stays, so a later collection retries
        }
        let p = self.gpath(stem, g, "pin");
        match self.fail("pin.remove", &p).and_then(|_| fs::remove_file(&p)) {
            Ok(()) => true,
            Err(e) => e.kind() == io::ErrorKind::NotFound,
        }
    }

    // ── reading ──

    fn pin_existing(&self, stem: &str, g: u128) -> Result<PinGuard, Miss> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.gpath(stem, g, "pin"))
            .map_err(|_| Miss::Damaged("pin missing".into()))?;
        match f.try_lock_shared() {
            Ok(()) => Ok(PinGuard { _file: f, generation: g }),
            Err(_) => Err(Miss::Damaged("pin unavailable".into())),
        }
    }

    /// Pin a named generation again, under the stem lock. Only for a
    /// completion that lost its publisher's pin; it can fail if the
    /// generation was collected in the gap.
    pub fn pin_generation(&self, stem: &str, g: u128) -> Result<PinGuard, Miss> {
        let _s = self.stem_lock(stem).map_err(|e| Miss::Io(e.to_string()))?;
        self.pin_existing(stem, g)
    }

    /// Read the head and pin its generation under the stem lock.
    pub fn pin_head(&self, stem: &str) -> Result<PinGuard, Miss> {
        let _p = prof::span("pin.head");
        let _s = self.stem_lock(stem).map_err(|e| Miss::Io(e.to_string()))?;
        let g = match self.head_state(stem) {
            HeadState::Valid(g) => g,
            HeadState::Absent => return Err(Miss::NoHead),
            HeadState::Unreadable(e) => return Err(Miss::Io(format!("head unreadable: {e}"))),
        };
        self.at("read.head_read", stem);
        self.pin_existing(stem, g)
    }

    /// The Mix payload of a pinned generation, validated against `desc`,
    /// with `decode` run on the payload while its digest is checked (see
    /// `verified`). Its result is returned only with a payload that passed
    /// every check.
    pub fn load_mix_with<T>(
        &self,
        stem: &str,
        pin: &PinGuard,
        desc: &Descriptor,
        decode: impl FnOnce(&[u8]) -> T,
    ) -> Result<(Envelope, bool, Payload, T), Miss> {
        let g = pin.generation;
        let p = prof::span("mix.read");
        let raw = super::cache::read_bounded(
            &self.gpath(stem, g, "mix"),
            super::cache::Limits { max_file_bytes: MAX_GENERATION_FILE, ..Default::default() },
        )
        .map_err(|e| Miss::Damaged(format!("mix unreadable: {e}")))?;
        drop(p);
        let damaged = |e: EnvelopeError| Miss::Damaged(format!("mix envelope: {e:?}"));
        let (e, text, at) = check_envelope(&raw).map_err(damaged)?;
        if e.kind != Kind::Mix || e.generation != g {
            return Err(Miss::Damaged("mix belongs to another generation".into()));
        }
        if text != desc.canonical() {
            return Err(Miss::DescriptorMismatch);
        }
        let p = prof::span("mix.verify_decode");
        let t = verified(&e, &raw[at..], decode).map_err(damaged)?;
        drop(p);
        let has_lr = self.gpath(stem, g, "lr").exists();
        Ok((e, has_lr, Payload { raw, at }, t))
    }

    /// Exact lookup: head, pin, Mix, descriptor. The pin is returned with the
    /// payload so the caller installs it without a gap.
    #[cfg(test)]
    pub fn lookup(&self, stem: &str, desc: &Descriptor) -> Result<(PinGuard, Envelope, bool, Payload), Miss> {
        self.lookup_with(stem, desc, |_| ()).map(|(pin, e, has_lr, p, ())| (pin, e, has_lr, p))
    }

    /// `lookup`, decoding the Mix as `load_mix_with` does.
    pub fn lookup_with<T>(
        &self,
        stem: &str,
        desc: &Descriptor,
        decode: impl FnOnce(&[u8]) -> T,
    ) -> Result<(PinGuard, Envelope, bool, Payload, T), Miss> {
        let pin = self.pin_head(stem)?;
        self.at("read.pinned", stem);
        let (e, has_lr, payload, t) = self.load_mix_with(stem, &pin, desc, decode)?;
        Ok((pin, e, has_lr, payload, t))
    }

    /// The L/R payload of an installed generation: the same generation, the
    /// same descriptor, decoded audio and Mix payload digest as the installed
    /// Mix. Read under the installed pin, never through the head.
    #[cfg(test)]
    pub fn read_lr(&self, inst: &Installed) -> Result<Payload, Miss> {
        self.read_lr_with(inst, |_| ()).map(|(p, ())| p)
    }

    /// `read_lr`, with `decode` run on the payload while its digest is
    /// checked (see `verified`). Its result is returned only with a payload
    /// that passed every check.
    pub fn read_lr_with<T>(&self, inst: &Installed, decode: impl FnOnce(&[u8]) -> T) -> Result<(Payload, T), Miss> {
        self.at("read.before_lr", &inst.stem);
        let p = prof::span("lr.read");
        let raw = super::cache::read_bounded(
            &self.gpath(&inst.stem, inst.generation, "lr"),
            super::cache::Limits { max_file_bytes: MAX_GENERATION_FILE, ..Default::default() },
        )
        .map_err(|e| Miss::Damaged(format!("channels unreadable: {e}")))?;
        drop(p);
        let damaged = |e: EnvelopeError| Miss::Damaged(format!("channels envelope: {e:?}"));
        let (e, _, at) = check_envelope(&raw).map_err(damaged)?;
        if e.kind != Kind::Lr
            || e.generation != inst.generation
            || e.descriptor != inst.envelope.descriptor
            || e.decoded != inst.envelope.decoded
            || e.mix != inst.envelope.mix
        {
            return Err(Miss::Damaged("channels do not belong to the installed analysis".into()));
        }
        let p = prof::span("lr.verify_decode");
        let t = verified(&e, &raw[at..], decode).map_err(damaged)?;
        drop(p);
        Ok((Payload { raw, at }, t))
    }

    /// Every committed head of one track (stems sharing `path_prefix`), with
    /// the Mix descriptor read from the envelope header only (never the rows).
    pub fn heads_for(&self, track: &Path) -> Vec<Found> {
        let prefix = format!("{}_", path_prefix(track));
        let Ok(entries) = fs::read_dir(&self.dir) else { return Vec::new() };
        let mut out: Vec<Found> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let stem = name.strip_suffix(".head")?.to_string();
                if !stem.starts_with(&prefix) {
                    return None;
                }
                let HeadState::Valid(g) = self.head_state(&stem) else { return None };
                let text = self.read_descriptor(&self.gpath(&stem, g, "mix"))?;
                let mix_bytes = fs::metadata(self.gpath(&stem, g, "mix")).ok()?.len();
                let lr_bytes = fs::metadata(self.gpath(&stem, g, "lr")).ok().map(|m| m.len());
                Some(Found { stem, text, mix_bytes, lr_bytes })
            })
            .collect();
        out.sort_by(|a, b| a.stem.cmp(&b.stem));
        out
    }

    fn read_descriptor(&self, p: &Path) -> Option<String> {
        let mut f = File::open(p).ok()?;
        let mut head = vec![0u8; HEADER];
        f.read_exact(&mut head).ok()?;
        let dlen = u32::from_le_bytes(head[12..16].try_into().ok()?) as usize;
        if dlen == 0 || dlen > MAX_DESCRIPTOR {
            return None;
        }
        head.resize(HEADER + dlen, 0);
        f.read_exact(&mut head[HEADER..]).ok()?;
        let (e, text, _) = parse_head(&head).ok()?;
        (e.kind == Kind::Mix && Descriptor::record_in(&text).is_some()).then_some(text)
    }

    // ── collection ──

    pub fn generations(&self, stem: &str) -> Vec<u128> {
        let prefix = format!("{stem}.g");
        let mut v: Vec<u128> = fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().into_string().ok()?;
                let rest = n.strip_prefix(&prefix)?;
                let (g, _) = rest.split_once('.')?;
                (g.len() == 32).then(|| u128::from_str_radix(g, 16).ok()).flatten()
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// Every stem in the namespace. A listing failure is an error, never an
    /// empty namespace; a namespace directory that does not exist is empty.
    pub fn stems(&self) -> io::Result<Vec<String>> {
        let entries = match self.fail("dir.list", &self.dir).and_then(|_| fs::read_dir(&self.dir)) {
            Ok(it) => it,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut v = Vec::new();
        for e in entries {
            if let Some(stem) = e?.file_name().into_string().ok().and_then(|n| n.strip_suffix(".coord").map(str::to_string)) {
                v.push(stem);
            }
        }
        v.sort();
        Ok(v)
    }

    /// Evict every stem (Clear All). Pinned generations stay and are reported.
    pub fn evict_all(&self) -> CollectReport {
        if let Some(r) = self.unsupported() {
            return r;
        }
        let mut total = CollectReport::default();
        match self.stems() {
            Ok(stems) => {
                for s in stems {
                    total.merge(self.evict_stem(&s));
                }
            }
            Err(e) => total.errors.push(format!("cannot list the cache folder: {e}")),
        }
        total
    }

    fn gen_bytes(&self, stem: &str, g: u128) -> u64 {
        ["mix", "lr"].iter().filter_map(|e| fs::metadata(self.gpath(stem, g, e)).ok()).map(|m| m.len()).sum()
    }

    /// Bytes of every generation payload in the namespace.
    pub fn bytes(&self) -> u64 {
        fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".mix") || n.ends_with(".lr")))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }

    fn unsupported(&self) -> Option<CollectReport> {
        match &self.capability {
            Capability::Supported(_) => None,
            Capability::Unsupported(why) => Some(CollectReport { unsupported: Some(why.clone()), ..Default::default() }),
        }
    }

    /// Delete the superseded generations of one stem that nobody pins.
    pub fn collect(&self, stem: &str) -> CollectReport {
        if let Some(r) = self.unsupported() {
            return r;
        }
        let mut r = CollectReport::default();
        let early = match self.head_state(stem) {
            HeadState::Unreadable(e) => return ambiguous(r, "selection", e),
            h => valid(&h),
        };
        let candidates: Vec<u128> = self.generations(stem).into_iter().filter(|g| Some(*g) != early).collect();
        self.at("col.selected", stem);
        let Ok(_s) = self.stem_lock(stem) else {
            r.errors.push(format!("{stem}: coordination lock unavailable; nothing deleted"));
            return r;
        };
        self.at("col.locked", stem);
        let head = match self.head_state(stem) {
            HeadState::Unreadable(e) => return ambiguous(r, "revalidation", e),
            h => valid(&h),
        };
        for g in candidates {
            if Some(g) != head {
                self.try_delete_gen(stem, g, &mut r);
            }
        }
        r
    }

    fn check_pin(&self, stem: &str, g: u128) -> PinCheck {
        let p = self.gpath(stem, g, "pin");
        match self.fail("pin.open_for_delete", &p).and_then(|_| OpenOptions::new().read(true).write(true).open(&p)) {
            Ok(f) => match self.fail("pin.lock", &p).map_err(TryLockError::Error).and_then(|_| f.try_lock()) {
                Ok(()) => PinCheck::Free(Some(f)),
                Err(TryLockError::WouldBlock) => PinCheck::Pinned,
                Err(TryLockError::Error(e)) => PinCheck::Error(format!("g{g:032x}: pin lock failed: {e}")),
            },
            // No pin file: an interrupted deletion or abandoned publication.
            // A lease is only taken by opening an existing pin file under this
            // stem lock, and a publisher creates its pin under it before any
            // payload exists, so nothing can hold a lease on `g` now.
            Err(e) if e.kind() == io::ErrorKind::NotFound => PinCheck::Free(None),
            Err(e) => PinCheck::Error(format!("g{g:032x}: pin open failed: {e}")),
        }
    }

    /// Under the stem lock: delete `g` if its pin can be taken exclusively.
    fn try_delete_gen(&self, stem: &str, g: u128, r: &mut CollectReport) -> bool {
        let excl = match self.check_pin(stem, g) {
            PinCheck::Free(f) => f,
            PinCheck::Pinned => {
                r.skipped_pinned.push(g);
                r.pinned_bytes += self.gen_bytes(stem, g);
                return false;
            }
            PinCheck::Error(e) => {
                r.errors.push(format!("{e}; generation preserved"));
                return false;
            }
        };
        self.at("col.before_delete", stem);
        let (removed, failed) = self.remove_gen_data(stem, g);
        drop(excl);
        r.deleted_bytes += removed;
        if !failed.is_empty() {
            r.partial.push(g);
            r.errors.push(format!("g{g:032x}: payload removal failed: {}; pin kept", failed.join("; ")));
            return false;
        }
        let p = self.gpath(stem, g, "pin");
        match self.fail("pin.remove", &p).and_then(|_| fs::remove_file(&p)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                r.errors.push(format!("g{g:032x}: payloads removed, pin file left: {e}; cleanup pending"));
                r.pin_pending.push(g);
                return false;
            }
        }
        r.deleted.push(g);
        true
    }

    /// Whole-stem eviction: every unpinned generation, and the head when its
    /// generation is unpinned. Coordination files stay (D5).
    pub fn evict_stem(&self, stem: &str) -> CollectReport {
        if let Some(r) = self.unsupported() {
            return r;
        }
        let mut r = CollectReport::default();
        let early = match self.head_state(stem) {
            HeadState::Unreadable(e) => return ambiguous(r, "selection", e),
            h => valid(&h),
        };
        let gens = self.generations(stem);
        self.at("evict.selected", stem);
        let Ok(_s) = self.stem_lock(stem) else {
            r.errors.push(format!("{stem}: coordination lock unavailable; nothing deleted"));
            return r;
        };
        let head = match self.head_state(stem) {
            HeadState::Unreadable(e) => return ambiguous(r, "revalidation", e),
            h => valid(&h),
        };
        let _ = early;
        let mut head_removed = false;
        let mut head_pinned = false;
        if let Some(h) = head {
            match self.check_pin(stem, h) {
                PinCheck::Free(f) => {
                    let hp = self.path(stem, "head");
                    match self.fail("head.remove", &hp).and_then(|_| fs::remove_file(&hp)) {
                        Ok(()) => head_removed = true,
                        Err(e) if e.kind() == io::ErrorKind::NotFound => head_removed = true,
                        Err(e) => r.errors.push(format!("head unlink failed: {e}; head and g{h:032x} preserved")),
                    }
                    drop(f);
                }
                PinCheck::Pinned => head_pinned = true,
                PinCheck::Error(e) => r.errors.push(format!("{e}; head and its generation preserved")),
            }
        }
        if head_removed {
            r.heads_removed += 1;
        }
        for g in gens {
            if Some(g) == head && !head_removed {
                if head_pinned {
                    r.skipped_pinned.push(g);
                    r.pinned_bytes += self.gen_bytes(stem, g);
                }
                continue;
            }
            self.try_delete_gen(stem, g, &mut r);
        }
        r
    }

    /// Fit the namespace to `budget` bytes: superseded generations first, then
    /// whole stems, oldest head first. Pinned bytes are reported, never forced.
    /// Returns the report and the bytes still in the namespace.
    pub fn trim(&self, budget: u64) -> (CollectReport, u64) {
        if let Some(r) = self.unsupported() {
            return (r, self.bytes());
        }
        let mut total = CollectReport::default();
        let stems = match self.stems() {
            Ok(s) => s,
            Err(e) => {
                total.errors.push(format!("cannot list the cache folder: {e}"));
                return (total, self.bytes());
            }
        };
        for s in &stems {
            total.merge(self.collect(s));
        }
        let mut by_age: Vec<(std::time::SystemTime, String)> = stems
            .into_iter()
            .filter_map(|s| fs::metadata(self.path(&s, "head")).and_then(|m| m.modified()).ok().map(|t| (t, s)))
            .collect();
        by_age.sort();
        for (_, s) in by_age {
            if self.bytes() <= budget {
                break;
            }
            total.merge(self.evict_stem(&s));
        }
        (total, self.bytes())
    }

    /// Remove `g`'s payload and temporary files (never its pin): the bytes of
    /// confirmed removals, and a description of each failure.
    fn remove_gen_data(&self, stem: &str, g: u128) -> (u64, Vec<String>) {
        let base = format!("{stem}.g{g:032x}.");
        let mut removed = 0u64;
        let mut failed = Vec::new();
        let entries = match fs::read_dir(&self.dir) {
            Ok(it) => it,
            Err(e) => return (0, vec![format!("cannot enumerate: {e}")]),
        };
        for e in entries {
            let e = match e {
                Ok(e) => e,
                Err(e) => {
                    failed.push(format!("enumeration: {e}"));
                    continue;
                }
            };
            let n = e.file_name().to_string_lossy().to_string();
            if !n.starts_with(&base) || n.ends_with(".pin") {
                continue;
            }
            let p = e.path();
            let len = match fs::metadata(&p) {
                Ok(m) => m.len(),
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => {
                    failed.push(format!("{n}: {e}"));
                    continue;
                }
            };
            match self.fail("payload.remove", &p).and_then(|_| fs::remove_file(&p)) {
                Ok(()) => removed += len,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => failed.push(format!("{n}: {e}")),
            }
        }
        (removed, failed)
    }

    #[cfg(test)]
    pub fn gen_path(&self, stem: &str, g: u128, ext: &str) -> PathBuf {
        self.gpath(stem, g, ext)
    }

    #[cfg(test)]
    pub fn head_path(&self, stem: &str) -> PathBuf {
        self.path(stem, "head")
    }
}

enum Refusal {
    Cancelled,
    Changed,
    Check(String),
}

fn valid(h: &HeadState) -> Option<u128> {
    match h {
        HeadState::Valid(g) => Some(*g),
        _ => None,
    }
}

fn ambiguous(mut r: CollectReport, at: &str, e: String) -> CollectReport {
    r.errors.push(format!("head unreadable at {at}: {e}; no deletion for this stem"));
    r
}

/// Test-only: make `Store::open` treat a directory as outside the validated
/// scope, whatever it really is, to exercise D6 on a directory where the lock
/// probe itself passes.
#[cfg(test)]
pub(crate) mod test_scope {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    static DIRS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    pub fn force_unvalidated(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        DIRS.lock().unwrap().push(std::fs::canonicalize(dir).unwrap());
    }
    pub(super) fn forced_unvalidated(dir: &Path) -> bool {
        DIRS.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|d| d == dir)
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
