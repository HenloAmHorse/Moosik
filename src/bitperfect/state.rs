//! The authoritative output-session state.
//!
//! # Why one owner
//!
//! Three different questions used to share one `bool`, and then several call
//! sites each built their own answer from whichever field was nearest. "Has
//! the user asked for bit-perfect output", "what is open right now", and "is
//! what reached the driver the source's own data" are independent facts, and
//! every place that guessed one from another produced a claim the audio could
//! not support.
//!
//! Everything visible now derives from [`OutputSessionState`] through one
//! [`Presentation`]. The toolbar badge, the transient status line, the
//! persistent panel, the tooltip, the gapless rollover message and the error
//! message all read the same object, so they cannot disagree.
//!
//! # Generations
//!
//! A decode thread, a device scan, a backend callback and an async seek all
//! outlive the session that started them. Each carries the `generation` it was
//! created for; a result arriving from a superseded generation may be logged
//! but may never mutate the current session. Without that, a slow scan
//! finishing after the user changed tracks would republish the previous
//! track's verdict onto the new one.
//!
//! # The boundary of every claim in this module
//!
//! [`Fidelity::PayloadExact`] means the sample or payload words Moosik handed
//! the driver are the source's own words. It says nothing about the USB link,
//! the driver's internals, or the DAC. A DAC's rate display, its DSD lock
//! indicator and its chassis temperature are sanity observations, not
//! checksums, and nothing here should be read as claiming otherwise.

// This module is the shared vocabulary the 1.4.3 output contract is written in,
// so some of it is declared before the code that constructs it exists. The
// lifecycle work uses most of it today; the Float32 policy work constructs the
// remaining `TransformDescription`/`Fidelity::ValueExact` variants, `AsioPcm`
// arrives with the ASIO PCM renderer, and `CpalDirect`/`AlsaNativeDsd` are only
// constructed on non-Windows targets. Everything here is exercised by this
// module's tests; the allow suppresses "never constructed on this target",
// not "never used at all".
#![allow(dead_code)]

use std::fmt;

use super::format::{ChannelLayout, PcmKind, SourceFormat};

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// A device identified by something stable, plus whatever it likes to be
/// called.
///
/// The friendly name is not an identity: two DACs of the same model share one,
/// and Windows renames endpoints. Reuse decisions compare `id`; only the UI
/// reads `display`.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Default)]
pub struct EndpointIdentity {
    pub id: String,
    pub display: String,
}

impl EndpointIdentity {
    pub fn new(id: impl Into<String>, display: impl Into<String>) -> Self {
        EndpointIdentity {
            id: id.into(),
            display: display.into(),
        }
    }

    /// For backends that expose no stable identifier: the name is all there
    /// is, and reuse is correspondingly weaker.
    pub fn from_name(name: impl Into<String>) -> Self {
        let n = name.into();
        EndpointIdentity {
            id: n.clone(),
            display: n,
        }
    }
}

impl fmt::Display for EndpointIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display)
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

/// What was done to the samples between the decoder and the driver.
///
/// Carried as data rather than prose so the fidelity state, the panel text and
/// the reuse key all describe the same thing.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TransformDescription {
    /// The decoded representation reached the driver unchanged.
    Identity,
    /// IEEE Float32 rewritten as signed Q1.31, every sample proven exactly
    /// representable. The numbers survive; the representation does not.
    FloatToQ31ValueExact,
    /// Float32 to Q1.31 with rounding and clamping. Lossy by construction.
    FloatToQ31Processed,
    /// Float64 narrowed to Float32 and then written as Q1.31. Lossy twice.
    ///
    /// It was described as `FloatToQ31Processed` — "Float32 → Q1.31" — which
    /// names the second half of what happened and silently omits the first.
    /// The narrowing to `f32` occurs in the decoder, before anything on this
    /// route can see the samples, so every check downstream is a check on
    /// numbers that have already been changed.
    Float64ToQ31Processed,
    /// A wider integer source rounded into a narrower container.
    NarrowingInteger { to_valid_bits: u16, dither: Dither },
    /// A DSD bitstream decimated to PCM.
    DsdDecimated { pcm_rate: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dither {
    None,
    Tpdf,
}

impl TransformDescription {
    pub fn is_identity(&self) -> bool {
        matches!(self, TransformDescription::Identity)
    }

    pub fn describe(&self) -> String {
        match self {
            TransformDescription::Identity => "Identity".into(),
            TransformDescription::FloatToQ31ValueExact => {
                "Float32 → Q1.31, value-exact proof passed".into()
            }
            TransformDescription::FloatToQ31Processed => {
                "Float32 → Q1.31, rounded ties-to-even and clamped".into()
            }
            TransformDescription::Float64ToQ31Processed => {
                "Float64 → Float32 → Q1.31, narrowed then rounded ties-to-even and clamped"
                    .into()
            }
            TransformDescription::NarrowingInteger {
                to_valid_bits,
                dither,
            } => match dither {
                Dither::None => format!("integer → {to_valid_bits}-bit, truncating round"),
                Dither::Tpdf => format!("integer → {to_valid_bits}-bit, TPDF dither"),
            },
            TransformDescription::DsdDecimated { pcm_rate } => {
                format!("DSD → {} PCM", super::fmt_khz(*pcm_rate))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Typed failures
// ---------------------------------------------------------------------------

/// Why an output route could not be established.
///
/// Typed end to end rather than flattened into a `String`, because exactly one
/// question is asked of it — "may this failure advance the fallback ladder?" —
/// and a string cannot answer it. Before 1.4.3 every PCM failure flattened, so
/// an unreadable file, a typo in an environment override and a DAC that
/// genuinely rejects a rate all took the same fallback, and the first two hid
/// the real problem behind something that looked like it worked.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FailureReason {
    SourceOpen(String),
    SourceParse(String),
    SourceRead(String),
    Decode(String),
    Seek(String),
    Configuration(String),
    UnsupportedExactRepresentation(String),
    DeviceUnavailable(String),
    DeviceFormatUnsupported(String),
    BackendReset(String),
    BackendWrite(String),
    BackendDead(String),
    SessionThreadStart(String),
    IntegrityFault(String),
}

impl FailureReason {
    /// Whether the *endpoint refused this format*, and a different format or
    /// route could therefore plausibly succeed. This is the only question that
    /// may advance the fallback ladder.
    ///
    /// Deliberately narrow, and narrower than it was: `DeviceUnavailable` is
    /// no longer included. A device that is gone, a COM error, a thread that
    /// would not start and a malformed override are not format rejections, and
    /// treating any of them as one converts a diagnosable fault into a silent
    /// downgrade. Only `DeviceFormatUnsupported` means "this endpoint works,
    /// it just will not take these bits".
    pub fn is_device_limitation(&self) -> bool {
        matches!(self, FailureReason::DeviceFormatUnsupported(_))
    }

    /// Whether the failure came from inside our own backend rather than from
    /// the device's answer to a format request. Never advances a ladder: the
    /// next rung runs the same code and fails the same way.
    pub fn is_backend_internal(&self) -> bool {
        matches!(
            self,
            FailureReason::BackendReset(_)
                | FailureReason::BackendWrite(_)
                | FailureReason::BackendDead(_)
                | FailureReason::SessionThreadStart(_)
                | FailureReason::IntegrityFault(_)
                | FailureReason::DeviceUnavailable(_)
        )
    }

    /// Whether the source itself is the problem, so no other route can help.
    pub fn is_source_problem(&self) -> bool {
        matches!(
            self,
            FailureReason::SourceOpen(_)
                | FailureReason::SourceParse(_)
                | FailureReason::SourceRead(_)
                | FailureReason::Decode(_)
                | FailureReason::Seek(_)
        )
    }

    pub fn message(&self) -> &str {
        match self {
            FailureReason::SourceOpen(m)
            | FailureReason::SourceParse(m)
            | FailureReason::SourceRead(m)
            | FailureReason::Decode(m)
            | FailureReason::Seek(m)
            | FailureReason::Configuration(m)
            | FailureReason::UnsupportedExactRepresentation(m)
            | FailureReason::DeviceUnavailable(m)
            | FailureReason::DeviceFormatUnsupported(m)
            | FailureReason::BackendReset(m)
            | FailureReason::BackendWrite(m)
            | FailureReason::BackendDead(m)
            | FailureReason::SessionThreadStart(m)
            | FailureReason::IntegrityFault(m) => m,
        }
    }

    /// A short kind label, for logs and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            FailureReason::SourceOpen(_) => "source open",
            FailureReason::SourceParse(_) => "source parse",
            FailureReason::SourceRead(_) => "source read",
            FailureReason::Decode(_) => "decode",
            FailureReason::Seek(_) => "seek",
            FailureReason::Configuration(_) => "configuration",
            FailureReason::UnsupportedExactRepresentation(_) => "unsupported exact representation",
            FailureReason::DeviceUnavailable(_) => "device unavailable",
            FailureReason::DeviceFormatUnsupported(_) => "device format unsupported",
            FailureReason::BackendReset(_) => "backend reset",
            FailureReason::BackendWrite(_) => "backend write",
            FailureReason::BackendDead(_) => "backend dead",
            FailureReason::SessionThreadStart(_) => "session thread start",
            FailureReason::IntegrityFault(_) => "integrity fault",
        }
    }
}

impl fmt::Display for FailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

/// Why a session that had made a claim stopped deserving it.
///
/// Distinct from [`FailureReason`]: a failure means playback never started on
/// this route, a fault means it started and then something went wrong. The
/// distinction matters because a fault latches — the DAC already received
/// something that was not in the file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultReason {
    Underrun,
    TornFrame,
    DecodeError,
    SourceRead,
    SourceFormatChange,
    BackendWrite,
    BackendSpace,
    BackendReset,
    BackendDead,
    CallbackLockMiss,
    PrimingTimeout,
    /// A sample that a value-exactness proof said could not occur, did.
    ValueExactViolation,
    SessionThreadFailed,
    /// The driver reported that it missed its own callback deadline.
    BackendOverload,
    /// A shared-mixer track's audio ran out well short of its declared length.
    /// An inference from how much played, not a report from the decoder.
    SharedEndedEarly,
}

impl FaultReason {
    pub fn describe(&self) -> &'static str {
        match self {
            FaultReason::Underrun => {
                "a dropout — the device played silence that was not in the file"
            }
            FaultReason::TornFrame => {
                "a torn frame in the output ring — the channels can no longer be told apart"
            }
            FaultReason::DecodeError => "a decode error mid-track",
            FaultReason::SourceRead => "a read error mid-track",
            FaultReason::SourceFormatChange => "the source changed format mid-stream",
            FaultReason::BackendWrite => "the audio device rejected a write",
            FaultReason::BackendSpace => "the audio device stopped reporting buffer space",
            FaultReason::BackendReset => "the driver reset the stream",
            FaultReason::BackendDead => "the audio backend stopped unexpectedly",
            FaultReason::CallbackLockMiss => {
                "the render callback could not reach the session in time"
            }
            FaultReason::PrimingTimeout => "the output never started producing audio",
            FaultReason::ValueExactViolation => {
                "a sample outside the proven value-exact range reached the output"
            }
            FaultReason::SessionThreadFailed => "the decoder thread stopped unexpectedly",
            FaultReason::BackendOverload => "the driver missed its own callback deadline",
            FaultReason::SharedEndedEarly => {
                "the audio ran out well before the end of the track — the file may be damaged"
            }
        }
    }

    /// Whether this fault means the backend itself must not be reused.
    pub fn kills_backend(&self) -> bool {
        matches!(
            self,
            FaultReason::BackendDead | FaultReason::BackendReset | FaultReason::BackendOverload
        )
    }
}

// ---------------------------------------------------------------------------
// Fidelity
// ---------------------------------------------------------------------------

/// What happened to the source data on its way to the driver.
///
/// Independent of which transport carried it: an exclusive stream can be
/// processed, and a shared stream is never exact. Only
/// [`Fidelity::PayloadExact`] may render the green diamond.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Fidelity {
    /// Nothing is playing.
    Idle,
    /// A route is being established. Deliberately not a claim: the window
    /// between pressing play and the driver accepting a format is exactly when
    /// a claim is least justified.
    Pending,
    /// The representation handed to the driver is the source's own. Integer
    /// samples kept every valid bit; Float32 words are bit-identical; DSD
    /// payload bits are the file's bits. Transport framing — DoP markers,
    /// protocol idle during pause, prefill and bounded drain — is carrier
    /// grammar, not source conversion.
    PayloadExact,
    /// The representation changed, but every decoded sample landed exactly on
    /// the destination lattice. Signed-zero and NaN payload identity are lost,
    /// so this is not bit-perfect and never green.
    ValueExact { transform: TransformDescription },
    /// At least one known alteration: rounding, clipping, dither, gain, EQ,
    /// ReplayGain, resampling, decimation or modulation.
    Processed {
        transform: TransformDescription,
        reason: String,
    },
    /// No known Moosik-side alteration, but the downstream path cannot be
    /// proved transparent. Correct for CPAL/CoreAudio/PipeWire/Pulse/dmix and
    /// native ALSA until real platform validation exists.
    Unverified { reason: String },
    /// The session began with a stronger claim and then broke it. Latched.
    Faulted { reason: FaultReason },
    /// The route could not be established at all.
    Failed { reason: FailureReason },
}

impl Fidelity {
    /// The single question the green diamond answers.
    pub fn is_payload_exact(&self) -> bool {
        matches!(self, Fidelity::PayloadExact)
    }

    pub fn is_faulted(&self) -> bool {
        matches!(self, Fidelity::Faulted { .. })
    }

    /// The user-facing reason, where there is one to give.
    pub fn reason(&self) -> Option<String> {
        match self {
            Fidelity::Idle | Fidelity::Pending | Fidelity::PayloadExact => None,
            Fidelity::ValueExact { transform } => Some(transform.describe()),
            Fidelity::Processed { transform, reason } => {
                if transform.is_identity() {
                    Some(reason.clone())
                } else {
                    Some(format!("{} — {reason}", transform.describe()))
                }
            }
            Fidelity::Unverified { reason } => Some(reason.clone()),
            Fidelity::Faulted { reason } => Some(reason.describe().to_string()),
            Fidelity::Failed { reason } => Some(reason.message().to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/// Which output path is open. Says nothing about fidelity on its own — an
/// exclusive transport can carry processed audio.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Transport {
    Inactive,
    Opening,
    Shared { endpoint: EndpointIdentity },
    WasapiExclusivePcm { endpoint: EndpointIdentity },
    WasapiExclusiveDop { endpoint: EndpointIdentity },
    AsioPcm { driver: EndpointIdentity },
    AsioNativeDsd { driver: EndpointIdentity },
    CpalDirect { endpoint: EndpointIdentity },
    AlsaNativeDsd { endpoint: EndpointIdentity },
    Dead { reason: FailureReason },
}

impl Transport {
    /// Whether this transport is capable of carrying payload-exact audio at
    /// all. A `true` here is a precondition, never a proof — the fidelity
    /// state is decided separately and can still be Processed or Faulted.
    ///
    /// CPAL and native ALSA are excluded: they request an exact format and very
    /// likely get it, but ALSA, PipeWire, Pulse, `dmix` and CoreAudio can each
    /// resample or mix downstream, and this process cannot observe which route
    /// it got. Requesting is not measuring.
    pub fn can_carry_exact(&self) -> bool {
        matches!(
            self,
            Transport::WasapiExclusivePcm { .. }
                | Transport::WasapiExclusiveDop { .. }
                | Transport::AsioPcm { .. }
                | Transport::AsioNativeDsd { .. }
        )
    }

    /// True when this route bypasses the shared `rodio` sink entirely and is
    /// fed by one of our own device streams.
    ///
    /// This is the routing question — which object owns pause, position,
    /// seek, end-of-track and the gapless hand-off — and it is deliberately
    /// separate from `can_carry_exact`, which is a fidelity question. CPAL
    /// direct output is a device stream we feed ourselves even though it can
    /// never *prove* exactness; shared output is not, however green the
    /// user's preference happens to be.
    pub fn is_device_stream(&self) -> bool {
        matches!(
            self,
            Transport::WasapiExclusivePcm { .. }
                | Transport::WasapiExclusiveDop { .. }
                | Transport::AsioPcm { .. }
                | Transport::AsioNativeDsd { .. }
                | Transport::CpalDirect { .. }
                | Transport::AlsaNativeDsd { .. }
        )
    }

    /// True while a route is actually open and able to move audio.
    pub fn is_live(&self) -> bool {
        !matches!(
            self,
            Transport::Inactive | Transport::Opening | Transport::Dead { .. }
        )
    }

    /// Whether this transport has any software gain stage at all. Where it has
    /// none, a volume control is not "ignored" — it does not exist.
    pub fn has_gain_stage(&self) -> bool {
        matches!(
            self,
            Transport::Shared { .. } | Transport::Inactive | Transport::Opening
        )
    }

    pub fn endpoint(&self) -> Option<&EndpointIdentity> {
        match self {
            Transport::Shared { endpoint }
            | Transport::WasapiExclusivePcm { endpoint }
            | Transport::WasapiExclusiveDop { endpoint }
            | Transport::AsioPcm { driver: endpoint }
            | Transport::AsioNativeDsd { driver: endpoint }
            | Transport::CpalDirect { endpoint }
            | Transport::AlsaNativeDsd { endpoint } => Some(endpoint),
            Transport::Inactive | Transport::Opening | Transport::Dead { .. } => None,
        }
    }

    /// The transport name shown to the user.
    ///
    /// Native DSD is never described as DoP: they are different transports with
    /// different ceilings, and the ASIO route exists precisely because it is
    /// not DoP.
    pub fn name(&self) -> &'static str {
        match self {
            Transport::Inactive => "none",
            Transport::Opening => "opening",
            Transport::Shared { .. } => "Shared / processed",
            Transport::WasapiExclusivePcm { .. } => "WASAPI Exclusive",
            Transport::WasapiExclusiveDop { .. } => "WASAPI Exclusive · DoP",
            Transport::AsioPcm { .. } => "ASIO PCM",
            Transport::AsioNativeDsd { .. } => "Native DSD (ASIO)",
            Transport::CpalDirect { .. } => "Direct output (cpal)",
            Transport::AlsaNativeDsd { .. } => "Native DSD (ALSA)",
            Transport::Dead { .. } => "unavailable",
        }
    }
}

// ---------------------------------------------------------------------------
// Negotiated output, carrier, and media source
// ---------------------------------------------------------------------------

/// What the backend actually agreed to, read back from the backend rather than
/// assumed from what was requested.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OutputFormat {
    pub endpoint: EndpointIdentity,
    pub sample_rate: u32,
    pub channels: u16,
    pub layout: ChannelLayout,
    pub container_bits: u16,
    pub valid_bits: u16,
    pub integer: bool,
    /// The device buffer, in frames — how much audio the endpoint still holds
    /// when we stop feeding it. The drain bound is derived from this.
    pub buffer_frames: u32,
    /// e.g. `24i/32 excl`.
    pub label: String,
}

impl OutputFormat {
    /// `192 kHz · stereo · 24 valid in 32 integer`
    pub fn describe(&self) -> String {
        let bits = if self.container_bits == self.valid_bits {
            format!(
                "{}-bit {}",
                self.valid_bits,
                if self.integer { "integer" } else { "float" }
            )
        } else {
            format!(
                "{} valid in {} {}",
                self.valid_bits,
                self.container_bits,
                if self.integer { "integer" } else { "float" }
            )
        };
        format!(
            "{} · {} · {}",
            super::fmt_khz(self.sample_rate),
            super::format::describe_channels(self.channels),
            bits
        )
    }
}

/// A protocol or generated carrier that is not the source: the DoP PCM stream
/// carrying DSD, or a modulated DSD stream carrying PCM.
///
/// Modelled separately from [`SourceFormat`] because it is not the source. The
/// 1.4.2 code passed a synthetic 24-bit `SourceFormat` describing the DoP
/// carrier into the open path, so the panel reported a DSD file as a 24-bit
/// integer source.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CarrierFormat {
    pub kind: CarrierKind,
    /// PCM carrier rate for DoP, DSD bit rate for native DSD.
    pub rate: u32,
    pub channels: u16,
    pub bits: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CarrierKind {
    /// DSD over PCM: 1-bit DSD packed into marked PCM words.
    Dop,
    /// Raw DSD bytes to a native DSD transport.
    NativeDsd,
}

impl CarrierFormat {
    pub fn describe(&self) -> String {
        match self.kind {
            CarrierKind::Dop => format!(
                "DoP · {} · {}-bit integer",
                super::fmt_khz(self.rate),
                self.bits
            ),
            CarrierKind::NativeDsd => {
                format!("Native DSD · {}", crate::dsd::fmt_mhz(self.rate))
            }
        }
    }
}

/// The media actually being played — the file, never the carrier.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MediaSource {
    pub format: SourceFormat,
    /// For DSD, the rate label the container declares (e.g. `DSD64`).
    pub dsd_label: Option<String>,
    /// The container declares more audio than the file holds.
    ///
    /// A source that cannot deliver what it promises cannot be carried
    /// exactly, whatever the transport does with the bytes that are there.
    pub truncated: bool,
}

impl MediaSource {
    pub fn pcm(format: SourceFormat) -> Self {
        MediaSource {
            format,
            dsd_label: None,
            truncated: false,
        }
    }

    pub fn dsd(format: SourceFormat, label: impl Into<String>) -> Self {
        MediaSource {
            format,
            dsd_label: Some(label.into()),
            truncated: false,
        }
    }

    /// Mark a source whose container declares more audio than the file holds.
    pub fn truncated(mut self) -> Self {
        self.truncated = true;
        self
    }

    pub fn describe(&self) -> String {
        match &self.dsd_label {
            Some(l) => format!(
                "{l} · {} · 1-bit",
                super::format::describe_channels(self.format.channels)
            ),
            None => self.format.describe(),
        }
    }
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

/// Everything between the decoder and the driver that could alter a sample.
///
/// Live, not a start-time snapshot: every volume, EQ, ReplayGain, resampler,
/// decimator and conversion change updates this in the same operation that
/// changes the audio.
#[derive(Clone, PartialEq, Debug)]
pub struct ProcessingState {
    /// Software gain actually applied on the active route.
    pub gain: f32,
    /// True when the active route physically has no software gain stage, so a
    /// volume control would have nothing to act on.
    pub gain_locked: bool,
    pub eq_active: bool,
    pub replaygain_active: bool,
    /// Any conversion applied to the samples themselves.
    pub transform: TransformDescription,
}

impl Default for ProcessingState {
    fn default() -> Self {
        ProcessingState {
            gain: 1.0,
            gain_locked: false,
            eq_active: false,
            replaygain_active: false,
            transform: TransformDescription::Identity,
        }
    }
}

impl ProcessingState {
    /// The processing state of a route with no gain stage and no DSP.
    pub fn transparent_locked() -> Self {
        ProcessingState {
            gain: 1.0,
            gain_locked: true,
            ..Default::default()
        }
    }

    /// Whether nothing here alters a sample.
    ///
    /// `gain` is compared bit-exactly on purpose: `0.999999` is not unity, and
    /// multiplying by it changes samples.
    pub fn is_transparent(&self) -> bool {
        self.gain.to_bits() == 1.0f32.to_bits()
            && !self.eq_active
            && !self.replaygain_active
            && self.transform.is_identity()
    }

    pub fn describe(&self) -> String {
        let vol = if self.gain_locked {
            "Volume locked".to_string()
        } else {
            format!("Volume {:.0}%", self.gain * 100.0)
        };
        let mut s = format!(
            "{vol} · EQ {} · ReplayGain {}",
            if self.eq_active { "active" } else { "bypassed" },
            if self.replaygain_active {
                "active"
            } else {
                "bypassed"
            }
        );
        if !self.transform.is_identity() {
            s.push_str(" · ");
            s.push_str(&self.transform.describe());
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// How far the user permits Moosik to go when an exact route is impossible.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum OutputPolicy {
    /// Never transform. Stop with the reason.
    #[serde(rename = "strict")]
    StrictExact,
    /// Exact first, then the highest-fidelity permitted fallback. The default
    /// for existing and general users.
    #[serde(rename = "prefer")]
    #[default]
    PreferExact,
    /// The user deliberately chose a high-quality conversion route.
    #[serde(rename = "hq")]
    HqProcessed,
}

impl OutputPolicy {
    pub fn label(&self) -> &'static str {
        match self {
            OutputPolicy::StrictExact => "Strict",
            OutputPolicy::PreferExact => "Automatic",
            OutputPolicy::HqProcessed => "HQ",
        }
    }

    pub fn describe(&self) -> &'static str {
        match self {
            OutputPolicy::StrictExact => {
                "Never transform the audio. If no exact route exists, stop and say why."
            }
            OutputPolicy::PreferExact => {
                "Use an exact route when one can be proved, otherwise the highest-fidelity \
                 route you allow. Choices are per track and never change your preference."
            }
            OutputPolicy::HqProcessed => {
                "Deliberately use a high-quality conversion route. Always labelled processed."
            }
        }
    }

    /// Whether a representation change onto an exactly-representable lattice is
    /// allowed (Float32 → Q1.31 after a complete proof).
    /// A stable wire code, so the decode thread can read the policy out of an
    /// atomic without a lock on a path that must not take one.
    pub fn code(self) -> u8 {
        match self {
            OutputPolicy::StrictExact => 0,
            OutputPolicy::PreferExact => 1,
            OutputPolicy::HqProcessed => 2,
        }
    }

    /// Inverse of `code`. An unknown value is the conservative default, not a
    /// panic: an atomic read that races a policy change must not take the
    /// process down.
    pub fn from_code(c: u8) -> Self {
        match c {
            0 => OutputPolicy::StrictExact,
            2 => OutputPolicy::HqProcessed,
            _ => OutputPolicy::PreferExact,
        }
    }

    pub fn allows_value_exact(&self) -> bool {
        !matches!(self, OutputPolicy::StrictExact)
    }

    /// Whether a lossy conversion is allowed.
    pub fn allows_processed(&self) -> bool {
        !matches!(self, OutputPolicy::StrictExact)
    }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PlaybackState {
    #[default]
    Stopped,
    Playing,
    Paused,
}

impl PlaybackState {
    pub fn is_active(&self) -> bool {
        matches!(self, PlaybackState::Playing | PlaybackState::Paused)
    }
}

// ---------------------------------------------------------------------------
// The authoritative snapshot
// ---------------------------------------------------------------------------

/// How loudly the UI should speak. Colour is never the only signal — every
/// badge has its own glyph and its own words.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Badge {
    /// Bit-perfect is switched off.
    Off,
    /// Requested, but nothing is playing through an exact route.
    Requested,
    /// Establishing a route.
    Pending,
    /// Payload exact. The only badge that renders the green diamond.
    PayloadExact,
    /// Value-exact: the numbers survived, the representation did not.
    ValueExact,
    /// Something altered the audio.
    Processed,
    /// Nothing known altered the audio, but it cannot be proved.
    Unverified,
    /// A claim was made and then broken.
    Faulted,
    /// The route could not be established.
    Failed,
}

impl Badge {
    /// The one place that decides whether a diamond is drawn.
    pub fn shows_diamond(&self) -> bool {
        matches!(self, Badge::PayloadExact)
    }

    pub fn glyph(&self) -> &'static str {
        match self {
            Badge::PayloadExact => "💎",
            Badge::Off | Badge::Requested | Badge::Pending | Badge::ValueExact => "◇",
            Badge::Processed | Badge::Unverified | Badge::Faulted => "⚠",
            Badge::Failed => "✕",
        }
    }
}

/// One authoritative snapshot of the output session. Mutated only through the
/// reducer methods below.
#[derive(Clone, PartialEq, Debug)]
pub struct OutputSessionState {
    /// The persisted user request. Never evidence, and never changed by a
    /// fallback — only by an explicit user action.
    pub requested: bool,
    pub policy: OutputPolicy,
    pub playback: PlaybackState,
    /// Incremented on every route change. Everything asynchronous carries the
    /// generation it was born in.
    pub generation: u64,
    pub transport: Transport,
    pub fidelity: Fidelity,
    /// A recoverable loss of integrity this session suffered, kept beside the
    /// fidelity rather than inside it.
    ///
    /// `Fidelity` holds one thing at a time and `fault()` is first-wins, so
    /// the two records were competing for the same slot: the code pushed the
    /// recoverable loss first — correctly, since it happened first — and the
    /// fatal reason that ended the session then arrived at a slot that was
    /// already taken and was dropped. A track that dropped out and then had
    /// its write rejected displayed "a dropout" and never said the device had
    /// stopped accepting audio. Three places in the source claimed the panel
    /// could say both.
    ///
    /// It can now, because they are two fields. The badge is the fatal reason,
    /// which is what ended the audio; the loss appears in the detail, where it
    /// is still true.
    pub revoked: Option<FaultReason>,
    /// The media being played — never the carrier.
    pub source: Option<MediaSource>,
    pub carrier: Option<CarrierFormat>,
    pub output: Option<OutputFormat>,
    pub processing: ProcessingState,
    /// False once a backend has died; such a backend may never be reused.
    pub backend_alive: bool,
}

impl Default for OutputSessionState {
    fn default() -> Self {
        OutputSessionState {
            requested: false,
            policy: OutputPolicy::default(),
            playback: PlaybackState::Stopped,
            generation: 0,
            transport: Transport::Inactive,
            fidelity: Fidelity::Idle,
            revoked: None,
            source: None,
            carrier: None,
            output: None,
            processing: ProcessingState::default(),
            backend_alive: true,
        }
    }
}

impl OutputSessionState {
    pub fn new(requested: bool, policy: OutputPolicy) -> Self {
        OutputSessionState {
            requested,
            policy,
            ..Default::default()
        }
    }

    // -- reducer -----------------------------------------------------------

    /// Begin establishing a route. Drops every previous claim immediately and
    /// starts a new generation, so results still in flight from the old one
    /// cannot land on this one. Returns the new generation.
    pub fn begin_open(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.transport = Transport::Opening;
        self.fidelity = Fidelity::Pending;
        self.output = None;
        self.carrier = None;
        self.processing = ProcessingState::default();
        self.backend_alive = true;
        // Nothing has happened to this route yet, so it answers for nothing.
        //
        // The recoverable record and the source both survived an open. While
        // "Verifying output…" was on screen the panel was still showing the
        // previous track's format and the previous session's dropout — a
        // specific claim about a file that was no longer playing, on a route
        // that did not exist yet. Cleared deliberately, not left to whoever
        // opens next: the caller publishes a source when it has one, and if it
        // never does, unknown is the truth.
        self.revoked = None;
        self.source = None;
        self.generation
    }

    /// Record a recoverable loss of integrity against `generation`.
    ///
    /// First-wins within a session, like the fatal record and for the same
    /// reason: the first loss is the one that explains when the route stopped
    /// being what it claimed, and every later one is a consequence.
    pub fn revoke(&mut self, generation: u64, reason: FaultReason) -> bool {
        if !self.accepts(generation) || self.revoked.is_some() {
            return false;
        }
        self.revoked = Some(reason);
        true
    }

    /// Whether a result born in `generation` may still mutate this session.
    pub fn accepts(&self, generation: u64) -> bool {
        generation == self.generation
    }

    /// Publish a live route. Every route-dependent field is replaced together,
    /// so nothing can survive from the previous route.
    ///
    /// `fidelity` is decided by the caller from what it actually proved, never
    /// derived from the transport: an exclusive transport carrying a converted
    /// stream is Processed, and saying so is the point of this release.
    pub fn opened(
        &mut self,
        transport: Transport,
        fidelity: Fidelity,
        output: Option<OutputFormat>,
        carrier: Option<CarrierFormat>,
        processing: ProcessingState,
    ) {
        debug_assert!(
            !fidelity.is_payload_exact() || transport.can_carry_exact(),
            "payload-exact claimed on a transport that cannot carry it"
        );
        debug_assert!(
            !fidelity.is_payload_exact() || processing.is_transparent(),
            "payload-exact claimed with processing in the path"
        );
        self.transport = transport;
        self.fidelity = fidelity;
        self.output = output;
        self.carrier = carrier;
        self.processing = processing;
        self.backend_alive = true;
        // A new route has nothing to answer for yet.
        self.revoked = None;
        if self.playback == PlaybackState::Stopped {
            self.playback = PlaybackState::Playing;
        }
    }

    /// Playback stopped. Every active claim is dropped; the request and the
    /// policy are not.
    pub fn stopped(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.playback = PlaybackState::Stopped;
        self.transport = Transport::Inactive;
        self.fidelity = Fidelity::Idle;
        self.source = None;
        self.carrier = None;
        self.output = None;
        self.processing = ProcessingState::default();
        self.backend_alive = true;
        self.revoked = None;
    }

    /// Pause retains transport and fidelity: nothing was altered, the device
    /// is simply being fed protocol silence.
    pub fn set_playback(&mut self, playback: PlaybackState) {
        self.playback = playback;
    }

    /// The route could not be established.
    pub fn failed(&mut self, reason: FailureReason) {
        self.transport = Transport::Dead {
            reason: reason.clone(),
        };
        self.fidelity = Fidelity::Failed { reason };
        self.output = None;
        self.carrier = None;
        self.playback = PlaybackState::Stopped;
        self.processing = ProcessingState::default();
        // A route that never opened has no evidence and no source of its own,
        // and it must not borrow the last one's. `halted` is the other case
        // and deliberately keeps everything: there, a session really was
        // playing and its reason is all the listener has to go on.
        self.revoked = None;
        self.source = None;
    }

    /// A realtime integrity fault, scoped to the generation that produced it.
    ///
    /// Latching is the point: a dropout means the DAC played silence that was
    /// not in the file, and no later good buffer undoes that. Only a genuinely
    /// new, successfully installed session may claim again. Returns whether the
    /// fault was accepted, so a stale generation can be logged as ignored.
    pub fn fault(&mut self, generation: u64, reason: FaultReason) -> bool {
        if !self.accepts(generation) || self.fidelity.is_faulted() {
            return false;
        }
        if reason.kills_backend() {
            self.backend_alive = false;
        }
        self.fidelity = Fidelity::Faulted { reason };
        true
    }

    /// Playback ended because the session failed.
    ///
    /// Deliberately not `stopped()`. Stopping clears the claim, the source and
    /// the transport, which is right when the user stops but wrong here: the
    /// reason is the only thing the user has to go on, and wiping it leaves an
    /// idle player with no account of why it is idle. The fault stays,
    /// everything describing the route stays, and only the playback state
    /// changes.
    pub fn halted(&mut self, reason: FaultReason) {
        self.fault(self.generation, reason);
        self.playback = PlaybackState::Stopped;
    }

    /// Mark the backend unusable — used when a backend dies outright.
    pub fn backend_died(&mut self, reason: FailureReason) {
        self.backend_alive = false;
        self.transport = Transport::Dead { reason };
    }

    /// Replace the current media source, keeping the open transport — the
    /// gapless boundary case.
    pub fn set_source(&mut self, source: MediaSource) {
        self.source = Some(source);
    }

    /// Say that the source is not known.
    ///
    /// A real answer, and a different one from leaving the previous track's
    /// description standing — which is a specific claim about the wrong file.
    pub fn clear_source(&mut self) {
        self.source = None;
    }

    /// Cross a gapless boundary: same transport, same device, new track.
    ///
    /// One operation, because the fields describe one thing. Rolling the title
    /// over and leaving fidelity and processing behind is how a 24-bit track
    /// following a 16-bit one, or a processed rung following an exact one,
    /// inherited a badge it had not earned.
    ///
    /// The generation advances, so a scan or a fault still in flight from the
    /// previous track cannot land on this one. Returns the new generation.
    pub fn roll_over(&mut self, fidelity: Fidelity, processing: ProcessingState) -> u64 {
        self.generation += 1;
        // The incoming track answers for nothing the outgoing one did.
        //
        // This is only safe because of the order the caller keeps: every
        // boundary the device has crossed is folded *before* any realtime
        // evidence is polled, so a dropout that happened in the outgoing track
        // has already been applied to it, and one that happens in the incoming
        // track arrives after this line. Clearing here without that ordering
        // would erase a real dropout — the callback that spans the hand-off
        // publishes the successor's loss before the UI has rolled over, and
        // the clear would land on top of it.
        self.revoked = None;
        self.fidelity = fidelity;
        self.processing = processing;
        self.playback = PlaybackState::Playing;
        self.generation
    }

    /// Update the live processing description in the same operation that
    /// changes the audio.
    pub fn set_processing(&mut self, processing: ProcessingState) {
        self.processing = processing;
    }

    // -- derived -----------------------------------------------------------

    /// The only question the green diamond answers.
    ///
    /// Four conditions, all required: something is actually playing, on a
    /// transport that can carry exact audio, whose fidelity is payload-exact,
    /// with nothing in the processing path that could alter a sample.
    pub fn shows_diamond(&self) -> bool {
        // A recoverable loss is still a loss. The DAC played something that
        // was not in the file; no later good buffer undoes it, and the claim
        // is gone for the rest of the session even though the music is not.
        self.revoked.is_none()
            && self.playback.is_active()
            && self.transport.can_carry_exact()
            && self.fidelity.is_payload_exact()
            && self.processing.is_transparent()
    }

    pub fn badge(&self) -> Badge {
        if self.shows_diamond() {
            return Badge::PayloadExact;
        }
        // A recoverable loss with no terminal reason behind it: the route
        // stopped being what it claimed and the track is still playing. Amber,
        // and never a claim of exactness — a session that dropped out cannot
        // go on describing itself as payload- or value-exact.
        if self.revoked.is_some()
            && matches!(
                self.fidelity,
                Fidelity::PayloadExact | Fidelity::ValueExact { .. } | Fidelity::Pending
            )
        {
            return Badge::Faulted;
        }
        match &self.fidelity {
            Fidelity::Failed { .. } => Badge::Failed,
            Fidelity::Faulted { .. } => Badge::Faulted,
            Fidelity::Pending => Badge::Pending,
            Fidelity::ValueExact { .. } => Badge::ValueExact,
            Fidelity::Processed { .. } => Badge::Processed,
            Fidelity::Unverified { .. } => Badge::Unverified,
            // Payload-exact fidelity that failed one of the other three
            // conditions is still not a diamond.
            Fidelity::PayloadExact => {
                if !self.playback.is_active() {
                    if self.requested {
                        Badge::Requested
                    } else {
                        Badge::Off
                    }
                } else {
                    Badge::Processed
                }
            }
            Fidelity::Idle => {
                if self.requested {
                    Badge::Requested
                } else {
                    Badge::Off
                }
            }
        }
    }

    /// Whether the volume control has anything to act on.
    ///
    /// Locked only while a live route physically has no software gain stage.
    /// Request-only, opening, failed, stopped and idle states are all unlocked
    /// — locking those would disable a working control for no reason the user
    /// could see.
    pub fn volume_locked(&self) -> bool {
        self.playback.is_active() && self.transport.is_live() && !self.transport.has_gain_stage()
    }

    /// Why the volume is locked, for the disabled-control tooltip.
    ///
    /// Says what is true of *this* route. An unverified direct route locks
    /// volume for the same structural reason as an exact one, and must not
    /// borrow the word "bit-perfect" from it.
    pub fn volume_lock_reason(&self) -> Option<String> {
        if !self.volume_locked() {
            return None;
        }
        Some(format!(
            "{} has no software volume stage — applying gain would mean altering the samples. \
             Your saved volume is unchanged and returns with normal playback.",
            self.transport.name()
        ))
    }

    /// One presentation object. Every visible surface reads this and nothing
    /// else, so no two surfaces can disagree about what is happening.
    pub fn presentation(&self) -> Presentation {
        let badge = self.badge();
        let headline = match badge {
            Badge::Off => "◇ Bit-perfect off".to_string(),
            Badge::Requested => "◇ Bit-perfect requested — no active output".to_string(),
            Badge::Pending => "◇ Verifying output…".to_string(),
            Badge::PayloadExact => format!("💎 Payload exact · {}", self.transport.name()),
            Badge::ValueExact => match &self.fidelity {
                Fidelity::ValueExact { transform } => {
                    format!("◇ Value-exact · {}", transform.describe())
                }
                _ => "◇ Value-exact".to_string(),
            },
            Badge::Processed => format!(
                "⚠ Processed · {}",
                self.fidelity.reason().unwrap_or_else(|| "unknown".into())
            ),
            Badge::Unverified => format!(
                "⚠ Exactness unverified · {}",
                self.fidelity.reason().unwrap_or_else(|| "unknown".into())
            ),
            // The reason the session *ended*, when there is one. A
            // recoverable loss with nothing behind it speaks for itself, and
            // it is the whole of what is wrong.
            Badge::Faulted => {
                let why = match &self.fidelity {
                    Fidelity::Faulted { reason } => reason.describe().to_string(),
                    _ => self
                        .revoked
                        .map(|r| r.describe().to_string())
                        .unwrap_or_else(|| "unknown".into()),
                };
                format!("⚠ Integrity fault · {why}")
            }
            Badge::Failed => format!(
                "✕ Output unavailable · {}",
                self.fidelity.reason().unwrap_or_else(|| "unknown".into())
            ),
        };

        let mut detail = Vec::new();
        if let Some(s) = &self.source {
            detail.push(format!("Source: {}", s.describe()));
        }
        // Said even when the badge is about something else — usually it is,
        // because a fatal reason outranks a recoverable one as the account of
        // why the audio ended without making the earlier loss untrue.
        // Said on every live route, whatever the badge says.
        //
        // This was gated on the fidelity already being faulted or failed,
        // which is the one case where the headline is likely to mention it
        // anyway — so the routes where the loss was the *only* record of it
        // were exactly the routes that hid it. A Processed or Unverified route
        // keeps its primary badge, because a dropout does not change what the
        // route is doing to the audio; what it changes is that the route
        // stopped delivering it, and that has to be somewhere the listener can
        // read.
        //
        // Not repeated when the headline is already this same sentence, which
        // is the recoverable-only case on an otherwise exact route.
        if let Some(r) = &self.revoked {
            let headline_is_the_loss =
                badge == Badge::Faulted && !matches!(self.fidelity, Fidelity::Faulted { .. });
            if !headline_is_the_loss {
                detail.push(format!("Integrity: {}", r.describe()));
            }
        }
        if !self.processing.transform.is_identity() {
            detail.push(format!(
                "Transform: {}",
                self.processing.transform.describe()
            ));
        }
        if let Some(c) = &self.carrier {
            detail.push(format!("Carrier: {}", c.describe()));
        }
        if self.transport.is_live() {
            let ep = self
                .transport
                .endpoint()
                .map(|e| e.display.clone())
                .unwrap_or_else(|| "not reported".into());
            detail.push(format!("Transport: {} · {ep}", self.transport.name()));
        }
        if self.playback.is_active() {
            match &self.output {
                Some(o) => detail.push(format!("Output: {}", o.describe())),
                None => detail.push("Output: the backend did not report a format".into()),
            }
            detail.push(format!("Processing: {}", self.processing.describe()));
        }

        Presentation {
            badge,
            headline,
            detail,
            volume_locked: self.volume_locked(),
            volume_lock_reason: self.volume_lock_reason(),
        }
    }
}

/// Everything the UI needs, derived once.
#[derive(Clone, PartialEq, Debug)]
pub struct Presentation {
    pub badge: Badge,
    pub headline: String,
    pub detail: Vec<String>,
    pub volume_locked: bool,
    pub volume_lock_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Stream reuse
// ---------------------------------------------------------------------------

/// Everything that must match before an open stream may carry another track.
///
/// The rule is not "are these the same" but "does the open stream still prove
/// the claim for the new source". A wider integer container may carry a
/// narrower integer source, because left-aligned canonical packing into a wider
/// field is exact. Nothing else may vary — and in particular a stream is never
/// shared merely because both routes end up writing 32-bit words. Native
/// integer, raw Float32, Float32→Q31 value-exact, Float32→Q31 processed,
/// generated DSD and DoP are all distinct keys.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StreamKey {
    pub endpoint: EndpointIdentity,
    pub transport: Transport,
    pub sample_rate: u32,
    pub channels: u16,
    pub layout: ChannelLayout,
    pub source_kind: PcmKind,
    /// What actually reaches the ring, after any transform.
    ///
    /// Distinct from `source_kind`, and the distinction is load-bearing. A
    /// Float32 source on the Q1.31 route arrives at the device as 32-bit
    /// integer; asking whether the open stream can carry *Float32* then
    /// answers "no, it is an integer stream" — which is true and irrelevant,
    /// because nothing is sending it floats. That answer made a Q1.31 stream
    /// unable to carry a second Float32 track, so every track boundary on an
    /// integer-only DAC forced a full device reopen instead of a gapless
    /// hand-off.
    ///
    /// The source's own kind still has to match, and so does the transform:
    /// those are what keep a converted stream from silently carrying a native
    /// integer track, or one conversion's claim from covering another's.
    pub payload_kind: PcmKind,
    pub transform: TransformDescription,
    pub out_container_bits: u16,
    pub out_valid_bits: u16,
    pub out_integer: bool,
    pub dop: bool,
}

impl StreamKey {
    /// Whether a stream opened for `self` can carry `next` without reopening,
    /// and without narrowing anything.
    pub fn can_carry(&self, next: &StreamKey) -> bool {
        if self.endpoint != next.endpoint
            || std::mem::discriminant(&self.transport) != std::mem::discriminant(&next.transport)
            || self.transport.endpoint() != next.transport.endpoint()
            || self.dop != next.dop
            || self.sample_rate != next.sample_rate
            || self.channels != next.channels
            || self.layout != next.layout
            || self.transform != next.transform
        {
            return false;
        }
        if self.source_kind.family() != next.source_kind.family() {
            return false;
        }
        // The container has to hold what will be *written into it*. On the
        // identity route that is the source; on a conversion it is the
        // conversion's output, and the transform equality checked above has
        // already established that both streams are running the same one.
        match next.payload_kind {
            PcmKind::Integer { valid_bits } => {
                self.out_integer && self.out_valid_bits >= valid_bits as u16
            }
            PcmKind::Float32 => !self.out_integer && self.out_valid_bits == 32,
            // Never exact anywhere, so never eligible for an exact stream.
            PcmKind::Float64 => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These drive the reducer that production drives. Nothing here inspects source
// text or asserts its own setup: each test puts the session through a real
// transition and reads the same derived values the UI reads.

#[cfg(test)]
mod tests_h3 {
    use super::*;

    /// The reason a session ended and the loss it suffered before ending are
    /// two facts, and the panel keeps both.
    ///
    /// `Fidelity` holds one thing at a time and `fault` is first-wins, so the
    /// two were competing for the same slot. The engine pushed the recoverable
    /// loss first, correctly, because it happened first — and the fatal reason
    /// then arrived at a slot that was already taken and was thrown away. A
    /// track that dropped out and then had its write rejected displayed "a
    /// dropout" and never said the device had stopped accepting audio, which
    /// is the part that explains why the music stopped.
    #[test]
    fn a_dropout_before_a_write_failure_does_not_hide_it() {
        let mut st = OutputSessionState::default();
        let g = st.generation;

        // A dropout, then the write that ends the track.
        assert!(st.revoke(g, FaultReason::Underrun));
        assert!(st.fault(g, FaultReason::BackendWrite));

        let p = st.presentation();
        assert!(
            p.headline.contains("write") || p.headline.contains("Write"),
            "the badge is what ended the track: {}",
            p.headline
        );
        assert!(
            p.detail.iter().any(|d| d.starts_with("Integrity:")),
            "and the dropout before it is still on the record: {:?}",
            p.detail
        );

        // First-wins within the session, on each record separately.
        assert!(!st.revoke(g, FaultReason::CallbackLockMiss));
        assert!(!st.fault(g, FaultReason::BackendDead));

        // And a fresh route answers for nothing.
        st.stopped();
        assert!(st.revoked.is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitperfect::format::{PcmKind, SourceFormat};

    fn ep(name: &str) -> EndpointIdentity {
        EndpointIdentity::new(format!("{{0.0.0.00000000}}.{name}"), name)
    }

    fn stereo(kind: PcmKind, rate: u32) -> SourceFormat {
        SourceFormat {
            kind,
            sample_rate: rate,
            channels: 2,
            layout: ChannelLayout(0x3),
        }
    }

    fn out(endpoint: &EndpointIdentity, valid: u16, container: u16, integer: bool) -> OutputFormat {
        OutputFormat {
            endpoint: endpoint.clone(),
            sample_rate: 44_100,
            channels: 2,
            layout: ChannelLayout(0x3),
            container_bits: container,
            valid_bits: valid,
            integer,
            buffer_frames: 1024,
            label: "test".into(),
        }
    }

    /// Open an exact WASAPI PCM route the way `Engine::bp_note_opened` does.
    fn open_exact(s: &mut OutputSessionState, endpoint: &EndpointIdentity) {
        s.begin_open();
        s.set_source(MediaSource::pcm(stereo(
            PcmKind::Integer { valid_bits: 24 },
            44_100,
        )));
        s.opened(
            Transport::WasapiExclusivePcm {
                endpoint: endpoint.clone(),
            },
            Fidelity::PayloadExact,
            Some(out(endpoint, 24, 32, true)),
            None,
            ProcessingState::transparent_locked(),
        );
    }

    // -- lifecycle ---------------------------------------------------------

    /// Loading a persisted `enabled = true` must produce a request and nothing
    /// else: no active output, no diamond, and a volume control that still
    /// works.
    #[test]
    fn a_persisted_request_starts_idle_and_unlocked() {
        let s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        assert!(s.requested);
        assert_eq!(s.playback, PlaybackState::Stopped);
        assert_eq!(s.transport, Transport::Inactive);
        assert_eq!(s.fidelity, Fidelity::Idle);
        assert!(
            !s.volume_locked(),
            "an idle session must not lock the volume"
        );
        assert!(!s.shows_diamond());
        let p = s.presentation();
        assert!(!p.badge.shows_diamond());
        assert!(p.headline.contains("requested"), "{}", p.headline);
        assert!(!p.volume_locked);
    }

    /// Stopping drops every active claim and unlocks the volume, and leaves the
    /// request and the policy exactly where the user put them.
    #[test]
    fn stopping_clears_the_claim_and_keeps_the_preference() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::StrictExact);
        open_exact(&mut s, &dac);
        assert!(s.shows_diamond());
        assert!(s.volume_locked());

        s.stopped();
        assert!(!s.shows_diamond());
        assert!(
            !s.volume_locked(),
            "a stopped session must not lock the volume"
        );
        assert_eq!(s.transport, Transport::Inactive);
        assert_eq!(s.fidelity, Fidelity::Idle);
        assert!(
            s.source.is_none() && s.output.is_none() && s.carrier.is_none(),
            "stop must clear the route-dependent fields"
        );
        assert!(s.requested, "stopping is not a preference change");
        assert_eq!(s.policy, OutputPolicy::StrictExact);
    }

    /// Pause keeps the transport and the claim — nothing was altered, the
    /// device is being fed protocol silence — and does not clear a fault.
    #[test]
    fn pausing_retains_the_claim_and_never_clears_a_fault() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        let g = s.generation;

        s.set_playback(PlaybackState::Paused);
        assert!(s.shows_diamond(), "a paused exact stream is still exact");
        assert!(s.volume_locked());

        assert!(s.fault(g, FaultReason::Underrun));
        s.set_playback(PlaybackState::Playing);
        s.set_playback(PlaybackState::Paused);
        assert!(
            s.fidelity.is_faulted(),
            "pause/resume must not clear a fault"
        );
        assert!(!s.shows_diamond());
    }

    /// Beginning a new open drops the previous claim before the new one is
    /// proved, and starts a generation that stale results cannot reach.
    #[test]
    fn opening_supersedes_the_previous_claim_and_generation() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        let old = s.generation;

        let new = s.begin_open();
        assert_ne!(old, new);
        assert!(!s.shows_diamond());
        assert_eq!(s.fidelity, Fidelity::Pending);
        assert!(
            s.output.is_none(),
            "the old negotiated format must not linger"
        );
        assert!(
            !s.accepts(old),
            "a superseded generation must not be accepted"
        );
        assert!(s.accepts(new));
    }

    /// A fault from a session that has since been replaced is ignored, not
    /// applied to its successor.
    #[test]
    fn a_stale_generation_cannot_fault_the_current_session() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        let stale = s.generation;

        open_exact(&mut s, &dac); // new track, new generation
        assert!(
            !s.fault(stale, FaultReason::Underrun),
            "a fault from the previous session must not land here"
        );
        assert!(s.shows_diamond(), "the current session is untouched");

        // ...and the current generation still can.
        let now = s.generation;
        assert!(s.fault(now, FaultReason::Underrun));
        assert!(!s.shows_diamond());
    }

    /// A fault latches for the session and the first cause is the one kept.
    #[test]
    fn a_fault_latches_and_keeps_its_first_cause() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        let g = s.generation;

        assert!(s.fault(g, FaultReason::Underrun));
        assert!(!s.fault(g, FaultReason::BackendWrite), "already faulted");
        assert_eq!(
            s.fidelity,
            Fidelity::Faulted {
                reason: FaultReason::Underrun
            }
        );

        // Only a genuinely new session may claim again.
        open_exact(&mut s, &dac);
        assert!(s.shows_diamond());
    }

    /// A backend that dies is not reusable, and says so through
    /// `backend_alive` rather than through a fidelity state a new session
    /// would reset.
    #[test]
    fn a_dead_backend_is_recorded_separately_from_the_session_claim() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        let g = s.generation;

        assert!(s.fault(g, FaultReason::BackendReset));
        assert!(
            !s.backend_alive,
            "a reset kills the backend, not just the session"
        );

        s.backend_died(FailureReason::BackendDead("render thread exited".into()));
        assert!(!s.backend_alive);
        assert!(matches!(s.transport, Transport::Dead { .. }));
    }

    /// Publishing a route replaces every route-dependent field together, so
    /// nothing survives from the previous route.
    #[test]
    fn publishing_a_route_leaves_no_stale_field() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);

        // A DoP route, with a carrier and a negotiated output.
        s.begin_open();
        s.set_source(MediaSource::dsd(
            stereo(PcmKind::Integer { valid_bits: 1 }, 2_822_400),
            "DSD64",
        ));
        s.opened(
            Transport::WasapiExclusiveDop {
                endpoint: dac.clone(),
            },
            Fidelity::PayloadExact,
            Some(out(&dac, 24, 32, true)),
            Some(CarrierFormat {
                kind: CarrierKind::Dop,
                rate: 176_400,
                channels: 2,
                bits: 24,
            }),
            ProcessingState::transparent_locked(),
        );
        assert!(s.carrier.is_some());

        // ...replaced by a processed shared route that has neither.
        s.begin_open();
        s.opened(
            Transport::Shared {
                endpoint: EndpointIdentity::from_name("not reported"),
            },
            Fidelity::Processed {
                transform: TransformDescription::DsdDecimated { pcm_rate: 176_400 },
                reason: "the device rejected the DoP carrier".into(),
            },
            None,
            None,
            ProcessingState::default(),
        );
        assert!(
            s.carrier.is_none(),
            "the DoP carrier must not survive the route change"
        );
        assert!(s.output.is_none());
        assert!(!s.shows_diamond());
    }

    // -- the diamond -------------------------------------------------------

    /// The headline invariant: only `PayloadExact` renders the diamond, on any
    /// transport, in any playback state, with any policy.
    #[test]
    fn only_payload_exact_ever_shows_a_diamond() {
        let dac = ep("DAC A");
        let transports = [
            Transport::Inactive,
            Transport::Opening,
            Transport::Shared {
                endpoint: dac.clone(),
            },
            Transport::WasapiExclusivePcm {
                endpoint: dac.clone(),
            },
            Transport::WasapiExclusiveDop {
                endpoint: dac.clone(),
            },
            Transport::AsioPcm {
                driver: dac.clone(),
            },
            Transport::AsioNativeDsd {
                driver: dac.clone(),
            },
            Transport::CpalDirect {
                endpoint: dac.clone(),
            },
            Transport::AlsaNativeDsd {
                endpoint: dac.clone(),
            },
            Transport::Dead {
                reason: FailureReason::BackendDead("gone".into()),
            },
        ];
        let fidelities = [
            Fidelity::Idle,
            Fidelity::Pending,
            Fidelity::PayloadExact,
            Fidelity::ValueExact {
                transform: TransformDescription::FloatToQ31ValueExact,
            },
            Fidelity::Processed {
                transform: TransformDescription::FloatToQ31Processed,
                reason: "rounded".into(),
            },
            Fidelity::Unverified {
                reason: "platform route not observable".into(),
            },
            Fidelity::Faulted {
                reason: FaultReason::Underrun,
            },
            Fidelity::Failed {
                reason: FailureReason::DeviceFormatUnsupported("no 32f".into()),
            },
        ];

        let mut green = 0usize;
        for t in &transports {
            for fd in &fidelities {
                for playback in [
                    PlaybackState::Stopped,
                    PlaybackState::Playing,
                    PlaybackState::Paused,
                ] {
                    for requested in [false, true] {
                        let mut s = OutputSessionState::new(requested, OutputPolicy::PreferExact);
                        s.transport = t.clone();
                        s.fidelity = fd.clone();
                        s.playback = playback;
                        s.processing = ProcessingState::transparent_locked();

                        let shown = s.presentation().badge.shows_diamond();
                        assert_eq!(
                            shown,
                            s.shows_diamond(),
                            "the badge and the predicate must agree"
                        );
                        if shown {
                            green += 1;
                            assert_eq!(
                                *fd,
                                Fidelity::PayloadExact,
                                "{fd:?} on {t:?} rendered a diamond"
                            );
                            assert!(t.can_carry_exact());
                            assert!(playback.is_active());
                        }
                    }
                }
            }
        }
        assert!(green > 0, "the exact combinations must still be reachable");
    }

    /// Processing anywhere in the path removes the diamond even when the
    /// fidelity state says payload-exact — the two must agree or neither is
    /// trustworthy.
    #[test]
    fn processing_in_the_path_removes_the_diamond() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        open_exact(&mut s, &dac);
        assert!(s.shows_diamond());

        for p in [
            ProcessingState {
                gain: 0.8,
                ..ProcessingState::transparent_locked()
            },
            ProcessingState {
                gain: 0.999_999_9,
                ..ProcessingState::transparent_locked()
            },
            ProcessingState {
                eq_active: true,
                ..ProcessingState::transparent_locked()
            },
            ProcessingState {
                replaygain_active: true,
                ..ProcessingState::transparent_locked()
            },
            ProcessingState {
                transform: TransformDescription::FloatToQ31Processed,
                ..ProcessingState::transparent_locked()
            },
        ] {
            let mut t = s.clone();
            t.set_processing(p.clone());
            assert!(!t.shows_diamond(), "{p:?} still rendered a diamond");
            assert!(!t.presentation().badge.shows_diamond());
        }
    }

    /// Native DSD is never described as DoP, and a value-exact route is never
    /// described as bit-perfect.
    #[test]
    fn transports_and_fidelities_are_named_honestly() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        s.begin_open();
        s.opened(
            Transport::AsioNativeDsd {
                driver: dac.clone(),
            },
            Fidelity::PayloadExact,
            None,
            Some(CarrierFormat {
                kind: CarrierKind::NativeDsd,
                rate: 2_822_400,
                channels: 2,
                bits: 1,
            }),
            ProcessingState::transparent_locked(),
        );
        let p = s.presentation();
        let all = format!("{} {}", p.headline, p.detail.join(" "));
        assert!(all.contains("Native DSD"), "{all}");
        assert!(
            !all.contains("DoP"),
            "native DSD must not be called DoP: {all}"
        );

        s.begin_open();
        s.opened(
            Transport::WasapiExclusivePcm {
                endpoint: dac.clone(),
            },
            Fidelity::ValueExact {
                transform: TransformDescription::FloatToQ31ValueExact,
            },
            Some(out(&dac, 32, 32, true)),
            None,
            ProcessingState {
                transform: TransformDescription::FloatToQ31ValueExact,
                ..ProcessingState::transparent_locked()
            },
        );
        let p = s.presentation();
        assert!(!p.badge.shows_diamond(), "value-exact is not payload-exact");
        assert!(p.headline.contains("Value-exact"), "{}", p.headline);
        assert!(!p.headline.contains("💎"));
    }

    /// A DSD source and its DoP carrier are reported separately. 1.4.2 passed
    /// the carrier in as the source, so a DSD64 file described itself as a
    /// 24-bit integer PCM source.
    #[test]
    fn a_dop_session_reports_the_dsd_source_not_the_carrier() {
        let dac = ep("SMSL USB DAC");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        s.begin_open();
        s.set_source(MediaSource::dsd(
            stereo(PcmKind::Integer { valid_bits: 1 }, 2_822_400),
            "DSD64",
        ));
        s.opened(
            Transport::WasapiExclusiveDop {
                endpoint: dac.clone(),
            },
            Fidelity::PayloadExact,
            Some(out(&dac, 24, 32, true)),
            Some(CarrierFormat {
                kind: CarrierKind::Dop,
                rate: 176_400,
                channels: 2,
                bits: 24,
            }),
            ProcessingState::transparent_locked(),
        );
        let d = s.presentation().detail.join("\n");
        assert!(d.contains("Source: DSD64"), "{d}");
        assert!(d.contains("Carrier: DoP"), "{d}");
        assert!(
            !d.contains("Source: 176.4"),
            "the carrier is not the source: {d}"
        );
    }

    /// A source that cannot describe its own speakers, or cannot deliver the
    /// audio it declares, blocks the exact claim regardless of transport.
    ///
    /// Three channels as L/R/LFE and three as L/R/C carry the same samples to
    /// different speakers, and "the bytes were unaltered" says nothing useful
    /// if they reached the wrong outputs.
    #[test]
    fn a_source_that_cannot_be_described_blocks_the_claim() {
        let unknown_5ch = MediaSource::dsd(
            SourceFormat {
                kind: PcmKind::Integer { valid_bits: 1 },
                sample_rate: 2_822_400,
                channels: 5,
                layout: ChannelLayout::UNSPECIFIED,
            },
            "DSD64",
        );
        assert!(!unknown_5ch.format.layout.is_specified());

        // Stereo without a named layout is still describable: there is only one
        // sensible reading of two channels.
        let stereo_unknown = MediaSource::pcm(SourceFormat {
            kind: PcmKind::Integer { valid_bits: 24 },
            sample_rate: 44_100,
            channels: 2,
            layout: ChannelLayout::UNSPECIFIED,
        });
        assert!(!stereo_unknown.format.layout.is_specified());

        // A truncated source is marked, and the mark survives cloning into the
        // session.
        let cut = stereo_unknown.clone().truncated();
        assert!(cut.truncated);
        assert!(!stereo_unknown.truncated);
    }

    // -- volume lock -------------------------------------------------------

    /// The lock follows the transport's gain stage, not the fidelity claim and
    /// not the user's preference.
    #[test]
    fn the_volume_lock_follows_the_gain_stage() {
        let dac = ep("DAC A");
        let cases: [(Transport, bool); 6] = [
            (
                Transport::Shared {
                    endpoint: dac.clone(),
                },
                false,
            ),
            (
                Transport::WasapiExclusivePcm {
                    endpoint: dac.clone(),
                },
                true,
            ),
            (
                Transport::WasapiExclusiveDop {
                    endpoint: dac.clone(),
                },
                true,
            ),
            (
                Transport::AsioNativeDsd {
                    driver: dac.clone(),
                },
                true,
            ),
            // Unverified routes lock for the same structural reason, and the
            // copy must not borrow the word "bit-perfect" from an exact one.
            (
                Transport::CpalDirect {
                    endpoint: dac.clone(),
                },
                true,
            ),
            (
                Transport::AlsaNativeDsd {
                    endpoint: dac.clone(),
                },
                true,
            ),
        ];
        for (t, want) in cases {
            let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
            s.transport = t.clone();
            s.playback = PlaybackState::Playing;
            s.fidelity = Fidelity::Unverified {
                reason: "test".into(),
            };
            assert_eq!(s.volume_locked(), want, "{t:?}");
            if want {
                let why = s
                    .volume_lock_reason()
                    .expect("a locked control needs a reason");
                assert!(why.contains(t.name()), "{why}");
                assert!(
                    !why.contains("bit-perfect"),
                    "an unverified route must not claim bit-perfect: {why}"
                );
            } else {
                assert!(s.volume_lock_reason().is_none());
            }
        }
    }

    /// Request-only, opening and failed states never lock the control.
    #[test]
    fn inactive_states_never_lock_the_volume() {
        let dac = ep("DAC A");
        let mut s = OutputSessionState::new(true, OutputPolicy::PreferExact);
        assert!(!s.volume_locked(), "idle");

        s.begin_open();
        assert!(!s.volume_locked(), "opening");

        s.failed(FailureReason::DeviceUnavailable("unplugged".into()));
        assert!(!s.volume_locked(), "failed");

        open_exact(&mut s, &dac);
        assert!(s.volume_locked(), "playing on an exclusive route");
        s.stopped();
        assert!(!s.volume_locked(), "stopped");
    }

    // -- typed errors ------------------------------------------------------

    /// Only a *format rejection* may advance a fallback ladder. Everything
    /// else is a problem to report, not to route around.
    ///
    /// `DeviceUnavailable` is on the wrong side of this line on purpose. A
    /// device that is unplugged, or held exclusively by another process, is
    /// not answering format questions — so trying more formats on it is not a
    /// fallback, it is the same failure repeated with a less useful message at
    /// the end of it.
    #[test]
    fn only_device_limitations_may_advance_the_ladder() {
        let device = [FailureReason::DeviceFormatUnsupported(
            "rejected 352.8 kHz".into(),
        )];
        let not_device = [
            FailureReason::DeviceUnavailable("in exclusive use".into()),
            FailureReason::SourceOpen("no such file".into()),
            FailureReason::SourceParse("bad header".into()),
            FailureReason::SourceRead("read error".into()),
            FailureReason::Decode("bad packet".into()),
            FailureReason::Seek("unseekable".into()),
            FailureReason::Configuration("MOOSIK_BP_FORMAT=nonsense".into()),
            FailureReason::UnsupportedExactRepresentation("f64 has no exact route".into()),
            FailureReason::BackendReset("driver reset".into()),
            FailureReason::BackendWrite("write failed".into()),
            FailureReason::BackendDead("thread exited".into()),
            FailureReason::SessionThreadStart("spawn failed".into()),
            FailureReason::IntegrityFault("torn frame".into()),
        ];
        for e in &device {
            assert!(e.is_device_limitation(), "{e:?}");
            assert!(!e.is_source_problem());
        }
        for e in &not_device {
            assert!(
                !e.is_device_limitation(),
                "{e:?} must not trigger a device fallback"
            );
            assert!(!e.kind().is_empty());
            assert!(!e.message().is_empty());
        }
        // Source problems are identified as such, so no route is retried on
        // bytes that will fail the same way.
        for e in &not_device[1..6] {
            assert!(e.is_source_problem(), "{e:?}");
        }
        // And a failure from inside our own backend is named as one, so it is
        // never mistaken for the device's answer to anything.
        for e in [
            FailureReason::BackendReset("driver reset".into()),
            FailureReason::BackendWrite("write failed".into()),
            FailureReason::BackendDead("thread exited".into()),
            FailureReason::SessionThreadStart("spawn failed".into()),
            FailureReason::IntegrityFault("torn frame".into()),
            FailureReason::DeviceUnavailable("in exclusive use".into()),
        ] {
            assert!(e.is_backend_internal(), "{e:?}");
            assert!(!e.is_device_limitation(), "{e:?}");
        }
    }

    // -- reuse -------------------------------------------------------------
    /// Two Float32 tracks share one Q1.31 stream.
    ///
    /// The compatibility question is about the *payload*, not the source's own
    /// family. Asked about a Float32 source, an integer stream used to answer
    /// "no, I am an integer stream" — true, irrelevant, and fatal: it meant a
    /// Q1.31 route could never carry a second Float32 track, so every track
    /// boundary on an integer-only DAC forced a full device close and reopen
    /// instead of a gapless hand-off.
    #[test]
    fn one_q31_stream_carries_a_whole_float_playlist() {
        let e = ep("DAC");
        let q31 = |rate: u32| StreamKey {
            endpoint: e.clone(),
            transport: Transport::WasapiExclusivePcm {
                endpoint: e.clone(),
            },
            sample_rate: rate,
            channels: 2,
            layout: ChannelLayout(0x3),
            source_kind: PcmKind::Float32,
            payload_kind: PcmKind::Integer { valid_bits: 32 },
            transform: TransformDescription::FloatToQ31Processed,
            out_container_bits: 32,
            out_valid_bits: 32,
            out_integer: true,
            dop: false,
        };
        assert!(
            q31(44_100).can_carry(&q31(44_100)),
            "a Q1.31 stream must carry the next Float32 track on the same route"
        );

        // A rate change still forces a reopen — that is a real device change.
        assert!(!q31(44_100).can_carry(&q31(48_000)));

        // A native integer track may not ride the converted stream: same
        // container, different claim, and the transform is what says so.
        let mut native = q31(44_100);
        native.source_kind = PcmKind::Integer { valid_bits: 24 };
        native.payload_kind = PcmKind::Integer { valid_bits: 24 };
        native.transform = TransformDescription::Identity;
        assert!(!q31(44_100).can_carry(&native));
        assert!(!native.can_carry(&q31(44_100)));

        // And a raw Float32 stream still may not carry a converted track, nor
        // the reverse: the transform differs, so the badge would.
        let mut raw = q31(44_100);
        raw.payload_kind = PcmKind::Float32;
        raw.transform = TransformDescription::Identity;
        raw.out_integer = false;
        assert!(!raw.can_carry(&q31(44_100)));
        assert!(!q31(44_100).can_carry(&raw));
    }

    fn key(endpoint: &EndpointIdentity, kind: PcmKind, valid: u16, integer: bool) -> StreamKey {
        StreamKey {
            endpoint: endpoint.clone(),
            transport: Transport::WasapiExclusivePcm {
                endpoint: endpoint.clone(),
            },
            sample_rate: 44_100,
            channels: 2,
            layout: ChannelLayout(0x3),
            source_kind: kind,
            // The identity route: what reaches the ring is what came out of
            // the decoder.
            payload_kind: kind,
            transform: TransformDescription::Identity,
            out_container_bits: if valid <= 16 { 16 } else { 32 },
            out_valid_bits: valid,
            out_integer: integer,
            dop: false,
        }
    }

    #[test]
    fn a_narrower_stream_may_not_carry_a_wider_track() {
        let dac = ep("DAC A");
        let open16 = key(&dac, PcmKind::Integer { valid_bits: 16 }, 16, true);
        let next24 = key(&dac, PcmKind::Integer { valid_bits: 24 }, 24, true);
        assert!(!open16.can_carry(&next24));
        assert!(next24.can_carry(&open16), "widening is exact");

        // 24 valid bits in a 32-bit container cannot carry a 32-bit source.
        let open24in32 = key(&dac, PcmKind::Integer { valid_bits: 24 }, 24, true);
        let next32 = key(&dac, PcmKind::Integer { valid_bits: 32 }, 32, true);
        assert_eq!(open24in32.out_container_bits, 32);
        assert!(!open24in32.can_carry(&next32));
    }

    /// A stream is never shared merely because both routes write 32-bit words.
    #[test]
    fn distinct_representations_never_share_a_stream() {
        let dac = ep("DAC A");
        let int32 = key(&dac, PcmKind::Integer { valid_bits: 32 }, 32, true);
        let float = key(&dac, PcmKind::Float32, 32, false);
        assert!(!int32.can_carry(&float));
        assert!(!float.can_carry(&int32));

        // Same family and width, different transform.
        let mut value_exact = float.clone();
        value_exact.transform = TransformDescription::FloatToQ31ValueExact;
        value_exact.out_integer = true;
        let mut processed = value_exact.clone();
        processed.transform = TransformDescription::FloatToQ31Processed;
        assert!(
            !value_exact.can_carry(&processed),
            "a value-exact stream must not carry a processed conversion"
        );

        // f64 is never eligible.
        let f64k = key(&dac, PcmKind::Float64, 32, false);
        assert!(!float.can_carry(&f64k));
    }

    /// A changed default endpoint forces a reopen. This is the case that was
    /// invisible while the target key was copied from the open stream.
    #[test]
    fn a_changed_endpoint_forces_a_reopen() {
        let a = ep("DAC A");
        let b = ep("DAC B");
        let open = key(&a, PcmKind::Integer { valid_bits: 24 }, 24, true);
        let moved = key(&b, PcmKind::Integer { valid_bits: 24 }, 24, true);
        assert!(!open.can_carry(&moved));

        // Same friendly name, different stable id — still a different device.
        let twin = EndpointIdentity::new("{0.0.0.0}.other", "DAC A");
        let same_name = key(&twin, PcmKind::Integer { valid_bits: 24 }, 24, true);
        assert!(
            !open.can_carry(&same_name),
            "identity is the id, not the friendly name"
        );
    }

    #[test]
    fn transports_never_share_a_stream() {
        let dac = ep("DAC A");
        let pcm = key(&dac, PcmKind::Integer { valid_bits: 24 }, 24, true);
        let mut dop = pcm.clone();
        dop.dop = true;
        dop.transport = Transport::WasapiExclusiveDop {
            endpoint: dac.clone(),
        };
        assert!(!pcm.can_carry(&dop));
        assert!(!dop.can_carry(&pcm));

        let mut asio = pcm.clone();
        asio.transport = Transport::AsioPcm {
            driver: dac.clone(),
        };
        assert!(!pcm.can_carry(&asio));
    }

    /// Colour is never the only signal: every badge carries its own glyph, and
    /// only one of them is the diamond.
    #[test]
    fn every_badge_is_distinguishable_without_colour() {
        let all = [
            Badge::Off,
            Badge::Requested,
            Badge::Pending,
            Badge::PayloadExact,
            Badge::ValueExact,
            Badge::Processed,
            Badge::Unverified,
            Badge::Faulted,
            Badge::Failed,
        ];
        let diamonds: Vec<Badge> = all.iter().copied().filter(|b| b.shows_diamond()).collect();
        assert_eq!(
            diamonds,
            vec![Badge::PayloadExact],
            "exactly one badge may render the diamond"
        );
        for b in all {
            assert!(!b.glyph().is_empty(), "{b:?} has no glyph");
            assert_eq!(
                b.glyph() == "\u{1F48E}",
                b.shows_diamond(),
                "{b:?}: the diamond glyph and the diamond predicate must agree"
            );
        }
    }

    // -- policy ------------------------------------------------------------

    /// The default policy is the one existing users get, and Strict is the
    /// only one that refuses every conversion.
    #[test]
    fn the_policy_ladder_permits_what_it_says() {
        assert_eq!(OutputPolicy::default(), OutputPolicy::PreferExact);
        assert!(!OutputPolicy::StrictExact.allows_value_exact());
        assert!(!OutputPolicy::StrictExact.allows_processed());
        for p in [OutputPolicy::PreferExact, OutputPolicy::HqProcessed] {
            assert!(p.allows_value_exact(), "{p:?}");
            assert!(p.allows_processed(), "{p:?}");
        }
        for p in [
            OutputPolicy::StrictExact,
            OutputPolicy::PreferExact,
            OutputPolicy::HqProcessed,
        ] {
            assert!(!p.label().is_empty());
            assert!(!p.describe().is_empty());
        }
    }
}
