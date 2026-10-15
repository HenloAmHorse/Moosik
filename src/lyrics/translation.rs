//! Explicit, bounded Chinese translation and safe provider supplementation.
//!
//! `translate` sends lyric text to MyMemory only when its caller opts in. It
//! uses the documented read-only `get` endpoint, without an account, key,
//! contact email, contribution endpoint or automatic retry. A generated sheet
//! is returned only after every requested line succeeds; the caller owns
//! preview, provenance, application and saving.
//!
//! API specification: <https://mymemory.translated.net/doc/spec.php>.
//! Limits and data terms: <https://mymemory.translated.net/doc/usagelimits.php>
//! and <https://mymemory.translated.net/terms-and-conditions>. These are external
//! service limits, not a promise of availability or confidential processing.

use super::{Hit, LyricLine, LyricSource, Lyrics};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const NAME: &str = "MyMemory machine translation";
pub const TERMS_URL: &str = "https://mymemory.translated.net/terms-and-conditions";
pub const LIMITS_URL: &str = "https://mymemory.translated.net/doc/usagelimits.php";
/// MyMemory's `q` limit is bytes, not Unicode characters.
pub const MAX_REQUEST_BYTES: usize = 500;
/// A conservative per-operation cap, below or equal to the anonymous daily
/// character allowance. Other applications may already have used that quota.
pub const MAX_SESSION_BYTES: usize = 5_000;
pub const MAX_UNIQUE_LINES: usize = 120;
const MAX_ROWS: usize = 1_000;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 4_000;
const MAX_PROVIDER_BYTES: usize = 512 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(12);
const SESSION_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const ENDPOINT: &str = "https://api.mymemory.translated.net/get";
const CANCELLED: &str = "Translation cancelled; no generated lines were applied.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLanguage {
    Japanese,
    Korean,
    English,
}

impl SourceLanguage {
    pub const ALL: [Self; 3] = [Self::Japanese, Self::Korean, Self::English];

    pub fn label(self) -> &'static str {
        match self {
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::English => "English",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::English => "en",
        }
    }
}

/// A local suggestion for the language selector, never API autodetection.
/// Han-only text is ambiguous; ASCII text can only suggest English, so the
/// user must still be able to choose the actual source language.
pub fn detect(lines: &[LyricLine]) -> Option<SourceLanguage> {
    let mut kana = false;
    let mut hangul = false;
    let mut han = false;
    let mut latin = false;
    let mut other_letters = false;
    for c in lines.iter().flat_map(|l| l.text.chars()) {
        kana |= is_kana(c);
        hangul |= is_hangul(c);
        han |= is_han(c);
        latin |= c.is_ascii_alphabetic();
        other_letters |= c.is_alphabetic()
            && !c.is_ascii_alphabetic()
            && !is_kana(c)
            && !is_hangul(c)
            && !is_han(c);
    }
    match (kana, hangul) {
        (true, false) => Some(SourceLanguage::Japanese),
        (false, true) => Some(SourceLanguage::Korean),
        (false, false) if latin && !han && !other_letters => Some(SourceLanguage::English),
        _ => None,
    }
}

fn is_kana(c: char) -> bool {
    // Shared punctuation such as ・ and the prolonged-sound mark alone are
    // not evidence that an otherwise English, Korean or Han-only row is
    // Japanese. Actual kana and kana iteration letters are sufficient.
    matches!(c as u32, 0x3041..=0x3096 | 0x309d..=0x309f
        | 0x30a1..=0x30fa | 0x30fd..=0x30ff | 0x31f0..=0x31ff
        | 0xff66..=0xff6f | 0xff71..=0xff9d)
}

fn is_hangul(c: char) -> bool {
    matches!(c as u32, 0x1100..=0x11ff | 0x3130..=0x318f | 0xa960..=0xa97f
        | 0xac00..=0xd7af | 0xd7b0..=0xd7ff)
}

fn is_han(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff
        | 0x20000..=0x2ebef | 0x30000..=0x323af)
}

/// Translate every nonblank row into Chinese (`zh-CN`). The existing Chinese
/// display conversion supplies Traditional/Simplified views of this stored
/// result. Identical trimmed source rows share one request, while output has
/// the exact input row count and order. Punctuation-only rows need no request.
///
/// This does not inspect or overwrite existing extras. For a fill-missing
/// operation, callers may clear the text of already translated rows in an
/// input clone. Progress is completed unique requests / total unique requests.
/// One declared source language applies to the entire operation, including
/// mixed-language rows; this does not infer or switch languages per row.
/// Cancellation is checked around each bounded request; an in-flight request
/// may take up to `REQUEST_TIMEOUT` to stop. No partial sheet is returned.
pub fn translate(
    lines: &[LyricLine],
    source: SourceLanguage,
    cancel: &AtomicBool,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<Option<String>>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .timeout_connect(Some(Duration::from_secs(5)))
        .https_only(true)
        .max_redirects(0)
        .http_status_as_error(false)
        .user_agent(concat!("moosik/", env!("MOOSIK_VERSION")))
        .build()
        .into();
    translate_with(lines, source, cancel, progress, |text, language| {
        let url = request_url(text, language);
        let mut response = agent.get(&url).call().map_err(fetch_error)?;
        let status = response.status().as_u16();
        if status != 200 {
            return Err(FetchError::Status(status));
        }
        response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES as u64)
            .read_to_string()
            .map_err(fetch_error)
    })
}

fn request_url(text: &str, source: SourceLanguage) -> String {
    format!(
        "{ENDPOINT}?q={}&langpair={}%7Czh-CN&mt=1",
        super::http::esc(text),
        source.code()
    )
}

/// Never propagate HTTP errors verbatim: they can contain a URL whose query
/// contains the user's lyrics. The transport seam uses the same finite errors.
#[derive(Debug)]
enum FetchError {
    Timeout,
    Status(u16),
    TooLarge,
    Network,
}

fn fetch_error(error: ureq::Error) -> FetchError {
    match error {
        ureq::Error::Timeout(_) => FetchError::Timeout,
        ureq::Error::StatusCode(status) => FetchError::Status(status),
        ureq::Error::BodyExceedsLimit(_) => FetchError::TooLarge,
        _ => FetchError::Network,
    }
}

impl FetchError {
    fn message(self) -> String {
        match self {
            Self::Timeout => "MyMemory timed out; no generated lines were applied.".into(),
            Self::Status(403 | 429) => quota_error(),
            Self::Status(status) => {
                format!("MyMemory request failed (HTTP {status}); no generated lines were applied.")
            }
            Self::TooLarge => {
                "MyMemory response exceeded the size limit; no generated lines were applied.".into()
            }
            Self::Network => {
                "Could not reach MyMemory securely; no generated lines were applied.".into()
            }
        }
    }
}

fn quota_error() -> String {
    "MyMemory refused the request or its free quota is exhausted. Try later; no generated lines were applied.".into()
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err(CANCELLED.into())
    } else {
        Ok(())
    }
}

fn translate_with(
    lines: &[LyricLine],
    source: SourceLanguage,
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
    mut fetch: impl FnMut(&str, SourceLanguage) -> Result<String, FetchError>,
) -> Result<Vec<Option<String>>, String> {
    check_cancel(cancel)?;
    if lines.len() > MAX_ROWS {
        return Err(format!(
            "Translation is limited to {MAX_ROWS} lyric rows per operation."
        ));
    }
    let mut unique = Vec::new();
    let mut ids: HashMap<&str, usize> = HashMap::new();
    let mut rows = Vec::with_capacity(lines.len());
    let mut bytes = 0usize;
    for line in lines {
        let text = line.text.trim();
        if !text.chars().any(char::is_alphanumeric) {
            rows.push(None);
            continue;
        }
        if text.chars().any(|c| c.is_control() && !c.is_whitespace()) || text.contains(['\n', '\r'])
        {
            return Err("A lyric row contains a line break or control character; edit it before translating.".into());
        }
        if text.len() > MAX_REQUEST_BYTES {
            return Err(format!(
                "One lyric row exceeds MyMemory's {MAX_REQUEST_BYTES}-byte limit. Split it into shorter rows before translating."
            ));
        }
        let id = match ids.get(text) {
            Some(&id) => id,
            None => {
                bytes += text.len();
                let id = unique.len();
                ids.insert(text, id);
                unique.push(text);
                id
            }
        };
        rows.push(Some(id));
    }
    if unique.len() > MAX_UNIQUE_LINES || bytes > MAX_SESSION_BYTES {
        return Err(format!(
            "Translation is limited to {MAX_UNIQUE_LINES} unique rows and {MAX_SESSION_BYTES} UTF-8 bytes per operation. Repeated rows count once."
        ));
    }
    if unique.is_empty() {
        return Err("There are no lyric words to translate.".into());
    }
    let started = Instant::now();
    let mut translated = Vec::with_capacity(unique.len());
    progress(0, unique.len());
    for (i, text) in unique.iter().enumerate() {
        check_cancel(cancel)?;
        if started.elapsed() >= SESSION_TIMEOUT {
            return Err(
                "Translation exceeded the operation time limit; no generated lines were applied."
                    .into(),
            );
        }
        let body = fetch(text, source).map_err(FetchError::message)?;
        check_cancel(cancel)?;
        if started.elapsed() >= SESSION_TIMEOUT {
            return Err(
                "Translation exceeded the operation time limit; no generated lines were applied."
                    .into(),
            );
        }
        let result = response_translation(&body)?;
        if !is_chinese_translation(&result)
            || result == text.split_whitespace().collect::<Vec<_>>().join(" ")
        {
            return Err("MyMemory did not return a Chinese translation for every requested row; no generated lines were applied.".into());
        }
        translated.push(result);
        progress(i + 1, unique.len());
    }
    check_cancel(cancel)?;
    Ok(rows
        .into_iter()
        .map(|id| id.map(|i| translated[i].clone()))
        .collect())
}

fn is_chinese_translation(text: &str) -> bool {
    let han = text.chars().filter(|&c| is_han(c)).count();
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    // A matching archive entry can be tagged Chinese while containing mostly
    // untranslated Latin words. Require a substantial Chinese component,
    // still allowing names inside a Chinese sentence. This cannot assess
    // semantic quality; the caller must show a preview before application.
    han > 0 && han * 5 >= letters * 2 && !text.chars().any(|c| is_kana(c) || is_hangul(c))
}

fn response_translation(body: &str) -> Result<String, String> {
    let invalid = || {
        "MyMemory returned an invalid translation response; no generated lines were applied."
            .to_string()
    };
    if body.len() > MAX_RESPONSE_BYTES {
        return Err(invalid());
    }
    let value: serde_json::Value = serde_json::from_str(body).map_err(|_| invalid())?;
    let status = value
        .get("responseStatus")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .ok_or_else(invalid)?;
    let details = value
        .get("responseDetails")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let raw = value
        .pointer("/responseData/translatedText")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let warning = format!("{details} {raw}").to_ascii_uppercase();
    if value
        .get("quotaFinished")
        .is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"))
        || matches!(status, 403 | 429)
        || warning.contains("MYMEMORY WARNING")
        || warning.contains("QUOTA")
        || warning.contains("LIMIT EXCEEDED")
        || warning.contains("USED ALL AVAILABLE")
    {
        return Err(quota_error());
    }
    if status != 200 || raw.is_empty() {
        return Err(invalid());
    }
    let decoded = decode_entities(raw);
    validate_translation_line(&decoded).map_err(|error| {
        format!("MyMemory translation cannot be saved as a lyric row: {error} No generated lines were applied.")
    })?;
    if contains_markup(&decoded) {
        return Err(invalid());
    }
    let clean = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.is_empty() {
        return Err(invalid());
    }
    Ok(clean)
}

/// Whether a manual or returned translation can be saved as one ordinary LRC
/// row without being reinterpreted as metadata, extra rows or word timing.
/// Empty rows and non-Chinese manual text are valid. This only scans syntax;
/// it never parses or allocates an expanded lyric sheet.
pub(crate) fn validate_translation_line(text: &str) -> Result<(), String> {
    if text.len() > MAX_OUTPUT_BYTES {
        return Err(format!(
            "A translation row is limited to {MAX_OUTPUT_BYTES} UTF-8 bytes."
        ));
    }
    if text
        .chars()
        .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
    {
        return Err("A translation row cannot contain line breaks or control characters.".into());
    }
    let leading = text.trim_start();
    if let Some(body) = leading
        .strip_prefix('[')
        .and_then(|s| s.split_once(']').map(|(body, _)| body))
        && body.contains(':')
    {
        return Err(
            "A translation row cannot begin with LRC metadata or a timestamp; enter ordinary text."
                .into(),
        );
    }
    if text.split('<').skip(1).any(|rest| {
        rest.split_once('>')
            .is_some_and(|(body, _)| is_time_token(body))
    }) {
        return Err(
            "A translation row cannot contain an LRC word-timing mark; enter ordinary text.".into(),
        );
    }
    Ok(())
}

/// Recognize timestamp syntax without integer arithmetic. Deliberately also
/// recognize values outside the valid minute/second range, so manual text
/// cannot become structural after someone edits one digit later.
fn is_time_token(body: &str) -> bool {
    let Some((minutes, rest)) = body.split_once(':') else {
        return false;
    };
    let digits = |s: &str| {
        let s = s.trim().strip_prefix('+').unwrap_or(s.trim());
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
    };
    let (seconds, fraction) = match rest.split_once(['.', ':']) {
        Some((seconds, fraction)) => (seconds, Some(fraction.trim())),
        None => (rest, None),
    };
    digits(minutes)
        && digits(seconds)
        && fraction.is_none_or(|f| !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()))
}

fn contains_markup(text: &str) -> bool {
    text.split('<').skip(1).any(|rest| {
        rest.starts_with(|c: char| c.is_ascii_alphabetic() || matches!(c, '/' | '!' | '?'))
            && rest.contains('>')
    })
}

/// MyMemory may return HTML entities in JSON strings. Decode once, preserving
/// unknown/malformed entities as literal text; never interpret them as HTML.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let Some(end) = rest.find(';').filter(|&end| end <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "nbsp" => Some(' '),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|s| u32::from_str_radix(s, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|s| s.parse::<u32>().ok()))
                .and_then(char::from_u32),
        };
        if let Some(c) = decoded {
            out.push(c);
            rest = &rest[end + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Align a provider's original sheet to the current original by text before
/// borrowing its Chinese translations. Current originals, timing, offsets,
/// word marks and extras are never changed. The returned vector has exactly
/// the current row count; existing nonblank translations get no replacement.
///
/// Require at least 95% original word-row agreement in both directions, with
/// monotonic alignment. Whitespace/punctuation/case may differ, but letters
/// and pronunciation annotations remain significant. Repeated originals must
/// have identical translations, otherwise their mapping is ambiguous.
pub fn supplement(current: &Lyrics, hit: &Hit) -> Result<Vec<Option<String>>, String> {
    let mismatch = || {
        "The provider's original lyrics do not agree closely enough with this sheet; no translations were attached.".to_string()
    };
    let original = hit
        .best_text()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(mismatch)?;
    let supplied = hit
        .translation
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| "This provider result has no translation.".to_string())?;
    preflight_current_rows(&current.lines)?;
    preflight_provider_sheet(original)?;
    preflight_provider_sheet(supplied)?;
    let candidate = super::parse(original, LyricSource::Online);
    if candidate.lines.len() > MAX_ROWS {
        return Err(mismatch());
    }
    let translations = provider_rows(&candidate, supplied)?;
    let wanted: Vec<_> = current
        .lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.text.trim().is_empty())
        .map(|(i, l)| (i, text_key(&l.text)))
        .filter(|(_, key)| !key.is_empty())
        .collect();
    let offered: Vec<_> = candidate
        .lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.text.trim().is_empty())
        .map(|(i, l)| (i, text_key(&l.text)))
        .filter(|(_, key)| !key.is_empty())
        .collect();
    if wanted.is_empty() || offered.is_empty() {
        return Err(mismatch());
    }
    let mut repeated: HashMap<&str, &Option<String>> = HashMap::new();
    for (i, key) in &offered {
        if let Some(previous) = repeated.insert(key, &translations[*i])
            && previous != &translations[*i]
        {
            return Err("Repeated provider lyric rows have different translations; their alignment is ambiguous.".into());
        }
    }
    let width = offered.len() + 1;
    let mut lcs = vec![0u16; (wanted.len() + 1) * width];
    for i in (0..wanted.len()).rev() {
        for j in (0..offered.len()).rev() {
            lcs[i * width + j] = if wanted[i].1 == offered[j].1 {
                1 + lcs[(i + 1) * width + j + 1]
            } else {
                lcs[(i + 1) * width + j].max(lcs[i * width + j + 1])
            };
        }
    }
    let agreement = lcs[0] as usize;
    if agreement * 100 < wanted.len() * 95 || agreement * 100 < offered.len() * 95 {
        return Err(mismatch());
    }
    let mut additions = vec![None; current.lines.len()];
    let (mut i, mut j) = (0, 0);
    while i < wanted.len() && j < offered.len() {
        if wanted[i].1 == offered[j].1 {
            let current_index = wanted[i].0;
            if current.lines[current_index]
                .translation
                .as_deref()
                .is_none_or(|t| t.trim().is_empty())
            {
                additions[current_index] = translations[offered[j].0].clone();
            }
            i += 1;
            j += 1;
        } else if lcs[(i + 1) * width + j] >= lcs[i * width + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if additions.iter().all(Option::is_none) {
        return Err(
            "The matching provider sheet has no usable missing Chinese translations.".into(),
        );
    }
    Ok(additions)
}

fn text_key(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn provider_size_error() -> String {
    format!(
        "Provider lyrics exceed the safety limit: at most {MAX_ROWS} expanded rows, {MAX_OUTPUT_BYTES} UTF-8 bytes per row and {MAX_PROVIDER_BYTES} expanded text bytes are allowed. No translations were attached."
    )
}

fn preflight_current_rows(lines: &[LyricLine]) -> Result<(), String> {
    if lines.len() > MAX_ROWS || lines.iter().any(|line| line.text.len() > MAX_OUTPUT_BYTES) {
        return Err(provider_size_error());
    }
    let mut bytes = 0usize;
    for line in lines {
        bytes = bytes
            .checked_add(line.text.len())
            .ok_or_else(provider_size_error)?;
        if bytes > MAX_PROVIDER_BYTES {
            return Err(provider_size_error());
        }
    }
    Ok(())
}

/// Bound parser expansion before `parse` builds its timestamp vector, clones
/// each refrain's text/word marks, or sorts and clones the resulting sheet.
/// The unstripped row remainder is an upper bound on its text and timing data,
/// so multiplying it by the timestamp count also bounds enhanced-LRC clones.
fn preflight_provider_sheet(sheet: &str) -> Result<(), String> {
    if sheet.len() > MAX_PROVIDER_BYTES {
        return Err(provider_size_error());
    }
    let (mut rows, mut bytes) = (0usize, 0usize);
    for raw in sheet.lines() {
        let mut rest = raw.trim_start();
        let mut stamps = 0usize;
        let mut meta_only = false;
        while let Some(after) = rest.strip_prefix('[') {
            let Some(end) = after.find(']') else { break };
            let body = &after[..end];
            if bounded_provider_stamp(body)? {
                stamps += 1;
                if stamps > MAX_ROWS {
                    return Err(provider_size_error());
                }
            } else if body.contains(':') {
                if stamps == 0 {
                    meta_only = true;
                }
            } else {
                break;
            }
            rest = &after[end + 1..];
        }
        if rest.len() > MAX_OUTPUT_BYTES {
            return Err(provider_size_error());
        }
        // `strip_word_marks` calls the same timestamp parser inside a row.
        // Guard its numeric arithmetic as well, even on untimed originals.
        for after in rest.split('<').skip(1) {
            if let Some((body, _)) = after.split_once('>') {
                bounded_provider_stamp(body)?;
            }
        }
        let copies = if stamps > 0 {
            stamps
        } else {
            usize::from(!meta_only)
        };
        rows = rows.checked_add(copies).ok_or_else(provider_size_error)?;
        let expanded = rest
            .len()
            .checked_mul(copies)
            .ok_or_else(provider_size_error)?;
        bytes = bytes
            .checked_add(expanded)
            .ok_or_else(provider_size_error)?;
        if rows > MAX_ROWS || bytes > MAX_PROVIDER_BYTES {
            return Err(provider_size_error());
        }
    }
    Ok(())
}

/// The shared parser computes minutes * 60_000 without checked arithmetic.
/// Reject numeric timestamps which could overflow before passing it any data.
fn bounded_provider_stamp(body: &str) -> Result<bool, String> {
    if !is_time_token(body) {
        return Ok(false);
    }
    let (minutes, rest) = body.split_once(':').unwrap_or_else(|| unreachable!());
    let minutes: u64 = minutes.trim().parse().map_err(|_| provider_size_error())?;
    let seconds = rest.split(['.', ':']).next().unwrap_or("");
    let seconds: u64 = seconds.trim().parse().map_err(|_| provider_size_error())?;
    if seconds >= 60 {
        return Ok(false);
    }
    if minutes > (u64::MAX - 59_999) / 60_000 {
        return Err(provider_size_error());
    }
    Ok(true)
}

/// Timestamp pairing is confined to originals/translations delivered together
/// by one provider. Cross-provider alignment in `supplement` uses text only.
fn provider_rows(original: &Lyrics, supplied: &str) -> Result<Vec<Option<String>>, String> {
    let invalid = || {
        "The provider translation cannot be paired unambiguously with its own original sheet."
            .to_string()
    };
    preflight_current_rows(&original.lines)?;
    preflight_provider_sheet(supplied)?;
    let translated = super::parse(supplied, LyricSource::Online);
    if translated.lines.len() > MAX_ROWS || original.is_synced() != translated.is_synced() {
        return Err(invalid());
    }
    let clean = |text: &str| -> Result<Option<String>, String> {
        let decoded = decode_entities(text);
        let text = decoded.trim();
        validate_translation_line(&decoded).map_err(|error| {
            format!("The provider translation cannot be saved as a lyric row: {error}")
        })?;
        Ok((is_chinese_translation(text) && !contains_markup(text)).then(|| text.to_string()))
    };
    if !original.is_synced() {
        if original.lines.len() != translated.lines.len() {
            return Err(invalid());
        }
        return translated.lines.iter().map(|l| clean(&l.text)).collect();
    }
    let mut originals: HashMap<u128, String> = HashMap::new();
    for line in &original.lines {
        if let Some(at) = line.at {
            let key = text_key(&line.text);
            if let Some(previous) = originals.insert(at.as_millis(), key.clone())
                && previous != key
            {
                return Err(invalid());
            }
        }
    }
    let mut by_time: HashMap<u128, Option<String>> = HashMap::new();
    for line in &translated.lines {
        if let Some(at) = line.at {
            let value = clean(&line.text)?;
            if let Some(previous) = by_time.insert(at.as_millis(), value.clone())
                && previous != value
            {
                return Err(invalid());
            }
        }
    }
    Ok(original
        .lines
        .iter()
        .map(|l| {
            l.at.and_then(|at| by_time.get(&at.as_millis()).cloned().flatten())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn lines(text: &[&str]) -> Vec<LyricLine> {
        text.iter()
            .map(|text| LyricLine {
                text: (*text).into(),
                ..Default::default()
            })
            .collect()
    }

    fn response(text: &str) -> String {
        json!({"responseStatus": 200, "quotaFinished": false,
            "responseData": {"translatedText": text}})
        .to_string()
    }

    #[test]
    fn script_suggestion_is_local_and_refuses_ambiguous_han() {
        assert_eq!(
            detect(&lines(&["夜が明ける", "Hello"])),
            Some(SourceLanguage::Japanese)
        );
        assert_eq!(
            detect(&lines(&["안녕", "Hello"])),
            Some(SourceLanguage::Korean)
        );
        assert_eq!(
            detect(&lines(&["Hello world"])),
            Some(SourceLanguage::English)
        );
        assert_eq!(
            detect(&lines(&["Hello・World"])),
            Some(SourceLanguage::English)
        );
        assert_eq!(
            detect(&lines(&["안녕・Hello"])),
            Some(SourceLanguage::Korean)
        );
        assert_eq!(detect(&lines(&["愛・夢"])), None);
        for text in [["愛夢"], ["안녕かな"], ["été"], ["123 !"]] {
            assert_eq!(detect(&lines(&text)), None);
        }
        assert_eq!(
            SourceLanguage::ALL.map(SourceLanguage::code),
            ["ja", "ko", "en"]
        );
    }

    #[test]
    fn repeated_chorus_and_blanks_keep_row_order_without_repeated_requests() {
        let input = lines(&["Sky", "", " Rain ", "Sky", "...", "Rain"]);
        let mut sent = Vec::new();
        let mut progress = Vec::new();
        let got = translate_with(
            &input,
            SourceLanguage::English,
            &AtomicBool::new(false),
            |done, total| progress.push((done, total)),
            |text, language| {
                assert_eq!(language, SourceLanguage::English);
                sent.push(text.to_string());
                Ok(response(if text == "Sky" { "天空" } else { "雨" }))
            },
        )
        .unwrap();
        assert_eq!(sent, ["Sky", "Rain"]);
        assert_eq!(progress, [(0, 2), (1, 2), (2, 2)]);
        assert_eq!(
            got,
            [
                Some("天空".into()),
                None,
                Some("雨".into()),
                Some("天空".into()),
                None,
                Some("雨".into())
            ]
        );
        assert_eq!(input[2].text, " Rain ");
    }

    #[test]
    fn request_encoding_does_not_add_contact_or_private_data_fields() {
        let url = request_url("a&b\" かな", SourceLanguage::Japanese);
        assert!(url.starts_with("https://api.mymemory.translated.net/get?q="));
        assert!(url.contains("a%26b%22+%E3%81%8B%E3%81%AA"));
        assert!(url.ends_with("&langpair=ja%7Czh-CN&mt=1"));
        assert!(!url.contains("&de=") && !url.contains("&key="));
    }

    #[test]
    fn all_bounds_are_preflighted_before_any_request() {
        let too_long = "あ".repeat(167); // 501 UTF-8 bytes, not 501 characters.
        let many = (0..121)
            .map(|i| LyricLine {
                text: format!("line {i}"),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let heavy = (0..11)
            .map(|i| LyricLine {
                text: format!("{i:02}{}", "a".repeat(498)),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        for input in [
            lines(&["valid", &too_long]),
            many,
            heavy,
            lines(&["first\nsecond"]),
        ] {
            let result = translate_with(
                &input,
                SourceLanguage::Japanese,
                &AtomicBool::new(false),
                |_, _| {},
                |_, _| panic!("preflight must finish before transmission"),
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn errors_quota_and_malformed_responses_never_return_a_partial_sheet() {
        let bad = [
            "{}".to_string(),
            "not JSON".to_string(),
            json!({"responseStatus": 429, "responseDetails": "secret words"}).to_string(),
            json!({"responseStatus": 200, "quotaFinished": true,
                "responseData": {"translatedText": "假译文"}})
            .to_string(),
            response("MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY"),
            response("secret source text"),
            response("日本語かな"),
            response("<b>中文</b>"),
        ];
        for bad in bad {
            let mut calls = 0;
            let result = translate_with(
                &lines(&["first", "second"]),
                SourceLanguage::English,
                &AtomicBool::new(false),
                |_, _| {},
                |_, _| {
                    calls += 1;
                    Ok(if calls == 1 {
                        response("第一行")
                    } else {
                        bad.clone()
                    })
                },
            );
            let error = result.unwrap_err();
            assert!(!error.contains("secret") && !error.contains("https://"));
        }
        for error in [
            FetchError::Network,
            FetchError::Timeout,
            FetchError::Status(503),
            FetchError::TooLarge,
        ] {
            assert!(!error.message().contains("https://"));
        }
    }

    #[test]
    fn entities_are_decoded_once_and_output_cannot_create_extra_rows() {
        assert_eq!(
            response_translation(&response("&#x5929;&#31354;&nbsp;&amp; &quot;雨&quot;")).unwrap(),
            "天空 & \"雨\""
        );
        assert_eq!(
            decode_entities("&amp;lt; &unknown; &#xD800;"),
            "&lt; &unknown; &#xD800;"
        );
        assert!(response_translation(&response("中文&#0;")).is_err());
        assert!(response_translation(&response("中文\n另一行")).is_err());
        assert!(response_translation(&"x".repeat(MAX_RESPONSE_BYTES + 1)).is_err());
        assert!(response_translation(&response(&"天".repeat(MAX_OUTPUT_BYTES))).is_err());
    }

    #[test]
    fn cancellation_before_during_and_after_progress_discards_results() {
        let cancel = AtomicBool::new(true);
        assert!(
            translate_with(
                &lines(&["a"]),
                SourceLanguage::English,
                &cancel,
                |_, _| {},
                |_, _| panic!("already cancelled")
            )
            .unwrap_err()
            .contains("cancelled")
        );
        cancel.store(false, Ordering::Relaxed);
        assert!(
            translate_with(
                &lines(&["a", "b"]),
                SourceLanguage::English,
                &cancel,
                |_, _| {},
                |_, _| {
                    cancel.store(true, Ordering::Relaxed);
                    Ok(response("天"))
                }
            )
            .unwrap_err()
            .contains("cancelled")
        );
        cancel.store(false, Ordering::Relaxed);
        assert!(
            translate_with(
                &lines(&["a"]),
                SourceLanguage::English,
                &cancel,
                |done, _| {
                    if done == 1 {
                        cancel.store(true, Ordering::Relaxed);
                    }
                },
                |_, _| Ok(response("天"))
            )
            .unwrap_err()
            .contains("cancelled")
        );
    }

    #[test]
    fn every_unchanged_row_is_rejected_even_names_or_han() {
        for unchanged in ["OpenAI", "Hello World", "爱", "爱  你"] {
            assert!(
                translate_with(
                    &lines(&["Sky", unchanged]),
                    SourceLanguage::English,
                    &AtomicBool::new(false),
                    |_, _| {},
                    |text, _| Ok(response(if text == "Sky" { "天空" } else { text }))
                )
                .is_err()
            );
        }
    }

    #[test]
    fn an_unchanged_repeated_refrain_never_installs_partial_translations() {
        let input = lines(&["Sky", "Hello World", "Sky", "Hello World", "Rain"]);
        let before = input.clone();
        let mut calls = Vec::new();
        let result = translate_with(
            &input,
            SourceLanguage::English,
            &AtomicBool::new(false),
            |_, _| {},
            |text, _| {
                calls.push(text.to_string());
                Ok(response(if text == "Sky" { "天空" } else { text }))
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, ["Sky", "Hello World"]);
        assert_eq!(input, before);
    }

    #[test]
    fn mostly_latin_archive_matches_do_not_pass_as_chinese() {
        assert!(!is_chinese_translation(
            "penian ferein OU潘托斯所有安卓sofou"
        ));
        assert!(!is_chinese_translation("Hello 中文 world"));
        assert!(is_chinese_translation("这是 OpenAI 的名字"));
        assert!(is_chinese_translation("今天天气不错。"));
    }

    fn hit(original: &str, translated: &str) -> Hit {
        Hit {
            synced: Some(original.into()),
            translation: Some(translated.into()),
            ..Default::default()
        }
    }

    #[test]
    fn provider_translation_follows_original_text_even_when_current_times_differ() {
        let mut current = super::super::parse(
            "[00:01]Hello world\n[00:04]The sky\n[00:08]Rain",
            LyricSource::Sidecar,
        );
        current.offset_ms = 170;
        current.lines[0].romaji = Some("confirmed".into());
        current.lines[0].marks.push((1, Duration::from_millis(500)));
        current.lines[1].translation = Some("用户原译文".into());
        let before = current.lines.clone();
        let got = supplement(
            &current,
            &hit(
                "[00:10]hello, world!\n[00:20]the sky\n[00:30]rain",
                "[00:10]你好世界\n[00:20]天空\n[00:30]雨",
            ),
        )
        .unwrap();
        assert_eq!(got, [Some("你好世界".into()), None, Some("雨".into())]);
        assert_eq!(current.lines, before);
        assert_eq!(current.offset_ms, 170);
    }

    #[test]
    fn same_timestamps_and_song_title_do_not_identify_the_original_version() {
        let current = super::super::parse(
            "[00:01]Hello\n[00:02]World\n[00:03]Sky",
            LyricSource::Sidecar,
        );
        for original in [
            "[00:01]Goodbye\n[00:02]World\n[00:03]Sky",
            "[00:01]Sky\n[00:02]World\n[00:03]Hello",
        ] {
            assert!(
                supplement(&current, &hit(original, "[00:01]甲\n[00:02]乙\n[00:03]丙")).is_err()
            );
        }
    }

    #[test]
    fn repeated_originals_require_agreeing_translations() {
        let current = super::super::parse(
            "[00:01]Hello\n[00:02]Sky\n[00:03]Hello",
            LyricSource::Sidecar,
        );
        let original = "[00:10]Hello\n[00:20]Sky\n[00:30]Hello";
        assert!(
            supplement(
                &current,
                &hit(original, "[00:10]你好\n[00:20]天空\n[00:30]再见")
            )
            .is_err()
        );
        assert!(supplement(&current, &hit(original, "[00:10]你好\n[00:20]天空")).is_err());
        let got = supplement(
            &current,
            &hit(original, "[00:10]你好\n[00:20]天空\n[00:30]你好"),
        )
        .unwrap();
        assert_eq!(got[0], got[2]);
    }

    #[test]
    fn near_complete_gate_works_in_both_directions_and_keeps_unmatched_row_empty() {
        let base = (0..20).map(|i| format!("row {i}")).collect::<Vec<_>>();
        let current = super::super::parse(&base.join("\n"), LyricSource::Sidecar);
        let mut candidate = base.clone();
        candidate[9] = "a changed version".into();
        let translated = (0..20)
            .map(|i| format!("第{i}行"))
            .collect::<Vec<_>>()
            .join("\n");
        let got = supplement(&current, &hit(&candidate.join("\n"), &translated)).unwrap();
        assert_eq!(got[9], None);
        assert_eq!(got.iter().flatten().count(), 19);
        candidate.push("added verse".into());
        assert!(
            supplement(
                &current,
                &hit(&candidate.join("\n"), &format!("{translated}\n新词"))
            )
            .is_err()
        );
    }

    #[test]
    fn annotations_and_duplicate_provider_timestamps_are_not_assumed_equivalent() {
        let current = super::super::parse("剣(つるぎ)\n空", LyricSource::Sidecar);
        assert!(supplement(&current, &hit("剣(けん)\n空", "剑\n天空")).is_err());
        let current = super::super::parse("[00:01]Hello\n[00:01]Sky", LyricSource::Sidecar);
        assert!(
            supplement(
                &current,
                &hit("[00:01]Hello\n[00:01]Sky", "[00:01]你好\n[00:01]天空")
            )
            .is_err()
        );
    }

    #[test]
    fn provider_supplement_keeps_blank_and_punctuation_rows_in_place() {
        let current = super::super::parse("Hello\n\n♪\nSky", LyricSource::Sidecar);
        let got = supplement(&current, &hit("Hello\nSky", "你好\n天空")).unwrap();
        assert_eq!(got, [Some("你好".into()), None, None, Some("天空".into())]);
    }

    #[test]
    fn line_validation_allows_manual_languages_but_rejects_lrc_structure() {
        for text in [
            "",
            "普通中文",
            "Hello World",
            "[Chorus]歌词",
            "中文 [00:01]",
            "中文 <3",
        ] {
            validate_translation_line(text).unwrap();
        }
        for text in [
            "[ti:title]中文",
            " [00:01]中文",
            "[offset:100]中文",
            "中文<00:02.500>歌词",
            "中文< +00 : 02 : 5 >歌词",
            "中文\n下一行",
            "中文\r下一行",
            "中文\t下一行",
            "中文\u{2028}下一行",
            "中文\0",
        ] {
            assert!(validate_translation_line(text).is_err(), "{text:?}");
        }
        assert!(validate_translation_line(&"天".repeat(1334)).is_err());
        validate_translation_line(&"a".repeat(MAX_OUTPUT_BYTES)).unwrap();
    }

    #[test]
    fn returned_literal_or_encoded_lrc_structure_is_rejected_before_preview() {
        for text in [
            "[ti:bad]中文",
            "[00:01]中文",
            "中文<00:02>歌词",
            "&#91;ar:bad&#93;中文",
            "中文&lt;00:02.123&gt;歌词",
        ] {
            assert!(response_translation(&response(text)).is_err());
            assert!(
                translate_with(
                    &lines(&["Sky"]),
                    SourceLanguage::English,
                    &AtomicBool::new(false),
                    |_, _| {},
                    |_, _| Ok(response(text))
                )
                .is_err()
            );
        }
        let current = super::super::parse("[00:01]Sky", LyricSource::Sidecar);
        assert!(supplement(&current, &hit("[00:01]Sky", "[00:01]&#91;ti:bad&#93;天空")).is_err());
    }

    #[test]
    fn provider_lrc_expansion_is_bounded_before_parsing_either_sheet() {
        let row_bomb = format!("{}Sky", "[00:01]".repeat(MAX_ROWS + 1));
        let byte_bomb = format!(
            "{}{}",
            "[00:01]".repeat(MAX_ROWS),
            "a".repeat(MAX_OUTPUT_BYTES)
        );
        let long_row = format!("[00:01]{}", "天".repeat(1334));
        let too_many_plain = "a\n".repeat(MAX_ROWS + 1);
        let current = super::super::parse("[00:01]Sky", LyricSource::Sidecar);
        for sheet in [row_bomb, byte_bomb, long_row, too_many_plain] {
            assert!(
                sheet.len() <= MAX_PROVIDER_BYTES,
                "fixture must bypass raw-byte limit"
            );
            assert!(preflight_provider_sheet(&sheet).is_err());
            // These public entry points must reject the small physical input
            // before a parser can allocate its amplified representation.
            assert!(supplement(&current, &hit(&sheet, "[00:01]天空")).is_err());
            assert!(supplement(&current, &hit("[00:01]Sky", &sheet)).is_err());
            assert!(provider_rows(&current, &sheet).is_err());
        }
    }

    #[test]
    fn provider_timestamp_arithmetic_is_bounded_before_parsing() {
        for sheet in [
            "[18446744073709551615:00]Sky",
            "[999999999999999999999999999999:00]Sky",
            "Sky<18446744073709551615:00>",
        ] {
            assert!(preflight_provider_sheet(sheet).is_err());
        }
        // The largest permitted number of timestamp repetitions is still
        // accepted when its expanded content remains within the byte budget.
        preflight_provider_sheet(&format!("{}a", "[00:01]".repeat(MAX_ROWS))).unwrap();
    }

    #[test]
    fn legitimate_repeated_timestamp_sheets_remain_usable() {
        let current = super::super::parse("[00:01]Sky\n[00:02]Sky", LyricSource::Sidecar);
        let got = supplement(
            &current,
            &hit("[ti:source]\n[00:10][00:20]Sky", "[00:10][00:20]天空"),
        )
        .unwrap();
        assert_eq!(got, [Some("天空".into()), Some("天空".into())]);
    }
}
