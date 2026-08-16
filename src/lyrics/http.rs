//! Shared HTTP for the lyric sources.

use std::time::Duration;

/// A lyric lookup is a nicety, not a feature worth hanging a thread on
/// indefinitely — and it runs on a worker, so a stuck socket would leak one.
const TIMEOUT: Duration = Duration::from_secs(12);

const UA: &str = concat!(
    "moosik/", env!("CARGO_PKG_VERSION"),
    " (https://github.com/HenloAmHorse/Moosik-Player)"
);

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .user_agent(UA)
        .build()
        .into()
}

/// GET a URL and parse JSON. `Ok(None)` for a 404, which several of these APIs
/// use to mean "no such track" — a normal answer, not a failure.
pub fn get_json(url: &str, referer: Option<&str>) -> Result<Option<serde_json::Value>, String> {
    let mut req = agent().get(url);
    if let Some(r) = referer { req = req.header("Referer", r); }
    match req.call() {
        Ok(mut resp) => {
            let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
            // An empty body is a miss, not a parse error worth surfacing.
            if body.trim().is_empty() { return Ok(None); }
            serde_json::from_str(&body).map(Some).map_err(|e| e.to_string())
        }
        Err(ureq::Error::StatusCode(404)) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Percent-encode a query-string value.
///
/// Written out rather than pulled in as a dependency: it is a few lines, and
/// the alternative drags a whole URL crate in for one query string.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' =>
                out.push(*b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_survives_japanese_and_separators() {
        assert_eq!(esc("a b"), "a+b");
        assert_eq!(esc("ロキ"), "%E3%83%AD%E3%82%AD");
        assert_eq!(esc("a&b=c"), "a%26b%3Dc");
        assert_eq!(esc("SWIPE×SWIPE"), "SWIPE%C3%97SWIPE");
    }
}
