//! Playing a result while it downloads (1.6.1). yt-dlp writes `<file>.part`
//! and renames it to `<file>` once it is all in; until then, `open(<file>)`
//! reads the part, and waits at the end of what has arrived for more.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RUNNING: u8 = 0;
const DONE: u8 = 1;
const FAILED: u8 = 2;

/// Left unread at the end of a download in progress: the last write may
/// not have landed whole.
const HOLD: u64 = 64 << 10;

/// The longest a reader waits for a download that has stopped growing.
const STALL: Duration = if cfg!(test) { Duration::from_millis(600) } else { Duration::from_secs(20) };

/// The longest `open_until` waits for a download to begin writing (YouTube
/// can take seconds to answer).
const BEGIN: Duration = Duration::from_secs(60);

/// A download in progress.
pub struct Growing {
    part: PathBuf,
    /// The whole file's size, as YouTube gives it.
    total: u64,
    state: AtomicU8,
}

/// Downloads in progress, by the file each will become.
static STREAMS: Mutex<Vec<(PathBuf, Arc<Growing>)>> = Mutex::new(Vec::new());

fn streams() -> std::sync::MutexGuard<'static, Vec<(PathBuf, Arc<Growing>)>> {
    STREAMS.lock().unwrap_or_else(|p| p.into_inner())
}

/// `file` is arriving in `part`, `total` bytes in all.
pub fn begin(file: &Path, part: PathBuf, total: u64) {
    let g = Arc::new(Growing { part, total, state: AtomicU8::new(RUNNING) });
    let mut s = streams();
    s.retain(|(f, _)| f != file);
    s.push((file.to_path_buf(), g));
}

/// `other` reads as `file` does while it arrives (a picture fetched for a
/// song kept elsewhere, moved beside it once it is in).
pub fn alias(file: &Path, other: &Path) {
    let mut s = streams();
    if let Some(g) = s.iter().find(|(f, _)| f == file).map(|(_, g)| g.clone()) {
        s.retain(|(f, _)| f != other);
        s.push((other.to_path_buf(), g));
    }
}

/// The download of `file` is over: readers already open read on to the
/// end, or (`ok` false) stop.
pub fn end(file: &Path, ok: bool) {
    let mut s = streams();
    let Some(g) = s.iter().find(|(f, _)| f == file).map(|(_, g)| g.clone()) else { return };
    g.state.store(if ok { DONE } else { FAILED }, Ordering::Relaxed);
    s.retain(|(_, h)| !Arc::ptr_eq(h, &g));
}

/// `file` is still arriving.
pub fn downloading(file: &Path) -> bool {
    streams().iter().any(|(f, _)| f == file)
}

/// A file to decode: the file, or while it downloads its part, read no
/// further than what has arrived (waiting there for more).
pub struct Source {
    file: File,
    growing: Option<Arc<Growing>>,
    pos: u64,
    len: Option<u64>,
    /// Asked while waiting: true gives up.
    give_up: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}

/// Open `path` to read, or the part it is arriving in.
pub fn open(path: &Path) -> io::Result<Source> {
    let g = streams().iter().find(|(f, _)| f == path).map(|(_, g)| g.clone());
    if let Some(g) = g
        && let Ok(file) = File::open(&g.part)
    {
        return Ok(Source { file, len: Some(g.total), growing: Some(g), pos: 0, give_up: None });
    }
    let file = File::open(path)?;
    Ok(Source { len: file.metadata().ok().map(|m| m.len()), file, growing: None, pos: 0, give_up: None })
}

/// `open`, but a download that has not begun writing is waited for; any
/// wait (here and in reads) gives up once `stop` says so.
pub fn open_until(path: &Path, stop: impl Fn() -> bool + Send + Sync + 'static) -> io::Result<Source> {
    let since = Instant::now();
    while downloading(path) && !part_of(path).is_some_and(|p| p.is_file()) {
        if stop() || since.elapsed() > BEGIN {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "the download has not begun"));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut s = open(path)?;
    s.give_up = Some(Box::new(stop));
    Ok(s)
}

/// Where `path` is arriving, while it is.
fn part_of(path: &Path) -> Option<PathBuf> {
    streams().iter().find(|(f, _)| f == path).map(|(_, g)| g.part.clone())
}

impl Source {
    /// How much can be read from here, once there is something.
    fn ready(&mut self) -> io::Result<u64> {
        let mut seen = (0, Instant::now());
        loop {
            let Some(g) = &self.growing else { return Ok(u64::MAX) };
            match g.state.load(Ordering::Relaxed) {
                DONE => {
                    self.growing = None;
                    continue;
                }
                FAILED => return Err(io::Error::other("the download stopped")),
                _ => {}
            }
            let have = self.file.metadata()?.len();
            let room = have.saturating_sub(HOLD).saturating_sub(self.pos);
            if room > 0 {
                return Ok(room);
            }
            if have != seen.0 {
                seen = (have, Instant::now());
            } else if seen.1.elapsed() > STALL {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "the download stalled"));
            }
            if self.give_up.as_ref().is_some_and(|stop| stop()) {
                return Err(io::Error::other("no longer wanted"));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Read for Source {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let room = self.ready()?;
        let n = (buf.len() as u64).min(room) as usize;
        let n = self.file.read(&mut buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Source {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        // The end is the whole file's, before it is all in too.
        let to = match (to, &self.growing) {
            (SeekFrom::End(d), Some(g)) => SeekFrom::Start(g.total.checked_add_signed(d).ok_or(io::ErrorKind::InvalidInput)?),
            (to, _) => to,
        };
        self.pos = self.file.seek(to)?;
        Ok(self.pos)
    }
}

impl symphonia::core::io::MediaSource for Source {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("moosik-stream-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Writes `data` into `part` a slice at a time, then renames it to
    /// `file` and ends the download, as yt-dlp does.
    fn arrive(file: PathBuf, part: PathBuf, data: Vec<u8>) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut f = std::fs::OpenOptions::new().append(true).open(&part).unwrap();
            for c in data.chunks(50_000) {
                std::thread::sleep(Duration::from_millis(15));
                f.write_all(c).unwrap();
            }
            drop(f);
            std::fs::rename(&part, &file).unwrap();
            end(&file, true);
        })
    }

    #[test]
    fn a_file_still_arriving_reads_whole_and_in_order() {
        let d = dir("whole");
        let (file, part) = (d.join("a.webm"), d.join("a.webm.part"));
        let data: Vec<u8> = (0..1_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&part, &data[..10]).unwrap();
        begin(&file, part.clone(), data.len() as u64);
        assert!(downloading(&file));
        let mut s = open(&file).unwrap();
        assert_eq!(symphonia::core::io::MediaSource::byte_len(&s), Some(1_000_000));
        let writer = arrive(file.clone(), part, data[10..].to_vec());
        let mut got = Vec::new();
        s.read_to_end(&mut got).unwrap();
        writer.join().unwrap();
        assert!(got == data, "read {} bytes", got.len());
        assert!(!downloading(&file));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_end_is_the_whole_file_and_seeking_ahead_waits_for_it() {
        let d = dir("seek");
        let (file, part) = (d.join("b.webm"), d.join("b.webm.part"));
        let data: Vec<u8> = (0..400_000u32).map(|i| (i % 256) as u8).collect();
        std::fs::write(&part, &data[..1000]).unwrap();
        begin(&file, part.clone(), data.len() as u64);
        let mut s = open(&file).unwrap();
        assert_eq!(s.seek(SeekFrom::End(-100)).unwrap(), 399_900);
        let writer = arrive(file.clone(), part, data[1000..].to_vec());
        let mut tail = Vec::new();
        s.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, &data[399_900..]);
        writer.join().unwrap();
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The decoders read YouTube's audio as it arrives: the first packet
    /// long before the end is in, and every packet in the end.
    /// MOOSIK_TEST_OPUS = a WebM/Opus file from YouTube.
    #[test]
    #[ignore]
    fn youtube_audio_decodes_while_it_arrives() {
        use symphonia::core::{formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint};
        let Ok(p) = std::env::var("MOOSIK_TEST_OPUS") else { return };
        let data = std::fs::read(&p).unwrap();
        let count = |src: Box<dyn symphonia::core::io::MediaSource>, first: &mut Option<Duration>| {
            let t = Instant::now();
            let mut hint = Hint::new();
            hint.with_extension("webm");
            let mut f = symphonia::default::get_probe()
                .format(&hint, MediaSourceStream::new(src, Default::default()), &FormatOptions::default(), &MetadataOptions::default())
                .unwrap()
                .format;
            let track = f.default_track().unwrap().clone();
            let mut dec = symphonia::default::get_codecs().make(&track.codec_params, &Default::default()).unwrap();
            let mut frames = 0;
            while let Ok(packet) = f.next_packet() {
                frames += dec.decode(&packet).unwrap().frames();
                first.get_or_insert(t.elapsed());
            }
            frames
        };
        let whole = count(Box::new(File::open(&p).unwrap()), &mut None);
        let d = dir("opus");
        let (file, part) = (d.join("a.webm"), d.join("a.webm.part"));
        std::fs::write(&part, &data[..2000]).unwrap();
        begin(&file, part.clone(), data.len() as u64);
        // About 500 KB a second.
        let (f2, rest) = (file.clone(), data[2000..].to_vec());
        let writer = std::thread::spawn(move || {
            let mut w = std::fs::OpenOptions::new().append(true).open(&part).unwrap();
            for c in rest.chunks(25_000) {
                std::thread::sleep(Duration::from_millis(50));
                w.write_all(c).unwrap();
            }
            drop(w);
            std::fs::rename(&part, &f2).unwrap();
            end(&f2, true);
        });
        let t = Instant::now();
        let mut first = None;
        let arriving = count(Box::new(open(&file).unwrap()), &mut first);
        let all_in = t.elapsed();
        writer.join().unwrap();
        eprintln!("first packet after {first:?}, all {arriving} frames after {all_in:?} ({whole} from the whole file)");
        assert_eq!(arriving, whole);
        assert!(first.unwrap() < all_in / 4, "the first packet came before most of the file");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_download_not_yet_writing_is_waited_for_or_given_up_on() {
        let d = dir("late");
        let (file, part) = (d.join("v.video.webm"), d.join("v.video.webm.part"));
        begin(&file, part.clone(), 100_000);
        // Nothing to open yet; told to stop, the wait ends at once.
        assert_eq!(open_until(&file, || true).err().map(|e| e.kind()), Some(io::ErrorKind::TimedOut));
        let late = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            std::fs::write(&part, vec![9u8; 100_000]).unwrap();
        });
        let mut s = open_until(&file, || false).unwrap();
        late.join().unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(s.read(&mut buf).unwrap(), 10);
        assert_eq!(buf, [9u8; 10]);
        // Given up on: a wait at the end stops.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = stop.clone();
        let mut s = open_until(&file, move || s2.load(Ordering::Relaxed)).unwrap();
        let mut all = vec![0u8; 100_000];
        s.read(&mut all).unwrap();
        stop.store(true, Ordering::Relaxed);
        let t = Instant::now();
        assert!(s.read(&mut all).is_err());
        assert!(t.elapsed() < STALL);
        end(&file, false);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_failed_or_stalled_download_ends_the_read() {
        let d = dir("fail");
        let (file, part) = (d.join("c.webm"), d.join("c.webm.part"));
        std::fs::write(&part, vec![1u8; 200_000]).unwrap();
        begin(&file, part.clone(), 1_000_000);
        let mut s = open(&file).unwrap();
        let mut buf = vec![0u8; 1_000_000];
        // What has arrived, less the held-back end.
        assert_eq!(s.read(&mut buf).unwrap() as u64, 200_000 - HOLD);
        let t = Instant::now();
        assert_eq!(s.read(&mut buf).unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(t.elapsed() >= STALL);
        let other = d.join("elsewhere.video.webm");
        alias(&file, &other);
        let mut s = open(&other).unwrap();
        s.read(&mut buf).unwrap();
        end(&file, false);
        assert!(s.read(&mut buf).is_err());
        // Over, for both names: the file itself, which is not there.
        assert!(!downloading(&other));
        assert!(open(&file).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
