// OS media integration: hardware/media-key handling and now-playing metadata.
//
// One thin wrapper over `souvlaki`, which maps to MPRIS (Linux), the System
// Media Transport Controls (Windows), and the Now Playing center (macOS).
//
// Media-key events arrive off the UI thread (a D-Bus thread on Linux, a WinRT
// event handler on Windows), so the attached callback only forwards them into
// a channel; the app drains it each frame and acts on the UI thread.

use std::sync::mpsc::{channel, Receiver, Sender, TryIter};
use std::time::Duration;

use souvlaki::{MediaControls, MediaMetadata, MediaPlayback, MediaPosition, PlatformConfig};
pub use souvlaki::{MediaControlEvent, SeekDirection};

#[derive(Clone, Copy, PartialEq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

pub struct MediaOs {
    /// `None` if the platform controls couldn't be created (e.g. no D-Bus
    /// session, or no window handle). All methods then no-op.
    controls: Option<MediaControls>,
    rx: Receiver<MediaControlEvent>,
}

impl MediaOs {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let controls = build(cc, tx);
        Self { controls, rx }
    }

    /// Drain media-key events received since the last frame.
    pub fn events(&self) -> TryIter<'_, MediaControlEvent> {
        self.rx.try_iter()
    }

    pub fn set_metadata(&mut self, title: &str, artist: &str, album: &str, duration: Option<Duration>) {
        if let Some(c) = self.controls.as_mut() {
            let _ = c.set_metadata(MediaMetadata {
                title: Some(title),
                artist: (!artist.is_empty()).then_some(artist),
                album: (!album.is_empty()).then_some(album),
                cover_url: None, // embedded art has no file:// URL; left for later
                duration,
            });
        }
    }

    pub fn set_playback(&mut self, state: PlaybackState, progress: Option<Duration>) {
        if let Some(c) = self.controls.as_mut() {
            let pos = progress.map(MediaPosition);
            let pb = match state {
                PlaybackState::Stopped => MediaPlayback::Stopped,
                PlaybackState::Playing => MediaPlayback::Playing { progress: pos },
                PlaybackState::Paused => MediaPlayback::Paused { progress: pos },
            };
            let _ = c.set_playback(pb);
        }
    }
}

fn build(cc: &eframe::CreationContext<'_>, tx: Sender<MediaControlEvent>) -> Option<MediaControls> {
    let hwnd = window_hwnd(cc);
    // On Windows an HWND is mandatory — MediaControls::new panics without one.
    #[cfg(windows)]
    if hwnd.is_none() {
        return None;
    }

    let mut controls = MediaControls::new(PlatformConfig {
        dbus_name: "moosik",
        display_name: "Moosik",
        hwnd,
    })
    .ok()?;

    controls.attach(move |event| { let _ = tx.send(event); }).ok()?;
    Some(controls)
}

#[cfg(windows)]
fn window_hwnd(cc: &eframe::CreationContext<'_>) -> Option<*mut std::ffi::c_void> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match cc.window_handle().ok()?.as_raw() {
        RawWindowHandle::Win32(h) => Some(h.hwnd.get() as *mut std::ffi::c_void),
        _ => None,
    }
}

#[cfg(not(windows))]
fn window_hwnd(_cc: &eframe::CreationContext<'_>) -> Option<*mut std::ffi::c_void> {
    None
}
