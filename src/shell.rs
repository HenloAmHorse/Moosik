//! The 1.6.0 shell: one window with three modes (Listen, Library, Analyze), a
//! shared header and player bar, and the motion between them.
//!
//! Motion runs only while something is changing: a mode switch, a track
//! change, or playback (when the window is redrawing anyway). A paused or
//! stopped player with nothing in flight draws nothing extra.
//!
//! The analyser keeps its own drawing. Analyze mode only hosts it: bar count,
//! methods and colour ramps are exactly those of the separate window.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use egui::{Color32, CornerRadius, Id, LayerId, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use serde::{Deserialize, Serialize};

use super::{apply_theme, lyrics, moosik_dir, save_appearance, save_playlist_store, LoopMode, MoosikApp, PlayState, ThemeMode};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub(crate) enum UiMode {
    #[default]
    Listen,
    Library,
    Analyze,
}

impl UiMode {
    const ALL: [UiMode; 3] = [UiMode::Listen, UiMode::Library, UiMode::Analyze];
    fn label(self) -> &'static str {
        match self {
            UiMode::Listen => "Listen",
            UiMode::Library => "Library",
            UiMode::Analyze => "Analyze",
        }
    }
}

/// What the shell shows and how much it moves. Persisted with the appearance.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct VisualSettings {
    /// The 1.6.0 layout; off restores the classic window.
    pub modern: bool,
    pub mode: UiMode,
    pub animations: bool,
    /// Multiplies every animation's duration: 0.5 is twice as fast.
    pub duration_scale: f32,
    /// The blurred cover behind everything.
    pub ambient: bool,
    pub ambient_strength: f32,
    /// Let the backdrop drift slowly while playing.
    pub ambient_drift: bool,
    pub cover_glow: bool,
    /// Glow and play button follow the music's low end while playing.
    pub beat_pulse: bool,
    /// The spectrum ribbon under the title in Listen.
    pub ribbon: bool,
    /// The current lyric lines drawn over the spectrum in Analyze.
    pub lyrics_over_spectrum: bool,
    pub lyrics_overlay_opacity: f32,
    /// Analyze's panels under the plot: the stereo image, and the harmony.
    pub stereo_panel: bool,
    pub harmony_panel: bool,
    /// Buttons give way and spring back when pressed, with a ripple.
    pub button_fx: bool,
    /// Pieces burst out of the transport buttons.
    pub confetti: bool,
    /// The cover can be dragged, thrown and poked.
    /// How far every spring, ripple and burst goes: 0 still, 1 as designed,
    /// 2 twice as wild.
    pub fx_intensity: f32,
    /// The extras that do nothing useful: a fainting speaker, the Konami
    /// code, ripples on the background and the like.
    pub toys: bool,
    /// Each extra switched on or off by itself, where it differs from the
    /// default, and the ones that have happened (their switches show).
    pub toy_set: std::collections::BTreeMap<String, bool>,
    pub toys_found: std::collections::BTreeSet<String>,
}

impl Default for VisualSettings {
    fn default() -> Self {
        Self {
            modern: true,
            mode: UiMode::Listen,
            animations: true,
            duration_scale: 1.0,
            ambient: true,
            ambient_strength: 1.0,
            ambient_drift: true,
            cover_glow: true,
            beat_pulse: true,
            ribbon: true,
            lyrics_over_spectrum: true,
            lyrics_overlay_opacity: 0.9,
            stereo_panel: false,
            harmony_panel: false,
            button_fx: true,
            confetti: true,
            fx_intensity: 1.0,
            toys: true,
            toy_set: Default::default(),
            toys_found: Default::default(),
        }
    }
}

/// One silly extra: its key, its name, how to come across it, and whether
/// it is on out of the box. One that is off by default never happens by
/// itself, so its switch is never locked.
struct Toy {
    key: &'static str,
    name: &'static str,
    hint: &'static str,
    on: bool,
}

const fn toy(key: &'static str, name: &'static str, hint: &'static str, on: bool) -> Toy {
    Toy { key, name, hint, on }
}

const TOYS: &[Toy] = &[
    toy("throw", "Throwable cover", "Drag the cover and let go.", true),
    toy("wrap", "Cover round the window", "Throw the cover out of the window.", true),
    toy("sparks", "Sparks", "Throw the cover really hard.", true),
    toy("flip", "Cover flip", "Click the cover five times in a row.", true),
    toy("vinyl", "Record mode", "Right-click the cover.", true),
    toy("doze", "Sleepy cover", "Leave Moosik alone for ten minutes.", true),
    toy("medal", "Play medals", "Play one track a hundred times.", true),
    toy("sway", "Wobbly window", "Shake the window around.", true),
    toy("konami", "Party code", "A famous cheat code, typed anywhere.", true),
    toy("ripples", "Background ripples", "Click on nothing at all.", true),
    toy("charge", "Charged play", "Hold the play button down.", true),
    toy("hammer", "Hammered play", "Press play ten times in three seconds.", true),
    toy("face", "Speaker face", "Point at the speaker.", true),
    toy("faint", "Fainting speaker", "Turn the volume all the way down.", true),
    toy("shades", "Speaker shades", "Turn the volume all the way up.", true),
    toy("knob_drop", "Falling seek knob", "Drag the seek bar off its end and let go.", true),
    toy("falling", "Tumbling rows", "Clear All in the Library, or delete a playlist.", true),
    toy("night", "Sleepy logo", "Open Moosik between midnight and five.", true),
    toy("gold", "Golden badge", "Switch bit-perfect on and watch the badge.", true),
    toy("modes", "Dizzy mode switch", "Switch modes six times in three seconds.", true),
    toy("yawn", "Yawning moon", "Let a sleep timer run out.", true),
    toy("bouquet", "Bookmark bouquet", "Bookmark one track ten times.", true),
    toy("ab_dizzy", "Dizzy A-B loop", "Let an A-B loop go round ten times.", true),
    toy("balloon", "Follow balloon", "Untick Follow under the lyrics.", true),
    toy("peek", "Peeking note", "Empty the queue and go near the box.", true),
    toy("row_eq", "Dancing row", "Play something with the Library open.", true),
    toy("fidget", "Fidgety sliders", "Point at Controls in Analyze.", true),
    toy("trail", "Spectrum doodles", "Draw a circle on the spectrum.", true),
    toy("fork", "Humming fork", "Strike the tuning fork in Analyze.", true),
    toy("answer", "The answer", "Save a playlist of 42 tracks.", true),
    toy("tea", "Tea count", "Open the Library's Queue tab.", true),
    toy("nice", "Nice", "Set the volume to 69%.", true),
    toy("seasons", "Holiday hats", "Open Moosik on a special day.", true),
    toy("logo_loop", "Loop-the-loop logo", "Click the logo seven times.", true),
    toy("headshake", "Head shake", "In Listen, press Esc five times fast.", true),
    toy("sky", "Sun and moon", "Switch between light and dark.", true),
    toy("bye", "Bye!", "Close the settings.", true),
    toy("trophy", "Trophy", "Find every other one.", true),
    toy("zoomies", "Zoomies", "Open Moosik on a screen of 120 Hz or more.", true),
    toy("pet", "Pet the cover", "Stroke the cover gently, back and forth.", true),
    toy("chord_pop", "Popping chords", "Open Harmony in Analyze while music plays.", true),
    toy("worried", "Worried scope", "Play something out of phase with Stereo open.", true),
    toy("mono_nap", "Napping scope", "Play something in mono with Stereo open.", true),
    toy("four", "Four chords", "Hear I-V-vi-IV with Harmony open.", true),
    toy("key_party", "Key-change party", "Hear a song change key with Harmony open.", true),
    toy("rumble", "Rumbly bass", "Play deep sub-bass with Stereo open.", true),
    toy("crystal_ball", "Crystal ball", "Let the chord model see a chord coming.", true),
    toy("mood", "Chord moods", "Hear major and minor chords with Harmony open.", true),
    toy("shrug", "Shrug", "Stump the chord model.", true),
    toy("chord_nap", "Napping chord", "Hold one chord for a long time.", true),
    toy("chord_rush", "Chord rush", "Hear four chord changes in two seconds.", true),
    toy("horns", "Little horns", "Hear a diminished chord with Harmony open.", true),
    toy("suspense", "Suspense", "Hear a suspended chord with Harmony open.", true),
    toy("bow", "Last bow", "Reach a song's final chord with Harmony open.", true),
    toy("tumbleweed", "Tumbleweed", "Sit through a long stretch with no chord.", true),
    toy("thanks", "Thank you", "Correct a chord the model got wrong.", true),
    toy("home", "Home chord", "Land on the key's own chord.", true),
    toy("sigh", "Sigh of relief", "Hear a chord resolve down a fifth.", true),
    toy("halo", "Halo", "Hear a major seventh chord.", true),
    toy("sparkle", "Sparkles", "Hear an augmented chord.", true),
    toy("stopwatch", "Stopwatch", "Let the chord model work a song out.", true),
    toy("ping_pong", "Ping-pong", "Hear the sound bounce between the speakers.", true),
    toy("smooth", "Smooth", "Hear a ii-V-I.", true),
    toy("apple", "Teacher's apple", "Correct five chords.", true),
    toy("on_repeat", "On repeat", "Play one song ten times in a row.", true),
    toy("centurion", "Centurion", "Grow the playlist past a hundred tracks.", true),
    toy("eyes", "Watching eyes", "Eyes on the analyser's controls that follow the pointer.", false),
    toy("six_seven", "6 7", "Palms up, see-sawing, at 67% volume or 6:07 into a track.", false),
];

impl MoosikApp {
    /// One extra, by key: on when the extras are, unless switched off by itself.
    fn toy(&self, key: &str) -> bool {
        self.toys_on() && self.appearance.visual.toy_set.get(key).copied()
            .unwrap_or_else(|| TOYS.iter().any(|t| t.key == key && t.on))
    }

    /// It happened: its switch shows from now on, and a note says so.
    fn found(&mut self, key: &'static str) {
        if self.toy(key) && self.appearance.visual.toys_found.insert(key.to_string()) {
            save_appearance(&moosik_dir(), &self.appearance);
            self.shell.found_toast = Some((key, Instant::now()));
        }
    }

    /// Every extra's switch, the found ones open and the rest locked.
    fn toys_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        let accent = self.track_accent;
        let mut changed = ui.checkbox(&mut self.appearance.visual.toys, "Silly extras").changed();
        let lockable = TOYS.iter().filter(|t| t.on).count();
        let found = TOYS.iter().filter(|t| t.on && self.appearance.visual.toys_found.contains(t.key)).count();
        // The trophy is the last one: it counts itself.
        if found + 1 == lockable && !self.appearance.visual.toys_found.contains("trophy") && self.toy("trophy") {
            self.found("trophy");
            self.celebrate(ui.ctx(), accent, true);
        }
        let found = TOYS.iter().filter(|t| t.on && self.appearance.visual.toys_found.contains(t.key)).count();
        if found == lockable && self.toy("trophy") {
            ui.label(RichText::new(format!("★ Found all {lockable}. Nothing left to find, and nothing useful found.")).size(12.0).strong().color(accent));
        } else {
            ui.label(RichText::new(format!("Found {found} of {lockable}")).size(12.0).color(tok.dim));
        }
        if found < lockable {
            ui.add_space(4.0);
            ui.label(RichText::new("🔒 A locked one opens once it has happened. Can't wait? Click its lock five times.")
                .size(12.0).strong().color(accent));
        }
        ui.add_space(6.0);
        let on = self.appearance.visual.toys;
        ui.add_enabled_ui(on, |ui| {
            for t in TOYS {
                let open = !t.on || self.appearance.visual.toys_found.contains(t.key);
                if open {
                    let mut v = self.appearance.visual.toy_set.get(t.key).copied().unwrap_or(t.on);
                    // Just unlocked, the box sits where the lock was: a sixth
                    // click from the same burst would untick it.
                    let fresh = self.shell.unlocked.is_some_and(|(k, at)| k == t.key && at.elapsed().as_secs_f32() < 0.8);
                    if fresh {
                        self.shell.animating = true;
                    }
                    if ui.add_enabled(!fresh, egui::Checkbox::new(&mut v, t.name)).on_hover_text(t.hint).changed() {
                        if v == t.on {
                            self.appearance.visual.toy_set.remove(t.key);
                        } else {
                            self.appearance.visual.toy_set.insert(t.key.to_string(), v);
                        }
                        changed = true;
                    }
                    continue;
                }
                ui.horizontal(|ui| {
                    let clicks = self.shell.lock_clicks.filter(|(k, _, at)| *k == t.key && at.elapsed().as_secs_f32() < 2.0);
                    let n = clicks.map_or(0, |(_, n, _)| n);
                    // Each click leaves it rattling a little harder.
                    let jig = if n > 0 {
                        self.shell.animating = true;
                        (ui.input(|i| i.time) as f32 * 40.0).sin() * n as f32 * 0.7
                    } else {
                        0.0
                    };
                    ui.add_space(3.0 + jig);
                    let lock = ui.add(egui::Button::new(RichText::new("🔒").size(13.0)).frame(false))
                        .on_hover_text(format!("{} more click{}", 5 - n, if n == 4 { "" } else { "s" }));
                    ui.add_space(3.0 - jig);
                    if lock.clicked() {
                        let n = n + 1;
                        if n >= 5 {
                            self.shell.lock_clicks = None;
                            self.shell.unlocked = Some((t.key, Instant::now()));
                            self.appearance.visual.toys_found.insert(t.key.to_string());
                            changed = true;
                            self.burst(lock.rect.center(), accent, 16, 0.6, None);
                        } else {
                            self.shell.lock_clicks = Some((t.key, n, Instant::now()));
                        }
                    }
                    ui.label(RichText::new(t.hint).size(12.0).italics().color(tok.faint));
                });
            }
        });
        if changed {
            save_appearance(&moosik_dir(), &self.appearance);
        }
    }
}

/// Colours of the shell for one theme. The accent comes from the cover.
#[derive(Clone, Copy)]
struct Tokens {
    bg: Color32,
    ink: Color32,
    dim: Color32,
    faint: Color32,
    line: Color32,
    glass: Color32,
    /// Behind the lyrics: dense enough that faint lines stay legible over any cover.
    scrim: Color32,
    chip: Color32,
    pill: Color32,
    pill_ink: Color32,
    ok: Color32,
    ok_bg: Color32,
    /// Veil over the backdrop per mode (Listen, Library, Analyze).
    veil: [f32; 3],
}

impl Tokens {
    fn new(dark: bool) -> Self {
        let a = |r, g, b, a: f32| premul(r, g, b, a);
        if dark {
            Tokens {
                bg: Color32::from_rgb(0x0a, 0x0c, 0x10),
                ink: Color32::from_rgb(0xee, 0xf0, 0xf4),
                dim: Color32::from_rgb(0x9a, 0xa2, 0xae),
                faint: Color32::from_rgb(0x6d, 0x75, 0x81),
                line: a(255, 255, 255, 0.08),
                glass: a(14, 17, 22, 0.62),
                scrim: a(10, 12, 16, 0.78),
                chip: a(255, 255, 255, 0.08),
                pill: Color32::from_rgb(0xee, 0xf0, 0xf4),
                pill_ink: Color32::from_rgb(0x0a, 0x0c, 0x10),
                ok: Color32::from_rgb(0x7f, 0xd7, 0x9c),
                ok_bg: a(63, 191, 114, 0.14),
                veil: [0.35, 0.72, 0.86],
            }
        } else {
            Tokens {
                bg: Color32::from_rgb(0xf2, 0xf3, 0xf5),
                ink: Color32::from_rgb(0x12, 0x14, 0x19),
                dim: Color32::from_rgb(0x55, 0x5c, 0x67),
                faint: Color32::from_rgb(0x7c, 0x83, 0x8e),
                line: a(18, 20, 25, 0.08),
                glass: a(255, 255, 255, 0.66),
                scrim: a(248, 249, 251, 0.86),
                chip: a(18, 20, 25, 0.06),
                pill: Color32::from_rgb(0x12, 0x14, 0x19),
                pill_ink: Color32::WHITE,
                ok: Color32::from_rgb(0x1d, 0x6b, 0x45),
                ok_bg: Color32::from_rgb(0xe2, 0xf4, 0xea),
                veil: [0.45, 0.78, 0.88],
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Tween {
    from: f32,
    to: f32,
    start: Instant,
    secs: f32,
}

#[derive(Clone, Copy)]
struct Particle {
    pos: Pos2,
    vel: Vec2,
    rot: f32,
    spin: f32,
    size: f32,
    color: Color32,
    born: Instant,
    life: f32,
    /// 0 confetti strip, 1 dot, 2 triangle, 3 ring.
    shape: u8,
}

/// The cover as a plaything: pulled off its place and sprung back.
#[derive(Clone, Copy, Default)]
struct CoverToy {
    /// Away from its place, and how fast.
    off: Vec2,
    vel: Vec2,
    /// Tilt in radians, and how fast it swings.
    ang: f32,
    angv: f32,
    /// A size wobble around 1, and its speed.
    squash: f32,
    squashv: f32,
    /// Where the pointer holds it, relative to its place, while dragged.
    grab: Option<Vec2>,
}

#[derive(Clone, Copy)]
struct CoverAnim {
    from: (Rect, f32),
    to: (Rect, f32),
    start: Instant,
    secs: f32,
}

type Chan<T> = std::sync::mpsc::Receiver<T>;

/// The chord model's side of Harmony. Weights are checked and loaded once a
/// session in the background; a download happens only when asked for; each
/// track is worked out once (then cached on disk) while it plays.
#[derive(Default)]
struct ChordAi {
    loading: Option<Chan<Result<std::sync::Arc<crate::chords::Model>, String>>>,
    checked: bool,
    model: Option<std::sync::Arc<crate::chords::Model>>,
    /// Bytes downloaded, cancel, outcome.
    install: Option<(std::sync::Arc<std::sync::atomic::AtomicU64>, std::sync::Arc<std::sync::atomic::AtomicBool>, Chan<Result<(), String>>)>,
    job: Option<(PathBuf, std::sync::Arc<std::sync::atomic::AtomicBool>, Chan<Result<Vec<crate::chords::Segment>, String>>)>,
    track: Option<(PathBuf, Result<Vec<crate::chords::Segment>, String>)>,
    error: Option<String>,
    /// When the running job started; how long the last one took, and when it finished.
    job_at: Option<Instant>,
    took: Option<(f32, Instant)>,
}

/// Per-session animation state; nothing here is persisted.
#[derive(Default)]
pub(crate) struct ShellState {
    tweens: HashMap<&'static str, Tween>,
    cover: Option<CoverAnim>,
    cover_mode: Option<UiMode>,
    /// The window the cover last laid out in: a resize follows at once,
    /// any other move glides.
    cover_screen: Option<Rect>,
    /// Where this frame's layout wants the cover; painted after the panels.
    cover_target: Option<(Rect, f32)>,
    /// Low-end energy of the playing music, 0–1, smoothed.
    energy: f32,
    drift: f32,
    last_frame: Option<Instant>,
    /// The backdrop that was showing when the track changed, faded out.
    ambient_prev: Option<(egui::TextureId, Instant)>,
    ambient_cur: Option<egui::TextureId>,
    ambient_track: Option<PathBuf>,
    /// The full classic toolbar, shown above the player bar.
    more_open: bool,
    /// When each shell button was last pressed, for its spring.
    pressed: HashMap<&'static str, Instant>,
    particles: Vec<Particle>,
    ripples: Vec<(Rect, Instant, Color32)>,
    /// Clickable things under the pointer, with how lit each is (0–1).
    hovers: HashMap<Id, (Rect, f32)>,
    /// This frame's round buttons, whose glow is a ring.
    round_ids: Vec<Id>,
    /// Clicks the shell's own buttons already answered, so the general ripple
    /// does not answer them again.
    fx_handled: Vec<Id>,
    rng: u32,
    /// The current line the lyrics overlay last showed, and when it changed.
    overlay_line: Option<(String, Instant)>,
    toy: CoverToy,
    /// Icons mid-spin: when, and how many turns.
    spins: HashMap<&'static str, (Instant, f32)>,
    /// Recent play-button presses, for the one that comes after many.
    play_presses: Vec<Instant>,
    /// The mode pill's last position, for its stretch.
    pill_prev: f32,
    /// When the speaker last fainted.
    fainted_at: Option<Instant>,
    /// The last keys, for a certain sequence.
    keys: Vec<egui::Key>,
    /// The party the sequence starts lasts until here.
    party_until: Option<Instant>,
    /// Rings on the background where it was clicked.
    rings: Vec<(Pos2, Instant)>,
    /// The volume while no secret code is under way, put back once it lands:
    /// its arrows are volume and seek shortcuts too.
    konami_vol: Option<f32>,
    /// Whether the seek bar was being dragged last frame, and the knob that
    /// fell off its end.
    seek_was: bool,
    knob_drop: Option<(Pos2, Instant)>,
    /// The lyrics' Follow tick, floating off.
    balloon: Option<(Rect, Instant)>,
    /// The last input or track change, for the cover that nods off.
    last_input: Option<Instant>,
    last_track: Option<usize>,
    /// The current track's play count as last seen, and a medal for a round
    /// hundred.
    plays_seen: Option<(usize, u32)>,
    medal: Option<(u32, Instant)>,
    /// Recent clicks on the cover; five turn it over.
    cover_clicks: Vec<Instant>,
    logo_clicks: Vec<Instant>,
    esc_presses: Vec<Instant>,
    flipped: bool,
    /// The cover as a spinning record, and how far round it is.
    vinyl: bool,
    vinyl_ang: f32,
    /// A hard throw flies free, without the spring, until this.
    fly_until: Option<Instant>,
    /// Back in from the other side: straight home, no overshoot, no wrap.
    homing: bool,
    /// Rows falling out of a cleared playlist, each from when.
    falling: Vec<(Rect, String, Instant)>,
    list_rect: Option<Rect>,
    /// The window's last position, and the lyrics card's lag behind it.
    win_pos: Option<Pos2>,
    sway: Vec2,
    sway_vel: Vec2,
    /// The bit-perfect badge's gold moment after the switch is turned on.
    was_exact: bool,
    gold_armed: Option<Instant>,
    gold: Option<Instant>,
    /// When the play button went down, for a charged press.
    play_held: Option<Instant>,
    play_fired: bool,
    /// The pointer's streak across the spectrum, and the last circle closed.
    trail: Vec<(Pos2, Instant)>,
    trail_pop: Option<Instant>,
    /// A-B loops completed; the playback loop counts them.
    pub(crate) ab_laps: u32,
    ab_dizzy: Option<Instant>,
    /// The Library inspector's tab, and its underline's two ends.
    insp_tab: usize,
    insp_ul: (f32, f32),
    /// How far the note is out of the empty library's box, 0–1.
    peek: f32,
    /// Analyze's controls drawer: shut by the user, and its width now.
    tools_hidden: bool,
    tools_w: f32,
    /// The tuning fork was struck; the bottom of the drawer was reached.
    fork_ding: Option<Instant>,
    tools_bottom: bool,
    /// The drawer's eyes, shut since.
    blink: Option<Instant>,
    /// Mode switches in the last few seconds, and the dizzy spell after.
    mode_hops: Vec<Instant>,
    mode_dizzy: Option<Instant>,
    /// 67 and 69 going off: where, when, which. And what set them off last.
    memes: Vec<(Pos2, Instant, u8)>,
    /// A lock in the extras being clicked open: which, how often, last when.
    lock_clicks: Option<(&'static str, u32, Instant)>,
    /// An extra just found, for the note that says so.
    found_toast: Option<(&'static str, Instant)>,
    unlocked: Option<(&'static str, Instant)>,
    /// A fast screen: the logo runs about for a moment at start.
    zoomies: Option<Instant>,
    /// Stroking the cover: the last direction, the turns, the hearts.
    pet_dir: f32,
    pet_turns: Vec<Instant>,
    hearts: Vec<(Pos2, Instant)>,
    /// Analyze's live panels: the analysis, when it last ran, for which track,
    /// and the PCM it read (the scope draws the stereo copy).
    live: crate::analysis::Live,
    live_at: Option<Instant>,
    /// When the stereo image was last worked out.
    stereo_at: Option<Instant>,
    live_path: Option<PathBuf>,
    live_pcm: Vec<[f32; 2]>,
    live_mono: Vec<f32>,
    scope_gain: f32,
    /// The chord last popped, and when; mono since when; four chords heard when.
    chord_at: Option<(String, Instant)>,
    /// When the chord last changed, newest last (a few seconds' worth).
    chord_changes: Vec<Instant>,
    /// Since when there has been no chord (for the tumbleweed).
    nc_since: Option<Instant>,
    /// The chord being corrected (its track and start), and the root picked so far.
    chord_fix: Option<(PathBuf, f32, usize)>,
    /// When the user last corrected a chord (for the thank-you).
    chord_thanks: Option<Instant>,
    /// The chord before the current one (for the sigh).
    chord_prev: Option<String>,
    /// The chord before that (for the ii-V-I).
    chord_prev2: Option<String>,
    /// Chords corrected this session (for the apple).
    fixes_made: u32,
    /// Which side the sound last leant to (-1 left, 1 right), each time it
    /// swapped sides recently, and when a ping-pong ball was served.
    pan_side: i8,
    pan_swaps: Vec<Instant>,
    ping_at: Option<Instant>,
    /// Plays of the current track in a row, and when hearts last rose off the cover.
    repeat_streak: u32,
    hearts_at: Option<Instant>,
    /// The playlist's length last frame (for the centurion).
    playlist_seen: Option<usize>,
    /// The chord model: its weights, an install, the track being worked out.
    chord_ai: ChordAi,
    mono_since: Option<Instant>,
    four_at: Option<(usize, Instant)>,
    /// The key now, and since when; the last key that held; the last change of key, from what.
    key_since: Option<(usize, bool, Instant)>,
    key_settled: Option<(usize, bool)>,
    key_party: Option<(Instant, String)>,
    /// The sun (or moon) crossing the sky after a theme switch: when, which, from where.
    sky: Option<(Instant, bool, Pos2)>,
    settings_was_open: bool,
    vol_meme: u8,
    meme_secs: f32,
    /// Frames with motion in flight ask for the next one.
    pub(crate) animating: bool,
}

fn ease_out_quart(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(4)
}

/// A little overshoot, for the mode pill.
fn ease_out_back(t: f32) -> f32 {
    let c1 = 1.20;
    let c3 = c1 + 1.0;
    1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_rect(a: Rect, b: Rect, t: f32) -> Rect {
    Rect::from_min_max(
        Pos2::new(lerp(a.min.x, b.min.x, t), lerp(a.min.y, b.min.y, t)),
        Pos2::new(lerp(a.max.x, b.max.x, t), lerp(a.max.y, b.max.y, t)),
    )
}

/// Premultiplied in gamma space, the way the eye reads a translucent layer.
/// (`from_rgba_unmultiplied` premultiplies in linear light, which makes a 5%
/// white veil look like 25% grey.)
fn premul(r: u8, g: u8, b: u8, a: f32) -> Color32 {
    let a = a.clamp(0.0, 1.0);
    let m = |c: u8| (c as f32 * a).round() as u8;
    Color32::from_rgba_premultiplied(m(r), m(g), m(b), (a * 255.0).round() as u8)
}

/// `a` moved `t` of the way to `b`, channel by channel.
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()), a.a())
}

/// A chart chord name ("Am7", "C#m", "Cmaj7") is minor: an "m" straight
/// after the root that does not start "maj".
/// The pitch class (0 = C) of a chart chord name's root: "Bb7" -> 10.
fn root_pc(name: &str) -> Option<usize> {
    let mut c = name.chars();
    let base = match c.next()? {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    Some(match c.next() {
        Some('#') => (base + 1) % 12,
        Some('b') => (base + 11) % 12,
        _ => base,
    })
}

/// Minor on the 2nd, major on the 5th, then major on the 1st: "Dm7 G7 Cmaj7".
fn is_two_five_one(a: &str, b: &str, c: &str) -> bool {
    match (root_pc(a), root_pc(b), root_pc(c)) {
        (Some(ra), Some(rb), Some(rc)) => is_minor(a) && !is_minor(b) && !is_minor(c) && ra == (rc + 2) % 12 && rb == (rc + 7) % 12,
        _ => false,
    }
}

fn is_minor(name: &str) -> bool {
    let rest = name.get(1..).unwrap_or("").trim_start_matches(['#', 'b']);
    rest.starts_with('m') && !rest.starts_with("maj")
}

fn with_alpha(c: Color32, a: f32) -> Color32 {
    let [r, g, b, _] = c.to_srgba_unmultiplied();
    premul(r, g, b, a)
}

impl ShellState {
    /// The value of a named tween, started towards `to` when the target
    /// changes. `secs` of zero snaps.
    fn tween(&mut self, key: &'static str, to: f32, secs: f32, ease: fn(f32) -> f32) -> f32 {
        let now = Instant::now();
        let tw = self.tweens.entry(key).or_insert(Tween { from: to, to, start: now, secs });
        if (tw.to - to).abs() > f32::EPSILON {
            let cur = Self::sample(tw, now, ease);
            *tw = Tween { from: cur, to, start: now, secs };
        }
        let v = Self::sample(tw, now, ease);
        if (v - tw.to).abs() > 1e-4 {
            self.animating = true;
        }
        v
    }

    /// 0-1, xorshift; good enough for confetti.
    fn rand(&mut self) -> f32 {
        if self.rng == 0 {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.subsec_nanos());
            self.rng = t | 1;
        }
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Restart a tween from `from` (used for things that replay, like a fade-in).
    fn restart(&mut self, key: &'static str, from: f32, to: f32, secs: f32) {
        self.tweens.insert(key, Tween { from, to, start: Instant::now(), secs });
    }

    fn sample(tw: &Tween, now: Instant, ease: fn(f32) -> f32) -> f32 {
        if tw.secs <= 0.0 {
            return tw.to;
        }
        let t = (now.duration_since(tw.start).as_secs_f32() / tw.secs).clamp(0.0, 1.0);
        lerp(tw.from, tw.to, ease(t))
    }
}

impl MoosikApp {
    /// The intensity knob, 0 when animations are off.
    fn fx_k(&self) -> f32 {
        let v = &self.appearance.visual;
        if v.animations { v.fx_intensity.clamp(0.0, 2.0) } else { 0.0 }
    }

    fn toys_on(&self) -> bool {
        self.appearance.visual.toys && self.appearance.visual.animations
    }

    /// The angle of an icon's spin now, overshooting a little at the end.
    fn spin_angle(&mut self, key: &'static str) -> f32 {
        let Some((at, turns)) = self.shell.spins.get(key).copied() else { return 0.0 };
        let secs = self.shell_secs(700.0) * turns.abs().max(0.5).sqrt();
        let t = if secs > 0.0 { at.elapsed().as_secs_f32() / secs } else { 1.0 };
        if t >= 1.0 {
            self.shell.spins.remove(key);
            return 0.0;
        }
        self.shell.animating = true;
        ease_out_back(t) * turns * std::f32::consts::TAU
    }

    fn shell_secs(&self, ms: f32) -> f32 {
        let v = &self.appearance.visual;
        if v.animations { ms / 1000.0 * v.duration_scale.clamp(0.1, 4.0) } else { 0.0 }
    }

    fn set_mode(&mut self, mode: UiMode) {
        if self.appearance.visual.mode != mode {
            self.appearance.visual.mode = mode;
            let secs = self.shell_secs(460.0);
            self.shell.restart("content", 0.0, 1.0, secs);
            save_appearance(&moosik_dir(), &self.appearance);
        }
    }

    /// Before the analyser ticks: in the shell, Analyze hosts it and it is
    /// otherwise closed.
    pub(crate) fn shell_prepare_spectrum(&mut self) {
        if !self.appearance.visual.modern {
            self.spectrum_window.embedded = false;
            self.spectrum_window.analyzer.celebrate = false;
            return;
        }
        let analyze = self.appearance.visual.mode == UiMode::Analyze;
        self.spectrum_window.embedded = analyze;
        self.spectrum_window.open = analyze;
    }

    /// The page colour of Moosik's own windows (Info, Tags, Lyrics) in the
    /// 1.6.0 look; None keeps the classic one.
    pub(crate) fn page_fill(&self) -> Option<Color32> {
        self.appearance.visual.modern.then(|| Tokens::new(self.appearance.theme.is_dark()).bg)
    }

    pub(crate) fn window_frame(&self, ctx: &egui::Context) -> egui::Frame {
        page_frame(self.page_fill(), &ctx.style())
    }

    pub(crate) fn shell_ui(&mut self, ctx: &egui::Context) {
        let dark = self.appearance.theme.is_dark();
        let tok = Tokens::new(dark);
        self.party_keys(ctx);
        if let Some(until) = self.shell.party_until {
            if Instant::now() < until {
                // The accent runs round the colour wheel.
                let h = (ctx.input(|i| i.time) as f32 * 0.35).fract();
                self.track_accent = egui::ecolor::Hsva::new(h, 0.7, 1.0, 1.0).into();
                self.shell.animating = true;
            } else {
                self.shell.party_until = None;
            }
        }
        let accent = self.track_accent;
        let now = Instant::now();
        let dt = self.shell.last_frame.map_or(0.0, |t| now.duration_since(t).as_secs_f32()).min(0.1);
        self.shell.last_frame = Some(now);
        self.shell.animating = false;
        let playing = self.play_state == PlayState::Playing;
        self.update_energy(playing);
        if playing && self.appearance.visual.ambient_drift && self.appearance.visual.animations {
            self.shell.drift += dt;
        }
        let mode = self.appearance.visual.mode;
        self.shell.cover_target = None;
        self.shell_toys_tick(ctx, dt);

        self.paint_backdrop(ctx, &tok, mode);
        self.shell_header(ctx, &tok, accent);
        self.shell_player_bar(ctx, &tok, accent, mode);
        self.settings_drawer(ctx, &tok);

        let content = {
            let tw = self.shell.tweens.get("content").copied();
            match tw {
                Some(tw) => ShellState::sample(&tw, now, ease_out_quart),
                None => 1.0,
            }
        };
        if content < 0.999 {
            self.shell.animating = true;
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
            let rise = (1.0 - content) * 16.0;
            let full = ui.max_rect();
            let area = Rect::from_min_max(full.min + Vec2::new(0.0, rise), full.max + Vec2::new(0.0, rise));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(area));
            child.multiply_opacity(content);
            match mode {
                UiMode::Listen => self.listen_ui(&mut child, &tok, accent),
                UiMode::Library => self.library_ui(&mut child, &tok, accent),
                UiMode::Analyze => self.analyze_ui(&mut child, &tok),
            }
        });

        // A pre-process analysis is minutes of work: it gets a party.
        if std::mem::take(&mut self.spectrum_window.analyzer.celebrate) {
            self.celebrate(ctx, accent, false);
        }
        self.paint_cover(ctx, &tok, accent, mode, dt);
        self.paint_fx(ctx, dt);

        // The classic toolbar can still open or close the analyser: follow it.
        if mode != UiMode::Analyze && self.spectrum_window.open {
            self.set_mode(UiMode::Analyze);
        }
        if self.shell.animating {
            ctx.request_repaint();
        }
    }

    /// Up up down down left right left right B A.
    fn party_keys(&mut self, ctx: &egui::Context) {
        use egui::Key::*;
        const CODE: [egui::Key; 10] = [ArrowUp, ArrowUp, ArrowDown, ArrowDown, ArrowLeft, ArrowRight, ArrowLeft, ArrowRight, B, A];
        if !self.toy("konami") {
            return;
        }
        let pressed: Vec<egui::Key> = ctx.input(|i| i.events.iter().filter_map(|e| match e {
            egui::Event::Key { key, pressed: true, repeat: false, .. } => Some(*key),
            _ => None,
        }).collect());
        let mut done = false;
        for key in pressed {
            self.shell.keys.push(key);
            let n = self.shell.keys.len();
            if n > CODE.len() {
                self.shell.keys.drain(..n - CODE.len());
            }
            if self.shell.keys == CODE {
                self.shell.keys.clear();
                self.shell.party_until = Some(Instant::now() + std::time::Duration::from_secs(12));
                self.celebrate(ctx, self.track_accent, true);
                done = true;
                self.found("konami");
            }
        }
        // The arrows have already done their usual jobs by now. Left and right
        // cancel out; up and down do too, unless the volume sat at the top,
        // so the level from before the code goes back. The snapshot is taken
        // after this frame's shortcuts, while no code is under way.
        let keys = &self.shell.keys;
        let under_way = (1..=keys.len().min(CODE.len())).any(|l| keys.ends_with(&CODE[..l]));
        if done {
            if let Some(v) = self.shell.konami_vol.take()
                && v != self.volume
                && self.engine.as_mut().is_some_and(|e| e.set_volume(v))
            {
                self.volume = v;
            }
        } else if !under_way {
            self.shell.konami_vol = Some(self.volume);
        }
    }

    /// Bookkeeping for the extras that watch the world rather than a widget.
    fn shell_toys_tick(&mut self, ctx: &egui::Context, dt: f32) {
        let now = Instant::now();
        // Any input, or a new track, wakes the cover up.
        if ctx.input(|i| !i.events.is_empty() || i.pointer.is_moving())
            || self.shell.last_track != self.current_index || self.shell.last_input.is_none()
        {
            self.shell.last_input = Some(now);
            self.shell.last_track = self.current_index;
        }
        if !self.toys_on() {
            self.shell.sway = Vec2::ZERO;
            self.shell.sway_vel = Vec2::ZERO;
            return;
        }
        // A shaken window leaves the cover and the lyrics card behind a little.
        let pos = ctx.input(|i| i.viewport().outer_rect.map(|r| r.min));
        if let (Some(prev), Some(cur)) = (self.shell.win_pos, pos)
            && self.toy("sway")
        {
            let d = cur - prev;
            if d != Vec2::ZERO && d.length() < 400.0 {
                if d.length() > 30.0 {
                    self.found("sway");
                }
                let k = self.fx_k();
                if self.shell.toy.grab.is_none() {
                    self.shell.toy.off -= d * 0.5 * k;
                }
                self.shell.sway -= d * 0.4 * k;
            }
        }
        self.shell.win_pos = pos;
        let (s, v) = (&mut self.shell.sway, &mut self.shell.sway_vel);
        if s.length() > 0.1 || v.length() > 1.0 {
            let h = dt / 4.0;
            for _ in 0..4 {
                let acc = -*s * 140.0 - *v * 9.0;
                *v += acc * h;
                *s += *v * h;
            }
            if s.length() > 120.0 {
                *s = *s / s.length() * 120.0;
            }
            self.shell.animating = true;
        } else {
            *s = Vec2::ZERO;
            *v = Vec2::ZERO;
        }
        // A fast screen, and the logo can't sit still for a moment.
        if self.shell.zoomies.is_none() {
            self.shell.zoomies = Some(if self.refresh_hz >= 119.0 && self.toy("zoomies") {
                self.found("zoomies");
                now
            } else {
                now - std::time::Duration::from_secs(60)
            });
        }
        // Settings closed, by any door: it waves.
        if std::mem::replace(&mut self.shell.settings_was_open, self.shell.more_open) && !self.shell.more_open
            && self.toy("bye")
        {
            self.found("bye");
            let r = ctx.screen_rect();
            self.shell.memes.push((Pos2::new(r.right() - 70.0, r.center().y), now, 1));
        }
        // Esc, Esc, Esc, Esc, Esc: no, no, no — the card shakes its head.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && self.toy("headshake")
            && self.appearance.visual.mode == UiMode::Listen
        {
            self.shell.esc_presses.retain(|t| now.duration_since(*t).as_secs_f32() < 2.5);
            self.shell.esc_presses.push(now);
            if self.shell.esc_presses.len() >= 5 {
                self.shell.esc_presses.clear();
                self.found("headshake");
                self.shell.sway_vel.x += 1600.0 * self.fx_k().max(0.3);
                self.shell.animating = true;
            }
        }
        // A track's hundredth play (and two hundredth…) gets a medal.
        if let Some(i) = self.current_index
            && let Some(t) = self.playlist.get(i)
        {
            let c = self.stats.get(&t.path).map_or(0, |s| s.count);
            let replayed = self.shell.plays_seen.is_some_and(|(pi, pc)| pi == i && c > pc);
            if replayed && c % 100 == 0 && self.toy("medal") {
                self.found("medal");
                self.shell.medal = Some((c, now));
                self.celebrate(ctx, self.track_accent, false);
            }
            // The same song ten times running: hearts rise off the cover.
            if self.shell.plays_seen.is_some_and(|(pi, _)| pi != i) {
                self.shell.repeat_streak = 0;
            }
            if replayed {
                self.shell.repeat_streak += 1;
                if self.shell.repeat_streak % 10 == 0 && self.toy("on_repeat") {
                    self.found("on_repeat");
                    self.shell.hearts_at = Some(now);
                }
            }
            self.shell.plays_seen = Some((i, c));
        }
        // A hundred tracks in the playlist: confetti.
        let n = self.playlist.len();
        if self.shell.playlist_seen.is_some_and(|m| m < 100) && n >= 100 && self.toy("centurion") {
            self.found("centurion");
            self.celebrate(ctx, self.track_accent, false);
        }
        self.shell.playlist_seen = Some(n);
        // Come back when the cover would nod off.
        let idle = self.idle_secs();
        if idle < DOZE_SECS {
            ctx.request_repaint_after(std::time::Duration::from_secs_f32(DOZE_SECS - idle + 0.1));
        }
    }

    fn idle_secs(&self) -> f32 {
        self.shell.last_input.map_or(0.0, |t| t.elapsed().as_secs_f32())
    }

    /// Clear All in the shell: the rows fall out of the list one by one.
    pub(crate) fn shell_playlist_cleared(&mut self) {
        let v = &self.appearance.visual;
        if !(v.modern && v.mode == UiMode::Library && self.toy("falling")) {
            return;
        }
        self.found("falling");
        let Some(list) = self.shell.list_rect else { return };
        let now = Instant::now();
        let row_h = 26.0;
        let top = list.top() + 64.0;
        let n = ((list.bottom() - top) / row_h).max(0.0) as usize;
        for (i, t) in self.playlist.iter().take(n.min(40)).enumerate() {
            let r = Rect::from_min_size(Pos2::new(list.left() + 14.0, top + i as f32 * row_h),
                Vec2::new(list.width() - 28.0, row_h - 4.0));
            self.shell.falling.push((r, t.title.clone(), now + std::time::Duration::from_secs_f32(i as f32 * 0.035)));
        }
    }

    /// Dragged off the very end and let go, the seek bar's knob falls and
    /// bounces.
    fn seek_toys(&mut self, ctx: &egui::Context, seek: Rect) {
        let was = std::mem::replace(&mut self.shell.seek_was, self.seeking);
        if !self.toys_on() {
            self.shell.knob_drop = None;
            return;
        }
        let knob_toy = self.toy("knob_drop");
        // The seek bar's own geometry.
        let h = if self.spectrum_window.waveform.is_some() { 48.0 } else { 22.0 };
        let (x0, x1) = (seek.left() + 58.0, seek.right() - 58.0);
        let y = seek.top() + h / 2.0;
        let x = x0 + self.seek_pos.clamp(0.0, 1.0) * (x1 - x0);
        let now_secs = self.elapsed().as_secs_f32();
        let last = std::mem::replace(&mut self.shell.meme_secs, now_secs);
        // Played into, not jumped over.
        if self.play_state == PlayState::Playing && now_secs - last < 1.5 {
            for (at, meme) in [(367.0f32, 67u8), (369.0, 69)] {
                if last < at && now_secs >= at && self.toy(if meme == 67 { "six_seven" } else { "nice" }) {
                    self.shell.memes.push((Pos2::new(seek.left() + 30.0, y - 16.0), Instant::now(), meme));
                }
            }
        }
        let painter = ctx.layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_seek_toys")));
        let accent = self.track_accent;
        if was && !self.seeking && self.seek_pos >= 0.99 && knob_toy {
            self.found("knob_drop");
            self.shell.knob_drop = Some((Pos2::new(x, y), Instant::now()));
        }
        if let Some((p, at)) = self.shell.knob_drop {
            let t = at.elapsed().as_secs_f32();
            if t > 1.6 {
                self.shell.knob_drop = None;
            } else {
                // Down to the floor, then two bounces, each lower.
                let fall = 40.0;
                let up = fall * (std::f32::consts::PI * 2.5 * t / 1.2).cos().abs() * (-2.2 * t / 1.2).exp();
                let q = Pos2::new(p.x + 12.0 * t, p.y + fall - up);
                painter.circle_filled(q, 6.0, with_alpha(accent, (1.6 - t).min(1.0)));
                self.shell.animating = true;
            }
        }
    }

    /// Held down, the play button charges; let go after a while, it fires.
    fn charge_play(&mut self, ui: &egui::Ui, play: &egui::Response, accent: Color32) {
        if !self.toy("charge") {
            self.shell.play_held = None;
            return;
        }
        let c = play.rect.center();
        // egui gives up on a press as a click after 0.8 s and stops calling the
        // button pressed, so past the first frame the hold is the raw button.
        let down = ui.input(|i| i.pointer.primary_down());
        if play.is_pointer_button_down_on() && self.shell.play_held.is_none() && !self.shell.play_fired {
            self.shell.play_held = Some(Instant::now());
        }
        if down && (self.shell.play_held.is_some() || self.shell.play_fired) {
            if self.shell.play_fired {
                return;
            }
            let Some(at) = self.shell.play_held else { return };
            // Full and still held: it goes off by itself.
            if at.elapsed().as_secs_f32() >= 1.7 {
                self.shell.play_fired = true;
                self.shell.play_held = None;
                self.fire_play(c, 1.0, accent);
                return;
            }
            let charge = (at.elapsed().as_secs_f32() - 0.25).clamp(0.0, 1.0);
            if charge > 0.0 {
                let a0 = -std::f32::consts::FRAC_PI_2;
                // Full, it trembles.
                let shake = if charge >= 1.0 { Vec2::new((ui.input(|i| i.time) as f32 * 60.0).sin() * 1.5, 0.0) } else { Vec2::ZERO };
                let pts = arc(c + shake, 30.0, a0, a0 + charge * std::f32::consts::TAU, 40);
                ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_charge")))
                    .add(egui::Shape::line(pts, Stroke::new(3.0, accent)));
            }
            self.shell.animating = true;
        } else {
            self.shell.play_fired = false;
            if let Some(at) = self.shell.play_held.take() {
                let held = at.elapsed().as_secs_f32();
                // Shorter than egui's longest click, the press was a click.
                let over = ui.input(|i| i.pointer.latest_pos()).is_some_and(|p| play.rect.expand(10.0).contains(p));
                if held >= 0.8 && over {
                    self.fire_play(c, (held - 0.25).clamp(0.0, 1.0), accent);
                }
            }
        }
    }

    fn fire_play(&mut self, c: Pos2, charge: f32, accent: Color32) {
        self.found("charge");
        self.burst(c, accent, (30.0 + 150.0 * charge) as usize, 1.0 + 2.2 * charge, None);
        // A shock ring.
        self.shell.rings.push((c, Instant::now()));
        self.shell.spins.insert("play", (Instant::now(), 1.0));
        self.toggle_play_pause();
    }

    /// Dragged across the spectrum, the pointer leaves a fading streak; once
    /// round in a circle, it pops. Drawn above; the analysis is untouched.
    fn paint_trail(&mut self, ctx: &egui::Context, canvas: Rect) {
        let now = Instant::now();
        let life = 0.8;
        let accent = self.track_accent;
        if let Some(p) = ctx.input(|i| if i.pointer.primary_down() { i.pointer.latest_pos() } else { None })
            && canvas.contains(p)
            && self.shell.trail.last().is_none_or(|(q, _)| q.distance(p) > 2.0)
        {
            self.shell.trail.push((p, now));
            let since = self.shell.trail_pop;
            let pts: Vec<Pos2> = self.shell.trail.iter()
                .filter(|(_, t)| since.is_none_or(|s| *t > s)).map(|(p, _)| *p).collect();
            if pts.len() >= 12 {
                let c = (pts.iter().fold(Vec2::ZERO, |s, p| s + p.to_vec2()) / pts.len() as f32).to_pos2();
                let r = pts.iter().map(|p| p.distance(c)).sum::<f32>() / pts.len() as f32;
                let mut turn = 0.0;
                for w in pts.windows(2) {
                    let mut d = (w[1] - c).angle() - (w[0] - c).angle();
                    if d > std::f32::consts::PI {
                        d -= std::f32::consts::TAU;
                    } else if d < -std::f32::consts::PI {
                        d += std::f32::consts::TAU;
                    }
                    turn += d;
                }
                if turn.abs() >= std::f32::consts::TAU && r > 15.0 {
                    self.shell.trail_pop = Some(now);
                    self.found("trail");
                    self.burst(c, accent, 30, 0.9, None);
                    self.shell.rings.push((c, now));
                }
            }
        }
        self.shell.trail.retain(|(_, at)| now.duration_since(*at).as_secs_f32() < life);
        if self.shell.trail.is_empty() {
            return;
        }
        self.shell.animating = true;
        let painter = ctx.layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_trail"))).with_clip_rect(canvas);
        let core = lerp_color(accent, Color32::WHITE, 0.5);
        for w in self.shell.trail.windows(2) {
            let ((a, ta), (b, tb)) = (w[0], w[1]);
            // A gap in time is a new stroke.
            if tb.duration_since(ta).as_secs_f32() > 0.1 {
                continue;
            }
            let f = 1.0 - now.duration_since(tb).as_secs_f32() / life;
            painter.line_segment([a, b], Stroke::new(9.0 * f, with_alpha(accent, 0.18 * f)));
            painter.line_segment([a, b], Stroke::new(2.5 * f, with_alpha(core, 0.9 * f)));
        }
    }

    fn update_energy(&mut self, playing: bool) {
        let target = if playing && self.appearance.visual.beat_pulse {
            let m = &self.spectrum_window.analyzer.magnitudes;
            let n = (m.len() / 10).max(1);
            let avg = m.iter().take(n).sum::<f32>() / n as f32;
            ((avg - 0.25) * 2.2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let e = &mut self.shell.energy;
        *e = if target > *e { lerp(*e, target, 0.6) } else { lerp(*e, target, 0.12) };
    }

    fn paint_backdrop(&mut self, ctx: &egui::Context, tok: &Tokens, mode: UiMode) {
        let screen = ctx.screen_rect();
        let painter = ctx.layer_painter(LayerId::background());
        painter.rect_filled(screen, 0.0, tok.bg);
        let v = self.appearance.visual.clone();
        let idx = match mode { UiMode::Listen => 0, UiMode::Library => 1, UiMode::Analyze => 2 };
        let mode_secs = self.shell_secs(360.0);
        let veil = self.shell.tween("veil", tok.veil[idx], mode_secs, ease_out_quart);
        if v.ambient {
            // The current track's backdrop; a change of track cross-fades.
            let path = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
            let tex = path.as_deref().and_then(|p| {
                self.art_cache.get_or_load(p, ctx);
                self.art_cache.ambient(p)
            });
            if path != self.shell.ambient_track {
                if let Some(old) = self.shell.ambient_cur {
                    self.shell.ambient_prev = Some((old, Instant::now()));
                }
                self.shell.ambient_track = path;
                self.shell.ambient_cur = tex;
            } else if self.shell.ambient_cur.is_none() {
                self.shell.ambient_cur = tex;
            }
            let strength = v.ambient_strength.clamp(0.0, 1.0) * if tok.bg.r() > 128 { 0.75 } else { 1.0 };
            let d = self.shell.drift;
            let grow = screen.size() * 0.18;
            let shift = Vec2::new((d * 0.05).sin() * 40.0, (d * 0.037).cos() * 30.0);
            let rect = screen.expand2(grow).translate(shift);
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            let cross = self.shell_secs(900.0);
            if let Some((old, at)) = self.shell.ambient_prev {
                let t = if cross > 0.0 { (at.elapsed().as_secs_f32() / cross).min(1.0) } else { 1.0 };
                if t >= 1.0 {
                    self.shell.ambient_prev = None;
                } else {
                    self.shell.animating = true;
                    painter.image(old, rect, uv, Color32::from_white_alpha(((1.0 - t) * strength * 255.0) as u8));
                }
            }
            if let Some(cur) = self.shell.ambient_cur {
                let t = self.shell.ambient_prev
                    .map_or(1.0, |(_, at)| if cross > 0.0 { (at.elapsed().as_secs_f32() / cross).min(1.0) } else { 1.0 });
                painter.image(cur, rect, uv, Color32::from_white_alpha((t * strength * 255.0) as u8));
            }
        }
        painter.rect_filled(screen, 0.0, with_alpha(tok.bg, veil));
    }

    fn shell_header(&mut self, ctx: &egui::Context, tok: &Tokens, accent: Color32) {
        let frame = egui::Frame::NONE.fill(tok.glass).inner_margin(Margin::symmetric(20, 0))
            .stroke(Stroke::new(1.0, tok.line));
        egui::TopBottomPanel::top("shell_header").exact_height(56.0).frame(frame).show_separator_line(false).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let width = ui.max_rect().width();
                // For nothing but the fun of it.
                // The mark is a wave that moves with the music, wriggles under
                // the pointer and throws confetti when clicked.
                let (lr, logo) = ui.allocate_exact_size(Vec2::new(26.0, 22.0), Sense::click());
                let logo = logo.on_hover_cursor(egui::CursorIcon::PointingHand);
                let wiggle = ui.ctx().animate_bool_with_time(logo.id, logo.hovered(), 0.25);
                let t = ui.input(|i| i.time) as f32;
                // Between midnight and five it is a slow, sleepy wave.
                let night = self.toy("night") && local_hour() < 5;
                if night {
                    self.found("night");
                }
                let amp = if night { 1.5 + 1.5 * wiggle } else { 3.0 + 4.0 * self.shell.energy + 2.5 * wiggle * self.fx_k() };
                let moving = self.appearance.visual.animations
                    && (wiggle > 0.0 || self.play_state == PlayState::Playing);
                let zoom = self.shell.zoomies.map_or(0.0, |at| (1.0 - at.elapsed().as_secs_f32() / 3.0).max(0.0));
                if zoom > 0.0 {
                    self.shell.animating = true;
                }
                let moving = moving || zoom > 0.0;
                let phase = if moving { t * (if night { 0.8 } else { 3.0 } + 5.0 * wiggle + 14.0 * zoom) } else { 0.0 };
                if night {
                    ui.painter().text(lr.right_top() + Vec2::new(1.0, 1.0), egui::Align2::LEFT_BOTTOM, "z",
                        egui::FontId::proportional(10.0), with_alpha(accent, 0.7));
                }
                if moving && wiggle > 0.0 {
                    self.shell.animating = true;
                }
                // Seven clicks and it loops the loop.
                let (sn, cs) = self.spin_angle("logo").sin_cos();
                let lc = lr.center();
                let pts: Vec<Pos2> = (0..=24).map(|i| {
                    let f = i as f32 / 24.0;
                    let v = Vec2::new((f - 0.5) * lr.width(), (f * std::f32::consts::TAU * 1.5 - phase).sin() * amp);
                    lc + Vec2::new(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
                }).collect();
                ui.painter().add(egui::Shape::line(pts, Stroke::new(2.2, accent)));
                if logo.clicked() {
                    self.shell.fx_handled.push(logo.id);
                    self.burst(lr.center(), accent, 40, 1.3, None);
                    let now = Instant::now();
                    self.shell.logo_clicks.retain(|t| now.duration_since(*t).as_secs_f32() < 3.0);
                    self.shell.logo_clicks.push(now);
                    if self.shell.logo_clicks.len() >= 7 && self.toy("logo_loop") {
                        self.shell.logo_clicks.clear();
                        self.found("logo_loop");
                        self.shell.spins.insert("logo", (now, 1.0));
                    }
                }
                if width > 780.0 {
                    ui.label(RichText::new("Moosik").size(15.0).strong().color(tok.ink));
                }
                ui.add_space(18.0);
                self.mode_switch(ui, tok);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dark = self.appearance.theme.is_dark();
                    let (icon, tip) = if dark { (Icon::Sun, "Light mode") } else { (Icon::Moon, "Dark mode") };
                    if self.fx_round(ui, "theme", icon, tip, 34.0, tok.chip, tok.chip, tok.ink, true).on_hover_text(tip).clicked() {
                        self.shell.spins.insert("theme", (Instant::now(), 1.0));
                        if self.toy("sky") {
                            self.found("sky");
                            self.shell.sky = Some((Instant::now(), !dark, ui.min_rect().right_center()));
                        }
                        self.appearance.theme = if dark { ThemeMode::Light } else { ThemeMode::Dark };
                        apply_theme(ui.ctx(), !dark);
                        save_appearance(&moosik_dir(), &self.appearance);
                    }
                    self.visual_settings_menu(ui, tok);
                    if let Some(i) = self.current_index.filter(|_| width > 900.0) {
                        let t = &self.playlist[i];
                        let fmt = match (t.sample_rate, t.bit_depth) {
                            (Some(sr), Some(b)) => format!("{} · {}-bit", fmt_rate(sr), b),
                            (Some(sr), None) => fmt_rate(sr),
                            _ => String::new(),
                        };
                        if !fmt.is_empty() {
                            chip(ui, &fmt, tok.chip, tok.dim);
                        }
                    }
                    if let Some(p) = self.engine.as_ref().map(|e| e.bp_state.presentation()) {
                        let exact = matches!(p.badge, crate::bitperfect::state::Badge::PayloadExact);
                        let (bg, fg) = if exact { (tok.ok_bg, tok.ok) } else { (tok.chip, tok.dim) };
                        use crate::bitperfect::state::Badge;
                        let short = match p.badge {
                            Badge::Off => "Bit-perfect off",
                            Badge::Requested => "Bit-perfect requested",
                            Badge::Pending => "Verifying output…",
                            Badge::PayloadExact => "Bit-perfect",
                            Badge::ValueExact => "Value-exact",
                            _ => "Processed",
                        };
                        // A button, and it looks like one: a chevron and an
                        // edge. Behind it the switch, the full output report
                        // and the route.
                        let badge = ui.add(egui::Button::new(RichText::new(format!("{short}  ▾")).size(11.5).color(fg).monospace())
                            .fill(bg).stroke(Stroke::new(1.0, with_alpha(fg, 0.35))).corner_radius(7))
                            .on_hover_text("Bit-perfect output: switch, report and device")
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        let pid = Id::new("shell_output_report");
                        if badge.clicked() {
                            ui.memory_mut(|m| m.toggle_popup(pid));
                        }
                        egui::popup_below_widget(ui, pid, &badge, egui::PopupCloseBehavior::CloseOnClickOutside, |ui| {
                            ui.set_min_width(360.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Bit-perfect output").strong().size(14.0));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    let dsd = self.dsd_active();
                                    // The switch is the request; the badge beside it says what landed.
                                    let on = self.bit_perfect;
                                    let sw = switch(ui, on, !dsd, accent, "Bit-perfect output");
                                    let sw = if dsd { sw.on_hover_text("DSD always takes its own route; this switch is for PCM") } else { sw };
                                    if sw.clicked() {
                                        self.toggle_bit_perfect();
                                        self.shell.gold_armed = (!on).then(Instant::now);
                                    }
                                });
                            });
                            // The device list can be long; the window may not be.
                            let max_h = (ui.ctx().screen_rect().height() - 140.0).max(160.0);
                            egui::ScrollArea::vertical().max_height(max_h).show(ui, |ui| {
                                self.status_lines_ui(ui);
                                ui.separator();
                                self.bp_route_ui(ui);
                            });
                        });
                        // Switched on and landed exact: a flash of gold.
                        let gold = Color32::from_rgb(255, 200, 70);
                        if exact && !self.shell.was_exact && self.toy("gold")
                            && self.shell.gold_armed.take().is_some_and(|t| t.elapsed().as_secs() < 15)
                        {
                            self.found("gold");
                            self.shell.gold = Some(Instant::now());
                            self.burst(badge.rect.center(), gold, 40, 0.9, None);
                        }
                        self.shell.was_exact = exact;
                        if let Some(at) = self.shell.gold {
                            let t = at.elapsed().as_secs_f32() / 1.2;
                            if t >= 1.0 {
                                self.shell.gold = None;
                            } else {
                                let r = badge.rect;
                                ui.painter().rect_filled(r.expand(3.0 * t), 7.0, with_alpha(gold, 0.45 * (1.0 - t)));
                                // A sheen sweeps across it.
                                let x = r.left() - 10.0 + (r.width() + 20.0) * t;
                                ui.painter().with_clip_rect(r).line_segment([Pos2::new(x - 6.0, r.bottom()), Pos2::new(x + 6.0, r.top())],
                                    Stroke::new(4.0, with_alpha(Color32::WHITE, 0.7 * (1.0 - t))));
                                self.shell.animating = true;
                            }
                        }
                    }
                });
            });
        });
    }

    /// Everything that is not playing music, in a drawer from the right:
    /// sections in a column, read top to bottom, instead of one long row.
    fn settings_drawer(&mut self, ctx: &egui::Context, tok: &Tokens) {
        if self.shell.more_open && ctx.memory(|m| m.focused().is_none()) && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.shell.more_open = false;
        }
        let frame = egui::Frame::NONE.fill(tok.glass).stroke(Stroke::new(1.0, tok.line))
            .inner_margin(Margin { left: 16, right: 10, top: 12, bottom: 12 });
        let width = (ctx.screen_rect().width() * 0.4).clamp(300.0, 380.0);
        egui::SidePanel::right("shell_settings").frame(frame).resizable(false).exact_width(width)
            .show_separator_line(false)
            .show_animated(ctx, self.shell.more_open, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Settings").size(18.0).strong().color(tok.ink));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new(RichText::new("✕").size(14.0)).frame(false))
                            .on_hover_text("Close (Esc)").clicked()
                        {
                            self.shell.more_open = false;
                        }
                    });
                });
                ui.add_space(8.0);
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_width(ui.available_width() - 6.0);
                    // The player bar holds these when there is room for them.
                    if shell_compact(ui.ctx().screen_rect().width()) {
                        settings_section(ui, tok, "Playback", |ui| {
                            ui.horizontal_wrapped(|ui| {
                                self.tb_transport(ui, false);
                                self.tb_ab(ui);
                                self.tb_bookmarks(ui);
                                self.tb_sleep(ui);
                            });
                        });
                    }
                    settings_section(ui, tok, "Output", |ui| {
                        ui.horizontal_wrapped(|ui| self.tb_bit_perfect(ui));
                        ui.label(RichText::new("The badge in the header has the full report.").size(11.0).color(tok.faint));
                    });
                    settings_section(ui, tok, "Loudness", |ui| self.rg_ui(ui));
                    settings_section(ui, tok, "Visuals and motion", |ui| self.visual_settings_ui(ui));
                    settings_section(ui, tok, "Silly extras", |ui| self.toys_ui(ui, tok));
                    settings_section(ui, tok, "Appearance", |ui| self.look_ui(ui));
                    settings_section(ui, tok, "Track", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            self.tb_windows(ui);
                            ui.add_space(8.0);
                            self.tb_log(ui);
                        });
                    });
                    settings_section(ui, tok, "Files", |ui| {
                        ui.horizontal_wrapped(|ui| self.tb_files(ui));
                    });
                    if self.play_state != PlayState::Stopped {
                        settings_section(ui, tok, "Signal", |ui| {
                            ui.horizontal_wrapped(|ui| self.tb_format(ui));
                        });
                    }
                });
            });
    }

    fn mode_switch(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        let seg = Vec2::new(92.0, 30.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(seg.x * 3.0 + 6.0, seg.y + 6.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 11, tok.chip);
        let cur = self.appearance.visual.mode;
        let idx = UiMode::ALL.iter().position(|m| *m == cur).unwrap_or(0) as f32;
        let secs = self.shell_secs(420.0);
        let x = self.shell.tween("pill", idx, secs, ease_out_back);
        // It stretches along the way it travels, like something soft.
        let dt = ui.input(|i| i.stable_dt).max(1e-3);
        let speed = ((x - self.shell.pill_prev) / dt).abs();
        self.shell.pill_prev = x;
        let st = (speed * 0.05).min(0.35) * self.fx_k();
        let pill = Rect::from_center_size(rect.min + Vec2::new(3.0 + x * seg.x + seg.x / 2.0, 3.0 + seg.y / 2.0),
            Vec2::new(seg.x * (1.0 + st), seg.y * (1.0 - st * 0.35)));
        painter.rect_filled(pill, 8, tok.pill);
        for (i, m) in UiMode::ALL.iter().enumerate() {
            let r = Rect::from_min_size(rect.min + Vec2::new(3.0 + i as f32 * seg.x, 3.0), seg);
            let resp = ui.interact(r, Id::new(("shell_mode", i)), Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, *m == cur, m.label()));
            // Ink follows the pill as it passes, not the selection.
            let over = (1.0 - (x - i as f32).abs()).clamp(0.0, 1.0);
            let ink = lerp_color(tok.dim, tok.pill_ink, over);
            painter.text(r.center(), egui::Align2::CENTER_CENTER, m.label(),
                egui::FontId::proportional(13.0), ink);
            if resp.clicked() {
                self.set_mode(*m);
                let now = Instant::now();
                self.shell.mode_hops.retain(|t| now.duration_since(*t).as_secs_f32() < 3.0);
                self.shell.mode_hops.push(now);
                if self.shell.mode_hops.len() >= 6 && self.toy("modes") {
                    self.found("modes");
                    self.shell.mode_hops.clear();
                    self.shell.mode_dizzy = Some(now);
                }
            }
        }
        if let Some(at) = self.shell.mode_dizzy {
            let accent = self.track_accent;
            let p = ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_mode_dizzy")));
            if paint_dizzy(&p, pill.center_top() + Vec2::new(0.0, 2.0), at.elapsed().as_secs_f32(), accent) {
                self.shell.animating = true;
            } else {
                self.shell.mode_dizzy = None;
            }
        }
    }

    fn visual_settings_menu(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        ui.menu_button(RichText::new("✨").size(15.0).color(tok.ink), |ui| {
            ui.set_min_width(260.0);
            ui.label(RichText::new("Visuals").strong());
            self.visual_settings_ui(ui);
        }).response.on_hover_text("Visuals and motion");
    }

    /// The shell's look and motion switches: in the ✨ menu and in settings.
    fn visual_settings_ui(&mut self, ui: &mut egui::Ui) {
        let before = self.appearance.visual.clone();
        {
            let v = &mut self.appearance.visual;
            ui.checkbox(&mut v.animations, "Animations");
            ui.add_enabled_ui(v.animations, |ui| {
                ui.add(egui::Slider::new(&mut v.duration_scale, 0.25..=3.0).text("duration ×"));
                ui.add(egui::Slider::new(&mut v.fx_intensity, 0.0..=2.0).text("intensity"));
            });
            ui.separator();
            ui.checkbox(&mut v.ambient, "Cover backdrop");
            ui.add_enabled_ui(v.ambient, |ui| {
                ui.add(egui::Slider::new(&mut v.ambient_strength, 0.0..=1.0).text("strength"));
                ui.checkbox(&mut v.ambient_drift, "Drift while playing");
            });
            ui.checkbox(&mut v.cover_glow, "Cover glow");
            ui.checkbox(&mut v.beat_pulse, "Follow the beat");
            ui.checkbox(&mut v.ribbon, "Spectrum ribbon (Listen)");
            ui.checkbox(&mut v.lyrics_over_spectrum, "Lyrics over the spectrum (Analyze)");
            ui.add_enabled_ui(v.lyrics_over_spectrum, |ui| {
                ui.add(egui::Slider::new(&mut v.lyrics_overlay_opacity, 0.1..=1.0).text("opacity"));
            });
            ui.separator();
            ui.checkbox(&mut v.button_fx, "Buttons spring and ripple");
            ui.checkbox(&mut v.confetti, "Confetti from buttons");
        }
        if self.appearance.visual != before {
            // Turning confetti off earns one last, bigger round of it.
            if before.confetti && !self.appearance.visual.confetti {
                self.celebrate(ui.ctx(), self.track_accent, true);
            }
            save_appearance(&moosik_dir(), &self.appearance);
        }
    }

    fn shell_player_bar(&mut self, ctx: &egui::Context, tok: &Tokens, accent: Color32, mode: UiMode) {
        let frame = egui::Frame::NONE.fill(tok.glass).stroke(Stroke::new(1.0, tok.line))
            .inner_margin(Margin { left: 20, right: 20, top: 8, bottom: 10 });
        egui::TopBottomPanel::bottom("shell_bar").exact_height(134.0).frame(frame).show_separator_line(false).show(ctx, |ui| {
            let full = ui.max_rect();
            // Left: the cover's Analyze slot, then title and artist.
            let slot = Rect::from_min_size(Pos2::new(full.left(), full.bottom() - 59.0), Vec2::splat(56.0));
            if mode == UiMode::Analyze {
                self.shell.cover_target = Some((slot, 9.0));
            }
            let move_secs = self.shell_secs(520.0);
            let pad = self.shell.tween("bar_pad", if mode == UiMode::Analyze { 72.0 } else { 0.0 }, move_secs, ease_out_quart);
            // Seek row across the top of the bar.
            let seek = Rect::from_min_max(full.min, Pos2::new(full.right(), full.top() + 50.0));
            let mut seek_ui = ui.new_child(egui::UiBuilder::new().max_rect(seek));
            self.seek_bar_ui(&mut seek_ui);
            self.seek_toys(ctx, seek);
            // Below the seek row with a gap, so the play button's ring never covers
            // the loudness preview.
            let row = Rect::from_min_max(Pos2::new(full.left(), full.bottom() - 62.0), full.max);
            let mut row_ui = ui.new_child(egui::UiBuilder::new().max_rect(row).layout(egui::Layout::left_to_right(egui::Align::Center)));
            let ui = &mut row_ui;
            ui.add_space(pad);
            let centre = full.center().x;
            let text_w = (centre - 112.0 - full.left() - pad - 12.0).min(360.0 - pad).max(40.0);
            ui.allocate_ui(Vec2::new(text_w, 56.0), |ui| {
                ui.set_width(text_w);
                ui.vertical(|ui| {
                    ui.add_space(8.0);
                    if let Some(i) = self.current_index {
                        let t = &self.playlist[i];
                        ui.add(egui::Label::new(RichText::new(&t.title).font(crate::name_font(&t.title, 15.0)).strong().color(tok.ink)).truncate());
                        ui.add(egui::Label::new(RichText::new(&t.artist).font(crate::name_font(&t.artist, 12.0)).color(tok.dim)).truncate());
                    } else {
                        ui.label(RichText::new("Nothing playing").size(14.0).color(tok.faint));
                    }
                });
            });
            let cur = ui.cursor().min.x;
            ui.add_space((centre - 112.0 - cur).max(8.0));
            let hover = tok.chip;
            if self.fx_round(ui, "prev", Icon::Prev, "Previous", 44.0, Color32::TRANSPARENT, hover, tok.ink, true)
                .on_hover_text("Previous").clicked() {
                self.burst_at_pointer(ui.ctx(), accent, 10, 0.7, Some(-1.0));
                self.prev_track();
            }
            let playing = self.play_state == PlayState::Playing;
            let ring = if playing { 2.0 + 3.0 * self.shell.energy } else { 0.0 };
            let (icon, name) = if playing { (Icon::Pause, "Pause") } else { (Icon::Play, "Play") };
            let play = self.fx_round(ui, "play", icon, name, 50.0, tok.ink, tok.ink, tok.bg, true)
                .on_hover_text(name);
            if ring > 0.0 {
                ui.painter().circle_stroke(play.rect.center(), 25.0 + ring * 0.5, Stroke::new(ring, with_alpha(accent, 0.25)));
            }
            if play.clicked() {
                self.burst(play.rect.center(), accent, 26, 1.0, None);
                self.toggle_play_pause();
                // Ten presses inside three seconds: it spins twice and throws
                // a party, for whoever is testing whether it notices.
                let now = Instant::now();
                self.shell.play_presses.retain(|t| now.duration_since(*t).as_secs_f32() < 3.0);
                self.shell.play_presses.push(now);
                if self.shell.play_presses.len() >= 10 && self.toy("hammer") {
                    self.found("hammer");
                    self.shell.play_presses.clear();
                    self.shell.spins.insert("play", (now, 2.0));
                    self.celebrate(ui.ctx(), accent, false);
                }
            }
            self.charge_play(ui, &play, accent);
            if self.fx_round(ui, "next", Icon::Next, "Next", 44.0, Color32::TRANSPARENT, hover, tok.ink, true)
                .on_hover_text("Next").clicked() {
                self.burst_at_pointer(ui.ctx(), accent, 10, 0.7, Some(1.0));
                self.skip_next();
            }
            // Stop closes the output stream, which in exclusive mode is the
            // only way to hand the device back to other programs.
            let stopped = self.play_state == PlayState::Stopped;
            if self.fx_round(ui, "stop", Icon::Stop, "Stop", 40.0, Color32::TRANSPARENT, hover, tok.dim, !stopped)
                .on_hover_text("Stop (releases the output device)").clicked() {
                self.burst_at_pointer(ui.ctx(), accent, 8, 0.5, None);
                self.stop();
            }
            let compact = shell_compact(full.width());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let on = with_alpha(accent, 0.22);
                let more_fill = if self.shell.more_open { tok.chip } else { Color32::TRANSPARENT };
                if self.fx_round(ui, "more", Icon::More, "All controls", 34.0, more_fill, hover, tok.ink, true)
                    .on_hover_text("Settings and everything else").clicked() {
                    self.shell.more_open = !self.shell.more_open;
                }
                // Same rule as classic: no gain stage on an exact route, so the
                // slider shows 100% and stays put.
                let (locked, why) = self.engine.as_ref()
                    .map(|e| (e.bp_state.volume_locked(), e.bp_state.volume_lock_reason()))
                    .unwrap_or((false, None));
                let mut vol = if locked { 1.0 } else { self.volume };
                ui.spacing_mut().slider_width = if compact { 64.0 } else { 96.0 };
                let slider = ui.add_enabled(!locked, egui::Slider::new(&mut vol, 0.0..=1.0).show_value(false));
                let (slider_rect, slider_hot) = (slider.rect, slider.hovered() || slider.dragged());
                // Too short to land on every percent by dragging; the wheel steps by one.
                let wheel = if slider.hovered() { ui.input(|i| i.raw_scroll_delta.y) } else { 0.0 };
                if !locked && wheel != 0.0 {
                    let v = ((vol * 100.0).round() + wheel.signum()).clamp(0.0, 100.0) / 100.0;
                    if self.engine.as_mut().is_some_and(|e| e.set_volume(v)) {
                        self.volume = v;
                    }
                }
                if locked {
                    slider.on_disabled_hover_text(why.unwrap_or_else(|| "Volume is locked on this route".into()));
                } else if slider.changed() && self.engine.as_mut().is_some_and(|e| e.set_volume(vol)) {
                    self.volume = vol;
                }
                let (r, spk) = ui.allocate_exact_size(Vec2::splat(26.0), Sense::hover());
                // Pointed at or dragged, the slider says how loud.
                if !locked && (slider_hot || spk.hovered()) {
                    ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_vol_pct")))
                        .text(slider_rect.center_top() - Vec2::new(0.0, 3.0), egui::Align2::CENTER_BOTTOM,
                            format!("{}%", (vol * 100.0).round()), egui::FontId::proportional(11.0), tok.ink);
                }
                // Its face is small; pointed at, it leans in to show it.
                let peek = ui.ctx().animate_bool_with_time(spk.id, spk.hovered(), self.shell_secs(180.0));
                // At zero the speaker faints: it keels over, and stars go round
                // its head for a while.
                let out = vol <= 0.0 && self.toy("faint");
                if out {
                    self.found("faint");
                }
                if out && self.shell.fainted_at.is_none() {
                    self.shell.fainted_at = Some(Instant::now());
                } else if !out {
                    self.shell.fainted_at = None;
                }
                let fall = self.shell.tween("faint", if out { 1.0 } else { 0.0 }, self.shell_secs(650.0), ease_out_back);
                let toys = self.toy("face");
                let shades = self.toy("shades") && vol >= 0.995;
                if toys && spk.hovered() {
                    self.found("face");
                }
                if shades {
                    self.found("shades");
                }
                // It sags as the volume goes down, then keels over at zero.
                let sag = if toys { (1.0 - vol).powf(1.5) } else { 0.0 };
                let c = r.center() + Vec2::new(0.0, 2.0 * sag + 3.0 * fall);
                let ang = 0.35 * sag * (1.0 - fall) - fall * std::f32::consts::FRAC_PI_2;
                let size = 22.0 * (1.0 + if toys { 0.8 * peek } else { 0.0 });
                paint_speaker(ui.painter(), c - Vec2::new(0.0, 10.0 * peek * f32::from(toys)), size, vol, toys, shades, tok.dim, tok.bg, ang);
                // On the right days, the speaker dresses up.
                let (month, day, _) = local_time();
                let hat = match (month, day) {
                    (10, 31) => 1,
                    (12, 24..=26) => 2,
                    _ => 0,
                };
                if hat != 0 && self.toy("seasons") && fall < 0.5 {
                    self.found("seasons");
                    let top = c - Vec2::new(0.0, 10.0 * peek * f32::from(toys) + size * 0.42);
                    let hp = ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_hat")));
                    if hat == 1 {
                        // A pumpkin, with its stalk.
                        let orange = Color32::from_rgb(242, 140, 40);
                        hp.add(egui::Shape::ellipse_filled(top, Vec2::new(5.5, 4.2), orange));
                        hp.add(egui::Shape::ellipse_stroke(top, Vec2::new(2.2, 4.2), Stroke::new(0.8, Color32::from_rgb(190, 95, 20))));
                        hp.line_segment([top - Vec2::new(0.0, 4.0), top + Vec2::new(1.5, -7.0)], Stroke::new(1.6, Color32::from_rgb(80, 130, 60)));
                    } else {
                        // A Santa hat, flopping to one side.
                        let red = Color32::from_rgb(214, 48, 49);
                        hp.add(egui::Shape::convex_polygon(vec![top + Vec2::new(-6.0, 1.0), top + Vec2::new(6.0, 1.0), top + Vec2::new(5.0, -9.0)],
                            red, Stroke::NONE));
                        hp.rect_filled(Rect::from_center_size(top + Vec2::new(0.0, 1.5), Vec2::new(13.0, 3.0)), 1, Color32::WHITE);
                        hp.circle_filled(top + Vec2::new(6.0, -9.0), 1.8, Color32::WHITE);
                    }
                }
                let pct = if locked { 0 } else { (vol * 100.0).round() as u8 };
                let meme = match pct {
                    67 if self.toy("six_seven") => 67,
                    69 if self.toy("nice") => 69,
                    _ => 0,
                };
                if meme != 0 && meme != self.shell.vol_meme
                    && !self.shell.memes.iter().any(|(_, _, m)| *m == meme)
                {
                    if meme == 69 {
                        self.found("nice");
                    }
                    self.shell.memes.push((r.center_top() - Vec2::new(0.0, 8.0), Instant::now(), meme));
                    self.shell.vol_meme = meme;
                }
                // Gone off once, it waits for the slider to leave the
                // neighbourhood: dragging to and fro over 67 does it once.
                if !(65..=71).contains(&pct) {
                    self.shell.vol_meme = 0;
                }
                if let Some(at) = self.shell.fainted_at
                    && paint_dizzy(ui.painter(), r.center(), at.elapsed().as_secs_f32(), accent)
                {
                    self.shell.animating = true;
                }
                ui.add_space(10.0);
                if compact {
                    return;
                }
                // Sleep, bookmarks and A-B, then the loop mode.
                let sleep_on = self.sleep_deadline.is_some() || self.sleep_end_of_track;
                let sleep = self.fx_round(ui, "sleep", Icon::Sleep, "Sleep timer", 34.0,
                    if sleep_on { on } else { Color32::TRANSPARENT }, hover, if sleep_on { accent } else { tok.ink }, true);
                let sleep_tip = match self.sleep_deadline {
                    Some(dl) => format!("Sleep in {}", Self::format_duration(dl.saturating_duration_since(Instant::now()))),
                    None if self.sleep_end_of_track => "Sleep at the end of this track".into(),
                    None => "Sleep timer".into(),
                };
                let sleep = sleep.on_hover_text(sleep_tip);
                let left = match self.sleep_deadline {
                    Some(dl) => Some(dl.saturating_duration_since(Instant::now())),
                    None if self.sleep_end_of_track && self.play_state == PlayState::Playing =>
                        self.current_duration().map(|d| d.saturating_sub(self.elapsed())),
                    None => None,
                };
                if self.toy("yawn") && left.is_some_and(|l| l.as_secs_f32() < 10.0) {
                    self.found("yawn");
                    // Nearly bedtime: the moon yawns.
                    let t = ui.input(|i| i.time) as f32;
                    let open = (t * std::f32::consts::PI / 2.5).sin().max(0.0).powf(0.7);
                    let m = sleep.rect.center() + Vec2::new(-2.8, 2.6);
                    ui.painter().add(egui::Shape::ellipse_filled(m, Vec2::new(1.8, 2.6) * open, tok.bg));
                    // And z's drift up off it.
                    let zp = ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_zzz")));
                    for k in 0..3 {
                        let f = (t * 0.5 + k as f32 / 3.0).fract();
                        let q = sleep.rect.center_top() + Vec2::new(6.0 + 8.0 * f + 2.0 * (f * 9.0).sin(), -4.0 - 26.0 * f);
                        zp.text(q, egui::Align2::CENTER_CENTER, "z", egui::FontId::proportional(9.0 + 6.0 * f),
                            with_alpha(accent, (1.0 - f) * (f * 6.0).min(1.0)));
                    }
                    self.shell.animating = true;
                }
                self.bar_popup(ui, "sleep", &sleep, true, |app, ui| app.sleep_ui(ui));
                let n_marks = self.current_index
                    .and_then(|i| self.bookmarks.for_track(&self.playlist[i].path))
                    .map_or(0, |m| m.len());
                let has_marks = n_marks > 0;
                let marks = self.fx_round(ui, "marks", Icon::Bookmark, "Bookmarks", 34.0,
                    if has_marks { on } else { Color32::TRANSPARENT }, hover, if has_marks { accent } else { tok.ink }, true)
                    .on_hover_text("Bookmarks for this track");
                if self.toy("bouquet") && n_marks >= 10 {
                    self.found("bouquet");
                    // Ten or more, and it grows a whole bunch of them.
                    let t = ui.input(|i| i.time) as f32;
                    let m = (n_marks - 9).min(9);
                    let p = ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_marks")));
                    for i in 0..m {
                        let f = if m == 1 { 0.5 } else { i as f32 / (m - 1) as f32 };
                        let a = -0.9 + 1.8 * f + 0.06 * (t * 2.0 + i as f32).sin();
                        paint_icon(&p, marks.rect.center() + Vec2::new(a.sin(), -a.cos()) * 17.0, 11.0, Icon::Bookmark, accent, a);
                    }
                }
                self.bar_popup(ui, "marks", &marks, false, |app, ui| app.bookmarks_ui(ui));
                let ab_on = self.ab_a.is_some();
                let ab_tip = match (self.ab_a, self.ab_b) {
                    (None, _) => "A-B repeat: set point A".to_string(),
                    (Some(a), None) => format!("A = {}: click to set B", Self::format_duration(a)),
                    (Some(a), Some(b)) => format!("Looping {}–{}: click to clear",
                        Self::format_duration(a), Self::format_duration(b)),
                };
                let ab = self.fx_round(ui, "ab", Icon::Ab, "A-B repeat", 34.0,
                    if ab_on { on } else { Color32::TRANSPARENT }, hover, if ab_on { accent } else { tok.ink },
                    self.current_index.is_some())
                    .on_hover_text(ab_tip);
                if ab.clicked() {
                    self.cycle_ab_repeat();
                }
                if self.ab_b.is_none() {
                    self.shell.ab_laps = 0;
                }
                if self.shell.ab_laps >= 10 && self.toy("ab_dizzy") {
                    self.found("ab_dizzy");
                    // Ten times round the same bars: dizzy.
                    self.shell.ab_laps = 0;
                    self.shell.spins.insert("ab", (Instant::now(), 3.0));
                    self.shell.ab_dizzy = Some(Instant::now());
                }
                if let Some(at) = self.shell.ab_dizzy {
                    if paint_dizzy(ui.painter(), ab.rect.center() - Vec2::new(0.0, 4.0), at.elapsed().as_secs_f32(), accent) {
                        self.shell.animating = true;
                    } else {
                        self.shell.ab_dizzy = None;
                    }
                }
                let (icon, tip, lit) = match self.loop_mode {
                    LoopMode::Sequential => (Icon::Sequential, "In order, stop at the end (click: repeat all)", false),
                    LoopMode::RepeatAll => (Icon::Repeat, "Repeat all (click: repeat one)", true),
                    LoopMode::RepeatOne => (Icon::RepeatOne, "Repeat one (click: in order)", true),
                };
                if self.fx_round(ui, "loop", icon, "Loop mode", 34.0,
                    if lit { on } else { Color32::TRANSPARENT }, hover, if lit { accent } else { tok.ink }, true)
                    .on_hover_text(tip).clicked() {
                    self.cycle_loop_mode();
                }
            });
        });
    }

    fn listen_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        let area = ui.max_rect().shrink2(Vec2::new(56.0, 28.0)).translate(self.shell.sway);
        let side = (area.height() - 150.0).min(area.width() * 0.40).min(480.0).max(96.0);
        let cover = Rect::from_min_size(area.min, Vec2::splat(side));
        self.shell.cover_target = Some((cover, 18.0));
        // Title, artist and the ribbon under the cover.
        let text = Rect::from_min_max(Pos2::new(cover.left(), cover.bottom() + 20.0), Pos2::new(cover.right(), area.bottom()));
        let mut tui = ui.new_child(egui::UiBuilder::new().max_rect(text));
        if let Some(i) = self.current_index {
            let t = &self.playlist[i];
            tui.add(egui::Label::new(RichText::new(&t.title).font(crate::name_font(&t.title, 30.0)).strong().color(tok.ink)).truncate());
            tui.add(egui::Label::new(RichText::new(format!("{} · {}", t.artist, t.album)).size(15.0).color(tok.dim)).truncate());
        }
        if self.appearance.visual.ribbon {
            let r = Rect::from_min_size(Pos2::new(text.left(), tui.cursor().min.y + 12.0), Vec2::new(cover.width(), 56.0));
            if r.bottom() <= area.bottom() + 28.0 {
                self.paint_ribbon(ui, r, accent);
            }
        }
        // Lyrics card on the right.
        let card = Rect::from_min_max(Pos2::new(cover.right() + 56.0, area.top()), area.max);
        if card.width() < 240.0 {
            return;
        }
        ui.painter().rect(card, 20, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        let inner = card.shrink2(Vec2::new(36.0, 28.0));
        let mut lui = ui.new_child(egui::UiBuilder::new().max_rect(inner));
        let pos = self.elapsed();
        if let Some(i) = self.current_index {
            let t = &self.playlist[i];
            let tref = lyrics::ui::TrackRef { path: &t.path, title: &t.title, artist: &t.artist, album: &t.album, duration: t.duration };
            if let Some(r) = self.lyrics_window.inline_viewer(&mut lui, tref, pos)
                && self.toy("balloon")
            {
                self.found("balloon");
                self.shell.balloon = Some((r, Instant::now()));
            }
        }
    }

    fn paint_ribbon(&mut self, ui: &egui::Ui, rect: Rect, accent: Color32) {
        let m = &self.spectrum_window.analyzer.magnitudes;
        if m.is_empty() {
            return;
        }
        let n = 64usize;
        let bw = rect.width() / n as f32;
        let mid = rect.center().y;
        let painter = ui.painter_at(rect.expand(8.0));
        for i in 0..n {
            let a = i * m.len() / n;
            let b = ((i + 1) * m.len() / n).max(a + 1).min(m.len());
            let v = m[a..b].iter().fold(0.0f32, |acc, x| acc.max(*x)).clamp(0.0, 1.0);
            let h = (v * v * rect.height() * 0.62).max(1.5);
            let x = rect.left() + i as f32 * bw;
            let bar = Rect::from_min_max(Pos2::new(x + 1.5, mid - h), Pos2::new(x + bw - 1.5, mid + h));
            painter.rect_filled(bar.expand(2.0), 3, with_alpha(accent, 0.12 * v));
            painter.rect_filled(bar, 2, with_alpha(accent, 0.35 + 0.65 * v));
        }
    }

    fn library_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        let area = ui.max_rect().shrink2(Vec2::new(20.0, 16.0));
        // A narrow window drops the inspector first, then the sidebar.
        let side_w = if area.width() < 980.0 { 0.0 } else { 210.0 };
        let inspector_w = if area.width() < 760.0 { 0.0 } else { 300.0 };
        let gap = 16.0;
        let side = Rect::from_min_size(area.min, Vec2::new(side_w, area.height()));
        let left = if side_w > 0.0 { side.right() + gap } else { area.left() };
        let right = if inspector_w > 0.0 { area.right() - inspector_w - gap } else { area.right() };
        let list = Rect::from_min_max(Pos2::new(left, area.top()), Pos2::new(right, area.bottom()));
        let insp = Rect::from_min_max(Pos2::new(right + gap, area.top()), area.max);
        self.shell.list_rect = Some(list);
        if side_w > 0.0 {
            ui.painter().rect(side, 18, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
            let mut sui = ui.new_child(egui::UiBuilder::new().max_rect(side.shrink(12.0)));
            self.playlist_sidebar(&mut sui, tok, accent);
        }
        ui.painter().rect(list, 18, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        let mut pui = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(14.0)));
        // With the sidebar, the saved playlists live there, not above the list.
        if side_w > 0.0 {
            self.playlist_list_ui(&mut pui);
        } else {
            self.playlist_ui(&mut pui);
        }
        if self.playlist.is_empty() {
            self.empty_box(ui, list, tok, accent);
        }
        if self.play_state == PlayState::Playing && self.toy("row_eq") {
            self.found("row_eq");
        }
        if inspector_w > 0.0 {
            ui.painter().rect(insp, 18, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
            self.inspector_ui(ui, insp.shrink(20.0), tok, accent);
        }
    }

    /// The saved playlists, one row each, under the queue.
    fn playlist_sidebar(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        ui.label(RichText::new("PLAYLISTS").size(11.0).strong().color(tok.dim).extra_letter_spacing(0.8));
        ui.add_space(6.0);
        let spin = (self.play_state == PlayState::Playing).then(|| ui.input(|i| i.time) as f32 * 3.0);
        let mut pick: Option<Option<usize>> = None;
        let mut active_row = None;
        let below = if self.show_save_playlist_input { 120.0 } else { 50.0 };
        egui::ScrollArea::vertical().id_salt("shell_playlists").auto_shrink([false, true])
            .max_height((ui.available_height() - below).max(60.0)).show(ui, |ui| {
                for row in 0..=self.playlist_store.len() {
                    let idx = row.checked_sub(1);
                    let (name, count) = match idx {
                        None => ("Current queue".to_string(), self.playlist.len()),
                        Some(i) => (self.playlist_store[i].name.clone(), self.playlist_store[i].paths.len()),
                    };
                    let active = self.active_saved_playlist == idx;
                    let (r, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::click());
                    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, active, &name));
                    let fill = if active { with_alpha(accent, 0.2) } else if resp.hovered() { tok.chip } else { Color32::TRANSPARENT };
                    let p = ui.painter_at(r);
                    p.rect_filled(r, 8, fill);
                    cassette(&p, Pos2::new(r.left() + 18.0, r.center().y), if active { accent } else { tok.dim },
                        if active { spin } else { None });
                    // The answer to everything gets a star.
                    let n = match count {
                        42 if self.toy("answer") => "42 ✦".to_string(),
                        67 if self.toy("six_seven") => "6 7".to_string(),
                        69 if self.toy("nice") => "69 · nice".to_string(),
                        _ => count.to_string(),
                    };
                    let n = p.text(Pos2::new(r.right() - 10.0, r.center().y), egui::Align2::RIGHT_CENTER, n,
                        egui::FontId::proportional(11.0), tok.faint);
                    let mut job = egui::text::LayoutJob::simple_singleline(name.clone(), egui::FontId::proportional(13.0), tok.ink);
                    job.wrap = egui::text::TextWrapping::truncate_at_width(n.left() - 6.0 - (r.left() + 36.0));
                    let g = ui.fonts(|f| f.layout_job(job));
                    p.galley(Pos2::new(r.left() + 36.0, r.center().y - g.size().y / 2.0), g, tok.ink);
                    let resp = if count == 42 && self.toy("answer") { resp.on_hover_text("The answer") } else { resp };
                    if count == 42 && idx.is_some() && self.toy("answer") {
                        self.found("answer");
                    }
                    if resp.clicked() && !active {
                        pick = Some(idx);
                    }
                    if active {
                        active_row = Some((r, name));
                    }
                }
            });
        match pick {
            Some(Some(i)) => self.load_saved_playlist(i),
            Some(None) => self.active_saved_playlist = None,
            None => {}
        }
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            if !self.playlist.is_empty() && ui.button("Save as…").on_hover_text("Save the queue as a playlist").clicked() {
                self.show_save_playlist_input = !self.show_save_playlist_input;
                if self.show_save_playlist_input && self.playlist_name_buf.is_empty() {
                    self.playlist_name_buf = "My Playlist".to_string();
                }
            }
            if let Some(idx) = self.active_saved_playlist {
                if !self.playlist.is_empty() && ui.button("Update").on_hover_text("Overwrite it with the queue").clicked() {
                    self.playlist_store[idx].paths = self.playlist.iter().map(|t| t.path.clone()).collect();
                    save_playlist_store(&self.playlist_store);
                    self.burst_at_pointer(ui.ctx(), accent, 8, 0.5, None);
                }
                if ui.button(RichText::new("Delete").color(Color32::from_rgb(200, 80, 80))).on_hover_text("Delete this playlist").clicked() {
                    // Off the shelf, tumbling.
                    if let Some((r, name)) = active_row
                        && self.toy("falling")
                    {
                        self.found("falling");
                        self.shell.falling.push((r, name, Instant::now()));
                    }
                    self.playlist_store.remove(idx);
                    self.active_saved_playlist = None;
                    save_playlist_store(&self.playlist_store);
                }
            }
        });
        if self.show_save_playlist_input {
            ui.add_space(6.0);
            let te = ui.add(egui::TextEdit::singleline(&mut self.playlist_name_buf).desired_width(f32::INFINITY));
            let mut save = te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                save |= ui.button("Save").clicked();
                if ui.button("Cancel").clicked() {
                    self.show_save_playlist_input = false;
                }
            });
            if save {
                let before = self.playlist_store.len();
                self.save_named_playlist();
                if self.playlist_store.len() > before {
                    self.burst(te.rect.center(), accent, 24, 0.9, None);
                }
            }
        }
    }

    /// The inspector: the playing track, or the queue in numbers.
    fn inspector_ui(&mut self, ui: &mut egui::Ui, inner: Rect, tok: &Tokens, accent: Color32) {
        let mut rects = [Rect::NOTHING; 2];
        let mut tui = ui.new_child(egui::UiBuilder::new().max_rect(inner));
        tui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            for (i, name) in ["Now playing", "Queue"].into_iter().enumerate() {
                let on = self.shell.insp_tab == i;
                let r = ui.add(egui::Button::new(RichText::new(name).size(13.0).strong()
                    .color(if on { tok.ink } else { tok.dim })).frame(false));
                if r.clicked() {
                    self.shell.insp_tab = i;
                }
                rects[i] = r.rect;
            }
        });
        // The underline crawls over like an inchworm: front end first.
        let target = rects[self.shell.insp_tab];
        let (mut a, mut b) = self.shell.insp_ul;
        if b <= a || self.fx_k() == 0.0 {
            (a, b) = (target.left(), target.right());
        }
        let dt = ui.input(|i| i.stable_dt).min(0.05);
        let (fast, slow) = (1.0 - (-dt * 24.0).exp(), 1.0 - (-dt * 9.0).exp());
        let (fa, fb) = if target.left() < a { (fast, slow) } else { (slow, fast) };
        a += (target.left() - a) * fa;
        b += (target.right() - b) * fb;
        if (a - target.left()).abs() + (b - target.right()).abs() > 0.5 {
            self.shell.animating = true;
        } else {
            (a, b) = (target.left(), target.right());
        }
        self.shell.insp_ul = (a, b);
        let y = target.bottom() + 3.0;
        ui.painter().rect_filled(Rect::from_min_max(Pos2::new(a, y), Pos2::new(b, y + 2.5)), 2, accent);

        let body = Rect::from_min_max(Pos2::new(inner.left(), y + 16.0), inner.max);
        let mut bui = ui.new_child(egui::UiBuilder::new().max_rect(body));
        if self.shell.insp_tab == 0 {
            let side = body.width().min(body.height() * 0.6);
            let cover = Rect::from_min_size(body.min, Vec2::splat(side));
            self.shell.cover_target = Some((cover, 12.0));
            bui.add_space(side + 16.0);
            if let Some(i) = self.current_index {
                let t = &self.playlist[i];
                bui.add(egui::Label::new(RichText::new(&t.title).font(crate::name_font(&t.title, 19.0)).strong().color(tok.ink)).wrap());
                bui.add(egui::Label::new(RichText::new(format!("{} · {}", t.artist, t.album)).size(13.0).color(tok.dim)).wrap());
                let mut bits = vec![t.codec.to_string()];
                if let Some(sr) = t.sample_rate {
                    bits.push(format!("{:.1} kHz", sr as f32 / 1000.0));
                }
                if let Some(d) = t.bit_depth {
                    bits.push(format!("{d}-bit"));
                }
                let plays = self.stats.get(&t.path).map(|s| s.count).unwrap_or(0);
                if plays > 0 {
                    bits.push(format!("played {plays}×"));
                }
                bui.add_space(4.0);
                bui.label(RichText::new(bits.join(" · ")).size(12.0).color(tok.faint));
            }
            return;
        }
        // The queue in numbers. ponytail: recounted every frame it shows;
        // cache it if a huge queue makes this tab slow.
        let n = self.playlist.len();
        if n == 0 {
            bui.label(RichText::new("Nothing queued yet.").size(13.0).color(tok.dim));
            return;
        }
        let secs: f32 = self.playlist.iter().filter_map(|t| t.duration).map(|d| d.as_secs_f32()).sum();
        let lossless = self.playlist.iter().filter(|t| t.lossless == Some(true)).count();
        let hires = self.playlist.iter().filter(|t| t.sample_rate.is_some_and(|r| r > 48_000) || t.bit_depth.is_some_and(|b| b > 16)).count();
        let mut artists: HashMap<&str, usize> = HashMap::new();
        for t in &self.playlist {
            *artists.entry(t.artist.as_str()).or_default() += 1;
        }
        // Ties go to the name first in order, so the pick holds still.
        let top = match artists.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))) {
            Some((a, c)) if *c > 1 => format!("{a} ({c})"),
            _ => "A tie, everyone once".to_string(),
        };
        let longest = self.playlist.iter().max_by_key(|t| t.duration).map(|t| t.display_title().to_string()).unwrap_or_default();
        let mins = (secs / 60.0).round() as u32;
        let rows = [
            ("Tracks", n.to_string()),
            ("Length", if mins >= 60 { format!("{} h {} min", mins / 60, mins % 60) } else { format!("{mins} min") }),
            ("Lossless", format!("{lossless} of {n}")),
            ("Hi-res", format!("{hires} of {n}")),
            ("Top artist", top),
            ("Longest", longest),
        ];
        egui::Grid::new("shell_queue_stats").num_columns(2).spacing(Vec2::new(14.0, 8.0)).show(&mut bui, |ui| {
            for (k, v) in rows {
                ui.label(RichText::new(k).size(12.0).color(tok.dim));
                ui.add(egui::Label::new(RichText::new(v).size(13.0).color(tok.ink)).truncate());
                ui.end_row();
            }
        });
        // Four minutes a cup.
        let cups = (secs / 240.0).round() as u32;
        if cups >= 1 && self.toy("tea") {
            self.found("tea");
            bui.add_space(12.0);
            let s = if cups == 1 { "" } else { "s" };
            bui.label(RichText::new(format!("Long enough for {cups} cup{s} of tea, steeped one after another.")).size(12.0).italics().color(tok.faint));
        }
    }

    /// An empty library: a box, and a note that peeks out of it when the
    /// pointer comes near. Clicking the box adds files.
    fn empty_box(&mut self, ui: &mut egui::Ui, list: Rect, tok: &Tokens, accent: Color32) {
        let c = Pos2::new(list.center().x, (list.top() + 200.0).min(list.center().y + 40.0));
        let bx = Rect::from_center_size(c, Vec2::new(84.0, 56.0));
        let resp = ui.interact(bx.expand(10.0), Id::new("shell_empty_box"), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Add music"));
        if resp.clicked() {
            self.add_files();
        }
        let near = ui.rect_contains_pointer(bx.expand(40.0)) && self.toy("peek");
        if near {
            self.found("peek");
        }
        let want = if near { 1.0 } else { 0.0 };
        let dt = ui.input(|i| i.stable_dt).min(0.05);
        self.shell.peek += (want - self.shell.peek) * (1.0 - (-dt * 10.0).exp());
        if (want - self.shell.peek).abs() > 0.01 || near {
            self.shell.animating = true;
        } else {
            self.shell.peek = want;
        }
        let peek = self.shell.peek;
        let ink = if resp.hovered() { tok.ink } else { tok.dim };
        // The note, only above the box's rim, as if inside it.
        let wob = if near { (ui.input(|i| i.time) as f32 * 5.0).sin() * 4.0 } else { 0.0 };
        let head = Pos2::new(c.x - 4.0 + wob, bx.top() + 14.0 - 40.0 * peek);
        let np = ui.painter().with_clip_rect(Rect::from_min_max(list.min, Pos2::new(list.right(), bx.top())));
        np.add(egui::Shape::ellipse_filled(head, Vec2::new(6.5, 5.0), accent));
        let stem = head + Vec2::new(5.5, 0.0);
        np.line_segment([stem, stem - Vec2::new(0.0, 22.0)], Stroke::new(2.2, accent));
        np.line_segment([stem - Vec2::new(0.0, 22.0), stem + Vec2::new(8.0, -14.0)], Stroke::new(2.2, accent));
        // The box, its flaps opening as the note comes up.
        let p = ui.painter();
        let st = Stroke::new(1.6, ink);
        p.rect_stroke(bx, 4, st, egui::StrokeKind::Middle);
        let open = 0.35 + 0.9 * peek;
        for (corner, dir) in [(bx.left_top(), -1.0f32), (bx.right_top(), 1.0)] {
            let ang = std::f32::consts::PI * 0.5 * (1.0 - open);
            let tip = corner + Vec2::new(dir * 26.0 * ang.cos(), -26.0 * ang.sin());
            p.line_segment([corner, tip], st);
        }
        p.text(Pos2::new(c.x, bx.bottom() + 18.0), egui::Align2::CENTER_TOP, "Click the box to add music",
            egui::FontId::proportional(12.0), tok.faint);
    }

    /// Three bars dancing on the playing row, in place of its drag handle.
    pub(crate) fn shell_row_eq(&self, painter: &egui::Painter, c: Pos2) -> bool {
        if !self.appearance.visual.modern || !self.toy("row_eq") || self.play_state != PlayState::Playing {
            return false;
        }
        let t = painter.ctx().input(|i| i.time) as f32;
        let e = self.shell.energy;
        for (i, ph) in [0.0f32, 2.1, 4.2].into_iter().enumerate() {
            let h = 3.0 + 9.0 * (0.35 + 0.65 * e) * (0.5 + 0.5 * (t * 9.0 + ph).sin());
            let x = c.x - 4.0 + i as f32 * 4.0;
            painter.rect_filled(Rect::from_min_max(Pos2::new(x - 1.2, c.y + 6.0 - h), Pos2::new(x + 1.2, c.y + 6.0)), 1, self.track_accent);
        }
        true
    }

    fn analyze_ui(&mut self, ui: &mut egui::Ui, tok: &Tokens) {
        let area = ui.max_rect().shrink2(Vec2::new(12.0, 10.0));
        ui.painter().rect(area, 12, with_alpha(tok.bg, 0.92), Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        let inner = area.shrink(10.0);
        let accent = self.track_accent;
        // A strip on top: a tuning fork, and the drawer's switch.
        let strip = Rect::from_min_size(inner.min, Vec2::new(inner.width(), 30.0));
        self.tuning_fork(ui, Pos2::new(strip.left() + 12.0, strip.center().y), tok, accent);
        ui.painter().text(Pos2::new(strip.left() + 32.0, strip.center().y), egui::Align2::LEFT_CENTER, "Analyzer",
            egui::FontId::proportional(13.0), tok.dim);
        let mut x = strip.left() + 104.0;
        for (key, label, tip) in [("stereo", "Stereo", "Where the sound sits between the speakers"), ("harmony", "Harmony", "Notes, chord and key")] {
            let on = if key == "stereo" { self.appearance.visual.stereo_panel } else { self.appearance.visual.harmony_panel };
            let w = 14.0 + label.len() as f32 * 7.2;
            let r = Rect::from_min_size(Pos2::new(x, strip.top() + 3.0), Vec2::new(w, 24.0));
            x += w + 6.0;
            let resp = ui.interact(r, Id::new(("shell_panel", key)), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(tip);
            resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, on, label));
            let fill = if on { with_alpha(accent, 0.22) } else if resp.hovered() { tok.chip } else { Color32::TRANSPARENT };
            ui.painter().rect(r, 12, fill, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
            ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(12.5), if on { tok.ink } else { tok.dim });
            if resp.clicked() {
                let v = &mut self.appearance.visual;
                if key == "stereo" { v.stereo_panel = !on } else { v.harmony_panel = !on }
                save_appearance(&moosik_dir(), &self.appearance);
            }
        }
        let open = !self.shell.tools_hidden;
        let r = Rect::from_min_size(Pos2::new(strip.right() - 108.0, strip.top() + 1.0), Vec2::new(108.0, 28.0));
        let resp = ui.interact(r, Id::new("shell_tools"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(if open { "Hide the controls" } else { "Show the controls" });
        resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, open, "Analyzer controls"));
        let fill = if open { with_alpha(accent, 0.22) } else if resp.hovered() { tok.chip } else { Color32::TRANSPARENT };
        ui.painter().rect(r, 9, fill, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        // Hovered, the sliders' knobs fidget.
        let wig = if resp.hovered() && self.toy("fidget") {
            self.found("fidget");
            self.shell.animating = true;
            ui.input(|i| i.time) as f32 * 6.0
        } else {
            0.0
        };
        let ic = Pos2::new(r.left() + 20.0, r.center().y);
        for (k, base) in [(-5.0f32, -2.5f32), (0.0, 2.5), (5.0, -1.0)] {
            let y = ic.y + k;
            ui.painter().line_segment([Pos2::new(ic.x - 7.0, y), Pos2::new(ic.x + 7.0, y)], Stroke::new(1.3, tok.ink));
            let x = ic.x + base + if wig != 0.0 { 3.0 * (wig + k).sin() } else { 0.0 };
            ui.painter().circle_filled(Pos2::new(x, y), 2.4, tok.ink);
        }
        ui.painter().text(Pos2::new(r.left() + 36.0, r.center().y), egui::Align2::LEFT_CENTER, "Controls",
            egui::FontId::proportional(13.0), tok.ink);
        if resp.clicked() {
            self.shell.tools_hidden = open;
            self.shell.tools_bottom = false;
        }

        let mut body = Rect::from_min_max(Pos2::new(inner.left(), strip.bottom() + 8.0), inner.max);
        // The live panels take a band at the bottom; the plot keeps its bars,
        // and the room it needs (more when the controls sit above it).
        let need = if open && body.width() < 760.0 { 480.0 } else { 240.0 };
        let deck = if (self.appearance.visual.stereo_panel || self.appearance.visual.harmony_panel) && body.height() > need {
            let h = (body.height() * 0.42).min(172.0);
            let d = Rect::from_min_max(Pos2::new(body.left(), body.bottom() - h), body.max);
            body.max.y = d.top() - 10.0;
            Some(d)
        } else {
            if self.appearance.visual.stereo_panel || self.appearance.visual.harmony_panel {
                ui.painter().text(Pos2::new(x + 4.0, strip.center().y), egui::Align2::LEFT_CENTER, "needs a taller window",
                    egui::FontId::proportional(11.5), tok.faint);
            }
            None
        };
        let wide = body.width() >= 760.0;
        let full = (body.width() * 0.3).clamp(300.0, 380.0);
        // The drawer slides; the plot takes what it leaves.
        let target = if open && wide { full } else { 0.0 };
        let dt = ui.input(|i| i.stable_dt).min(0.05);
        let f = if self.fx_k() == 0.0 { 1.0 } else { 1.0 - (-dt * 14.0).exp() };
        let w = &mut self.shell.tools_w;
        *w += (target - *w) * f;
        if (target - *w).abs() < 0.5 {
            *w = target;
        } else {
            self.shell.animating = true;
        }
        let w = *w;
        if open && !wide {
            // Too narrow for both side by side: the controls on top, as in the window.
            let mut sui = ui.new_child(egui::UiBuilder::new().max_rect(body));
            self.spectrum_window.embedded_ui(&mut sui);
        } else {
            if w > 1.0 {
                let drawer = Rect::from_min_max(Pos2::new(body.right() - w, body.top()), body.max);
                ui.painter().rect(drawer, 12, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
                // Laid out at full width and clipped, so the rows do not
                // rewrap while it slides.
                let room = Rect::from_min_size(drawer.min + Vec2::splat(12.0), Vec2::new(full - 24.0, drawer.height() - 24.0));
                let mut dui = ui.new_child(egui::UiBuilder::new().max_rect(room));
                dui.set_clip_rect(drawer.shrink(1.0));
                self.tools_drawer(&mut dui, tok, accent);
            }
            let plot = Rect::from_min_max(body.min, Pos2::new(body.right() - if w > 1.0 { w + 12.0 } else { 0.0 }, body.bottom()));
            let mut pui = ui.new_child(egui::UiBuilder::new().max_rect(plot));
            self.spectrum_window.embedded_plot_ui(&mut pui);
        }
        if let Some(d) = deck {
            self.live_ui(ui, d, tok, accent);
        }
        if self.appearance.visual.lyrics_over_spectrum
            && let Some(canvas) = self.spectrum_window.canvas
        {
            self.paint_lyrics_overlay(ui.ctx(), canvas);
        }
        if self.toy("trail")
            && let Some(canvas) = self.spectrum_window.canvas
        {
            self.paint_trail(ui.ctx(), canvas);
        }
    }

    /// The analyser's controls, with a pair of eyes keeping watch.
    /// Feed the live analysis from the analyser's buffers, at most fifteen
    /// times a second and only while playing. The buffers are copied out
    /// first: the audio side only try-locks them and drops what it cannot place.
    fn tick_live(&mut self) {
        let path = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
        if path != self.shell.live_path {
            self.shell.live.reset();
            self.shell.live_path = path;
            self.shell.live_pcm.clear();
            self.shell.mono_since = None;
            self.shell.key_since = None;
            self.shell.key_settled = None;
        }
        if self.play_state != PlayState::Playing {
            return;
        }
        let n = crate::analysis::N;
        let now = Instant::now();
        let rate = self.spectrum_window.analyzer.live_rate;
        // The stereo image every frame (one 8192-point FFT), so the scope and
        // the band capsules move smoothly.
        if self.appearance.visual.stereo_panel {
            let g = self.spectrum_window.stereo_buf.lock().unwrap_or_else(|p| p.into_inner());
            self.shell.live_pcm.clear();
            if g.channels == 2 {
                self.shell.live_pcm.extend_from_slice(&g.frames[g.frames.len().saturating_sub(n)..]);
            }
            drop(g);
            let dt = self.shell.stereo_at.map_or(0.016, |t| now.duration_since(t).as_secs_f32()).min(0.5);
            self.shell.stereo_at = Some(now);
            self.shell.live.stereo(&self.shell.live_pcm, rate, dt);
        }
        // Harmony at a slower pace: chords don't change that fast.
        let dt = self.shell.live_at.map_or(0.07, |t| now.duration_since(t).as_secs_f32());
        if dt < 0.066 {
            return;
        }
        self.shell.live_at = Some(now);
        let dt = dt.min(0.5);
        if self.appearance.visual.harmony_panel {
            let g = self.spectrum_window.sample_buf.lock().unwrap_or_else(|p| p.into_inner());
            self.shell.live_mono.clear();
            self.shell.live_mono.extend_from_slice(&g[g.len().saturating_sub(n)..]);
            drop(g);
            self.shell.live.harmony(&self.shell.live_mono, rate, dt);
        }
    }

    /// The stereo and harmony panels under the plot.
    fn live_ui(&mut self, ui: &mut egui::Ui, deck: Rect, tok: &Tokens, accent: Color32) {
        self.tick_live();
        let (st, ha) = (self.appearance.visual.stereo_panel, self.appearance.visual.harmony_panel);
        let half = Rect::from_min_size(deck.min, Vec2::new((deck.width() - 10.0) / 2.0, deck.height()));
        if st {
            self.stereo_panel(ui, if ha { half } else { deck }, tok, accent);
        }
        if ha {
            self.harmony_panel(ui, if st { half.translate(Vec2::new(half.width() + 10.0, 0.0)) } else { deck }, tok, accent);
        }
    }

    fn stereo_panel(&mut self, ui: &mut egui::Ui, r: Rect, tok: &Tokens, accent: Color32) {
        let p = ui.painter().with_clip_rect(r);
        p.rect(r, 12, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        let small = egui::FontId::proportional(11.0);
        p.text(r.left_top() + Vec2::new(12.0, 14.0), egui::Align2::LEFT_CENTER, "Stereo", egui::FontId::proportional(13.0), tok.ink);
        let live = &self.shell.live;
        if self.shell.live_pcm.len() < crate::analysis::N {
            let why = if self.play_state == PlayState::Playing { "Needs a two-channel track playing." } else { "Play something to see it." };
            p.text(r.center(), egui::Align2::CENTER_CENTER, why, egui::FontId::proportional(12.5), tok.dim);
            return;
        }
        let bal = if live.balance.abs() < 0.5 { "centre".to_string() } else {
            format!("{} {:.1} dB", if live.balance > 0.0 { "R" } else { "L" }, live.balance.abs())
        };
        p.text(r.right_top() + Vec2::new(-12.0, 14.0), egui::Align2::RIGHT_CENTER,
            format!("Width {:.0}%   Balance {bal}", live.width * 100.0), small.clone(), tok.dim);

        // The scope: mid up, side across. Mono is a vertical line.
        let side = r.height() - 28.0 - 22.0;
        let sq = Rect::from_min_size(Pos2::new(r.left() + 12.0, r.top() + 28.0), Vec2::splat(side));
        let c = sq.center();
        let faint = Stroke::new(1.0, tok.line);
        p.line_segment([sq.center_top(), sq.center_bottom()], faint);
        p.line_segment([sq.left_top(), sq.right_bottom()], faint);
        p.line_segment([sq.right_top(), sq.left_bottom()], faint);
        p.text(sq.left_top() + Vec2::splat(4.0), egui::Align2::LEFT_TOP, "L", small.clone(), tok.faint);
        p.text(sq.right_top() + Vec2::new(-4.0, 4.0), egui::Align2::RIGHT_TOP, "R", small.clone(), tok.faint);
        let pcm = &self.shell.live_pcm;
        let tail = &pcm[pcm.len().saturating_sub(1024)..];
        let peak = tail.iter().map(|[l, r]| l.abs().max(r.abs())).fold(1e-4, f32::max);
        let g = &mut self.shell.scope_gain;
        *g = if *g <= 0.0 { peak } else { *g + (peak - *g) * 0.2 };
        let k = side * 0.45 / g.max(1e-4) / std::f32::consts::SQRT_2;
        let pts: Vec<Pos2> = tail.iter().map(|[l, rr]| c + Vec2::new((rr - l) * k, -(l + rr) * k)).collect();
        p.add(egui::Shape::line(pts, Stroke::new(1.0, with_alpha(accent, 0.6))));
        // Correlation under the scope: -1 left end, +1 right end.
        let bar = Rect::from_min_size(Pos2::new(sq.left(), sq.bottom() + 8.0), Vec2::new(side, 6.0));
        p.rect_filled(bar, 3, tok.chip);
        let x = bar.left() + (live.corr.clamp(-1.0, 1.0) + 1.0) / 2.0 * bar.width();
        let col = if live.corr < 0.0 { Color32::from_rgb(232, 96, 96) } else { accent };
        p.rect_filled(Rect::from_min_max(Pos2::new(bar.center().x.min(x), bar.top()), Pos2::new(bar.center().x.max(x), bar.bottom())), 3, col);
        p.line_segment([bar.center_top() - Vec2::Y * 2.0, bar.center_bottom() + Vec2::Y * 2.0], Stroke::new(1.0, tok.dim));
        p.text(bar.right_center() + Vec2::new(6.0, 0.0), egui::Align2::LEFT_CENTER, format!("{:+.2}", live.corr), small.clone(), tok.dim);

        // Out of phase: the scope worries.
        let t = ui.input(|i| i.time) as f32;
        if live.corr < -0.3 && self.toy("worried") {
            self.found("worried");
            paint_worry(&p, sq.right_bottom() - Vec2::new(16.0, 18.0), t, tok.ink);
        }
        // Mono for three seconds: it naps.
        let live = &self.shell.live;
        if live.width < 0.004 && live.corr > 0.995 {
            let since = *self.shell.mono_since.get_or_insert_with(Instant::now);
            if since.elapsed().as_secs_f32() > 3.0 && self.toy("mono_nap") {
                self.found("mono_nap");
                for i in 0..3 {
                    let ph = (t * 0.5 + i as f32 / 3.0).fract();
                    p.text(sq.center_top() + Vec2::new(8.0 + 14.0 * ph, 16.0 - 14.0 * ph), egui::Align2::CENTER_CENTER, "z",
                        egui::FontId::proportional(9.0 + 5.0 * ph), with_alpha(tok.dim, 1.0 - ph));
                }
            }
        } else {
            self.shell.mono_since = None;
        }

        // The stage: one row a band, low at the bottom; each a capsule where
        // the band sits, as wide as it is.
        let live = &self.shell.live;
        let stage = Rect::from_min_max(Pos2::new(sq.right() + 52.0, sq.top()), Pos2::new(r.right() - 14.0, r.bottom() - 12.0));
        if stage.width() < 60.0 {
            return;
        }
        p.line_segment([stage.center_top(), stage.center_bottom()], faint);
        let row = stage.height() / crate::analysis::BANDS.len() as f32;
        let top = live.bands.iter().map(|b| b.level).fold(-120.0, f32::max);
        // Sub-bass the loudest band of all: the bottom rows rumble.
        let rumble = live.bands[0].level.is_finite() && live.bands[0].level >= top - 1.5 && self.toy("rumble");
        if rumble {
            self.found("rumble");
            self.shell.animating = true;
        }
        let t = ui.input(|i| i.time) as f32;
        let live = &self.shell.live;
        for (i, (b, &f)) in live.bands.iter().zip(&crate::analysis::BANDS).enumerate() {
            let y = stage.bottom() - (i as f32 + 0.5) * row;
            let label = if f >= 1000.0 { format!("{}k", f as u32 / 1000) } else { format!("{}", f as u32) };
            p.text(Pos2::new(stage.left() - 8.0, y), egui::Align2::RIGHT_CENTER, label, egui::FontId::proportional(9.5), tok.faint);
            if !b.level.is_finite() {
                continue;
            }
            let a = ((b.level - top + 48.0) / 48.0).clamp(0.0, 1.0);
            if a <= 0.0 {
                continue;
            }
            let cx = stage.center().x + b.pan.clamp(-1.0, 1.0) * stage.width() / 2.0;
            let hw = 3.0 + b.width.clamp(0.0, 1.0) * stage.width() / 2.0;
            let col = if b.corr < 0.0 { Color32::from_rgb(232, 96, 96) } else { accent };
            let shake = if rumble && i < 2 {
                Vec2::new((t * 61.0 + i as f32).sin() * 2.0, (t * 47.0 + 1.3 * i as f32).sin() * 1.5)
            } else {
                Vec2::ZERO
            };
            let cap = Rect::from_center_size(Pos2::new(cx, y) + shake, Vec2::new(2.0 * hw, (row * 0.6).min(8.0)));
            p.rect_filled(cap, 4, with_alpha(col, 0.25 + 0.65 * a));
        }
        // The sound swaps sides four times in three seconds: a ball is served.
        let now = Instant::now();
        let side = if live.balance > 3.0 { 1 } else if live.balance < -3.0 { -1 } else { 0 };
        if side != 0 && side != self.shell.pan_side {
            if self.shell.pan_side != 0 {
                self.shell.pan_swaps.push(now);
            }
            self.shell.pan_side = side;
        }
        self.shell.pan_swaps.retain(|t| now.duration_since(*t).as_secs_f32() < 3.0);
        if self.shell.pan_swaps.len() >= 4 && self.shell.ping_at.is_none() && self.toy("ping_pong") {
            self.found("ping_pong");
            self.shell.ping_at = Some(now);
            self.shell.pan_swaps.clear();
        }
        if let Some(at) = self.shell.ping_at {
            let q = now.duration_since(at).as_secs_f32() / 2.4;
            if q >= 1.0 {
                self.shell.ping_at = None;
            } else {
                self.shell.animating = true;
                // Three crossings, bouncing off the table at the bottom.
                let leg = (q * 3.0).fract();
                let dir = if (q * 3.0) as u32 % 2 == 0 { 1.0 } else { -1.0 };
                let x = stage.center().x + dir * (leg * 2.0 - 1.0) * stage.width() * 0.45;
                let y = stage.bottom() - 6.0 - (leg * std::f32::consts::PI).sin() * stage.height() * 0.5;
                p.circle_filled(Pos2::new(x, y), 3.5, with_alpha(Color32::from_rgb(0xff, 0xf4, 0xe0), 1.0 - q * q));
            }
        }
    }

    /// Moves the chord model along: load it once, finish an install, start or
    /// collect the current track's chords.
    fn tick_chords(&mut self) {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        let ai = &mut self.shell.chord_ai;
        if !ai.checked && ai.loading.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = std::thread::Builder::new().name("chord-model-load".into()).spawn(move || {
                let dir = crate::chords::install::dir(&moosik_dir());
                let r = if crate::chords::install::installed(&dir) { crate::chords::Model::load(&dir).map(Arc::new) } else { Err(String::new()) };
                let _ = tx.send(r);
            });
            ai.loading = Some(rx);
        }
        if let Some(rx) = &ai.loading
            && let Ok(r) = rx.try_recv()
        {
            ai.loading = None;
            ai.checked = true;
            match r {
                Ok(m) => ai.model = Some(m),
                Err(e) if !e.is_empty() => ai.error = Some(e),
                Err(_) => {}
            }
        }
        if let Some((_, _, rx)) = &ai.install
            && let Ok(r) = rx.try_recv()
        {
            ai.install = None;
            match r {
                Ok(()) => ai.checked = false, // load it now
                Err(e) => ai.error = Some(e),
            }
        }
        let Some(model) = ai.model.clone() else { return };
        let path = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
        let ai = &mut self.shell.chord_ai;
        if let Some((p, _, rx)) = &ai.job
            && let Ok(r) = rx.try_recv()
        {
            if r.is_ok() {
                ai.took = ai.job_at.map(|at| (at.elapsed().as_secs_f32(), Instant::now()));
            }
            ai.track = Some((p.clone(), r));
            ai.job = None;
        }
        let Some(path) = path else { return };
        let done = ai.track.as_ref().is_some_and(|(p, _)| *p == path);
        let running = ai.job.as_ref().is_some_and(|(p, _, _)| *p == path);
        if !done && !running {
            if let Some((_, cancel, _)) = ai.job.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            let cancel = Arc::new(AtomicBool::new(false));
            let (tx, rx) = std::sync::mpsc::channel();
            let (p, c) = (path.clone(), cancel.clone());
            let _ = std::thread::Builder::new().name("chords".into()).spawn(move || {
                let cache = moosik_dir().join("cache").join("chords");
                let _ = tx.send(crate::chords::analyze(&model, &p, &cache, &c));
            });
            ai.job = Some((path, cancel, rx));
            ai.job_at = Some(Instant::now());
        }
    }

    /// Download the chord model (the user asked).
    fn start_chord_install(&mut self) {
        use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
        let done = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let (d, c) = (done.clone(), cancel.clone());
        let _ = std::thread::Builder::new().name("chord-model-install".into()).spawn(move || {
            let dir = crate::chords::install::dir(&moosik_dir());
            let _ = tx.send(crate::chords::install::install(&dir, &c, |n, _| d.store(n, Ordering::Relaxed)));
        });
        self.shell.chord_ai.error = None;
        self.shell.chord_ai.install = Some((done, cancel, rx));
    }

    fn harmony_panel(&mut self, ui: &mut egui::Ui, r: Rect, tok: &Tokens, accent: Color32) {
        self.tick_chords();
        let p = ui.painter().with_clip_rect(r);
        p.rect(r, 12, tok.scrim, Stroke::new(1.0, tok.line), egui::StrokeKind::Inside);
        let title = p.text(r.left_top() + Vec2::new(12.0, 14.0), egui::Align2::LEFT_CENTER, "Harmony", egui::FontId::proportional(13.0), tok.ink);
        // Five chords corrected: an apple for the teacher.
        if self.shell.fixes_made >= 5 && self.toy("apple") {
            self.found("apple");
            let c = title.right_center() + Vec2::new(10.0, 1.0);
            p.circle_filled(c, 4.5, Color32::from_rgb(0xd9, 0x3b, 0x3b));
            p.line_segment([c - Vec2::new(0.0, 4.0), c - Vec2::new(-1.0, 7.0)], Stroke::new(1.2, Color32::from_rgb(0x7a, 0x52, 0x2e)));
            p.circle_filled(c + Vec2::new(2.5, -6.0), 1.6, Color32::from_rgb(0x5c, 0xb8, 0x5c));
        }
        let small = egui::FontId::proportional(11.0);
        let now = Instant::now();
        // A key holds once it has lasted five seconds; a new one holding
        // after it is a little party. Keys passed through on the way don't
        // count, nor does a flip to the relative major or minor (same notes).
        let sig = |root: usize, minor: bool| (root + if minor { 3 } else { 0 }) % 12;
        if let Some(k) = self.shell.live.key.filter(|k| k.fit > 0.6) {
            match self.shell.key_since {
                Some((root, minor, since)) if sig(root, minor) == sig(k.root, k.minor) => {
                    if since.elapsed().as_secs_f32() > 5.0 && self.shell.key_settled.is_none_or(|(r0, m0)| sig(r0, m0) != sig(root, minor)) {
                        if let Some((r0, m0)) = self.shell.key_settled
                            && self.toy("key_party")
                        {
                            self.found("key_party");
                            self.shell.key_party = Some((now, crate::analysis::Key { root: r0, minor: m0, fit: 1.0 }.name()));
                        }
                        self.shell.key_settled = Some((root, minor));
                    }
                }
                _ => self.shell.key_since = Some((k.root, k.minor, now)),
            }
        }
        let live = &self.shell.live;
        if let Some(k) = live.key {
            let anchor = r.right_top() + Vec2::new(-12.0, 14.0);
            let party = self.shell.key_party.as_ref().map(|(at, was)| (now.duration_since(*at).as_secs_f32() / 1.6, was.clone()));
            match party.filter(|(q, _)| *q < 1.0) {
                Some((q, was)) => {
                    self.shell.animating = true;
                    let g = p.layout_no_wrap(format!("Key {}", k.name()), small.clone(), accent);
                    let lt = anchor - Vec2::new(g.size().x, g.size().y / 2.0);
                    // The label turns over once; confetti flies off it.
                    let a = std::f32::consts::TAU * ease_out_back(q.min(0.5) * 2.0);
                    p.add(egui::epaint::TextShape::new(lt + g.size() / 2.0 - Vec2::angled(a) * g.size().x / 2.0
                        - Vec2::angled(a + std::f32::consts::FRAC_PI_2) * g.size().y / 2.0, g.clone(), accent).with_angle(a));
                    for i in 0..10 {
                        let ang = std::f32::consts::PI * (0.15 + 0.8 * (i as f32 * 0.618).fract()); // downward: the panel's top edge is close
                        let d = Vec2::angled(ang) * (8.0 + 40.0 * q) + Vec2::Y * 30.0 * q * q;
                        let col = if i % 2 == 0 { accent } else { tok.ink };
                        p.rect_filled(Rect::from_center_size(lt + Vec2::new(g.size().x / 2.0, 0.0) + d, Vec2::splat(3.0)), 1, with_alpha(col, 1.0 - q));
                    }
                    p.text(anchor + Vec2::new(0.0, 14.0), egui::Align2::RIGHT_CENTER, format!("was {was}"),
                        egui::FontId::proportional(10.0), with_alpha(tok.dim, 1.0 - q));
                }
                None => {
                    p.text(anchor, egui::Align2::RIGHT_CENTER,
                        format!("Key {} {}", k.name(), if k.fit > 0.6 { "" } else { "(guessing)" }).trim_end(), small.clone(), tok.dim);
                }
            }
        }
        // The chord, popping in when it changes: the model's when it has
        // worked the track out, otherwise the live guess.
        let elapsed = self.elapsed().as_secs_f32();
        let path = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
        let segs: Option<Vec<crate::chords::Segment>> = self.shell.chord_ai.track.as_ref()
            .filter(|(p, _)| Some(p) == path.as_ref()).and_then(|(_, r)| r.as_ref().ok()).cloned();
        let live = &self.shell.live;
        let four = live.four_chords();
        let hist_len = live.history.len();
        // `sure`: the model's confidence in the chord (1 for the live guess,
        // which says so in its own words). Below 0.5 it is right under 40% of
        // the time, so the chord is shown faint with a question mark.
        let (name, lit, hist, next, sure): (Option<String>, Vec<usize>, Vec<String>, Option<(String, f32, f32)>, f32) = match &segs {
            Some(segs) => {
                let (cur, nxt) = crate::chords::at(segs, elapsed);
                let cur = cur.filter(|s| s.name != "N");
                let before = segs.partition_point(|s| s.end <= elapsed);
                let hist = segs[..before].iter().rev().filter(|s| s.name != "N").take(5).map(|s| crate::chords::display(s.name)).collect();
                (
                    cur.map(|s| crate::chords::display(s.name)),
                    cur.map(|s| crate::chords::tones(s.name)).unwrap_or_default(),
                    hist,
                    nxt.filter(|s| s.name != "N").map(|s| (crate::chords::display(s.name), s.start - elapsed, s.conf)),
                    cur.map_or(1.0, |s| s.conf),
                )
            }
            None => (
                live.chord.map(|c| c.name()),
                live.chord.map(|c| c.tones()).unwrap_or_default(),
                live.history.iter().rev().skip(1).take(5).map(|c| c.name()).collect(),
                None,
                1.0,
            ),
        };
        let cur_seg = segs.as_ref().and_then(|s| crate::chords::at(s, elapsed).0).map(|s| (s.start, s.name));
        let wheel_d = (r.height() - 40.0).min(r.width() * 0.45);
        let left = Rect::from_min_max(r.min + Vec2::new(12.0, 28.0), Pos2::new(r.right() - wheel_d - 24.0, r.bottom() - 10.0));
        let cc = Pos2::new(left.center().x, left.top() + 34.0);
        match &name {
            Some(ch) => {
                self.shell.nc_since = None;
                if self.shell.chord_at.as_ref().is_none_or(|(c, _)| c != ch) {
                    self.shell.chord_prev2 = self.shell.chord_prev.take();
                    self.shell.chord_prev = self.shell.chord_at.take().map(|(c, _)| c);
                    self.shell.chord_at = Some((ch.clone(), now));
                    self.shell.chord_changes.retain(|t| now.duration_since(*t).as_secs_f32() < 2.0);
                    self.shell.chord_changes.push(now);
                }
                let age = self.shell.chord_at.as_ref().map_or(1.0, |(_, at)| now.duration_since(*at).as_secs_f32());
                let pop = self.toy("chord_pop");
                let s = if pop && age < 0.4 {
                    self.found("chord_pop");
                    self.shell.animating = true;
                    0.7 + 0.3 * ease_out_back(age / 0.4)
                } else {
                    1.0
                };
                let (shown, mut ink) = match sure {
                    x if x < 0.5 => (format!("{ch}?"), tok.faint),
                    x if x < 0.7 => (ch.clone(), tok.dim),
                    _ => (ch.clone(), tok.ink),
                };
                // Major chords warm, minor ones cool.
                if sure >= 0.7 && self.toy("mood") {
                    self.found("mood");
                    let warm = Color32::from_rgb(0xff, 0x9a, 0x4a);
                    let cool = Color32::from_rgb(0x5a, 0x9c, 0xff);
                    ink = mix(ink, if is_minor(ch) { cool } else { warm }, 0.45);
                }
                // Four changes in two seconds: the chord gets flustered.
                let rush = self.shell.chord_changes.len() >= 4 && self.toy("chord_rush");
                let wob = if rush && age < 1.2 {
                    self.found("chord_rush");
                    self.shell.animating = true;
                    Vec2::new((age * 40.0).sin() * 3.0 * (1.0 - age / 1.2), 0.0)
                } else {
                    Vec2::ZERO
                };
                let g = p.layout_no_wrap(shown, egui::FontId::proportional(40.0 * s), ink);
                let gr = egui::Align2::CENTER_CENTER.anchor_size(cc + wob, g.size());
                p.galley(gr.min, g, ink);
                if rush && age < 1.2 {
                    p.text(gr.right_top() + Vec2::new(4.0, 2.0), egui::Align2::LEFT_TOP, "!!", small.clone(), with_alpha(accent, 1.0 - age / 1.2));
                }
                // Not a clue: a little shrug under it.
                if sure < 0.5 && self.toy("shrug") {
                    self.found("shrug");
                    p.text(gr.left_center() - Vec2::new(8.0, 0.0), egui::Align2::RIGHT_CENTER, r"¯\_(ツ)_/¯", small.clone(), tok.faint);
                }
                // Diminished: a little devil's horns on the name.
                if (ch.contains("dim") || ch.contains("m7b5")) && self.toy("horns") {
                    self.found("horns");
                    for dx in [-0.3f32, 0.3] {
                        let b = Pos2::new(gr.center().x + dx * gr.width(), gr.top() + 4.0);
                        p.add(egui::Shape::convex_polygon(vec![b + Vec2::new(-3.0, 0.0), b + Vec2::new(dx.signum() * 2.0, -8.0), b + Vec2::new(3.0, 0.0)],
                            with_alpha(Color32::from_rgb(0xe0, 0x4a, 0x4a), 0.85), Stroke::NONE));
                    }
                }
                // Suspended: it waits to resolve, and so do the dots.
                if ch.contains("sus") && self.toy("suspense") {
                    self.found("suspense");
                    self.shell.animating = true;
                    let n = ((age * 2.5) as usize % 3) + 1;
                    p.text(gr.right_center() + Vec2::new(4.0, 6.0), egui::Align2::LEFT_CENTER, ".".repeat(n), egui::FontId::proportional(18.0), tok.dim);
                }
                // The song's last chord takes a bow.
                let left_s = self.current_duration().map_or(f32::MAX, |d| d.as_secs_f32() - elapsed);
                if segs.is_some() && next.is_none() && left_s < 8.0 && self.toy("bow") {
                    self.found("bow");
                    self.shell.animating = true;
                    let q = (1.0 - left_s / 8.0).clamp(0.0, 1.0);
                    p.text(gr.center_bottom() + Vec2::new(0.0, 4.0), egui::Align2::CENTER_TOP, "fin.", egui::FontId::proportional(11.0 + 3.0 * q), with_alpha(accent, q));
                }
                // The key's own chord: home.
                let key = self.shell.live.key.filter(|k| k.fit > 0.6);
                if key.is_some_and(|k| root_pc(ch) == Some(k.root) && is_minor(ch) == k.minor) && self.toy("home") {
                    self.found("home");
                    let b = gr.left_bottom() + Vec2::new(-12.0, -8.0);
                    let col = with_alpha(accent, 0.8);
                    p.add(egui::Shape::convex_polygon(vec![b + Vec2::new(-6.0, -5.0), b + Vec2::new(0.0, -11.0), b + Vec2::new(6.0, -5.0)], col, Stroke::NONE));
                    p.rect_filled(Rect::from_min_max(b + Vec2::new(-4.5, -5.0), b + Vec2::new(4.5, 2.0)), 0, col);
                    p.rect_filled(Rect::from_min_max(b + Vec2::new(-1.2, -2.0), b + Vec2::new(1.2, 2.0)), 0, tok.scrim);
                }
                // Down a fifth from a major chord: the tension lets go.
                let resolved = self.shell.chord_prev.as_deref().is_some_and(|pc| {
                    !is_minor(pc) && root_pc(pc).zip(root_pc(ch)).is_some_and(|(a, b)| a == (b + 7) % 12)
                });
                if resolved && age < 1.6 && self.toy("sigh") {
                    self.found("sigh");
                    self.shell.animating = true;
                    let q = age / 1.6;
                    p.text(gr.left_top() + Vec2::new(-6.0 - 10.0 * q, 4.0 - 8.0 * q), egui::Align2::RIGHT_BOTTOM, "ahh~",
                        egui::FontId::proportional(11.0), with_alpha(tok.dim, 1.0 - q));
                }
                // ii - V - I: the most worn path in jazz.
                let two_five_one = matches!((self.shell.chord_prev2.as_deref(), self.shell.chord_prev.as_deref()),
                    (Some(a), Some(b)) if is_two_five_one(a, b, ch));
                if two_five_one && age < 2.0 && self.toy("smooth") {
                    self.found("smooth");
                    self.shell.animating = true;
                    let q = age / 2.0;
                    p.text(gr.right_bottom() + Vec2::new(6.0, -4.0), egui::Align2::LEFT_BOTTOM, "smooth.",
                        egui::FontId::proportional(11.0), with_alpha(accent, (1.0 - q) * (q * 8.0).min(1.0)));
                }
                // Major seventh: dreamy enough for a halo.
                if ch.contains("maj7") && self.toy("halo") {
                    self.found("halo");
                    let c = Pos2::new(gr.center().x, gr.top() - 2.0);
                    let pts: Vec<Pos2> = (0..32)
                        .map(|i| {
                            let a = i as f32 / 32.0 * std::f32::consts::TAU;
                            c + Vec2::new(a.cos() * 16.0, a.sin() * 3.5)
                        })
                        .collect();
                    p.add(egui::Shape::closed_line(pts, Stroke::new(1.5, with_alpha(Color32::from_rgb(0xff, 0xd8, 0x6a), 0.85))));
                }
                // Augmented: a little extra sparkle.
                if ch.contains("aug") && self.toy("sparkle") {
                    self.found("sparkle");
                    self.shell.animating = true;
                    let t = ui.input(|i| i.time) as f32;
                    for (k, d) in [Vec2::new(-0.55, -0.45), Vec2::new(0.6, -0.35), Vec2::new(0.5, 0.5), Vec2::new(-0.6, 0.4)].into_iter().enumerate() {
                        let a = ((t * 3.0 + k as f32 * 1.7).sin() * 0.5 + 0.5).powi(2);
                        let c = gr.center() + Vec2::new(d.x * gr.width(), d.y * gr.height());
                        let s = 2.0 + 2.5 * a;
                        let col = with_alpha(Color32::from_rgb(0xff, 0xf0, 0xa0), a);
                        p.line_segment([c - Vec2::X * s, c + Vec2::X * s], Stroke::new(1.0, col));
                        p.line_segment([c - Vec2::Y * s, c + Vec2::Y * s], Stroke::new(1.0, col));
                    }
                }
                // One chord for ages: it nods off.
                if age > 12.0 && self.toy("chord_nap") {
                    self.found("chord_nap");
                    self.shell.animating = true;
                    let t = ui.input(|i| i.time) as f32;
                    for k in 0..3 {
                        let q = (t * 0.5 + k as f32 / 3.0).fract();
                        let at = gr.right_top() + Vec2::new(6.0 + 14.0 * q, 10.0 - 22.0 * q);
                        p.text(at, egui::Align2::CENTER_CENTER, "z", egui::FontId::proportional(9.0 + 6.0 * q), with_alpha(tok.dim, 1.0 - q));
                    }
                }
                if pop && age < 0.9 {
                    let q = age / 0.9;
                    paint_note(&p, cc + Vec2::new(34.0 + 10.0 * q, -6.0 - 26.0 * q), 5.0, with_alpha(accent, 1.0 - q));
                }
            }
            None => {
                let since = *self.shell.nc_since.get_or_insert(now);
                let quiet = now.duration_since(since).as_secs_f32();
                // A long stretch with no chord: a tumbleweed rolls through.
                if quiet > 6.0 && self.toy("tumbleweed") {
                    self.found("tumbleweed");
                    self.shell.animating = true;
                    let q = ((quiet - 6.0) / 5.0).fract();
                    let x = left.left() + (left.width() + 20.0) * q - 10.0;
                    let y = cc.y + 16.0 - ((q * 9.0 * std::f32::consts::PI).sin().abs() * 8.0);
                    let c = Pos2::new(x, y);
                    let col = with_alpha(Color32::from_rgb(0xb8, 0x93, 0x5a), 0.8);
                    p.circle_stroke(c, 6.0, Stroke::new(1.2, col));
                    for k in 0..4 {
                        let a = q * 30.0 + k as f32 * 0.8;
                        p.line_segment([c + Vec2::angled(a) * 5.5, c - Vec2::angled(a) * 5.5], Stroke::new(0.8, col));
                    }
                }
                p.text(cc, egui::Align2::CENTER_CENTER, "N.C.", egui::FontId::proportional(24.0), tok.faint);
                // The "next" line takes this spot when the model knows what's coming.
                if next.is_none() {
                    p.text(cc + Vec2::Y * 24.0, egui::Align2::CENTER_TOP, "no chord", small.clone(), tok.faint);
                }
            }
        }
        // What comes next (the model knows the whole song).
        if let Some((n, secs, next_sure)) = &next {
            let at = cc + Vec2::new(0.0, 30.0);
            let g = p.layout_no_wrap(format!("next  {n}"), small.clone(), tok.dim);
            p.galley(at - Vec2::new(g.size().x / 2.0, 0.0), g.clone(), tok.dim);
            if *secs < 1.0 && self.toy("crystal_ball") {
                // The crystal ball clouds over, then clears as the chord
                // arrives; when the model is unsure of it, it stays foggy.
                self.found("crystal_ball");
                self.shell.animating = true;
                let q = 1.0 - secs.clamp(0.0, 1.0);
                let bc = at + Vec2::new(g.size().x / 2.0 + 12.0, g.size().y / 2.0);
                p.circle_filled(bc, 6.5, with_alpha(accent, 0.15 + 0.35 * q));
                p.circle_stroke(bc, 6.5, Stroke::new(1.0, with_alpha(accent, 0.8)));
                let t = ui.input(|i| i.time) as f32;
                let clear = if *next_sure < 0.7 { q * 0.3 } else { q };
                for k in 0..3 {
                    let a = t * 4.0 + k as f32 * std::f32::consts::TAU / 3.0;
                    p.circle_filled(bc + Vec2::angled(a) * 3.0 * (1.0 - clear), 1.2, with_alpha(tok.ink, 0.4 + 0.6 * clear));
                }
                if *next_sure < 0.7 {
                    for k in 0..3 {
                        let a = -t * 1.5 + k as f32 * 2.1;
                        p.circle_filled(bc + Vec2::angled(a) * 2.5, 3.0, with_alpha(tok.faint, 0.35));
                    }
                }
                p.line_segment([bc + Vec2::new(-5.0, 8.0), bc + Vec2::new(5.0, 8.0)], Stroke::new(2.0, with_alpha(tok.dim, 0.8)));
            }
        }
        // The ones before it, fading.
        let mut x = left.left();
        let y = left.bottom() - 30.0;
        for (i, c) in hist.iter().enumerate() {
            let a = 1.0 - i as f32 / 5.5;
            let g = p.layout_no_wrap(c.clone(), egui::FontId::proportional(12.0), with_alpha(tok.dim, a));
            if x + g.size().x > left.right() {
                break;
            }
            p.galley(Pos2::new(x, y - g.size().y / 2.0), g.clone(), tok.dim);
            x += g.size().x + 10.0;
        }
        // Where the chords come from, and how to get the better source.
        self.chord_source_line(ui, &p, Rect::from_min_max(Pos2::new(left.left(), left.bottom() - 16.0), left.right_bottom()), tok, accent, segs.is_some());
        // I-V-vi-IV: the most famous four chords in pop.
        if four && self.toy("four") && self.shell.four_at.is_none_or(|(n, _)| n != hist_len) {
            self.shell.four_at = Some((hist_len, now));
            self.found("four");
            self.celebrate(ui.ctx(), accent, false);
        }
        if let Some((_, at)) = self.shell.four_at
            && now.duration_since(at).as_secs_f32() < 2.0
        {
            let q = now.duration_since(at).as_secs_f32() / 2.0;
            self.shell.animating = true;
            p.text(cc + Vec2::new(0.0, 34.0 - 8.0 * q), egui::Align2::CENTER_CENTER, "4 chords!",
                egui::FontId::proportional(13.0), with_alpha(accent, 1.0 - q * q));
        }
        // Wrong chord? Click it and say what it is; the model's word gives way.
        let mut opened = false;
        if let (Some(path), Some((start, cur))) = (&path, cur_seg) {
            let hit = Rect::from_center_size(cc, Vec2::new(140.0, 48.0));
            let resp = ui.interact(hit, ui.id().with("chord_fix"), egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text("Wrong chord? Click to correct it");
            if resp.clicked() {
                let root = crate::chords::ROOTS.iter().position(|r| cur.split(':').next() == Some(*r)).unwrap_or(0);
                self.shell.chord_fix = Some((path.clone(), start, root));
                opened = true;
            }
        }
        self.chord_fix_popup(ui.ctx(), cc + Vec2::new(0.0, 30.0), path.as_ref(), opened, tok);
        if let Some(at) = self.shell.chord_thanks
            && self.toy("thanks")
        {
            let q = now.duration_since(at).as_secs_f32() / 1.5;
            if q < 1.0 {
                self.found("thanks");
                self.shell.animating = true;
                let at = cc + Vec2::new(0.0, -24.0 - 12.0 * q);
                p.text(at, egui::Align2::CENTER_CENTER, "thank you!", small.clone(), with_alpha(accent, 1.0 - q));
                paint_note(&p, at + Vec2::new(36.0, -4.0 - 6.0 * q), 4.0, with_alpha(accent, 1.0 - q));
            }
        }

        // The wheel: the twelve notes round the circle of fifths, each as
        // long as it sounds; the chord's notes lit.
        let live = &self.shell.live;
        let wc = Pos2::new(r.right() - 12.0 - wheel_d / 2.0, r.top() + 28.0 + (r.height() - 38.0) / 2.0);
        let (r0, r1) = (wheel_d * 0.14, wheel_d * 0.36);
        p.circle_stroke(wc, r1, Stroke::new(1.0, tok.line));
        for i in 0..12 {
            let pc = (i * 7) % 12;
            let a = -std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::TAU / 12.0;
            let d = Vec2::angled(a);
            let v = live.chroma[pc].clamp(0.0, 1.0);
            let on = lit.contains(&pc);
            let col = if on { accent } else { tok.dim };
            p.line_segment([wc + d * r0, wc + d * (r0 + (r1 - r0) * v)], Stroke::new(if on { 4.0 } else { 3.0 }, with_alpha(col, 0.35 + 0.65 * v)));
            let tonic = live.key.is_some_and(|k| k.root == pc);
            let ink = if on { tok.ink } else if tonic { accent } else { tok.faint };
            p.text(wc + d * (r1 + 9.0), egui::Align2::CENTER_CENTER, crate::spectrum::NOTE_NAMES[pc], egui::FontId::proportional(10.0), ink);
        }
    }

    /// The chord picker: a root, then the chord; the choice is kept (see
    /// `chords::correct`). Escape, Cancel or a click elsewhere closes it.
    fn chord_fix_popup(&mut self, ctx: &egui::Context, at: Pos2, path: Option<&PathBuf>, opened: bool, tok: &Tokens) {
        use crate::chords::ROOTS;
        const QUALS: [(&str, &str); 10] = [("", "maj"), ("m", "min"), ("7", "7"), ("maj7", "maj7"), ("m7", "min7"),
            ("dim", "dim"), ("m7b5", "hdim7"), ("sus2", "sus2"), ("sus4", "sus4"), ("aug", "aug")];
        let Some((track, start, mut root)) = self.shell.chord_fix.clone() else { return };
        if path != Some(&track) {
            self.shell.chord_fix = None;
            return;
        }
        let mut pick = None;
        let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let area = egui::Area::new(Id::new("chord_fix")).order(egui::Order::Foreground).pivot(egui::Align2::CENTER_TOP).fixed_pos(at)
            .show(ctx, |ui| egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(egui::RichText::new("What chord is it?").size(12.0).color(tok.ink));
                ui.horizontal(|ui| {
                    for (i, r) in ROOTS.iter().enumerate() {
                        if ui.selectable_label(i == root, *r).clicked() {
                            root = i;
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    for (shown, q) in QUALS {
                        if ui.button(format!("{}{shown}", ROOTS[root])).clicked() {
                            pick = Some(format!("{}:{q}", ROOTS[root]));
                        }
                    }
                    if ui.button("N.C.").clicked() {
                        pick = Some("N".into());
                    }
                });
                if ui.small_button("Cancel").clicked() {
                    close = true;
                }
            }));
        if !opened && area.response.clicked_elsewhere() {
            close = true;
        }
        if let Some(name) = pick {
            if let Some((_, Ok(segs))) = self.shell.chord_ai.track.as_mut().filter(|(p, _)| *p == track)
                && let Some(i) = segs.iter().position(|s| (s.start - start).abs() < 0.01)
            {
                match crate::chords::correct(&track, &moosik_dir().join("cache").join("chords"), segs, i, &name) {
                    Ok(()) => {
                        self.shell.chord_thanks = Some(Instant::now());
                        self.shell.fixes_made += 1;
                    }
                    Err(e) => self.shell.chord_ai.error = Some(e),
                }
            }
            close = true;
        }
        match &mut self.shell.chord_fix {
            Some(f) if !close => f.2 = root,
            _ => self.shell.chord_fix = None,
        }
    }

    /// One line under the chords: the model's state, or a button to get it.
    fn chord_source_line(&mut self, ui: &mut egui::Ui, p: &egui::Painter, row: Rect, tok: &Tokens, accent: Color32, from_model: bool) {
        use std::sync::atomic::Ordering;
        let small = egui::FontId::proportional(10.5);
        // How long the model took, for a few seconds after (a cache hit is instant).
        let took = self.shell.chord_ai.took.filter(|(_, at)| at.elapsed().as_secs_f32() < 5.0 && self.toy("stopwatch"));
        if took.is_some() {
            self.found("stopwatch");
            self.shell.animating = true;
        }
        let ai = &self.shell.chord_ai;
        let path = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone());
        let (text, button): (String, Option<&str>) = if from_model {
            match took {
                Some((s, _)) if s < 0.3 => ("Chords from the chord model \u{b7} remembered this one".into(), None),
                Some((s, _)) => (format!("Chords from the chord model \u{b7} worked out in {s:.1} s"), None),
                None => ("Chords from the chord model".into(), None),
            }
        } else if let Some((done, _, _)) = &ai.install {
            let pct = done.load(Ordering::Relaxed) * 100 / crate::chords::install::WHEEL_BYTES;
            (format!("Downloading the chord model\u{2026} {pct}%"), Some("Cancel"))
        } else if ai.job.as_ref().is_some_and(|(p, _, _)| Some(p) == path.as_ref()) {
            ("Guessing while the model listens to the whole song\u{2026}".into(), None)
        } else if let Some((_, Err(e))) = ai.track.as_ref().filter(|(p, _)| Some(p) == path.as_ref()) {
            (format!("Guessing: {e}"), None)
        } else if ai.model.is_some() || ai.loading.is_some() {
            ("Guessing".into(), None)
        } else {
            let why = ai.error.as_deref().map(|e| format!("Guessing. {e} ")).unwrap_or_else(|| "Guessing.".into());
            (why, Some("Get the chord model (27 MB, MIT)"))
        };
        let g = p.layout_no_wrap(text, small.clone(), tok.faint);
        let w = g.size().x;
        p.galley(Pos2::new(row.left(), row.center().y - g.size().y / 2.0), g, tok.faint);
        let Some(label) = button else { return };
        let g = p.layout_no_wrap(label.to_string(), small, accent);
        let br = Rect::from_min_size(Pos2::new(row.left() + w + 6.0, row.center().y - g.size().y / 2.0 - 2.0), g.size() + Vec2::new(0.0, 4.0));
        let resp = ui.interact(br, ui.id().with("chord_model_button"), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, label));
        let col = if resp.hovered() { tok.ink } else { accent };
        p.galley(br.min + Vec2::new(0.0, 2.0), g, col);
        p.line_segment([br.left_bottom(), br.right_bottom()], Stroke::new(1.0, with_alpha(col, 0.5)));
        if resp.clicked() {
            if let Some((_, cancel, _)) = &self.shell.chord_ai.install {
                cancel.store(true, Ordering::Relaxed);
            } else {
                self.start_chord_install();
            }
        }
    }

    fn tools_drawer(&mut self, ui: &mut egui::Ui, tok: &Tokens, accent: Color32) {
        let (head, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 26.0), Sense::hover());
        ui.painter().text(head.left_center(), egui::Align2::LEFT_CENTER, "Controls",
            egui::FontId::proportional(16.0), tok.ink);
        if self.toy("eyes") {
            let eyes = Rect::from_min_max(Pos2::new(head.right() - 40.0, head.top()), head.max);
            let poke = ui.interact(eyes, Id::new("shell_eyes"), Sense::click());
            if poke.clicked() {
                self.shell.blink = Some(Instant::now());
            }
            let shut = self.shell.blink.is_some_and(|at| at.elapsed().as_secs_f32() < 0.35);
            if shut {
                self.shell.animating = true;
            }
            let look = ui.input(|i| i.pointer.hover_pos());
            for dx in [-9.0f32, 9.0] {
                let c = Pos2::new(eyes.center().x + dx, eyes.center().y);
                if shut {
                    // Squeezed shut: > <
                    let d = if dx < 0.0 { 1.0 } else { -1.0 };
                    let st = Stroke::new(1.6, tok.ink);
                    ui.painter().line_segment([c + Vec2::new(-3.5 * d, -3.5), c + Vec2::new(3.5 * d, 0.0)], st);
                    ui.painter().line_segment([c + Vec2::new(3.5 * d, 0.0), c + Vec2::new(-3.5 * d, 3.5)], st);
                    continue;
                }
                ui.painter().circle(c, 7.0, tok.ink, Stroke::new(1.0, tok.line));
                let to = look.map_or(Vec2::ZERO, |p| p - c);
                let pupil = c + if to.length() > 0.5 { to.normalized() * to.length().min(60.0) / 60.0 * 3.5 } else { Vec2::ZERO };
                ui.painter().circle_filled(pupil, 3.0, tok.bg);
            }
        }
        ui.add_space(6.0);
        egui::ScrollArea::vertical().id_salt("shell_tools_scroll").auto_shrink([false, false]).show(ui, |ui| {
            self.spectrum_window.embedded_controls_ui(ui);
            ui.add_space(18.0);
            let end = ui.vertical_centered(|ui| {
                ui.label(RichText::new("That's every knob there is.").size(11.0).italics().color(tok.faint));
            }).response.rect;
            // Scrolled all the way down: a little something for the effort.
            let scrolled = ui.max_rect().top() < ui.clip_rect().top() - 20.0;
            if scrolled && ui.clip_rect().contains_rect(end) && !self.shell.tools_bottom {
                self.shell.tools_bottom = true;
                self.burst(end.center_top(), accent, 16, 0.7, None);
            }
            ui.add_space(8.0);
        });
    }

    /// A struck A: 440 Hz and a little octave, dying away. Mixed into the
    /// shared output when it is open, else on a short stream of its own;
    /// an exclusive route keeps the device, and the fork stays quiet.
    fn ring_a(&self) {
        let sr = 48_000u32;
        let gain = 0.2 * self.volume.clamp(0.0, 1.0);
        let tau = std::f32::consts::TAU;
        let s: Vec<f32> = (0..sr as usize * 2).map(|i| {
            let t = i as f32 / sr as f32;
            let env = (-2.5 * t).exp() * (t * 300.0).min(1.0);
            gain * env * ((tau * 440.0 * t).sin() + 0.3 * (-7.0 * t).exp() * (tau * 880.0 * t).sin())
        }).collect();
        let buf = rodio::buffer::SamplesBuffer::new(1, sr, s);
        if let Some((_, h)) = self.engine.as_ref().and_then(|e| e.rodio_out.as_ref()) {
            let _ = h.play_raw(buf);
            return;
        }
        std::thread::spawn(move || {
            if let Ok((_stream, h)) = rodio::OutputStream::try_default()
                && h.play_raw(buf).is_ok()
            {
                std::thread::sleep(std::time::Duration::from_millis(2100));
            }
        });
    }

    /// A tuning fork that hums along to the bass. Struck, it rings an A.
    fn tuning_fork(&mut self, ui: &egui::Ui, c: Pos2, tok: &Tokens, accent: Color32) {
        let r = Rect::from_center_size(c, Vec2::new(20.0, 26.0));
        let resp = ui.interact(r, Id::new("shell_fork"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Tuning fork"));
        if resp.clicked() && self.toy("fork") {
            self.found("fork");
            self.shell.fork_ding = Some(Instant::now());
            self.shell.rings.push((c, Instant::now()));
            self.ring_a();
        }
        let ding = self.shell.fork_ding.map(|at| at.elapsed().as_secs_f32()).filter(|t| *t < 1.4);
        if ding.is_none() {
            self.shell.fork_ding = None;
        }
        let playing = self.play_state == PlayState::Playing && self.toy("fork");
        let amp = ding.map_or(0.0, |t| 2.2 * (1.0 - t / 1.4)) + if playing { 1.4 * self.shell.energy } else { 0.0 };
        let t = ui.input(|i| i.time) as f32;
        let buzz = amp * (t * 70.0).sin();
        if amp > 0.01 {
            self.shell.animating = true;
        }
        let col = if ding.is_some() { accent } else { tok.dim };
        let st = Stroke::new(1.8, col);
        let p = ui.painter();
        for side in [-1.0f32, 1.0] {
            let x = c.x + side * (4.0 + buzz);
            p.line_segment([Pos2::new(x, c.y - 10.0), Pos2::new(x, c.y + 2.0)], st);
        }
        p.add(egui::Shape::line(arc(Pos2::new(c.x, c.y + 2.0), 4.0, 0.0, std::f32::consts::PI, 10), st));
        p.line_segment([Pos2::new(c.x, c.y + 6.0), Pos2::new(c.x, c.y + 12.0)], st);
        if let Some(t) = ding {
            let a = (1.0 - t / 1.4).clamp(0.0, 1.0);
            ui.ctx().layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_fork_note")))
                .text(Pos2::new(c.x + 14.0, c.y - 12.0 - 26.0 * t), egui::Align2::LEFT_BOTTOM, "A = 440 Hz",
                    egui::FontId::proportional(12.0), with_alpha(accent, a));
        }
    }

    /// The current lyric line and its neighbours over the spectrum: drawn,
    /// not laid out, on a layer above it, so the analyser below keeps every
    /// pixel and every mouse interaction.
    fn paint_lyrics_overlay(&mut self, ctx: &egui::Context, canvas: Rect) {
        let Some(i) = self.current_index else { return };
        let t = &self.playlist[i];
        let tref = lyrics::ui::TrackRef { path: &t.path, title: &t.title, artist: &t.artist, album: &t.album, duration: t.duration };
        let pos = self.elapsed();
        let lines = self.lyrics_window.nearby_lines(tref, pos, 1, 2);
        let Some(cur) = lines.iter().find(|l| l.0 == 0) else { return };
        let now = Instant::now();
        if self.shell.overlay_line.as_ref().map(|(s, _)| s) != Some(&cur.1) {
            self.shell.overlay_line = Some((cur.1.clone(), now));
        }
        let since = self.shell.overlay_line.as_ref().map_or(1.0, |(_, at)| now.duration_since(*at).as_secs_f32());
        let secs = self.shell_secs(380.0);
        let k = if secs > 0.0 { ease_out_quart((since / secs).min(1.0)) } else { 1.0 };
        if k < 1.0 {
            self.shell.animating = true;
        }
        let alpha = self.appearance.visual.lyrics_overlay_opacity.clamp(0.1, 1.0);
        let accent = self.track_accent;
        let painter = ctx.layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_lyrics_overlay")))
            .with_clip_rect(canvas);
        // The plot is dark in either theme: light ink on a dark shadow.
        let ink = Color32::from_rgb(0xf4, 0xf5, 0xf8);
        let shadow = Color32::from_rgba_premultiplied(0, 0, 0, (200.0 * alpha) as u8);
        // A soft band behind the text, fading out downwards.
        let band = Rect::from_min_max(canvas.min, Pos2::new(canvas.right(), canvas.top() + canvas.height().min(260.0)));
        let mut mesh = egui::Mesh::default();
        let top = Color32::from_rgba_premultiplied(0, 0, 0, (150.0 * alpha) as u8);
        mesh.colored_vertex(band.left_top(), top);
        mesh.colored_vertex(band.right_top(), top);
        mesh.colored_vertex(band.right_bottom(), Color32::TRANSPARENT);
        mesh.colored_vertex(band.left_bottom(), Color32::TRANSPARENT);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);
        // An outline, so the text holds over the brightest heat-map colours.
        let outline = |painter: &egui::Painter, at: Pos2, g: &std::sync::Arc<egui::Galley>, c: Color32| {
            for d in [Vec2::new(-1.2, 0.0), Vec2::new(1.2, 0.0), Vec2::new(0.0, -1.2), Vec2::new(0.0, 1.2), Vec2::new(0.0, 2.0)] {
                painter.galley_with_override_text_color(at + d, g.clone(), c);
            }
        };
        let max_w = canvas.width() - 48.0;
        let mut y = canvas.top() + 18.0 + (1.0 - k) * 14.0;
        for (off, text, translation, upto) in &lines {
            let (size, a) = match off {
                0 => (26.0, 1.0),
                -1 => (17.0, 0.38),
                1 => (18.0, 0.62),
                _ => (16.0, 0.42),
            };
            // A short plot gets smaller words, so the lines leave some of it.
            let size = size * (canvas.height() / 380.0).clamp(0.6, 1.0);
            let a = a * alpha * if *off == 0 { 0.35 + 0.65 * k } else { 1.0 };
            let font = crate::name_font(text, size);
            let lit = with_alpha(lerp_color(accent, Color32::WHITE, 0.4), a);
            let base = with_alpha(ink, a);
            let mut job = egui::text::LayoutJob::default();
            job.wrap = egui::text::TextWrapping::truncate_at_width(max_w);
            job.halign = egui::Align::Center;
            let fmt = |c| egui::TextFormat { font_id: font.clone(), color: c, ..Default::default() };
            match upto {
                Some(u) => {
                    job.append(&text[..*u], 0.0, fmt(lit));
                    job.append(&text[*u..], 0.0, fmt(base));
                }
                None => job.append(text, 0.0, fmt(base)),
            }
            let g = ctx.fonts(|f| f.layout_job(job));
            let at = Pos2::new(canvas.center().x, y);
            outline(&painter, at, &g, shadow);
            painter.galley(at, g.clone(), base);
            y += g.size().y + 2.0;
            if let Some(tr) = translation {
                let tf = crate::name_font(tr, size * 0.72);
                let mut job = egui::text::LayoutJob::simple_singleline(tr.clone(), tf, with_alpha(ink, a * 0.8));
                job.wrap = egui::text::TextWrapping::truncate_at_width(max_w);
                job.halign = egui::Align::Center;
                let g = ctx.fonts(|f| f.layout_job(job));
                let at = Pos2::new(canvas.center().x, y);
                outline(&painter, at, &g, shadow);
                painter.galley(at, g.clone(), base);
                y += g.size().y;
            }
            y += 6.0;
        }
    }

    /// A round shell button that gives way when pressed and springs back past
    /// its size. Fixed footprint: only the drawing scales.
    #[allow(clippy::too_many_arguments)]
    fn fx_round(&mut self, ui: &mut egui::Ui, key: &'static str, icon: Icon, label: &str, size: f32,
                fill: Color32, hover: Color32, ink: Color32, enabled: bool) -> egui::Response {
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), if enabled { Sense::click() } else { Sense::hover() });
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, label));
        let lift = ui.ctx().animate_bool_with_time(resp.id, enabled && resp.hovered(), self.shell_secs(140.0));
        self.shell.round_ids.push(resp.id);
        if resp.clicked() {
            self.shell.pressed.insert(key, Instant::now());
            self.shell.fx_handled.push(resp.id);
        }
        let mut scale = 1.0;
        let k = self.fx_k();
        if self.appearance.visual.button_fx && self.appearance.visual.animations
            && let Some(at) = self.shell.pressed.get(key)
        {
            let t = at.elapsed().as_secs_f32() / self.appearance.visual.duration_scale.clamp(0.1, 4.0);
            let depth = (0.18 * k).min(0.36);
            scale = if t < 0.07 {
                1.0 - depth * (t / 0.07)
            } else {
                // A damped spring from 0.82: past 1 by ~7%, then settling.
                let u = t - 0.07;
                1.0 - depth * (-5.0 * u).exp() * (14.0 * u).cos()
            };
            if t < 1.0 {
                self.shell.animating = true;
            } else {
                self.shell.pressed.remove(key);
            }
        }
        if enabled && resp.is_pointer_button_down_on() {
            scale *= 0.94;
        }
        // Hovered, it swells a little and its disc fills in.
        scale *= 1.0 + 0.07 * lift * k;
        let ang = self.spin_angle(key);
        let r = size / 2.0 * scale;
        let bg = if fill == Color32::TRANSPARENT { hover.gamma_multiply(lift) } else { fill };
        let painter = ui.painter();
        if bg != Color32::TRANSPARENT {
            painter.circle_filled(rect.center(), r, bg);
        }
        let ink = if enabled { ink } else { with_alpha(ink, 0.35) };
        paint_icon(painter, rect.center(), size * 0.5 * scale, icon, ink, ang);
        if enabled {
            resp.on_hover_cursor(egui::CursorIcon::PointingHand)
        } else {
            resp
        }
    }

    /// A popup above a player-bar button, toggled by it. `close_on_pick`
    /// closes it on any click inside too, for a list of choices.
    fn bar_popup(&mut self, ui: &mut egui::Ui, key: &'static str, resp: &egui::Response, close_on_pick: bool,
                 body: impl FnOnce(&mut Self, &mut egui::Ui)) {
        let id = Id::new(("shell_bar_popup", key));
        if resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(id));
        }
        let close = if close_on_pick { egui::PopupCloseBehavior::CloseOnClick } else { egui::PopupCloseBehavior::CloseOnClickOutside };
        egui::popup_above_or_below_widget(ui, id, resp, egui::AboveOrBelow::Above, close, |ui| body(self, ui));
    }

    fn burst_at_pointer(&mut self, ctx: &egui::Context, accent: Color32, n: usize, power: f32, dir: Option<f32>) {
        if let Some(p) = ctx.pointer_interact_pos() {
            self.burst(p, accent, n, power, dir);
        }
    }

    /// Pieces flying out of `at`, coloured around the accent. `dir` leans
    /// them left (-1) or right (1); `None` is all round, mostly upwards.
    fn burst(&mut self, at: Pos2, accent: Color32, n: usize, power: f32, dir: Option<f32>) {
        let v = &self.appearance.visual;
        if v.confetti && v.animations {
            let k = self.fx_k();
            self.spray(at, accent, (n as f32 * k).round() as usize, power * (0.6 + 0.4 * k), dir);
        }
    }

    /// Confetti cannons from both bottom corners and a shower from the middle.
    /// `forced` is the farewell when confetti is switched off: it ignores the
    /// setting and goes all out.
    fn celebrate(&mut self, ctx: &egui::Context, accent: Color32, forced: bool) {
        let v = &self.appearance.visual;
        if !forced && !(v.confetti && v.animations) {
            return;
        }
        let k = if forced { 3.5 } else { self.fx_k() };
        let n = |c: f32| (c * k).round() as usize;
        let s = ctx.screen_rect();
        let power = 2.4 * (0.7 + 0.3 * k.min(2.0));
        self.spray(Pos2::new(s.left() + 20.0, s.bottom() - 20.0), accent, n(90.0), power, Some(0.75));
        self.spray(Pos2::new(s.right() - 20.0, s.bottom() - 20.0), accent, n(90.0), power, Some(-0.75));
        self.spray(s.center(), accent, n(70.0), 1.6, None);
        if forced {
            // And a downpour along the top edge.
            for i in 0..10 {
                let x = s.left() + s.width() * (i as f32 + 0.5) / 10.0;
                self.spray(Pos2::new(x, s.top() + 10.0), accent, 45, 0.9, Some(3.49));
            }
            self.spray(Pos2::new(s.center().x, s.bottom() - 10.0), accent, 160, 3.0, Some(0.0));
        }
    }

    /// The pieces themselves, whatever the setting says.
    fn spray(&mut self, at: Pos2, accent: Color32, n: usize, power: f32, dir: Option<f32>) {
        let [ar, ag, ab, _] = accent.to_srgba_unmultiplied();
        let palette = [
            accent,
            Color32::WHITE,
            Color32::from_rgb(ab, ar, ag),
            Color32::from_rgb(ag, ab, ar),
            lerp_color(accent, Color32::WHITE, 0.5),
        ];
        let now = Instant::now();
        let s = &mut self.shell;
        for _ in 0..n {
            let spread = match dir {
                Some(d) => -std::f32::consts::FRAC_PI_2 + d * 0.9 + (s.rand() - 0.5) * 1.6,
                None => -std::f32::consts::FRAC_PI_2 + (s.rand() - 0.5) * std::f32::consts::PI * 1.5,
            };
            let speed = (160.0 + 300.0 * s.rand()) * power;
            let color = palette[(s.rand() * palette.len() as f32) as usize % palette.len()];
            let p = Particle {
                pos: at,
                vel: Vec2::new(spread.cos(), spread.sin()) * speed,
                rot: s.rand() * std::f32::consts::TAU,
                spin: (s.rand() - 0.5) * 18.0,
                size: 3.0 + 5.0 * s.rand(),
                color,
                born: now,
                life: 0.7 + 0.6 * s.rand(),
                shape: (s.rand() * 4.0) as u8 % 4,
            };
            s.particles.push(p);
        }
    }

    /// A ripple for any other button clicked in the shell, then the particles;
    /// both on a layer above the panels.
    fn paint_fx(&mut self, ctx: &egui::Context, dt: f32) {
        let v = self.appearance.visual.clone();
        if v.button_fx && v.animations
            && let Some(id) = ctx.viewport(|vp| vp.interact_widgets.clicked)
            && !self.shell.fx_handled.contains(&id)
            && let Some(r) = ctx.read_response(id)
            && r.rect.width() < 400.0 && r.rect.height() < 120.0
        {
            self.shell.ripples.push((r.rect, Instant::now(), self.track_accent));
            // A few pieces from every button, not only the transport.
            let at = ctx.pointer_interact_pos().filter(|p| r.rect.contains(*p)).unwrap_or(r.rect.center());
            self.burst(at, self.track_accent, 7, 0.5, None);
        }
        self.shell.fx_handled.clear();
        let painter = ctx.layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_fx")));
        let now = Instant::now();
        let scale = v.duration_scale.clamp(0.1, 4.0);
        let k = self.fx_k();
        // A click on nothing at all leaves a ring on the background.
        // On the press, not the click: egui calls a click that moves a drag,
        // and a ring should not need a still hand.
        if self.toy("ripples") && ctx.input(|i| i.pointer.primary_pressed())
            && let Some(p) = ctx.input(|i| i.pointer.press_origin())
        {
            // Copied out first: read_response inside viewport() takes the
            // context lock twice and hangs the app.
            let hovered: Vec<Id> = ctx.viewport(|vp| vp.interact_widgets.hovered.iter().copied().collect());
            let on_widget = hovered.iter().any(|id| ctx.read_response(*id).is_some_and(|r| r.sense.senses_click() || r.sense.senses_drag()));
            if !on_widget {
                self.found("ripples");
                self.shell.rings.push((p, now));
                self.burst(p, self.track_accent, 6, 0.4, None);
            }
        }
        if let Some((key, at)) = self.shell.found_toast {
            let t = at.elapsed().as_secs_f32();
            if t > 3.0 {
                self.shell.found_toast = None;
            } else if let Some(toy) = TOYS.iter().find(|x| x.key == key) {
                let a = (t * 6.0).min(1.0).min((3.0 - t) * 2.0).clamp(0.0, 1.0);
                let rise = 12.0 * (1.0 - ease_out_quart((t * 3.0).min(1.0)));
                let vis = &ctx.style().visuals;
                let at = Pos2::new(ctx.screen_rect().center().x, ctx.screen_rect().bottom() - 150.0 + rise);
                let g = painter.layout_no_wrap(format!("🔓 Found: {}", toy.name), egui::FontId::proportional(13.0), with_alpha(vis.text_color(), a));
                let r = Rect::from_center_size(at, g.size() + Vec2::new(28.0, 14.0));
                painter.rect(r, 14, with_alpha(vis.window_fill, 0.95 * a), Stroke::new(1.0, with_alpha(self.track_accent, a)), egui::StrokeKind::Inside);
                painter.galley(r.center() - g.size() / 2.0, g, Color32::WHITE);
                self.shell.animating = true;
            }
        }
        if let Some((at, dark, from)) = self.shell.sky {
            let t = at.elapsed().as_secs_f32() / (1.4 * scale);
            if t >= 1.0 {
                self.shell.sky = None;
            } else {
                paint_sky(&painter, ctx.screen_rect(), from, t, dark, ctx.style().visuals.panel_fill);
                self.shell.animating = true;
            }
        }
        if !self.shell.hearts.is_empty() {
            let pink = Color32::from_rgb(240, 98, 146);
            self.shell.hearts.retain(|(c, at)| {
                let t = at.elapsed().as_secs_f32() / 1.3;
                if t >= 1.0 {
                    return false;
                }
                let q = *c + Vec2::new(6.0 * (t * 9.0).sin(), -50.0 * ease_out_quart(t));
                paint_heart(&painter, q, 5.0 + 3.0 * t, with_alpha(pink, 1.0 - t * t));
                true
            });
            self.shell.animating = true;
        }
        if !self.shell.memes.is_empty() {
            let accent = self.track_accent;
            let ink = ctx.style().visuals.text_color();
            self.shell.memes.retain(|(c, at, meme)| paint_meme(&painter, *c, at.elapsed().as_secs_f32(), *meme, ink, accent));
            self.shell.animating = true;
        }
        self.shell.rings.retain(|(p, at)| {
            let t = now.duration_since(*at).as_secs_f32() / (0.9 * scale);
            if t >= 1.0 {
                return false;
            }
            let a = 1.0 - t;
            for (i, lag) in [0.0f32, 0.18].iter().enumerate() {
                let tt = (t - lag).max(0.0);
                let rr = 4.0 + 70.0 * k * ease_out_quart(tt);
                painter.circle_stroke(*p, rr, Stroke::new(2.0 - i as f32, with_alpha(self.track_accent, 0.5 * a)));
            }
            true
        });
        if !self.shell.rings.is_empty() {
            self.shell.animating = true;
        }
        // Anything clickable under the pointer lights up, so a chip that is a
        // button reads as one.
        if v.button_fx && v.animations {
            let step = dt / (0.14 * scale);
            let hovered: Vec<Id> = ctx.viewport(|vp| vp.interact_widgets.hovered.iter().copied().collect());
            for h in self.shell.hovers.values_mut() {
                h.1 -= step;
            }
            for id in hovered {
                if let Some(r) = ctx.read_response(id)
                    && r.enabled() && r.sense.senses_click()
                    && r.rect.width() < 400.0 && r.rect.height() < 120.0
                {
                    let h = self.shell.hovers.entry(id).or_insert((r.rect, 0.0));
                    *h = (r.rect, (h.1 + 2.0 * step).min(1.0));
                }
            }
            self.shell.hovers.retain(|_, h| h.1 > 0.0);
            let accent = self.track_accent;
            for (id, (rect, a)) in &self.shell.hovers {
                if *a < 1.0 {
                    self.shell.animating = true;
                }
                let a = ease_out_quart(*a);
                let stroke = Stroke::new(1.5, with_alpha(accent, 0.6 * a));
                if self.shell.round_ids.contains(id) {
                    painter.circle_stroke(rect.center(), rect.width() / 2.0 + 3.0 * a * k, stroke);
                } else {
                    let r = rect.expand(2.5 * a * k);
                    let corner = (rect.height() / 2.0).min(10.0) + 2.0;
                    painter.rect_filled(r, corner, with_alpha(accent, 0.10 * a));
                    painter.rect_stroke(r, corner, stroke, egui::StrokeKind::Outside);
                }
            }
        } else {
            self.shell.hovers.clear();
        }
        self.shell.round_ids.clear();
        self.shell.ripples.retain(|(rect, at, c)| {
            let t = now.duration_since(*at).as_secs_f32() / (0.55 * scale);
            if t >= 1.0 {
                return false;
            }
            // The button itself bounces: a flash over it that squashes in,
            // springs out past its edge and settles.
            let u = t * 0.55 * scale;
            let bounce = 5.0 * k * (-7.0 * u).exp() * (18.0 * u - 0.6).sin();
            let base = (rect.height() / 2.0).min(12.0);
            painter.rect_filled(rect.expand(bounce), base + bounce.max(0.0), with_alpha(*c, 0.30 * (1.0 - t).powi(2)));
            let grow = 2.0 + 14.0 * k * ease_out_quart(t);
            let corner = base + grow;
            painter.rect_stroke(rect.expand(grow), corner, Stroke::new(2.0 * (1.0 - t), with_alpha(*c, 0.7 * (1.0 - t))),
                egui::StrokeKind::Outside);
            true
        });
        // Rows out of a cleared playlist, one after another, tumbling.
        if !self.shell.falling.is_empty() {
            let ink = ctx.style().visuals.text_color();
            let accent = self.track_accent;
            self.shell.falling.retain(|(r, title, at)| {
                let t = if now >= *at { now.duration_since(*at).as_secs_f32() } else { 0.0 };
                if t > 1.6 {
                    return false;
                }
                let c = r.center();
                let turn = if title.len() % 2 == 0 { 0.5 } else { -0.5 } * t;
                let down = Vec2::new(30.0 * turn, 0.5 * 2200.0 * t * t);
                let (sn, cs) = turn.sin_cos();
                let xf = |p: Pos2| {
                    let v = p - c;
                    c + down + Vec2::new(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
                };
                let a = 1.0 - t / 1.6;
                painter.add(rounded_quad(*r, 6.0, None, with_alpha(accent, 0.22 * a), &xf));
                painter.text(xf(r.left_center() + Vec2::new(10.0, 0.0)), egui::Align2::LEFT_CENTER, title,
                    egui::FontId::proportional(13.0), with_alpha(ink, a));
                true
            });
            self.shell.animating = true;
        }
        // The lyrics' Follow tick floats off like a balloon and drifts back.
        if let Some((r, at)) = self.shell.balloon {
            let t = at.elapsed().as_secs_f32();
            if t >= 3.2 {
                self.shell.balloon = None;
            } else {
                let h = if t < 1.6 { (t / 1.6).powi(2) } else { 1.0 - ease_out_quart((t - 1.6) / 1.6) };
                let home = Pos2::new(r.left() + 9.0, r.center().y);
                let b = home + Vec2::new((t * 3.0).sin() * 12.0 * h, -230.0 * h - 16.0 * h.sqrt());
                let size = 4.0 + 7.0 * h.sqrt();
                let accent = self.track_accent;
                // The string, wavy, down to where it was tied.
                let tail = b + Vec2::new(0.0, size + 14.0 * h.sqrt());
                let pts: Vec<Pos2> = (0..=8).map(|i| {
                    let f = i as f32 / 8.0;
                    b + Vec2::new((f * 9.0 + t * 6.0).sin() * 2.0, size) + (tail - b - Vec2::new(0.0, size)) * f
                }).collect();
                painter.add(egui::Shape::line(pts, Stroke::new(1.0, with_alpha(accent, 0.8))));
                painter.add(egui::Shape::ellipse_filled(b, Vec2::new(size, size * 1.15), accent));
                painter.add(egui::Shape::line(vec![b + Vec2::new(-0.45, 0.0) * size, b + Vec2::new(-0.1, 0.35) * size,
                    b + Vec2::new(0.5, -0.4) * size], Stroke::new(1.6, Color32::WHITE)));
                self.shell.animating = true;
            }
        }
        let gravity = 900.0;
        self.shell.particles.retain_mut(|p| {
            let age = now.duration_since(p.born).as_secs_f32();
            if age >= p.life {
                return false;
            }
            p.vel.y += gravity * dt;
            p.vel *= (1.0 - 1.6 * dt).max(0.0);
            p.pos += p.vel * dt;
            p.rot += p.spin * dt;
            let c = with_alpha(p.color, 1.0 - (age / p.life).powi(2));
            let (sn, cs) = p.rot.sin_cos();
            let at = |x: f32, y: f32| p.pos + Vec2::new(x * cs - y * sn, x * sn + y * cs);
            match p.shape {
                0 => {
                    // A strip that flutters: its width follows the spin.
                    let w = p.size * 0.5 * (p.rot * 1.7).sin().abs().max(0.2);
                    let h = p.size;
                    painter.add(egui::Shape::convex_polygon(vec![at(-w, -h), at(w, -h), at(w, h), at(-w, h)], c, Stroke::NONE));
                }
                1 => { painter.circle_filled(p.pos, p.size * 0.45, c); }
                2 => {
                    let r = p.size * 0.8;
                    painter.add(egui::Shape::convex_polygon(
                        vec![at(0.0, -r), at(r * 0.87, r * 0.5), at(-r * 0.87, r * 0.5)], c, Stroke::NONE));
                }
                _ => { painter.circle_stroke(p.pos, p.size * 0.7, Stroke::new(1.5, c)); }
            }
            true
        });
        if !self.shell.particles.is_empty() || !self.shell.ripples.is_empty() {
            self.shell.animating = true;
        }
    }

    /// Stroked gently back and forth, the cover gives off hearts.
    fn pet_cover(&mut self, ctx: &egui::Context, r: Rect) {
        let (pos, vx, down) = ctx.input(|i| (i.pointer.hover_pos(), i.pointer.velocity().x, i.pointer.any_down()));
        let Some(p) = pos.filter(|p| r.contains(*p) && !down && self.toy("pet")) else {
            self.shell.pet_dir = 0.0;
            self.shell.pet_turns.clear();
            return;
        };
        // Gently: moving, but not flung.
        if !(40.0..900.0).contains(&vx.abs()) {
            return;
        }
        let dir = vx.signum();
        if self.shell.pet_dir != 0.0 && dir != self.shell.pet_dir {
            let now = Instant::now();
            self.shell.pet_turns.retain(|t| now.duration_since(*t).as_secs_f32() < 2.5);
            self.shell.pet_turns.push(now);
            if self.shell.pet_turns.len() >= 3 {
                self.found("pet");
                self.shell.hearts.push((p, now));
            }
        }
        self.shell.pet_dir = dir;
    }

    fn paint_cover(&mut self, ctx: &egui::Context, tok: &Tokens, accent: Color32, mode: UiMode, dt: f32) {
        let Some(target) = self.shell.cover_target else { return };
        let secs = self.shell_secs(520.0);
        let now = Instant::now();
        let shown = match self.shell.cover {
            Some(a) => {
                let t = if a.secs > 0.0 { (now.duration_since(a.start).as_secs_f32() / a.secs).min(1.0) } else { 1.0 };
                (lerp_rect(a.from.0, a.to.0, ease_out_quart(t)), lerp(a.from.1, a.to.1, ease_out_quart(t)))
            }
            None => target,
        };
        self.pet_cover(ctx, shown.0);
        if let Some(at) = self.shell.hearts_at {
            let q = now.duration_since(at).as_secs_f32() / 3.5;
            if q >= 1.0 {
                self.shell.hearts_at = None;
            } else {
                self.shell.animating = true;
                let p = ctx.layer_painter(LayerId::new(egui::Order::Foreground, Id::new("shell_hearts")));
                for k in 0..6 {
                    let f = ((q * 1.6 - k as f32 * 0.12).max(0.0)).min(1.0);
                    if f <= 0.0 {
                        continue;
                    }
                    let x = shown.0.left() + shown.0.width() * (0.15 + 0.14 * k as f32) + 6.0 * (f * 9.0 + k as f32).sin();
                    let c = Pos2::new(x, shown.0.bottom() - shown.0.height() * 0.9 * f);
                    paint_heart(&p, c, 5.0 + (k % 3) as f32, with_alpha(Color32::from_rgb(0xff, 0x6b, 0x8b), (1.0 - f) * (f * 10.0).min(1.0)));
                }
            }
        }
        let mode_changed = self.shell.cover_mode != Some(mode);
        self.shell.cover_mode = Some(mode);
        let screen = ctx.screen_rect();
        let resized = self.shell.cover_screen.is_some_and(|s| s != screen);
        self.shell.cover_screen = Some(screen);
        let anim = match self.shell.cover {
            // A mode switch moves the cover from wherever it is now.
            Some(_) if mode_changed => CoverAnim { from: shown, to: target, start: now, secs },
            // A resize follows at once.
            Some(a) if a.to != target && resized => CoverAnim { from: target, to: target, start: now, secs: 0.0 },
            // A target that moves mid-flight (the content rises in after a
            // mode switch) is followed by the same flight; once landed, a
            // new layout (the ⋯ panel) starts a shorter one.
            Some(a) if a.to != target => {
                if now.duration_since(a.start).as_secs_f32() < a.secs {
                    CoverAnim { to: target, ..a }
                } else {
                    CoverAnim { from: shown, to: target, start: now, secs: self.shell_secs(320.0) }
                }
            }
            Some(a) => a,
            None => CoverAnim { from: target, to: target, start: now, secs: 0.0 },
        };
        self.shell.cover = Some(anim);
        let t = if anim.secs > 0.0 { (now.duration_since(anim.start).as_secs_f32() / anim.secs).min(1.0) } else { 1.0 };
        if t < 1.0 {
            self.shell.animating = true;
        }
        let rect = lerp_rect(anim.from.0, anim.to.0, ease_out_quart(t));
        let radius = lerp(anim.from.1, anim.to.1, ease_out_quart(t));
        let art = self.current_index.and_then(|i| self.playlist.get(i)).map(|t| t.path.clone())
            .and_then(|p| self.art_cache.get_or_load(&p, ctx));
        let glow = self.appearance.visual.cover_glow;
        let energy = self.shell.energy;
        let big = (rect.width() / 400.0).clamp(0.0, 1.0);
        let toy_on = self.toy("throw");
        if !toy_on {
            self.shell.toy = CoverToy::default();
        }
        let toy = self.shell.toy;
        // Where the toy has the cover now, and what it does to its shape.
        let home = rect;
        let rect = rect.translate(toy.off);
        let speed = toy.vel.length();
        let k = self.fx_k();
        let lift = if toy.grab.is_some() { 1.0 + 0.03 * k } else { 1.0 };
        // Squashed along the way it moves and taller across it, like
        // something soft pushed through the air; plus the boing.
        let stretch = (speed / 9000.0).min(0.10) * k;
        let dir = if speed > 1.0 { toy.vel / speed } else { Vec2::X };
        let pulse = 1.0 + toy.squash;
        // Five clicks turn it over; right-click makes it a record.
        let flip = self.shell.tween("flip", if self.shell.flipped && self.toy("flip") { 1.0 } else { 0.0 }, self.shell_secs(700.0), ease_out_back);
        let sx = (flip * std::f32::consts::PI).cos();
        let back = sx < 0.0;
        let vinyl = self.shell.tween("vinyl", if self.shell.vinyl && self.toy("vinyl") { 1.0 } else { 0.0 }, self.shell_secs(500.0), ease_out_quart);
        if vinyl > 0.0 && self.play_state == PlayState::Playing {
            self.shell.vinyl_ang = (self.shell.vinyl_ang + dt * 3.5 * vinyl) % std::f32::consts::TAU;
            self.shell.animating = true;
        } else if vinyl < 1.0 && self.shell.vinyl_ang != 0.0 {
            // Off again, it turns back upright.
            let a = self.shell.vinyl_ang;
            let a = if a > std::f32::consts::PI { a - std::f32::consts::TAU } else { a };
            self.shell.vinyl_ang = if a.abs() < 0.002 { 0.0 } else { a * (1.0 - (dt * 8.0).min(1.0)) };
            self.shell.animating = true;
        }
        let radius = lerp(radius, rect.width().min(rect.height()) / 2.0, vinyl);
        // Left alone long enough, it nods off.
        let doze = self.toy("doze") && self.idle_secs() > DOZE_SECS;
        if doze {
            self.found("doze");
        }
        let now_t = ctx.input(|i| i.time) as f32;
        let nod = if doze { 0.06 * (now_t * 1.3).sin() } else { 0.0 };
        if doze {
            self.shell.animating = true;
        }
        let spin = toy.ang + self.shell.vinyl_ang + nod;
        let back_text: Vec<(String, f32)> = match self.current_index.and_then(|i| self.playlist.get(i)).filter(|_| back) {
            Some(t) => {
                let plays = self.stats.get(&t.path).map_or(0, |s| s.count);
                let fmt = match (t.sample_rate, t.bit_depth) {
                    (Some(sr), Some(b)) => format!("{} · {}-bit", fmt_rate(sr), b),
                    (Some(sr), None) => fmt_rate(sr),
                    _ => String::new(),
                };
                vec![(t.title.clone(), 1.25), (t.artist.clone(), 1.0), (t.album.clone(), 0.9), (fmt, 0.8),
                     (format!("played {plays}×"), 0.8)]
            }
            None => Vec::new(),
        };
        let medal = self.shell.medal.filter(|(_, at)| at.elapsed().as_secs_f32() < 6.0);
        if medal.is_some() {
            self.shell.animating = true;
        } else {
            self.shell.medal = None;
        }
        let mut resp = None;
        egui::Area::new(Id::new("shell_cover")).order(egui::Order::Middle).fixed_pos(rect.min)
            .interactable(toy_on).show(ctx, |ui| {
                if toy_on {
                    let r = ui.interact(rect, Id::new("shell_cover_toy"), Sense::click_and_drag())
                        .on_hover_cursor(if toy.grab.is_some() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
                    resp = Some(r);
                }
                let r = CornerRadius::same(radius as u8);
                // Local point -> screen: squash and stretch, tilt, then place.
                let c = rect.center();
                let (sn, cs) = spin.sin_cos();
                let xf = |p: Pos2| {
                    let v = (p - c) * lift * pulse;
                    let v = Vec2::new(v.x * sx.abs().max(0.02), v.y);
                    let along = v.dot(dir);
                    let across = v - dir * along;
                    let v = dir * along * (1.0 - stretch * 0.6) + across * (1.0 + stretch);
                    c + Vec2::new(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
                };
                let still = toy.off.length() < 0.3 && spin.abs() < 0.002 && toy.squash.abs() < 0.002 && toy.grab.is_none()
                    && sx > 0.999 && vinyl == 0.0;
                if glow && big > 0.2 {
                    // The glow sits under the cover and spreads as it lifts.
                    let up = toy.off.length().min(200.0) / 200.0;
                    let drop = Vec2::new(0.0, (18.0 + 20.0 * up) * big);
                    let blur = (50.0 + 40.0 * energy + 40.0 * up) * big;
                    let color = with_alpha(accent, (0.30 + 0.35 * energy) * big * (1.0 - 0.4 * up));
                    if still {
                        let shadow = egui::epaint::Shadow {
                            offset: [0, drop.y as i8],
                            blur: blur as u8,
                            spread: 0,
                            color,
                        };
                        ui.painter().add(shadow.as_shape(rect, r));
                    } else {
                        // Moving, it is drawn through the same squash and tilt as
                        // the cover: stacked soft layers instead of the blur.
                        let layers = 10;
                        let [cr, cg, cb, ca] = color.to_array();
                        for i in (0..layers).rev() {
                            let f = (i as f32 + 0.5) / layers as f32;
                            let grow = blur * 0.5 * f;
                            let w = (1.0 - f) * (1.0 - f) * 2.2 / layers as f32;
                            let m = |c: u8| (c as f32 * w).round() as u8;
                            let c = Color32::from_rgba_premultiplied(m(cr), m(cg), m(cb), m(ca));
                            let xg = |p: Pos2| xf(p) + drop;
                            ui.painter().add(rounded_quad(rect.expand(grow), radius + grow, None, c, &xg));
                        }
                    }
                }
                if back {
                    // The back: what is playing, in words.
                    ui.painter().add(rounded_quad(rect, radius, None, lerp_color(accent, Color32::BLACK, 0.78), &xf));
                    let a = (-sx).clamp(0.0, 1.0);
                    let base = (rect.height() * 0.06).clamp(9.0, 26.0);
                    let n = back_text.len() as f32;
                    for (j, (line, s)) in back_text.iter().enumerate() {
                        if line.is_empty() {
                            continue;
                        }
                        let line: String = if line.chars().count() > 30 { line.chars().take(29).chain(['…']).collect() } else { line.clone() };
                        let y = (j as f32 - (n - 1.0) / 2.0) * base * 1.6;
                        ui.painter().text(xf(rect.center() + Vec2::new(0.0, y)), egui::Align2::CENTER_CENTER, &line,
                            crate::name_font(&line, base * s), with_alpha(Color32::WHITE, a * if j == 0 { 1.0 } else { 0.75 }));
                    }
                } else { match art {
                    Some((tex, w, h)) => {
                        let fit = super::fit_rect_preserve(rect, w, h);
                        if still {
                            ui.painter().rect_filled(rect, r, Color32::BLACK);
                            egui::Image::new((tex, fit.size())).corner_radius(r).paint_at(ui, fit);
                        } else {
                            ui.painter().add(rounded_quad(rect, radius, None, Color32::BLACK, &xf));
                            ui.painter().add(rounded_quad(fit, radius.min(fit.width() / 2.0), Some(tex), Color32::WHITE, &xf));
                        }
                    }
                    None => {
                        ui.painter().add(rounded_quad(rect, radius, None, with_alpha(accent, 0.35), &xf));
                        ui.painter().text(xf(rect.center()), egui::Align2::CENTER_CENTER, "♪",
                            egui::FontId::proportional(rect.height() * 0.4 * lift * pulse), tok.ink);
                    }
                }
                if vinyl > 0.0 {
                    // Grooves, a label and the hole.
                    let cc = xf(rect.center());
                    let rr = rect.width().min(rect.height()) / 2.0 * lift * pulse;
                    for g in [0.93, 0.82, 0.71, 0.6] {
                        ui.painter().circle_stroke(cc, rr * g, Stroke::new(1.0, with_alpha(Color32::BLACK, 0.4 * vinyl)));
                    }
                    ui.painter().circle_filled(cc, rr * 0.17, with_alpha(accent, vinyl));
                    ui.painter().circle_filled(cc, rr * 0.035, with_alpha(Color32::BLACK, vinyl));
                } }
                if doze {
                    // zZ, rising from a corner.
                    for i in 0..3 {
                        let f = (now_t * 0.45 + i as f32 / 3.0).fract();
                        let at = xf(rect.right_top()) + Vec2::new(4.0 + f * 22.0, -f * 46.0);
                        ui.painter().text(at, egui::Align2::CENTER_CENTER, if i % 2 == 0 { "z" } else { "Z" },
                            egui::FontId::proportional(13.0 + f * 12.0), with_alpha(tok.ink, (f * std::f32::consts::PI).sin()));
                    }
                }
                if let Some((plays, at)) = medal {
                    // A medal for a round hundred, pinned to the corner.
                    let t = at.elapsed().as_secs_f32();
                    let pop = ease_out_back((t * 2.5).min(1.0)) * (6.0 - t).clamp(0.0, 1.0);
                    let m = xf(rect.right_top()) + Vec2::new(-14.0, 14.0);
                    let gold = Color32::from_rgb(255, 200, 70);
                    let rr = 30.0 * pop;
                    for s in [-1.0f32, 1.0] {
                        ui.painter().add(egui::Shape::convex_polygon(vec![m + Vec2::new(s * 4.0, 0.0) * pop,
                            m + Vec2::new(s * 16.0, 40.0) * pop, m + Vec2::new(s * 6.0, 34.0) * pop], with_alpha(accent, pop.min(1.0)), Stroke::NONE));
                    }
                    ui.painter().circle(m, rr, gold, Stroke::new(2.5 * pop, lerp_color(gold, Color32::BLACK, 0.3)));
                    ui.painter().text(m - Vec2::new(0.0, 4.0 * pop), egui::Align2::CENTER_CENTER, plays.to_string(),
                        egui::FontId::proportional((17.0 * pop).max(1.0)), Color32::from_rgb(70, 45, 0));
                    ui.painter().text(m + Vec2::new(0.0, 11.0 * pop), egui::Align2::CENTER_CENTER, "plays",
                        egui::FontId::proportional((8.0 * pop).max(1.0)), Color32::from_rgb(70, 45, 0));
                }
            });
        if toy_on {
            self.step_cover_toy(ctx, resp, home, dt, accent);
        }
    }

    /// The cover as a toy: it follows a drag a little late, swings with the
    /// motion, can be thrown, and springs home past its place when let go.
    fn step_cover_toy(&mut self, ctx: &egui::Context, resp: Option<egui::Response>, home: Rect, dt: f32, accent: Color32) {
        let Some(resp) = resp else { return };
        let k = self.fx_k();
        let now = Instant::now();
        if self.toy("vinyl") && resp.secondary_clicked() {
            self.found("vinyl");
            self.shell.vinyl = !self.shell.vinyl;
        }
        if self.toy("flip") && resp.clicked() {
            if self.shell.flipped {
                self.shell.flipped = false;
            } else {
                self.shell.cover_clicks.retain(|t| now.duration_since(*t).as_secs_f32() < 2.0);
                self.shell.cover_clicks.push(now);
                if self.shell.cover_clicks.len() >= 5 {
                    self.shell.cover_clicks.clear();
                    self.shell.flipped = true;
                    self.found("flip");
                }
            }
        }
        if resp.drag_stopped() {
            self.found("throw");
        }
        let (sparks, wraps) = (self.toy("sparks"), self.toy("wrap"));
        let mut wrapped = false;
        let t = &mut self.shell.toy;
        if resp.drag_started() {
            t.grab = Some(t.off);
        }
        if let Some(g) = t.grab.as_mut() {
            *g += resp.drag_delta();
        }
        let mut poke = false;
        if resp.drag_stopped() {
            t.grab = None;
            // Thrown hard, it flies free for a moment.
            if wraps && t.vel.length() > 2500.0 {
                self.shell.fly_until = Some(now + std::time::Duration::from_millis(600));
            }
        } else if resp.clicked() {
            // A poke: it jumps, spins a little and squashes back.
            t.vel += Vec2::new(0.0, -450.0) * k;
            t.angv += if t.ang >= 0.0 { -6.0 } else { 6.0 } * k;
            t.squashv -= 3.5 * k;
            poke = true;
        }
        let dt = dt.clamp(1.0 / 400.0, 1.0 / 30.0);
        let flying = t.grab.is_none() && self.shell.fly_until.is_some_and(|u| now < u);
        // Held, it trails the pointer on a stiff spring; let go, a looser one
        // brings it home with an overshoot or two.
        let (target, stiff, c) = match t.grab {
            Some(g) => (g, 520.0, 26.0),
            None if flying => (Vec2::ZERO, 0.0, 0.8),
            // Critically damped: from a screen away, the usual spring would
            // fling it out past the far edge and round again.
            None if self.shell.homing => (Vec2::ZERO, 170.0, 26.0),
            None => (Vec2::ZERO, 170.0, 9.0),
        };
        // Substeps keep the spring stable at low frame rates.
        let n = 4;
        let h = dt / n as f32;
        for _ in 0..n {
            let acc = (target - t.off) * stiff - t.vel * c;
            t.vel += acc * h;
            t.off += t.vel * h;
            // It leans into horizontal motion, and a hovered cover tips up
            // on the side the pointer is on, like a card picked up.
            let mut lean = (t.vel.x * 0.0005 * k).clamp(-0.25, 0.25);
            if t.grab.is_none() && resp.hovered()
                && let Some(p) = ctx.pointer_hover_pos()
            {
                lean -= (p.x - (home.center().x + t.off.x)) / home.width().max(1.0) * 0.06 * k;
            }
            let aacc = (lean - t.ang) * 260.0 - t.angv * 11.0;
            t.angv += aacc * h;
            t.ang += t.angv * h;
            let sacc = -t.squash * 300.0 - t.squashv * 10.0;
            t.squashv += sacc * h;
            t.squash += t.squashv * h;
        }
        // Out past an edge, it comes back in from the other side.
        if t.grab.is_some() || t.off.length() < 4.0 {
            self.shell.homing = false;
        }
        if wraps && t.grab.is_none() && !self.shell.homing {
            let s = ctx.screen_rect();
            let p = home.center() + t.off;
            let (w, h) = (s.width() + home.width(), s.height() + home.height());
            let before = t.off;
            if p.x < s.left() - home.width() / 2.0 {
                t.off.x += w;
            } else if p.x > s.right() + home.width() / 2.0 {
                t.off.x -= w;
            }
            if p.y < s.top() - home.height() / 2.0 {
                t.off.y += h;
            } else if p.y > s.bottom() + home.height() / 2.0 {
                t.off.y -= h;
            }
            if t.off != before {
                wrapped = true;
                self.shell.fly_until = None;
                self.shell.homing = true;
                let v = t.vel.length();
                if v > 3000.0 {
                    t.vel *= 3000.0 / v;
                }
            }
        }
        let moving = flying || t.grab.is_some() || t.off.length() > 0.2 || t.vel.length() > 2.0
            || t.ang.abs() > 0.001 || t.angv.abs() > 0.01 || t.squash.abs() > 0.001 || t.squashv.abs() > 0.01;
        if !moving {
            *t = CoverToy::default();
        }
        if moving {
            self.shell.animating = true;
        }
        if poke {
            self.burst(home.center() + self.shell.toy.off, accent, 18, 0.8, None);
        }
        if wrapped {
            self.found("wrap");
        }
        // Thrown hard, it leaves sparks behind.
        let v = self.shell.toy.vel;
        if sparks && v.length() > 1600.0 {
            self.found("sparks");
            self.burst(home.center() + self.shell.toy.off, accent, 2, 0.3, None);
        }
    }
}

/// An eighth note: a head and a stem with a flag.
fn paint_note(p: &egui::Painter, c: Pos2, r: f32, col: Color32) {
    p.circle_filled(c, r, col);
    let top = c + Vec2::new(r * 0.9, -r * 3.2);
    p.line_segment([c + Vec2::new(r * 0.9, 0.0), top], Stroke::new(r * 0.35, col));
    p.line_segment([top, top + Vec2::new(r * 1.1, r * 1.0)], Stroke::new(r * 0.35, col));
}

/// A worried little face with a drop of sweat.
fn paint_worry(p: &egui::Painter, c: Pos2, t: f32, ink: Color32) {
    p.circle_stroke(c, 9.0, Stroke::new(1.2, ink));
    p.circle_filled(c + Vec2::new(-3.2, -2.0), 1.3, ink);
    p.circle_filled(c + Vec2::new(3.2, -2.0), 1.3, ink);
    p.circle_stroke(c + Vec2::new(0.0, 3.6), 1.8, Stroke::new(1.0, ink));
    let drop = c + Vec2::new(10.0, -6.0 + 3.0 * (t * 2.0).fract());
    let blue = Color32::from_rgb(110, 170, 240);
    p.circle_filled(drop, 2.2, blue);
    p.add(egui::Shape::convex_polygon(vec![drop + Vec2::new(-2.1, -0.6), drop + Vec2::new(0.0, -5.0), drop + Vec2::new(2.1, -0.6)], blue, Stroke::NONE));
}

/// A heart of size `r`, centred at `c`: two lobes and a point.
fn paint_heart(p: &egui::Painter, c: Pos2, r: f32, col: Color32) {
    let lobe = r * 0.55;
    p.circle_filled(c + Vec2::new(-lobe, -lobe * 0.4), lobe, col);
    p.circle_filled(c + Vec2::new(lobe, -lobe * 0.4), lobe, col);
    p.add(egui::Shape::convex_polygon(vec![c + Vec2::new(-r * 1.08, -lobe * 0.15), c + Vec2::new(r * 1.08, -lobe * 0.15), c + Vec2::new(0.0, r * 1.1)],
        col, Stroke::NONE));
}

/// An open hand, palm out, the wrist at `wrist`, pointing along `ang`.
fn paint_hand(p: &egui::Painter, wrist: Pos2, ang: f32, col: Color32) {
    let up = Vec2::angled(ang);
    let side = up.rot90();
    let palm = wrist + up * 6.0;
    p.circle_filled(palm, 4.6, col);
    let st = Stroke::new(2.2, col);
    // Four fingers fanned a little, the middle ones longest.
    for (i, len) in [5.0, 6.5, 6.5, 5.0].into_iter().enumerate() {
        let off = (i as f32 - 1.5) * 2.4;
        let base = palm + side * off + up * 3.0;
        let dir = (up + side * off * 0.06).normalized();
        p.line_segment([base, base + dir * len], st);
    }
    let thumb = palm - side * 4.0;
    p.line_segment([thumb, thumb + (up * 0.6 - side).normalized() * 4.5], st);
    p.line_segment([wrist, palm], Stroke::new(3.0, col));
}

/// The sun rising (or the moon) along an arc from the theme switch to the
/// left side of the window. `t` runs 0..1.
fn paint_sky(p: &egui::Painter, screen: Rect, from: Pos2, t: f32, dark: bool, bg: Color32) {
    let a = (t * 6.0).min(1.0).min((1.0 - t) * 4.0).clamp(0.0, 1.0);
    let e = t * t * (3.0 - 2.0 * t);
    let x = from.x + (screen.left() + 80.0 - from.x) * e;
    let y = from.y + 30.0 + (std::f32::consts::PI * e).sin() * 70.0;
    let c = Pos2::new(x, y);
    if dark {
        // A crescent: a pale disc with a bite of background out of it.
        p.circle_filled(c, 10.0, with_alpha(Color32::from_rgb(236, 232, 210), a));
        p.circle_filled(c + Vec2::new(4.5, -3.0), 8.5, with_alpha(bg, a));
    } else {
        let sun = Color32::from_rgb(250, 196, 60);
        p.circle_filled(c, 8.0, with_alpha(sun, a));
        for i in 0..8 {
            let d = Vec2::angled(i as f32 * std::f32::consts::TAU / 8.0 + t * 3.0);
            p.line_segment([c + d * 11.0, c + d * 15.0], Stroke::new(2.0, with_alpha(sun, a)));
        }
    }
}

/// 67: two palms up, see-sawing, "6" and "7" over them. 69: nice.
/// False once it is over.
fn paint_meme(p: &egui::Painter, c: Pos2, t: f32, meme: u8, ink: Color32, accent: Color32) -> bool {
    const LIFE: f32 = 2.6;
    if t >= LIFE {
        return false;
    }
    let a = (t * 5.0).min(1.0).min((LIFE - t) * 2.5).clamp(0.0, 1.0);
    let pop = 0.6 + 0.4 * ease_out_back(t * 4.0).min(1.2);
    if meme == 1 {
        // "bye~", drifting off to the right with a little wave.
        let q = c + Vec2::new(30.0 * t, -14.0 * t + 3.0 * (t * 12.0).sin());
        p.text(q, egui::Align2::CENTER_CENTER, "bye~", egui::FontId::proportional(14.0 * pop), with_alpha(ink, a));
        paint_hand(p, q + Vec2::new(-30.0, 9.0), -std::f32::consts::FRAC_PI_2 + (t * 12.0).sin() * 0.45, with_alpha(accent, a));
        return true;
    }
    if meme == 69 {
        let q = c - Vec2::new(0.0, 10.0 * t.min(1.0));
        p.text(q, egui::Align2::CENTER_BOTTOM, "nice.", egui::FontId::proportional(14.0 * pop), with_alpha(ink, a));
        for k in 0..3 {
            let ang = t * 2.0 + k as f32 * std::f32::consts::TAU / 3.0;
            let s = q + Vec2::new(26.0 * ang.cos(), -10.0 + 8.0 * ang.sin());
            p.circle_filled(s, 1.6 + (t * 9.0 + k as f32).sin().abs() * 1.2, with_alpha(accent, a));
        }
        return true;
    }
    for (i, label) in ["6", "7"].into_iter().enumerate() {
        let side = if i == 0 { -1.0 } else { 1.0 };
        // Out of step: one up while the other is down.
        let bob = 6.0 * (t * std::f32::consts::TAU * 2.2).sin() * side;
        let h = c + Vec2::new(14.0 * side * pop, bob);
        let col = with_alpha(ink, a);
        // A palm, face up, and its thumb.
        p.add(egui::Shape::ellipse_filled(h, Vec2::new(7.0, 3.0) * pop, col));
        p.line_segment([h + Vec2::new(5.0 * side, -1.0), h + Vec2::new(8.0 * side, -5.0)], Stroke::new(2.0, col));
        p.text(h - Vec2::new(0.0, 6.0), egui::Align2::CENTER_BOTTOM, label,
            egui::FontId::proportional(13.0 * pop), with_alpha(accent, a));
    }
    true
}

/// A little cassette; its reels turn when `spin` (an angle) is given.
fn cassette(p: &egui::Painter, c: Pos2, col: Color32, spin: Option<f32>) {
    let body = Rect::from_center_size(c, Vec2::new(20.0, 13.0));
    p.rect_stroke(body, 2, Stroke::new(1.3, col), egui::StrokeKind::Inside);
    for dx in [-4.5f32, 4.5] {
        let rc = c + Vec2::new(dx, -1.0);
        p.circle_stroke(rc, 2.6, Stroke::new(1.1, col));
        let a = spin.unwrap_or(0.0);
        for k in 0..3 {
            let ang = a + k as f32 * std::f32::consts::TAU / 3.0;
            p.line_segment([rc, rc + Vec2::angled(ang) * 2.6], Stroke::new(1.0, col));
        }
    }
    p.line_segment([Pos2::new(body.left() + 5.0, body.bottom() - 2.5), Pos2::new(body.right() - 5.0, body.bottom() - 2.5)],
        Stroke::new(1.0, col));
}

/// A titled card in the settings drawer.
fn settings_section(ui: &mut egui::Ui, tok: &Tokens, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.label(RichText::new(title.to_uppercase()).size(11.0).strong().color(tok.dim).extra_letter_spacing(0.8));
    ui.add_space(4.0);
    egui::Frame::NONE.fill(tok.scrim).stroke(Stroke::new(1.0, tok.line)).corner_radius(12)
        .inner_margin(Margin::same(12)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            body(ui);
        });
    ui.add_space(14.0);
}

/// A secondary window's main area: the page colour with room round the
/// edges, or the classic panel when `fill` is None.
pub(crate) fn page_frame(fill: Option<Color32>, style: &egui::Style) -> egui::Frame {
    fill.map_or_else(|| egui::Frame::central_panel(style), |f| egui::Frame::NONE.fill(f).inner_margin(Margin::symmetric(20, 14)))
}

/// A button label for a secondary window: the 1.6.0 look drops the leading
/// emoji ("💾 Save" -> "Save"); the classic one keeps it.
pub(crate) fn plain(fill: Option<Color32>, label: &str) -> &str {
    match label.split_once(' ') {
        Some((icon, rest)) if fill.is_some() && !icon.chars().any(char::is_alphanumeric) => rest,
        _ => label,
    }
}

/// A secondary window's toolbar or status bar, on the same page colour.
pub(crate) fn bar_frame(fill: Option<Color32>, style: &egui::Style) -> egui::Frame {
    fill.map_or_else(|| egui::Frame::side_top_panel(style), |f| egui::Frame::NONE.fill(f).inner_margin(Margin::symmetric(20, 8)))
}

/// Below this width the player bar keeps only transport and volume; the rest
/// moves to the settings drawer.
fn shell_compact(width: f32) -> bool {
    width < 1000.0
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let [ar, ag, ab, aa] = a.to_array();
    let [br, bg, bb, ba] = b.to_array();
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgba_premultiplied(m(ar, br), m(ag, bg), m(ab, bb), m(aa, ba))
}

fn fmt_rate(sr: u32) -> String {
    if sr % 1000 == 0 { format!("{} kHz", sr / 1000) } else { format!("{:.1} kHz", sr as f32 / 1000.0) }
}

fn chip(ui: &mut egui::Ui, text: &str, bg: Color32, fg: Color32) -> egui::Response {
    ui.add(egui::Button::new(RichText::new(text).size(11.5).color(fg).monospace())
        .fill(bg).stroke(Stroke::NONE).corner_radius(7).sense(Sense::hover()))
}

/// The shell's own glyphs, drawn as strokes so they share one weight and do
/// not depend on which emoji a font happens to have.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Icon {
    Prev,
    Play,
    Pause,
    Next,
    Stop,
    Repeat,
    RepeatOne,
    Sequential,
    Ab,
    Bookmark,
    Sleep,
    More,
    Sun,
    Moon,
}

fn arc(c: Pos2, r: f32, a0: f32, a1: f32, n: usize) -> Vec<Pos2> {
    (0..=n).map(|i| {
        let a = a0 + (a1 - a0) * i as f32 / n as f32;
        c + Vec2::new(a.cos(), a.sin()) * r
    }).collect()
}

/// `icon` centred on `c`, `size` across, on a 24-unit grid.
fn paint_icon(p: &egui::Painter, c: Pos2, size: f32, icon: Icon, ink: Color32, ang: f32) {
    use egui::Shape;
    let u = size / 24.0;
    let (sn, cs) = ang.sin_cos();
    let pt = |x: f32, y: f32| {
        let v = Vec2::new((x - 12.0) * u, (y - 12.0) * u);
        c + Vec2::new(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
    };
    let st = Stroke::new(1.9 * u, ink);
    let fill = |pts: Vec<Pos2>| Shape::convex_polygon(pts, ink, Stroke::NONE);
    let line = |pts: Vec<Pos2>| Shape::line(pts, st);
    let bar = |x0: f32, y0: f32, x1: f32, y1: f32| fill(vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)]);
    // The part of a circle r8 at (12,12) outside one r7 at (16.5,7.5); the
    // angles are where the two meet.
    let crescent = || {
        let mut pts = arc(pt(12.0, 12.0), 8.0 * u, 0.21 + ang, 4.50 + ang, 24);
        pts.extend(arc(pt(16.5, 7.5), 7.0 * u, 3.63 + ang, 1.076 + ang, 16));
        Shape::closed_line(pts, st)
    };
    let shapes: Vec<Shape> = match icon {
        Icon::Play => vec![fill(vec![pt(8.0, 5.0), pt(19.5, 12.0), pt(8.0, 19.0)])],
        Icon::Pause => vec![bar(6.5, 5.0, 10.0, 19.0), bar(14.0, 5.0, 17.5, 19.0)],
        Icon::Prev => vec![bar(5.0, 6.0, 7.5, 18.0), fill(vec![pt(19.0, 6.0), pt(19.0, 18.0), pt(8.5, 12.0)])],
        Icon::Next => vec![bar(16.5, 6.0, 19.0, 18.0), fill(vec![pt(5.0, 6.0), pt(15.5, 12.0), pt(5.0, 18.0)])],
        Icon::Stop => vec![bar(6.5, 6.5, 17.5, 17.5)],
        Icon::Repeat | Icon::RepeatOne => {
            let mut v = vec![
                line(vec![pt(4.5, 13.0), pt(4.5, 10.0), pt(7.5, 7.0), pt(18.5, 7.0)]),
                line(vec![pt(15.5, 4.0), pt(18.5, 7.0), pt(15.5, 10.0)]),
                line(vec![pt(19.5, 11.0), pt(19.5, 14.0), pt(16.5, 17.0), pt(5.5, 17.0)]),
                line(vec![pt(8.5, 14.0), pt(5.5, 17.0), pt(8.5, 20.0)]),
            ];
            if icon == Icon::RepeatOne {
                v.push(line(vec![pt(10.8, 10.6), pt(12.4, 9.6), pt(12.4, 14.4)]));
            }
            v
        }
        Icon::Sequential => vec![
            line(vec![pt(4.0, 12.0), pt(19.0, 12.0)]),
            line(vec![pt(14.0, 7.0), pt(19.0, 12.0), pt(14.0, 17.0)]),
            line(vec![pt(4.0, 7.0), pt(9.0, 7.0)]),
            line(vec![pt(4.0, 17.0), pt(9.0, 17.0)]),
        ],
        Icon::Ab => vec![
            line(vec![pt(3.5, 17.0), pt(7.0, 7.0), pt(10.5, 17.0)]),
            line(vec![pt(5.0, 13.5), pt(9.0, 13.5)]),
            line(vec![pt(13.5, 7.0), pt(13.5, 17.0), pt(18.0, 17.0)]),
            Shape::line(arc(pt(17.5, 9.5), 2.5 * u, ang - std::f32::consts::FRAC_PI_2, ang + std::f32::consts::FRAC_PI_2, 8), st),
            Shape::line(arc(pt(17.8, 14.5), 2.7 * u, ang - std::f32::consts::FRAC_PI_2, ang + std::f32::consts::FRAC_PI_2, 8), st),
            line(vec![pt(13.5, 7.0), pt(17.5, 7.0)]),
            line(vec![pt(13.5, 12.0), pt(17.8, 12.0)]),
        ],
        Icon::Bookmark => vec![Shape::closed_line(vec![pt(7.0, 4.0), pt(17.0, 4.0), pt(17.0, 20.0), pt(12.0, 15.5), pt(7.0, 20.0)], st)],
        Icon::Sleep | Icon::Moon => {
            let mut v = vec![crescent()];
            if icon == Icon::Sleep {
                v.push(line(vec![pt(16.0, 3.5), pt(20.5, 3.5), pt(16.0, 8.0), pt(20.5, 8.0)]));
            }
            v
        }
        Icon::More => [6.0, 12.0, 18.0].iter().map(|x| Shape::circle_filled(pt(*x, 12.0), 1.9 * u, ink)).collect(),
        Icon::Sun => {
            let mut v = vec![Shape::circle_stroke(pt(12.0, 12.0), 4.2 * u, st)];
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::FRAC_PI_4 + ang;
                let d = Vec2::new(a.cos(), a.sin());
                v.push(line(vec![pt(12.0, 12.0) + d * 7.0 * u, pt(12.0, 12.0) + d * 9.5 * u]));
            }
            v
        }
    };
    p.extend(shapes);
}

/// A rounded rectangle as a triangle fan, every point passed through `xf`,
/// so it can be tilted and stretched with its texture.
fn rounded_quad(rect: Rect, radius: f32, tex: Option<egui::TextureId>, color: Color32, xf: &dyn Fn(Pos2) -> Pos2) -> egui::Shape {
    use std::f32::consts::{FRAC_PI_2, PI};
    let r = radius.min(rect.width() / 2.0).min(rect.height() / 2.0).max(0.0);
    let mut mesh = match tex {
        Some(t) => egui::Mesh::with_texture(t),
        None => egui::Mesh::default(),
    };
    let uv = |p: Pos2| match tex {
        Some(_) => Pos2::new((p.x - rect.left()) / rect.width(), (p.y - rect.top()) / rect.height()),
        None => egui::epaint::WHITE_UV,
    };
    let c = rect.center();
    mesh.vertices.push(egui::epaint::Vertex { pos: xf(c), uv: uv(c), color });
    let corners = [
        (Pos2::new(rect.right() - r, rect.top() + r), -FRAC_PI_2),
        (Pos2::new(rect.right() - r, rect.bottom() - r), 0.0),
        (Pos2::new(rect.left() + r, rect.bottom() - r), FRAC_PI_2),
        (Pos2::new(rect.left() + r, rect.top() + r), PI),
    ];
    for (cc, a0) in corners {
        for p in arc(cc, r, a0, a0 + FRAC_PI_2, 6) {
            mesh.vertices.push(egui::epaint::Vertex { pos: xf(p), uv: uv(p), color });
        }
    }
    let n = mesh.vertices.len() as u32 - 1;
    for i in 1..=n {
        mesh.add_triangle(0, i, if i == n { 1 } else { i + 1 });
    }
    egui::Shape::mesh(mesh)
}

/// The volume icon. With `face`, the speaker has one: bright at full volume,
/// sleepier and glummer as it goes down, and x_x at zero.
#[allow(clippy::too_many_arguments)]
fn paint_speaker(p: &egui::Painter, c: Pos2, size: f32, vol: f32, face: bool, shades: bool, ink: Color32, hole: Color32, ang: f32) {
    use egui::Shape;
    let u = size / 24.0;
    let (sn, cs) = ang.sin_cos();
    let pt = |x: f32, y: f32| {
        let v = Vec2::new((x - 12.0) * u, (y - 12.0) * u);
        c + Vec2::new(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
    };
    let st = |w: f32, col: Color32| Stroke::new(w * u, col);
    p.add(Shape::convex_polygon(
        vec![pt(2.5, 8.5), pt(7.0, 8.5), pt(13.0, 3.5), pt(13.0, 20.5), pt(7.0, 15.5), pt(2.5, 15.5)], ink, Stroke::NONE));
    // Waves: two loud, one middling, none at zero.
    for (i, (r, from)) in [(4.0, 0.01), (7.5, 0.5)].iter().enumerate() {
        let a = ((vol - from) / 0.25).clamp(0.0, 1.0);
        if a > 0.0 {
            let pts: Vec<Pos2> = arc(Pos2::new(13.0, 12.0), *r, -0.8 - i as f32 * 0.05, 0.8 + i as f32 * 0.05, 12)
                .into_iter().map(|q| pt(q.x, q.y)).collect();
            p.add(Shape::line(pts, st(1.9, with_alpha(ink, a))));
        }
    }
    if !face {
        if vol <= 0.0 {
            p.add(Shape::line_segment([pt(15.5, 9.0), pt(21.0, 15.0)], st(1.9, ink)));
            p.add(Shape::line_segment([pt(21.0, 9.0), pt(15.5, 15.0)], st(1.9, ink)));
        }
        return;
    }
    let sleepy = 1.0 - vol.clamp(0.0, 1.0);
    if shades {
        // Full blast: shades.
        for ex in [8.4f32, 11.2] {
            p.add(Shape::convex_polygon(vec![pt(ex - 1.35, 9.1), pt(ex + 1.35, 9.1), pt(ex + 1.1, 11.4), pt(ex - 1.1, 11.4)],
                hole, Stroke::NONE));
            p.add(Shape::line_segment([pt(ex - 0.85, 9.6), pt(ex - 0.25, 9.6)], st(0.45, Color32::WHITE)));
        }
        p.add(Shape::line_segment([pt(6.8, 9.3), pt(12.8, 9.3)], st(0.7, hole)));
    }
    for ex in [8.4, 11.2].into_iter().filter(|_| vol < 0.995) {
        let e = pt(ex, 10.3);
        if vol <= 0.0 {
            // x_x
            let d = 1.1;
            p.add(Shape::line_segment([pt(ex - d, 10.3 - d), pt(ex + d, 10.3 + d)], st(0.9, hole)));
            p.add(Shape::line_segment([pt(ex + d, 10.3 - d), pt(ex - d, 10.3 + d)], st(0.9, hole)));
        } else if sleepy < 0.45 {
            p.add(Shape::circle_filled(e, 1.15 * u, hole));
        } else {
            // Lids coming down: a half eye, then a line.
            let open = ((0.9 - sleepy) / 0.45).clamp(0.0, 1.0);
            p.add(Shape::line_segment([pt(ex - 1.2, 10.0), pt(ex + 1.2, 10.0)], st(0.9, hole)));
            if open > 0.0 {
                let pts: Vec<Pos2> = arc(Pos2::new(ex, 10.0), 1.15, 0.0, std::f32::consts::PI, 6)
                    .into_iter().map(|q| pt(q.x, 10.0 + (q.y - 10.0) * open)).collect();
                p.add(Shape::line(pts, st(0.9, hole)));
            }
        }
    }
    // The mouth: a smile at full volume, flat midway, a frown low; a little
    // open "o" when out cold.
    if vol <= 0.0 {
        p.add(Shape::circle_stroke(pt(9.8, 14.0), 0.8 * u, st(0.8, hole)));
    } else {
        let curve = 1.4 * (1.0 - 2.0 * sleepy);
        let pts: Vec<Pos2> = (0..=6).map(|i| {
            let f = i as f32 / 6.0;
            let x = 8.3 + 3.0 * f;
            pt(x, 13.6 + curve * (1.0 - (2.0 * f - 1.0).powi(2)))
        }).collect();
        p.add(Shape::line(pts, st(0.9, hole)));
    }
}

/// After this long without input the cover nods off.
const DOZE_SECS: f32 = 600.0;

/// The hour on the wall clock.
fn local_hour() -> u16 {
    local_time().2
}

/// Month, day and hour, local.
fn local_time() -> (u16, u16, u16) {
    #[cfg(windows)]
    {
        // SAFETY: GetLocalTime only writes the struct it is given.
        unsafe {
            let mut st: windows_sys::Win32::Foundation::SYSTEMTIME = std::mem::zeroed();
            windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut st);
            (st.wMonth, st.wDay, st.wHour)
        }
    }
    #[cfg(not(windows))]
    {
        // ponytail: UTC hour and no date off Windows; the holiday hats stay home.
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        (0, 0, (secs / 3600 % 24) as u16)
    }
}

/// Stars going round a head for four seconds; false once they are gone.
fn paint_dizzy(p: &egui::Painter, c: Pos2, t: f32, color: Color32) -> bool {
    if t >= 4.0 {
        return false;
    }
    let fade = (1.0 - (t - 3.0).max(0.0)).min(t * 3.0).clamp(0.0, 1.0);
    for i in 0..3 {
        let a = t * 5.0 + i as f32 * std::f32::consts::TAU / 3.0;
        paint_star(p, c + Vec2::new(a.cos() * 10.0, -12.0 + a.sin() * 3.0), 3.2, with_alpha(color, fade));
    }
    true
}

/// A small five-pointed star.
fn paint_star(p: &egui::Painter, c: Pos2, r: f32, color: Color32) {
    let pts: Vec<Pos2> = (0..10).map(|i| {
        let a = -std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::PI / 5.0;
        let rr = if i % 2 == 0 { r } else { r * 0.45 };
        c + Vec2::new(a.cos(), a.sin()) * rr
    }).collect();
    // Not convex: a fan of triangles from the centre.
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(c, color);
    for q in &pts {
        mesh.colored_vertex(*q, color);
    }
    for i in 1..=10u32 {
        mesh.add_triangle(0, i, if i == 10 { 1 } else { i + 1 });
    }
    p.add(mesh);
}

/// An on/off switch with a sliding knob.
fn switch(ui: &mut egui::Ui, on: bool, enabled: bool, accent: Color32, label: &str) -> egui::Response {
    let size = Vec2::new(38.0, 22.0);
    let (rect, resp) = ui.allocate_exact_size(size, if enabled { Sense::click() } else { Sense::hover() });
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, enabled, on, label));
    let t = ui.ctx().animate_bool_with_time(resp.id, on, 0.18);
    let off = ui.visuals().widgets.inactive.bg_fill;
    let track = lerp_color(off, accent, t);
    let track = if enabled { track } else { track.gamma_multiply(0.5) };
    let p = ui.painter();
    p.rect_filled(rect, 11.0, track);
    let x = lerp(rect.left() + 11.0, rect.right() - 11.0, t);
    let knob = 8.0 + if resp.hovered() && enabled { 1.0 } else { 0.0 };
    p.circle_filled(Pos2::new(x, rect.center().y), knob, Color32::WHITE);
    if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp }
}

#[cfg(test)]
mod tests {
    #[test]
    fn minor_chords_are_told_from_major_ones() {
        for m in ["Am", "C#m7", "Bbm", "Ebm(maj7)", "Fm6"] {
            assert!(super::is_minor(m), "{m}");
        }
        for m in ["A", "Cmaj7", "Bb", "G7", "Ddim", "Ebmaj9", "F#sus4", "C/E"] {
            assert!(!super::is_minor(m), "{m}");
        }
    }

    #[test]
    fn window_buttons_drop_their_emoji_only_in_the_new_look() {
        let fill = Some(super::Color32::BLACK);
        assert_eq!(super::plain(fill, "💾 Save"), "Save");
        assert_eq!(super::plain(fill, "🔍 Find lyrics"), "Find lyrics");
        assert_eq!(super::plain(fill, "Find lyrics"), "Find lyrics");
        assert_eq!(super::plain(fill, "✖"), "✖");
        assert_eq!(super::plain(None, "💾 Save"), "💾 Save");
    }

    #[test]
    fn a_two_five_one_is_heard_in_any_key() {
        assert!(super::is_two_five_one("Dm7", "G7", "Cmaj7"));
        assert!(super::is_two_five_one("C#m", "F#", "B"));
        assert!(super::is_two_five_one("Gm7", "C7", "F"));
        assert!(!super::is_two_five_one("D", "G7", "C"));
        assert!(!super::is_two_five_one("Dm7", "G7", "Am"));
        assert!(!super::is_two_five_one("Em7", "G7", "C"));
    }

    #[test]
    fn a_chord_names_its_root() {
        for (name, pc) in [("C", 0), ("C#m7", 1), ("Eb", 3), ("F#sus4", 6), ("Bb7", 10), ("B", 11), ("Cb", 11), ("G/B", 7)] {
            assert_eq!(super::root_pc(name), Some(pc), "{name}");
        }
        assert_eq!(super::root_pc("N.C."), None);
    }

    /// A key misspelt at a call site would leave that extra off for good.
    #[test]
    fn every_extra_asked_for_is_in_the_table() {
        let src = include_str!("shell.rs");
        let keys: Vec<&str> = super::TOYS.iter().map(|t| t.key).collect();
        for (i, k) in keys.iter().enumerate() {
            assert!(!keys[..i].contains(k), "{k} is listed twice");
        }
        for pat in ["toy(\"", "found(\""] {
            for part in src.split(pat).skip(1) {
                let key = &part[..part.find('"').unwrap()];
                if key.contains(' ') || key == "{k}" {
                    continue;
                }
                assert!(keys.contains(&key), "{pat}{key}\" is not in TOYS");
            }
        }
    }
}
