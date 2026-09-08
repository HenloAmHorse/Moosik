# Everything else, in detail

Album art, lyrics, tag editing, appearance and the player itself. The README
gives each of these a line; this is the long form. Nothing here has been
shortened.

See also [BIT_PERFECT.md](BIT_PERFECT.md) and [SPECTRUM.md](SPECTRUM.md).

## Album Art

- **Playlist thumbnails** — 28×28 cover art thumbnail in every playlist row; hover for 1 second to see a 512px preview
- **Transparent overlay** — art rendered behind the spectrum at adjustable opacity; Contain / Cover / Stretch fit modes
- **Mask mode** — spectrum bars act as a cut-out window into the art, each bar textured with the art region it covers; brightness can track bar magnitude dynamically or be fixed
- **Art Settings panel** — collapsible section in the spectrum window; global settings with optional per-track overrides
- **Spectrum placeholder** — configurable ♪ glyph when a track has no embedded art
- **Persistent** — settings stored in `~/.moosik/art_settings.json`

## Lyrics

- **🎤 Lyrics window** — its own window, so it can live on a second screen. The current line highlights and centres itself, sung lines dim, and a slider sets the text size
- **Reads `.lrc` beside the track, or the file's own tags** (`USLT` / Vorbis `LYRICS` / iTunes atom)
- **Online lookup via [LRCLIB](https://lrclib.net), then NetEase Cloud Music** — no account, no API key, and both serve *synced* lyrics, not just plain text. Two sources because one wasn't enough: on a real set of thirteen failing tracks LRCLIB returned zero hits for nine, since a Japanese/Vocaloid library is largely absent from it. NetEase had synced lyrics for nearly all of them
- **A source that can't answer never reports "no lyrics"** — a rate limit or an outage says nothing about whether the lyrics exist, so it's reported as such rather than sending you off to transcribe by hand
- **Matching built for messy tags** — the lookup is a ladder (exact → without album → structured search on normalised fields → normalised title), filtering and ranking on duration throughout. Normalisation folds full-width Latin, strips bracketed decoration (`【初音ミク】…【オリジナルMV】`), cuts at `feat.`, and keeps only the first artist credit. This is what finds a track whose artist field holds the producer where the database has the vocalist
- **Manual search** — type anything, see the hits with synced-vs-plain marked, pick one
- **Tap-along sync editor** — play the song and press Space as each line starts; a plain sheet becomes a timed one in a few minutes. Re-times existing sheets too, and resumes at the first untimed line
- **Sidecar-only writes** — lyrics are saved as `.lrc`; the audio file is never modified, which also makes this safe for DSD

## Tag Editing

- **🏷 Tags window** — edit title, artist, album, album artist, track/disc numbers, year, genre, composer and comment
- **Full tag viewer** — every key the file carries, including ones the editor doesn't touch (which are preserved on save)
- **Safe writes** — the change goes to a copy which then replaces the original by rename, so an interrupted write can't damage the file; the result is read back and verified before it's accepted
- **DSD is read-only** — DSF keeps its ID3 blob behind a header pointer, and a bad write there costs the audio, not the metadata

## Theme & Appearance

- **Cohesive theme, light or dark** — built from the app's own identity colors (the **Eigengrau** #16161d base and **Kugelblitz** #94b1ff accent): a tinted surface ramp, rounded corners, and accent-coloured selection and hover, applied across both the egui widgets and the custom-painted surfaces (seek bar, playlist rows, status text). A light variant is a click away
- **Per-track accent** — the now-playing bar, current-row marker, and seek fill take on an accent sampled from the current track's cover art (blended toward the brand blue so it never clashes); art-less or grayscale covers fall back to the brand accent
- **🎨 Look menu** — optional, persisted appearance controls (`~/.moosik/appearance.json`), all defaulting to the shipped look:
  - **Theme** — dark (default) or light
  - **Font** — pick any font installed on the system. Families are read from each file's own name table, so the list reads *Fira Code Medium* rather than `FiraCode-Medium`, and every face in a `.ttc` collection is offered; a filter box keeps a few hundred families navigable. CJK fallbacks stay installed behind whatever you pick, so a Latin-only font won't tofu your Japanese tags
  - **Text size** — a 70–160% UI-scale slider (applied on click), scaling all text and chrome
  - **Accent source** — album-art accent or the fixed brand accent

## Player

- **Gapless playback** — consecutive tracks play with no silence between them, in both normal and bit-perfect mode. An exact stream stays gapless only while the next track still fits it exactly — same endpoint, rate, channel count, channel layout and sample family, and a container wide enough for its precision. Anything else re-opens the device at the boundary rather than narrowing the next track to fit
- **Playlist search** — a 🔍 box filters the playlist by title, artist, or album as you type (`Ctrl+F` to focus, `Esc` to clear)
- **Column sorting** — sort by Title / Artist / Album / Time, plus **Plays** and **Recent** from play history; click a column to sort, click again to reverse
- **Sleep timer** — stop after 15–90 minutes or at the end of the current track
- **A-B repeat** — loop a section of the current track, with markers on the seek bar
- **Bookmarks** — save and jump back to positions per track
- **Play statistics** — per-track play count and last-played, shown in the Info window and (optionally) as a column in the playlist
- Waveform seek bar with click-to-seek
- Volume control (volume and loop mode persist across launches)
- Metadata display (title, artist, album, cover art) via `lofty`
- CJK font fallback (Japanese, Chinese, Korean tags display correctly)
- **OS media integration** — hardware media keys and the system now-playing panel (MPRIS on Linux, SMTC on Windows, Now Playing on macOS): play/pause/next/previous/stop, live title/artist/album and playback position
- **ReplayGain** — loudness normalization (Track / Album), reading ReplayGain tags when present and falling back to Moosik's own measured LUFS for untagged files; clip-prevention toggle; bypassed in bit-perfect mode
- Momentary LUFS display
- Stereo correlation meter
- **Session log** — every run normally creates a timestamped file in `~/.moosik/logs/` (creation can fail; the 🗎 Log button then says why), recording the machine, GPU and analysis timings and a full backtrace on any panic. **WASAPI exclusive** and **native ASIO / ALSA DSD** additionally log the device and the format, rate and buffer they negotiated; normal shared-mode playback and non-Windows CPAL PCM/DoP do not yet report theirs. The **🗎 Log** button opens the folder, and older sessions are pruned back to roughly ten
  - *Before sharing one:* a log can contain your user name and file paths, track file names, audio device / GPU / driver details, and panic backtraces. Read it first — there is no redaction step yet
  - Retention is per-process: running two copies of Moosik at once can prune a log the other is still writing
