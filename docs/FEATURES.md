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
- **Karaoke highlighting** — where a sheet times its individual words (enhanced LRC, `<mm:ss.xx>` inside a line), each word lights as it is sung. That timing is kept on load and written back on save. Sheets without it simply highlight the current line; no timing is guessed
- **Japanese readings: Original / Furigana / Romanization / Both** — for lyrics containing Japanese. **Both** puts the reading under each line or under each word (**Lines / Words**)
  - **Where readings come from:** romaji supplied by the lyrics source (NetEase's romaji sheet) is used where there is one. Otherwise it is generated locally from a built-in Japanese dictionary and shown in *italics*, since a dictionary cannot know a song's chosen reading of a kanji
  - **Word by word, with a fallback:** a supplied reading is placed under the words by aligning it with the generated one. Where it cannot be placed word by word, the line shows it whole rather than guessing
  - **Original** shows the lyrics exactly as stored
- **Korean romanization** — local Revised-Romanization-style pronunciation rules, cached per unchanged line. **Romanization** replaces the displayed line; **Both** keeps the Korean original above its whole-line reading. Supplied/corrected readings take priority; generated guesses are italic and editable with **✎**. No Korean furigana or inferred karaoke alignment is claimed
  - No network, new dependency or large Korean dictionary is required. This is a rule-based helper, **not a complete pronunciation/morphology dictionary**: compound-boundary insertion, irregular/name readings, cross-space changes and a singer's chosen pronunciation can need correction. Hanja and mixed Latin text are preserved, not guessed as Japanese
  - Supplied Latin-only alternate readings are also shown whole, rather than forced through Japanese word alignment
- **Furigana written into the lyrics wins** — an annotation such as `剣(つるぎ)` is the reading used in every view, even where an online source's romaji disagrees
- **Correcting readings (✎)** — edit any line's romaji; a change you make is kept, and ↺ returns a line to its generated or annotated reading
  - **Overriding the lyrics' own furigana is deliberate.** If your reading would replace one the lyrics annotate, or you tick **Use this reading instead of the lyrics'** on a supplied reading that disagrees with it, the window shows both readings and asks. **Keep the lyrics' reading** changes nothing; **Use my reading** makes yours win in every view, and the Japanese is then shown without the replaced annotation
  - The lyric text itself is never changed
- **Text size** — the lyrics window's size slider is remembered across restarts (written when the slider is let go)
- **Translations** — shown under each line when the **Translation** switch is on, regardless of whether the original is Japanese, Korean or English
  - **Translate… → Find supplied translation** searches NetEase independently of the original lookup (including when LRCLIB already supplied the originals). It sends track metadata, not the current lyric sheet. A near-complete monotonic match of original text is required; timestamps alone cannot identify a version. Existing translated rows are retained; ambiguous repeated rows are rejected. Coverage is provider-dependent
  - **Translate to Chinese** offers explicit Japanese/Korean/English source selection. One selected source applies to the entire request, including mixed-language rows. The suggested language is local and can be corrected; Han-only or Japanese/Korean mixtures require a choice
  - Online translation uses [MyMemory](https://mymemory.translated.net/doc/spec.php), without an API key. **It sends lyric text to a third party.** Its [data terms](https://mymemory.translated.net/terms-and-conditions) describe long-term translation-segment storage and partner sharing. Do not submit private lyrics. Consent is required, there is no automatic translation on track change, and editing the original sheet clears its prior consent
  - Requests are HTTPS-only, deduplicated for repeated rows, and bounded: 500 UTF-8 bytes per row, 120 unique rows / 5,000 UTF-8 bytes per operation, 12 seconds per request and five minutes per operation. These app caps do not reserve the service's [daily quota](https://mymemory.translated.net/doc/usagelimits.php). Quota, network, malformed/non-Chinese output or cancellation yields no partially applied sheet; an in-flight network call is not interruptible
  - **Context-aware AI (OpenAI, your key)** is an optional second provider, off until you configure it: choose a model (documented IDs are offered, none is preselected; an ID OpenAI does not offer is reported, never replaced), enter your own API key in a masked field (kept in memory for the session only, never saved) or press **Use OPENAI_API_KEY**, choose **Faithful** or **More fluent (meaning kept)**, and optionally a **Review pass**. It asks for Traditional Chinese with Taiwan usage and stores each new AI row in canonical Traditional character forms (the bundled OpenCC s2tw conversion — character forms only, not a check of vocabulary or quality); 简体 remains a display conversion, and existing, supplied, manual and Basic rows are never converted
  - **It sends the whole original sheet** — every row, even when only rows without a translation will be filled — plus the title and artist, to OpenAI's [Responses API](https://developers.openai.com/api/docs/guides/structured-outputs) with `store: false`. Not sent: file paths, audio, settings, existing translations. `store: false` is not zero retention; see OpenAI's [data controls](https://developers.openai.com/api/docs/guides/your-data). Usage is billed to your key ([pricing](https://developers.openai.com/api/docs/pricing)). Agreement is to the exact provider, model, style, review setting and source language, and lapses when any of them or the sheet changes
  - At most two requests per press (translation, then the optional review), never retried; each is limited to 25,000 output tokens (reasoning tokens count and are billed) and 180 s, the whole operation to 360 s; at most 400 sheet rows (blank rows included), 1,000 bytes per row and 16 KiB of original text in all, counting spaces — checked before anything is copied or sent. A cancelled or timed-out request may still be billed. If the review fails, the preview says it is an unreviewed draft
  - The answer must name every requested row exactly once by its row ID and pass the same structural checks as manual text; anything else, a refusal or a cut-off answer is rejected whole. Rows left unchanged, not clearly Chinese, or marked uncertain by the model — even every requested row — are flagged in the preview, never trusted and applied only by **Apply**; model notes are shown, not saved. A result made under different settings is never applied. Agreement lapses when the provider, model, style, review setting, source language, replace setting or the sheet changes, and changing back does not restore it. Editing AI rows by hand keeps their machine-assisted attribution, and an unreviewed draft keeps its "unreviewed draft" note, however much of each row is edited; only clearing every translation row, or a Replace-mode result that overwrites every row, drops the earlier attribution
  - **On this computer (prototype)** translates with a downloadable local model, with no key, no account and no network once installed. Choose the model in the panel: **Hy-MT2 7B** and **Hy-MT2 1.8B** (Tencent translation models, Apache-2.0), **Hy-MT2 30B-A3B** (the largest, 18.2 GB of weights; its standard graphics-card route requires 19 GiB available RAM including the reserve) or **Qwen2.5 7B** (the first prototype, kept for comparison). Japanese, Korean, English or Chinese into Traditional Chinese (Taiwan) or English. Each model is a one-time download (about 4.7 GB, 1.2 GB, 18.2 GB, 4.7 GB) checked against pinned SHA-256s; the panel lists the download, disk, measured memory and limitations before anything is fetched, shows speed and time left while downloading, and resumes after a cancel
  - **Use less RAM (experimental)** is an optional route for the 30B on a graphics card with at least 7 GiB of dedicated memory. It uses the same model and context, with more weights on the card and buffered loading. Startup requires **16 GiB available RAM and 21 GiB commit headroom**, including 2 GiB kept free on both measures. On the reference 32 GB / RTX 2060 SUPER 8 GB computer, 10–150-row synthetic sheets peaked at about 13.4 GiB process RAM and 18.3 GiB process commit; a 10-row sheet took about 27 s. This reduces RAM at the cost of more graphics memory and commit, and can be slower. There is no additional model download. Smaller models remain available for computers below those requirements. These are measurements on one computer, not a translation-quality score
  - **Use files I already have…** takes an installed pack folder (for example from another Moosik) or the model file itself: it is copied, never moved or changed, kept only if it matches the pinned checksum, and nothing in it is run. A second model reuses the runtime another installed model already has
  - **Faithful** asks each model for an accurate translation of each line (the default). **More fluent (meaning kept)** also asks for natural lyric phrasing with nothing added or dropped and no rhyme at the meaning's cost; it is not a singable adaptation. The whole sheet (or, for a long song, consecutive parts of it) goes to the model in lyric order, so each line is translated in the context of the lines around it
  - **Local translation workbench** — custom instructions (for example, avoid added 啊/吧 filler particles) can be explicitly saved with the seed and variation setting. Feedback is session-only, reused until cleared, and cleared on a sheet change; it does not train model weights or become a permanent preference. **Draw another** changes the seed and starts an explicit request; Conservative retains the model's baseline sampling, Balanced/Free increase temperature and may reduce accuracy. A new seed does not guarantee different wording, and equal seeds do not guarantee identical text across hardware. Revisions carry previous translations as context. **Lock** and unchecked **Revise** rows are preserved by the host even if the model rewrites them. Up to eight previous drafts stay in session memory, bound to the same sheet/model/language/device; switching restores that draft's instructions/seed. Compare previous/current lines, copy individual previous lines into the draft, and choose **Adopt** rows before Apply. One-step **Undo last local Apply** refuses to overwrite newer edits. Draft history is not written to disk; Apply still requires Save to persist the translation. No extra model pass is added automatically.
  - **Which model for this computer?** (in the panel) reads this computer's memory and graphics card when opened and suggests three things separately: *Quality first* (the best-judged model that fits), *Balanced* (fits with room to spare and answers a 10-row sheet in about 15 s) and *Largest that fits* (size is not quality). Each shows its route, memory and time, and whether those were measured or estimated. Startup requirements use the same available-RAM and commit rule as the actual launch, in GiB, including the reserve; an unreadable inventory says so. Within a model, a route that can start with the current headroom takes priority. The ranking is provisional — no local model has been approved for lyrics. **Use this** only selects the model and route; nothing is downloaded or run. The model you choose is remembered
  - Each result names the model that made it, and a result made with one model or setting is never applied as another's. Output is machine translation and may mistranslate; a preference for one model on particular songs is not independent validation of general lyric accuracy
  - **Preview → Apply translation → Save** are separate actions. Machine-assisted provenance persists in the translation sidecar. Generated text may mistranslate poetic lyrics; it is not a verified human translation. Existing rows are replaced only when explicitly requested and adopted; overwritten sidecars are backed up on Save
  - **Edit / paste translation…** edits a translation for each original row, or imports an exactly row-matched plain-text sheet (including blank rows). Apply changes only the translations; Save writes only the affected sidecar, unless the original sheet/readings also changed. Structural LRC tokens are refused in new translation text to prevent reinterpretation after reload
  - **繁體 / 简体 / As is** change display only, using the existing Chinese conversion and Chinese-first font families. Stored source text remains unchanged; the preview uses the same conversion/family selection
- **Separate files, beside `song.lrc`:** `song.romaji.lrc` (romaji a source supplied or you corrected), `song.translation.lrc`, and `song.reading-override.lrc` (the lines where you chose your reading over the lyrics' furigana). Each mirrors `song.lrc` line by line, so `song.lrc` stays the plain sheet other players read
- **Saving is all or nothing**
  - Every file about to change is backed up first; each is replaced whole; if any step fails, those already changed are put back, and the window says so and names the backups
  - A romaji or translation file overwritten or removed by a save is kept as `….old` (then `.old.2`, …). Nothing removes these automatically yet
  - Only one save of a track runs at a time **within one copy of Moosik**. Two separately running copies are not prevented from saving the same track at once
- **The Sync editor keeps each line's romaji, translation and word timing** when it retimes a sheet; retyping a line drops that line's word timing

## Tag Editing

- **🏷 Tags window** — edit title, artist, album, album artist, track/disc numbers, year, genre, composer and comment
- **Full tag viewer** — every key the file carries, including ones the editor doesn't touch (which are preserved on save)
- **Safe writes** — the change goes to a copy which then replaces the original by rename, so an interrupted write can't damage the file; the result is read back and verified before it's accepted
- **DSD is read-only** — DSF keeps its ID3 blob behind a header pointer, and a bad write there costs the audio, not the metadata

## Interface (1.6.0)

- **Three modes in one window** — **Listen** (cover, title, a spectrum ribbon and the lyrics card), **Library** (playlists, the playlist, and a Now playing / Queue inspector) and **Analyze** (the analyser with its controls in a drawer, plus the Stereo and Harmony panels). One player bar serves all three
- **Settings drawer (⋯)** — output, loudness, visuals and motion, silly extras, appearance, track windows and files in one panel; below 1000 px wide the player bar's extra controls move into it
- **Motion** — the cover glides between modes, a blurred cover backdrop cross-fades on track change, buttons spring; animation duration and intensity are sliders, and each effect has its own switch. An idle window does not redraw
- **Info, Tags and Lyrics windows** take the same page colour and margins; their buttons drop the emoji
- **Classic layout** — **1.6.0 layout (off = classic)** in Appearance
- **Silly extras** — small animations hidden around the player, each found by doing what its hint says and then switchable on its own; one switch turns them all off
- **Debug console window** (Windows) — release builds open no console; this setting opens one showing what Moosik prints, attached to the terminal it was started from if any. The window's close button is removed because closing a console closes Moosik

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
