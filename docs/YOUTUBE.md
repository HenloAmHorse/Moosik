# YouTube and YouTube Music (1.6.1)

**Library → Sources → YouTube** searches YouTube or YouTube Music and plays,
queues or downloads what it finds. In a narrow window, where the Library has
no sidebar, a **Queue / YouTube** switch sits above the list instead.

## What it needs

Moosik does not talk to YouTube itself. Two programs do, and Moosik downloads
them only when you press **Get yt-dlp and Deno**:

- **yt-dlp** ([github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp)),
  The Unlicense. It finds and fetches the audio. Its Windows build bundles
  Python and other components under their own licenses.
- **Deno** ([github.com/denoland/deno](https://github.com/denoland/deno)),
  MIT. Since late 2025, yt-dlp needs a JavaScript runtime for full YouTube
  support; Deno is the one it recommends.

Both come from the latest release on the projects' GitHub pages: about 67 MB
to download and 140 MB on disk, kept in `~/.moosik/tools/youtube/`. Each file
is checked against the SHA-256 its own release publishes. That catches a
damaged or swapped download; it is not a signature, so it trusts GitHub and
the two projects. Files are written aside and moved into place, so a failed
update leaves the working copy as it was. yt-dlp is its one-folder build,
which starts about half a second sooner each time than the one-file
`yt-dlp.exe` (that unpacks itself on every run); an install from before
switches over at the next **Update**.

YouTube changes often, and an older yt-dlp then stops working. **Update** (at
the foot of the YouTube view) fetches the latest of both. **Settings →
YouTube → Remove** deletes them.

Neither program is part of Moosik or covered by its notices; a short notice
is written beside them when they are installed.

## Searching

- **Videos** searches all of YouTube; **Music** searches YouTube Music's
  songs. A search takes about 2–3 seconds and shows up to 25 results.
- A pasted link to one video (youtube.com, youtu.be, Shorts, or
  music.youtube.com) shows that video alone instead of searching for it.
- A pasted playlist link, or a YouTube Music album link (`playlist?list=`
  or `browse/MPREb_…`), shows its songs in order, up to 200. **Queue all**
  adds them to the end of the queue one at a time, so they arrive in the
  album's order; **Stop** adds no more. A watch link that is also on a
  playlist stays that one video.
- Live streams are left out: they have no end to fetch.
- YouTube Music's song search does not say who the artist is or how long the
  song is; both are filled in once a song is fetched.
- Thumbnails come from YouTube's image server, `i.ytimg.com`.

## Play, Queue, Download

- **Play** starts the song once enough of it is in (about 768 KB, some 45
  seconds of YouTube's Opus) and the rest arrives while it plays;
  **Queue** adds it to the end of the queue the same way. The row shows
  the progress; **Cancel** stops it. Starting takes a few seconds, most of
  it yt-dlp working out the video; on a congested connection, where each
  new connection to YouTube waits seconds for an answer, it can take much
  longer, and that wait is the connection's.
- The spectrum's analysis of a song played while it arrives starts once
  its audio is all in.
- A song played (or queued) from YouTube shows **Save** on its cover in
  Listen: it keeps the cached copy in your downloads folder, as Download
  would have, video and cover included.
- **Download** saves a copy to your music folder, `Music\Moosik` unless you
  choose another in **Settings → YouTube**, as `Artist - Title [id].ext`.
- Audio stays exactly as YouTube delivers it: usually Opus in WebM, sometimes
  AAC in M4A. It is lossy and Moosik shows it as lossy. Nothing is converted,
  so no ffmpeg is needed.
- A result played before and still cached plays at once (it says *ready*
  in the results): Moosik does not ask yt-dlp again, which would spend
  3–5 seconds working out the video.
- An empty search box offers the last eight searches; **Clear** forgets
  them.
- Fetching is as fast as the connection: yt-dlp already downloads at its
  full speed, and splitting a download into parallel parts measured no
  faster. The audio and its video come from one look at YouTube, not one
  each, and the HLS lists yt-dlp would also ask for (nothing here plays
  from them) are skipped: one slow request fewer.
- Played and queued audio is kept in `~/.moosik/cache/youtube/`, up to 1 GB.
  The oldest goes first, but never anything on the playlist. **Settings →
  YouTube → Clear** empties it.

## Music videos

**Audio / Audio + video**, at the top right of the YouTube view, chooses what
Play, Queue and Download bring; Moosik remembers the choice. With video, one
run of yt-dlp brings both: the audio first, which plays as soon as enough is
in, then the video (the row says *Video* while it comes), which shows while
it arrives, behind **Listen** and behind the spectrum, following the song:
pause, seek and skip, and the picture keeps up. The audio is still the file
the engine plays, bit-perfect or not; the video is silent and only follows
it.

Windows decodes the video (Media Foundation). On a Windows N edition without
the Media Feature Pack there is none: Moosik still starts and plays, and a
video says the pack is missing instead of showing.

- **The video's size** is chosen beside **Audio + video** (and in
  **Settings → YouTube → Music videos**): 480p, 720p (the default), 1080p,
  1440p (2K) or 2160p (4K), for what is fetched next. A song played before
  whose kept video is smaller than the size now chosen gets a new one; one
  YouTube has nothing taller of is not fetched again. A video larger than
  the screen is scaled down to the screen's size on the graphics card, so
  a 4K video on a 1440p screen costs what a 1440p one does to show.

- **Watch** (on the cover, or **V**) shows the video whole, on black, with
  the lyrics as subtitles and the song's title and artist at the top left
  for its first seconds. Still for three seconds while it plays, the bars
  and the pointer step aside; any move brings them back. A double click on
  the picture goes full screen. **Esc** leaves full screen, then theater;
  **Back** or **V** leaves theater.
- A song from YouTube without its video gets **Get video** on its cover,
  which fetches it in the background and shows how far it has got.
- **Settings → YouTube → Music videos**: the size; how bright the video
  shows; behind Listen, behind the spectrum, or both. Watch works either
  way.
- The video fades in over the cover's backdrop. Black bars (a 4:3 video in
  a 16:9 frame, a letterboxed film) are found and left out.
- It is decoded on the graphics card where it can (Direct3D 11, which also
  turns it into RGB), else on the processor; **Decode on the processor** in
  Settings keeps the card out of it. A card that gives up part way leaves
  the rest to the processor. It is decoded only while it shows: in the
  Library it waits, and catches up when it is back. Measured with a 720p25
  video, the whole of Moosik: 6.3% of one core in Listen, nothing extra in
  the Library.
- The video is YouTube's VP9 picture stream, at most the chosen height and
  30 frames a second where offered, kept beside its song as `<song>.video.webm` (in
  the cache, or beside a download in your music folder). Windows decodes
  it, with the VP9 Video Extensions that come with Windows 11; nothing more
  is installed. Without them the video is not shown and Moosik says why.
- It is bigger than the audio (Rick Astley's 3½-minute video: 17 MB at
  720p, 29 MB at 1080p, 144 MB at 1440p, 342 MB at 2160p) and shows while
  it arrives. A connection slower than the video (2K wants about 6 Mbit/s,
  4K about 15) brings it in behind the song: the picture holds, then
  catches up as it comes. A video YouTube offers at the chosen height only
  at 60 frames a second comes a size smaller instead.
- Any song with a `<song>.video.webm` beside it gets its video, not only
  YouTube's; your own music videos work as `<song>.video.mp4` or
  `.video.mkv` too (an MP4 cut into fragments, as YouTube's are, does not).
  Imports skip these files.
- Behind the spectrum it takes the place of the cover; in the spectrogram
  and waterfall views, which fill the plot, it is hidden.
- The decoder alone, at 25 fps: 720p 6% of one core on the graphics card
  (19% on the processor), 1080p 15% (33%).
  Rick Astley's video is 18 MB at 720p and 31 MB at 1080p.
- 2160p on the graphics card, in one run with 720p at 10%: 24% of one
  core shown at a 2560×1440 screen's size, 46% at its own 3840×2160 (and
  not keeping up), which is why it is scaled to the screen.

## Covers

A song from YouTube has no picture of its own, so Moosik gives it the
video's: the largest thumbnail YouTube has, its middle square, kept beside
the song as `<song>.cover.jpg` (and copied along with a download). It is
fetched once, when the song arrives or first plays, and from then on it is
the cover everywhere: Listen, the backdrop and its colour, the spectrum.

## Titles and artists

A video's own title usually says who and what, wrapped in decoration:
"Artist - Song (Official Video)", "Artist「Song」MV". Moosik takes the song
and artist from it without the decoration, so the player and lyric lookups
see the song. A bracket that is part of the name ("(feat. …)", "(Live at
…)") stays; channels lose " - Topic" and "VEVO". YouTube Music's own track
and artist are used as they are.

A song with no lyrics of its own, YouTube's or not, is looked up online (as
**Find lyrics** would) the first time it plays, so theater has subtitles
without asking; **Settings → Track → Find lyrics automatically** turns that
off. What is found shows at once and is written to disk only if you save
it.

yt-dlp cannot write tags into a WebM file without ffmpeg, so Moosik keeps its
own record of what each file is, in `~/.moosik/youtube.json`: title, artist,
album and the video it came from. It is used only for a file that has no tags
of its own, and the Info window shows the source link. A downloaded file
moved elsewhere loses the record and shows its file name instead.

## Privacy and terms

Searching, fetching and thumbnails talk to YouTube from your computer, as a
browser would; Moosik adds no account and sends nothing else. yt-dlp runs
hidden, reads no yt-dlp configuration of yours, keeps its cache in Moosik's
folder, and ends when Moosik does, even in the middle of a fetch.

YouTube's terms of service limit downloading. Use this for what you have the
right to play or keep. Nothing behind a paywall, a sign-in or DRM is
available; YouTube Music Premium-only audio is not.

## Signing in

**Sign in**, at the top of the YouTube view, opens Google's sign-in page in
a small window of a Chromium-family browser found here (Chrome, Brave,
Edge, Vivaldi or Chromium, on Windows, Linux or macOS), with a profile of
Moosik's own, apart from the one you browse with. Once Google knows you
(the browser notes the account in that profile), the window closes by
itself, or close it yourself. Moosik then starts the browser again,
unseen, on the same profile, asks it for the account's cookies over its
DevTools port (on this computer only) and writes them to
`tools/youtube/cookies.txt` for yt-dlp. The window itself has no DevTools
port, since Google will not sign in a browser that has one ("This browser
or app may not be secure"). Asking the browser itself is what
gets past the cookie encryption newer Chrome and Edge use, which yt-dlp
cannot read from their files. **● Signed in** signs out and deletes both the
cookies.txt and that profile. Without such a browser, a cookies.txt from
any browser works; Settings → YouTube → Sign in also reads Firefox (or the
others) directly.
Signed in, yt-dlp uses your YouTube account, so
age-restricted, members-only and private videos play, private playlists
open, and the YouTube view offers **Liked videos**, **Liked music** and
**Watch later**.

- **Settings → Firefox, Edge, Chrome, Brave:** yt-dlp reads the sign-in from that
  browser on this computer for each request. Firefox works while open;
  Chrome and Edge may need closing first, and newer versions encrypt it in
  a way yt-dlp cannot always read.
- **cookies.txt:** a file exported from any browser (with an extension such
  as "Get cookies.txt LOCALLY"), signed in to youtube.com. Always works;
  sign in again and export again when it expires.
- Nothing about the sign-in leaves the computer except to YouTube, through
  yt-dlp. Moosik saves only which browser, or the file's path.
- YouTube can limit accounts it sees used by downloaders; a second account
  is the safe choice.

## When it fails

The panel shows yt-dlp's own reason. Common ones:

- *Private video*, *Sign in to confirm your age*, members-only, or *The
  playlist does not exist* for a private list: sign in (above); the message
  says so when you are signed out.
- A sign-in that fails to read ("cookie", "DPAPI"): close the browser and
  try again, or use a cookies.txt.
- *Sign in to confirm you're not a bot*: YouTube is limiting this
  connection; waiting usually helps.
- Anything else after a YouTube change: press **Update**. A failure that
  reads like an older yt-dlp meeting a changed YouTube says so ("try
  Update").
