# The analysis cache (1.5.3)

Moosik keeps each finished Pre-process analysis on disk so a track you play again is shown at once instead of being analysed again. 1.5.3 replaces the way those files are named, written, found and removed. This page states what the new cache guarantees, what it does not, and what happens to the files earlier versions wrote.

The cache lives in your profile, in `.moosik/cache/`. Everything this version writes is in its subfolder `.moosik/cache/v2/`.

1.5.5 adds a distinct **True CQT** analysis identity inside the
same store. It does not change the namespace or retire matching 1.5.3/1.5.4
FFT/Superlet caches. The old `Cqt` setting remains FFT constant-Q weighting;
its cached results are not offered as exact True CQT hits. See [CQT.md](CQT.md).

## What changed, in one paragraph

An earlier version knew a track only by its file path. If you replaced a file with different audio at the same path, it could show the old analysis for the new file. 1.5.3 also checks the file's size, its last-modified time and a sample of its contents. Each analysis is now published as a whole, in one step, so a crash cannot leave the spectrum and its Left/Right channels out of step. Something you are looking at, or that is still being read, is never deleted to make room.

## Decisions this follows

The owner chose these on 2026-10-13. They are listed here so the behaviour can be checked against them.

| | Decision | What it means in practice |
|---|---|---|
| D1 | **Legacy files are kept; never used automatically** (L1 + L1a) | Files written by 1.5.2.1 and earlier (and by 1.5.3 test builds) stay where they are. They are never treated as a match for your current settings, because they cannot say whether the audio file has changed since. They are shown only as **legacy analyses**, clearly labelled, which you can choose to display yourself. Nothing is imported, converted, moved or deleted from them. |
| D2 | **Other programs may still change a file while it is analysed**, with a check just before saving | Moosik does not lock your music files. Just before an analysis is saved, Moosik checks that the file is still what it read. A change caught there means nothing is saved or shown. A change that lands in the moment between that check and the save is **not** caught. |
| D3 | **A track is identified by path, size, last-modified time and a content sample** | Changing a track's tags changes its last-modified time, so it may be analysed again; that costs time, not correctness. The content sample is 16 windows of 64 KiB, or the whole file when it is at most 1 MiB. An edit confined to bytes outside those windows, which also keeps the size and the time, is not detected. |
| D4 | **A separate folder and a versioned format** | New analyses go to `cache/v2/` in a new file format. Older versions do not read them, and this version never uses theirs as matches. |
| D5 | **Two small coordination files per analysed track are kept** | `<name>.coord` and `<name>.pub` are empty files used for locking. Moosik never deletes them, so a stale lock can never be mistaken for a free one. |
| D6 | **No deletion where locking has not been validated** | Deletion (the size limit, **Trim now**, **Clear Cache**, **Clear All**) runs only when the cache folder is on a local fixed NTFS disk on Windows **and** a lock check in that folder passes. On a network drive, removable drive or other file system, nothing is deleted and the space in use is reported. That includes Moosik's own unfinished files: a save that is cancelled or refused there leaves its files and says cleanup is pending, and no lock-check file is created there. |
| D7 | **Crash-safe, not power-loss-safe** | If Moosik or the computer process stops mid-save, the next start sees either the previous analysis or the complete new one, never a mix of the two. After a **power loss**, an older analysis may come back; it is used only if it still matches the file. |
| D8 | **What is in use is never deleted, and the overrun is reported** | The analysis on screen, a finished analysis on its way to the screen, and Left/Right channels being read are protected. If that is more than the size limit allows, Moosik says how much is over and how much is in use, instead of deleting it. |

## What is stored

For each analysis (one track at one set of analysis settings), `cache/v2/` holds:

- `<stem>.head`: names the current version of the analysis;
- `<stem>.g<id>.mix`: the spectrum (Mix);
- `<stem>.g<id>.lr`: the Left and Right channels, when they were analysed;
- `<stem>.g<id>.pin`: a lock file that marks the version as in use;
- `<stem>.coord`, `<stem>.pub`: the coordination files (D5).

The spectrum and channel data inside each file are exactly what earlier versions stored: the same 12-bit levels and the same packing. Nothing about analysis quality changed. The envelope around them is new. It records:
- which analysis it belongs to: every setting that affects the result, the sample rate and frame timing, and the file's identity (D3);
- a SHA-256 digest of the decoded audio;
- a SHA-256 digest of the spectrum the channels belong to;
- a SHA-256 digest of the data itself.

A file whose envelope, length or digest does not check out is not used at all. It is never partly used. The data digest is checked on a second thread while the data is decoded; what was decoded is kept only if the digest matched, and the load finishes only after both are done.

## How a version becomes current

1. One analysis at a time takes the right to publish for that track and those settings. A second one, for example from a second Moosik window, shows its result without saving it.
2. The new version's files are written under a new ID and read back to verify them. Nothing points at them yet.
3. Under the track's lock, Moosik checks that the analysis was not cancelled and that the audio file it read is unchanged (D2).
4. Only then is `<stem>.head` replaced, in one step, by a file naming the new version.

Readers take the head and the version's lock together, so a version cannot be deleted between being found and being opened. The Left/Right channels are always read from the same version as the spectrum on screen, never from whatever version is newest.

## When a finished analysis is shown

It is shown only if the track that is selected now is still the file that was analysed: the same path, with the same size, time and sample now. If the file was replaced while it was being analysed, nothing from that analysis is shown for the new file, neither the spectrum nor the waveform or statistics.

## Cache controls

- **Re-analyze** analyses the track again. Nothing is deleted first: the new analysis replaces the old one only when it has been saved completely. The old one is then removed once nothing uses it.
- **Clear Cache** removes this track's analysis at your settings. If it is on screen, it is taken off screen first. Anything still reading it keeps it, and that is reported.
- **Clear All** removes every analysis this version made. Legacy files from older versions are kept and counted in the message.
- **Trim now** and the size limit remove superseded versions first, then whole analyses, least recently analysed first. Legacy files are not counted against the limit and are never removed.

Every message says what was actually done. It counts only space that was really freed, and it names anything that could not be deleted, anything left for a later cleanup, and anything kept because it is in use.

## Legacy analyses (files from earlier versions)

Files directly in `.moosik/cache/` are legacy. When the analysis at your settings is not on screen, Moosik can offer a legacy analysis of the same track as an alternative:
- **Legacy · settings from its record**: written by a 1.5.3 test build, which recorded its exact settings.
- **Legacy · nominal settings from its file name**: written by 1.5.2.1 or earlier, which recorded settings only to two decimals in the name.

Either kind can be shown if you choose it. Neither is ever used automatically. Neither is checked against the audio file, so if the file has been replaced, a legacy analysis can describe the old audio. The label says so. Removing legacy files is left to you. This version never does it.

## Known limits, stated plainly

- **A change between the last check and the save is not caught** (D2). This is the known check-to-commit window. Saving is not atomic with respect to other programs writing your music files.
- **A sampled content check can miss a change** confined to unsampled bytes that also keeps the size and the last-modified time (D3).
- **Tag edits can cause re-analysis** (D3).
- **Power loss can bring back an older valid analysis** (D7).
- **The size limit can be exceeded** while analyses are in use (D8), and on folders where deletion is disabled (D6).
- **The coordination files are never removed** by Moosik (D5): two empty files per analysed track. A future offline tidy-up could reclaim them; none is included.
- **Replacing the cache folder itself, or one of its parent folders, while Moosik runs** is not detected and is not supported.
- **After a crash, unfinished leftovers** (an uncommitted version, or a version whose lock file could not be removed) are cleaned up by the next save, trim or clear of that track's analysis. There is no background cleaner.
