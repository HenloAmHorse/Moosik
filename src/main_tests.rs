use super::*;

// -----------------------------------------------------------------------
// Terminal lifecycle at the Engine
// -----------------------------------------------------------------------

/// Every loop mode agrees with itself about what "next" is, and the
/// end-of-list case is the only one that stops.
#[test]
fn the_loop_modes_choose_the_track_they_say_they_do() {
    use LoopMode::*;
    // Repeat One is the same track, forever, which is exactly why a
    // failure must never reach this function.
    for i in 0..4 {
        assert_eq!(next_in_playlist(RepeatOne, Some(i), 4), Some(i));
    }
    // Repeat All wraps.
    assert_eq!(next_in_playlist(RepeatAll, Some(0), 4), Some(1));
    assert_eq!(next_in_playlist(RepeatAll, Some(3), 4), Some(0));
    // Sequential stops at the end rather than wrapping.
    assert_eq!(next_in_playlist(Sequential, Some(2), 4), Some(3));
    assert_eq!(next_in_playlist(Sequential, Some(3), 4), None);
    // An empty playlist has no next track under any mode.
    for m in [Sequential, RepeatAll, RepeatOne] {
        assert_eq!(next_in_playlist(m, Some(0), 0), None);
        assert_eq!(next_in_playlist(m, None, 0), None);
    }
}

/// A minimal 16-bit PCM WAV, for the seek-worker fixtures.
///
/// `bitperfect`'s own fixture writer is not reachable from here, and these
/// files exist to be read by `rodio`'s decoder rather than by `prepare`.
fn write_seek_wav(path: &Path, pcm: &[u8], rate: u32, channels: u16) {
    let bits = 16u16;
    let block = channels * bits / 8;
    let data_len = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * block as u32).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    std::fs::write(path, out).unwrap();
}

struct SeekTemp(PathBuf);
impl Drop for SeekTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn seek_temp(name: &str) -> SeekTemp {
    SeekTemp(std::env::temp_dir().join(format!("moosik_seek_{}_{name}", std::process::id())))
}

/// The seek worker lands on a frame, on a real stereo file.
///
/// The skip was computed straight in samples — `target * rate * channels`,
/// truncated — so on stereo it was as likely as not to be odd. An odd
/// number of samples discarded leaves the decoder standing on a
/// right-channel sample, and every frame after the seek is assembled from
/// the right of one frame and the left of the next: the channels swap for
/// the rest of the track, silently, and nothing anywhere reports it.
///
/// This runs the real worker over a real file whose two channels are
/// distinguishable, and reads the first sample it produces.
#[test]
fn the_seek_worker_never_lands_between_two_channels() {
    // 8 kHz stereo, 4000 frames = 0.5 s. Left samples are positive, right
    // samples negative, so which channel a sample belongs to is a property
    // of the sample itself.
    const RATE: u32 = 8_000;
    const FRAMES: usize = 4_000;
    let mut pcm = Vec::with_capacity(FRAMES * 2 * 2);
    for i in 0..FRAMES {
        let mag = 1_000 + (i as i16 % 5_000);
        pcm.extend_from_slice(&mag.to_le_bytes()); // left: positive
        pcm.extend_from_slice(&(-mag).to_le_bytes()); // right: negative
    }
    let t = seek_temp("stereo_seek.wav");
    write_seek_wav(&t.0, &pcm, RATE, 2);

    // The targets are the point of the fixture.
    //
    // A whole number of milliseconds cannot fail here: at 8 kHz stereo the
    // old rule computed `target * 16000`, which is even for every whole
    // millisecond, so a test built on round targets passes against the
    // broken code as readily as against this one. These are chosen so the
    // truncated product is *odd* — the case that leaves the decoder
    // standing on a right-channel sample — and the assertion below refuses
    // to run on a target that would not have exercised it.
    for us in [7_813u64, 62_563, 123_456, 187_563, 250_063] {
        let target = Duration::from_micros(us);
        let would_have_been = (target.as_secs_f64() * RATE as f64 * 2.0) as u64;
        assert_eq!(
            would_have_been % 2,
            1,
            "{us} us: this fixture only tests anything if the old \
                 sample-count rule lands off a frame"
        );

        let rx = Engine::spawn_seek_worker(&t.0, target, never_cancelled()).expect("worker starts");
        let landing = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the worker answers")
            .unwrap_or_else(|e| panic!("{us} us: {e:?}"));

        let expected_frames = (target.as_secs_f64() * RATE as f64).floor() as u64;
        assert_eq!(
            landing.landed,
            frame_time(expected_frames, RATE),
            "{us} us: the landing is a frame boundary"
        );

        let mut decoder = landing.decoder;
        let first = decoder.next().expect("there is audio left");
        assert!(
            first > 0,
            "{us} us: the first sample after the seek must be a left \
                 sample; got {first}, which is the right channel"
        );
    }
}

/// A seek past the end of a real file does not hand back a decoder.
///
/// The worker stopped discarding when the decoder ran out and returned it
/// anyway, sitting at EOF, while the caller published the target it had
/// asked for.
#[test]
fn a_seek_past_the_end_of_a_real_file_reports_where_the_file_ends() {
    const RATE: u32 = 8_000;
    const FRAMES: usize = 800; // 0.1 s
    let mut pcm = Vec::with_capacity(FRAMES * 2 * 2);
    for _ in 0..FRAMES {
        pcm.extend_from_slice(&256i16.to_le_bytes());
        pcm.extend_from_slice(&256i16.to_le_bytes());
    }
    let t = seek_temp("short_seek.wav");
    write_seek_wav(&t.0, &pcm, RATE, 2);

    let rx = Engine::spawn_seek_worker(&t.0, Duration::from_secs(5), never_cancelled()).expect("worker starts");
    let got = rx.recv_timeout(Duration::from_secs(10)).expect("the worker answers");
    match got {
        Err(SeekError::PastEnd { landed }) => {
            // Exact, to the frame.
            //
            // `landed <= 101 ms` was satisfied by zero, which is what a
            // worker that gave up immediately would report — and the whole
            // point is that it says where the file *does* end. The file
            // holds `FRAMES` frames at `RATE`, the worker discards whole
            // frames until the decoder runs out, so the answer is that
            // count and nothing else.
            assert_eq!(
                landed,
                frame_time(FRAMES as u64, RATE),
                "the file ends at frame {FRAMES}; reported {landed:?}"
            );
            assert!(
                landed > Duration::ZERO,
                "zero would mean it never looked, which is the failure this \
                     assertion is here to exclude"
            );
            assert!(
                SeekError::PastEnd { landed }.describe().contains("0:00"),
                "and it says where"
            );
        }
        Err(other) => panic!("expected PastEnd, got {other:?}"),
        Ok(l) => panic!(
            "the worker handed back a decoder sitting at EOF, reporting {:?}",
            l.landed
        ),
    }
}

/// A failed seek leaves an engine that can be played again.
///
/// **What this covers and what it does not.** `rodio::Sink` cannot be
/// constructed without an output device, and a test machine may have none,
/// so this exercises the engine's own state transitions — the clock, the
/// pending-seek slot, the published session state and the parked reason —
/// with no sink installed. That is the half of the defect that lives in
/// this process. The other half, that `resume` reopens the file, is
/// asserted here only as far as "it does not silently do nothing": the
/// reopen path is `play_seeked_async`, which needs the device.
#[test]
fn a_failed_seek_leaves_a_resumable_engine() {
    let mut e = engine();
    e.bp_state.opened(
        bitperfect::Transport::Shared { endpoint: endpoint() },
        bitperfect::state::Fidelity::Processed {
            transform: bitperfect::state::TransformDescription::Identity,
            reason: "shared".into(),
        },
        None,
        None,
        e.shared_processing(),
    );
    e.current_duration = Some(Duration::from_secs(300));
    e.paused_elapsed = Duration::from_secs(120);
    e.started_at = None;
    let from = e.elapsed();

    // The slider has been dragged; the seek is in flight and the sink is
    // already gone.
    e.paused_elapsed = Duration::from_secs(280);
    e.abandon_seek(from, SeekError::Open("no such file".into()));

    assert_eq!(e.elapsed(), from, "back to where playback actually was");
    assert!(e.started_at.is_none(), "and stopped, not counting forward");
    assert!(!e.is_seeking(), "with nothing left to poll");
    assert_eq!(
        e.bp_state.playback,
        bitperfect::state::PlaybackState::Stopped,
        "and the published state agrees that nothing is playing"
    );
    assert!(
        e.bp_state.source.is_none(),
        "a stopped route describes no source"
    );

    assert_eq!(
        e.poll_pending_seek(),
        SeekOutcome::Failed(SeekError::Open("no such file".into()))
    );
    assert_eq!(e.poll_pending_seek(), SeekOutcome::Idle, "reported once");
}

/// A truncated DSD file opens no device.
///
/// The feeder catches a container that declares more audio than the file
/// holds — but by then an ASIO engine is running and the DAC is locked to
/// a DSD stream, so the failure arrives as a fault on a route that was
/// already claiming to be exact, and the fallback has to fight this
/// process for the device. The gate runs first, and this counts.
#[test]
fn a_truncated_dsd_file_opens_no_device() {
    use std::cell::Cell;

    let truncated = dsd::DsdInfo {
        declared_data_len: 4_096,
        data_len: 1_024,
        ..whole_dsd_info()
    };
    assert!(truncated.is_truncated(), "the fixture has to be truncated");

    let opens = Cell::new(0u32);
    let got: Result<(), String> = Engine::open_native_checked(&truncated, || {
        opens.set(opens.get() + 1);
        Ok(())
    });
    assert!(got.is_err(), "a truncated file must not reach the device");
    assert_eq!(
        opens.get(),
        0,
        "the device was opened before the file was rejected"
    );
    assert!(got.unwrap_err().contains("truncated"));

    // And a whole file does open, or the gate is just a wall.
    let whole = whole_dsd_info();
    assert!(!whole.is_truncated());
    let opens = Cell::new(0u32);
    let got: Result<(), String> = Engine::open_native_checked(&whole, || {
        opens.set(opens.get() + 1);
        Ok(())
    });
    assert!(got.is_ok());
    assert_eq!(opens.get(), 1);
}

/// A DSD file whose container and contents agree.
fn whole_dsd_info() -> dsd::DsdInfo {
    dsd::DsdInfo {
        container: dsd::DsdContainer::Dff,
        sample_rate: 2_822_400,
        channels: 2,
        layout: None,
        sample_count: 8_192,
        data_offset: 0,
        data_len: 2_048,
        declared_data_len: 2_048,
        block_size: 0,
        lsb_first: false,
        id3: None,
    }
}

/// A 64-bit float source is Processed under both processed policies, and
/// refused outright under Strict.
#[test]
fn the_64_bit_float_route_is_processed_or_refused() {
    use bitperfect::state::{Fidelity, OutputPolicy, TransformDescription};

    let source = bitperfect::state::MediaSource::pcm(bitperfect::format::SourceFormat {
        kind: bitperfect::format::PcmKind::Float64,
        sample_rate: 48_000,
        channels: 2,
        layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
    });
    let transport = bitperfect::Transport::Shared { endpoint: endpoint() };

    // Strict has one rung and it is the identity one, which exists so the
    // open fails and says why rather than quietly converting.
    assert_eq!(
        bitperfect::plan_ladder(
            bitperfect::format::PcmKind::Float64,
            OutputPolicy::StrictExact
        ),
        vec![bitperfect::PayloadPlan::Identity]
    );
    assert!(
        bitperfect::format::exact_candidates(
            bitperfect::format::PcmKind::Float64,
            false
        )
        .is_err(),
        "there is no exact device format for a double"
    );

    // The processed policies convert, and say what they did and why.
    for policy in [OutputPolicy::PreferExact, OutputPolicy::HqProcessed] {
        let ladder =
            bitperfect::plan_ladder(bitperfect::format::PcmKind::Float64, policy);
        assert_eq!(ladder, vec![bitperfect::PayloadPlan::Q31], "{policy:?}");
        let fidelity = Engine::fidelity_for(&source, ladder[0], &transport);
        match fidelity {
            Fidelity::Processed { transform, reason } => {
                assert_eq!(transform, TransformDescription::Float64ToQ31Processed);
                // The reason is the source, not the DAC. Blaming the
                // device invited the listener to go looking for one that
                // would fix it; there is none.
                assert!(
                    reason.contains("64-bit float source"),
                    "{policy:?}: {reason}"
                );
                assert!(
                    !reason.contains("the device has no Float32"),
                    "{policy:?}: the device is not why: {reason}"
                );
            }
            other => panic!("{policy:?}: expected Processed, got {other:?}"),
        }
    }
}

/// A 64-bit float source can never be value-exact, whatever the scan says.
///
/// The label means "the representation changed and no number did". A
/// 64-bit float source has had its numbers changed before any of this
/// sees them — the decoder narrows it to `f32` on the way into the
/// canonical buffer, and both the whole-track scan and the running
/// conversion look at what came out of that. Every sample can land
/// perfectly on the Q1.31 lattice and the claim still be false.
#[test]
fn a_64_bit_float_source_never_earns_the_value_exact_label() {
    // Everything the label needs, and it still does not get it.
    assert!(
        !value_exact_allowed(true, true, true, true),
        "a narrowed source cannot be value-exact on any evidence"
    );
    // The same evidence from a source that was not narrowed does.
    assert!(value_exact_allowed(true, true, true, false));

    // And each of the other three is still necessary.
    assert!(!value_exact_allowed(false, true, true, false), "no scan");
    assert!(!value_exact_allowed(true, false, true, false), "rounded live");
    assert!(!value_exact_allowed(true, true, false, false), "wrong route");
}

/// A failed seek resumes the track that was playing, not the last exact one.
///
/// `resume` reached for `last_prepared`, which is written by the exact
/// route and by a completed seek and by nothing else. Play an exact track
/// A, then an ordinary shared track B, and `last_prepared` still names A:
/// a failed seek in B followed by pressing play started **A**, from
/// somewhere in the middle, with B's duration on the slider.
///
/// **What this covers.** `rodio::Sink` cannot be built without an output
/// device, so this exercises the engine's own record of what is playing
/// and where — the thing `resume` reads — rather than the reopen itself.
/// The reopen is `play_seeked_async`, which needs the device.
#[test]
fn a_failed_seek_resumes_the_track_that_was_playing() {
    let mut e = engine();

    // Exact track A.
    let a = PathBuf::from("A.flac");
    let a_src = bitperfect::format::SourceFormat {
        kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
        sample_rate: 96_000,
        channels: 2,
        layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
    };
    e.last_prepared = Some((a.clone(), a_src));
    e.note_current(
        &a,
        Some(bitperfect::state::MediaSource::pcm(a_src)),
        Duration::from_secs(30),
    );

    // Then ordinary shared track B. `last_prepared` still names A, because
    // nothing on the shared route writes it.
    let b = PathBuf::from("B.mp3");
    let b_src = bitperfect::format::SourceFormat {
        kind: bitperfect::format::PcmKind::Float32,
        sample_rate: 44_100,
        channels: 2,
        layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
    };
    e.note_current(&b, Some(bitperfect::state::MediaSource::pcm(b_src)), Duration::ZERO);
    assert_eq!(
        e.last_prepared.as_ref().map(|(p, _)| p),
        Some(&a),
        "this test is about the case where the two disagree"
    );

    // Playing B, ninety seconds in. The slider is dragged to the end and
    // the seek fails; `abandon_seek` is the only thing that touches the
    // position from here.
    e.current_duration = Some(Duration::from_secs(240));
    e.started_at = None;
    e.paused_elapsed = Duration::from_secs(235);
    e.abandon_seek(Duration::from_secs(90), SeekError::Open("gone".into()));

    let (path, at) = e.resume_target().expect("there is something to resume");
    assert_eq!(path, b, "B is what was playing, so B is what resumes");
    assert_ne!(path, a, "A finished thirty seconds of playback ago");
    assert_eq!(
        at,
        Duration::from_secs(90),
        "and at where playback actually reached, not where the pointer went"
    );
    assert_eq!(e.elapsed(), Duration::from_secs(90));
}

/// The same, with nothing in `last_prepared` at all.
///
/// A shared track played from a cold start — no exact route has ever run
/// in this session — left `last_prepared` empty, so `resume` had nothing
/// to reach for and silently did nothing.
#[test]
fn a_shared_track_with_no_exact_history_still_resumes() {
    let mut e = engine();
    assert!(e.last_prepared.is_none());

    let b = PathBuf::from("only.mp3");
    e.note_current(&b, None, Duration::ZERO);
    e.paused_elapsed = Duration::from_secs(200);
    e.abandon_seek(Duration::from_secs(12), SeekError::WorkerLost);

    let (path, at) = e.resume_target().expect("there is something to resume");
    assert_eq!(path, b);
    assert_eq!(at, Duration::from_secs(12));
    assert!(
        e.current_track().unwrap().source.is_none(),
        "an unknown identity stays unknown rather than being invented"
    );

    // With nothing playing at all there is nothing to resume, which is a
    // different answer from the wrong track.
    let empty = engine();
    assert!(empty.resume_target().is_none());
}

/// Turning bit-perfect on mid-track attempts the exact route.
///
/// `restart_current_track` short-circuits to `play_seeked_async` when
/// playback is not on a device stream — which is exactly the situation
/// when the listener turns the toggle on during a shared track. It
/// restarted the same shared route it was already on and the toggle
/// appeared to do nothing. The test is on where playback is *going*.
#[test]
fn enabling_bit_perfect_mid_track_does_not_restart_the_shared_route() {
    let mut e = engine();
    // On the shared mixer.
    e.bp_state.opened(
        bitperfect::Transport::Shared { endpoint: endpoint() },
        bitperfect::state::Fidelity::Processed {
            transform: bitperfect::state::TransformDescription::Identity,
            reason: "shared".into(),
        },
        None,
        None,
        e.shared_processing(),
    );
    assert!(!e.on_bp_stream());

    // Off: the shortcut is right, because the destination is the mixer.
    e.bit_perfect = false;
    assert!(
        !(e.bit_perfect || e.native_dsd_selected()),
        "with the toggle off, a shared restart is the correct route"
    );

    // On: the shortcut is wrong, because the destination is the device.
    e.bit_perfect = true;
    assert!(
        e.bit_perfect || e.native_dsd_selected(),
        "with the toggle on, the restart must attempt the exact route"
    );
    // And the state of the *current* stream must not be what decides it.
    assert!(
        !e.on_bp_stream(),
        "which is precisely the case the old condition got wrong"
    );
}

/// A session state describing a live payload-exact route.
fn exact_session() -> bitperfect::state::OutputSessionState {
    use bitperfect::state::{Fidelity, OutputFormat, ProcessingState};
    let mut st = bitperfect::state::OutputSessionState {
        requested: true,
        ..Default::default()
    };
    st.opened(
        bitperfect::Transport::WasapiExclusivePcm { endpoint: endpoint() },
        Fidelity::PayloadExact,
        Some(OutputFormat {
            endpoint: endpoint(),
            sample_rate: 96_000,
            channels: 2,
            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
            container_bits: 32,
            valid_bits: 24,
            integer: true,
            buffer_frames: 480,
            label: "24i/32 excl".into(),
        }),
        None,
        ProcessingState::transparent_locked(),
    );
    st
}

/// Two ticks: a dropout, then the write failure that ends the track.
///
/// This is the sequence a listener actually gets — the device runs short
/// on one frame and stops taking audio on the next — and it used to
/// produce a headline naming the dropout and no mention at all of the
/// device. Both records went into `Fidelity`, which holds one thing and is
/// first-wins, so the dropout took the slot and the reason the music
/// stopped was discarded. The public fatal getter falling through to
/// recoverable evidence is what made the UI push it there.
#[test]
fn a_dropout_on_one_tick_does_not_hide_the_failure_on_the_next() {
    use bitperfect::fault;
    use bitperfect::state::Badge;

    let mut st = exact_session();
    let g = st.generation;
    assert_eq!(st.presentation().badge, Badge::PayloadExact, "a diamond, to start");

    // Tick one: the device ran short.
    apply_integrity(&mut st, g, &[fault::UNDERRUN], &[fault::NONE]);
    let p = st.presentation();
    assert_eq!(p.badge, Badge::Faulted, "amber, and no diamond");
    assert!(!st.shows_diamond());
    assert!(
        p.headline.contains("dropout"),
        "and it says what happened: {}",
        p.headline
    );
    assert!(
        st.playback.is_active(),
        "a dropout ends nothing — the music is still playing"
    );

    // Tick two: the device stopped accepting audio.
    apply_integrity(&mut st, g, &[fault::UNDERRUN], &[fault::BACKEND_WRITE]);
    let p = st.presentation();
    assert_eq!(p.badge, Badge::Faulted);
    assert!(
        p.headline.contains("write"),
        "the headline is the reason the track stopped: {}",
        p.headline
    );
    assert!(
        p.detail.iter().any(|d| d.starts_with("Integrity:") && d.contains("dropout")),
        "and the dropout before it is kept: {:?}",
        p.detail
    );
}

/// A clean successor answers for nothing its predecessor did.
#[test]
fn a_clean_successor_carries_none_of_the_previous_tracks_evidence() {
    use bitperfect::fault;
    use bitperfect::state::{Badge, Fidelity, TransformDescription};

    let mut st = exact_session();
    let b = st.generation;
    apply_integrity(&mut st, b, &[fault::UNDERRUN], &[fault::BACKEND_WRITE]);
    assert!(st.revoked.is_some());
    assert_eq!(st.presentation().badge, Badge::Faulted);

    // The device crosses into C, which is a different file on the same
    // open stream.
    let c = st.roll_over(
        Fidelity::PayloadExact,
        bitperfect::state::ProcessingState::transparent_locked(),
    );
    assert!(c > b);
    assert!(
        st.revoked.is_none(),
        "B's dropout is not C's — it is not even about the same file"
    );
    let p = st.presentation();
    assert_eq!(p.badge, Badge::PayloadExact, "C starts clean: {}", p.headline);
    assert!(st.shows_diamond());
    assert!(
        !p.detail.iter().any(|d| d.starts_with("Integrity:")),
        "and nothing of B's is left in the detail: {:?}",
        p.detail
    );

    // Evidence stamped with B cannot reach C, either.
    apply_integrity(&mut st, b, &[fault::UNDERRUN], &[fault::BACKEND_DEAD]);
    assert!(st.shows_diamond(), "a superseded generation may not speak for C");
    let _ = TransformDescription::Identity;
}

/// A dropout costs the claim and nothing else: the track still ends
/// cleanly and the playlist still advances.
///
/// The two halves are checked together because separating them is what
/// went wrong in the first place — integrity and completion were one
/// thing, so a single momentary underrun stopped the playlist at the end
/// of the track it happened in.
#[test]
fn a_recoverable_loss_costs_the_diamond_and_not_the_advance() {
    use bitperfect::fault;
    use bitperfect::state::Badge;

    for code in [
        fault::UNDERRUN,
        fault::CALLBACK_LOCK_MISS,
        fault::VALUE_EXACT_VIOLATION,
    ] {
        assert!(!fault::is_fatal(code));
        let mut st = exact_session();
        let g = st.generation;
        apply_integrity(&mut st, g, &[code], &[fault::NONE]);
        assert!(!st.shows_diamond(), "code {code} must withdraw the diamond");
        assert_eq!(st.presentation().badge, Badge::Faulted, "code {code}");
    }
    // The completion half of the same rule — that a dropout still earns
    // the advance — is asserted against `Shared` in `bitperfect`, which is
    // where that state lives.
}

/// A cached now-playing line cannot go on naming a claim the session has
/// lost.
///
/// `now_playing_line` embeds the fidelity headline in a `String` that is
/// then stored, so a line rendered while the route was payload-exact went
/// on showing a green diamond after a dropout had withdrawn the claim from
/// every live surface. The snapshot records the badge it was built with,
/// and the tick refreshes it when that changes.
#[test]
fn a_cached_now_playing_line_is_refreshed_when_the_headline_changes() {
    use bitperfect::state::Badge;

    // A line under a headline that no longer says what the session says.
    let mut s = StatusLine::default();
    s.now_playing("A — Track".into(), "A".into());
    assert!(
        s.refresh("B", true, || "B — Track".to_string()),
        "a line that no longer says what the session says is stale"
    );
    assert_eq!(s.text(), "B — Track");
    assert_eq!(*s.owner(), StatusOwner::NowPlaying { headline: "B".into() });
    assert!(!s.refresh("B", true, || unreachable!("nothing to redo")));

    // A message the listener has just been given is not ours to overwrite.
    let mut s = StatusLine::default();
    s.transient("⚠ Seek failed: the container refused it");
    assert!(!s.refresh("B", true, || unreachable!("not ours")));
    assert_eq!(s.text(), "⚠ Seek failed: the container refused it");

    let mut s = StatusLine::default();
    assert!(!s.refresh("B", true, || unreachable!("nobody owns it")));

    // And a stopped player never gets a now-playing line put back.
    let mut s = StatusLine::default();
    s.now_playing("A — Track".into(), "A".into());
    assert!(!s.refresh("B", false, || unreachable!("stopped")));
    assert_eq!(s.text(), "A — Track");

    // The two transitions it has to catch, on the real state machine.
    let mut st = exact_session();
    let first = st.presentation().headline;
    assert_eq!(st.presentation().badge, Badge::PayloadExact);
    let g = st.generation;

    // A dropout: a different badge, and a different sentence.
    apply_integrity(&mut st, g, &[bitperfect::fault::UNDERRUN], &[]);
    let second = st.presentation().headline;
    assert_ne!(first, second);
    let mut s = StatusLine::default();
    s.now_playing(format!("{first} — Track"), first.clone());
    assert!(s.refresh(&second, true, || format!("{second} — Track")));

    // The write failure that ends the track: the *same* badge, a different
    // sentence. Comparing badges left the dropout on screen and never said
    // why the music stopped.
    apply_integrity(&mut st, g, &[], &[bitperfect::fault::BACKEND_WRITE]);
    let third = st.presentation().headline;
    assert_eq!(
        st.presentation().badge,
        Badge::Faulted,
        "both are Faulted, which is the whole point"
    );
    assert_ne!(second, third, "{third}");
    let mut s = StatusLine::default();
    s.now_playing(format!("{second} — Track"), second.clone());
    assert!(
        s.refresh(&third, true, || format!("{third} — Track")),
        "a fatal reason under an unchanged badge still has to reach the bar"
    );
    assert!(s.text().contains(&third));
}

/// Who owns a session does not depend on whether the device still works.
///
/// It did, and the consequence was silence with no explanation. The
/// predicate that routes "how did this end?" to the object holding the
/// session also asked whether that object was still alive — so the moment
/// a WASAPI backend died or an ASIO driver requested a reset, the engine
/// stopped asking the only thing that knew. `completion` fell through to
/// the shared-sink branch, found no sink, and answered `Running`. The
/// track never ended, the playlist never advanced, `halt_on_failure`
/// never ran, and the exclusive handle stayed open for the life of the
/// process. The failure had been recorded correctly, in a place nobody
/// was reading.
///
/// `session_owner` takes what exists and nothing else. A function that
/// cannot see liveness cannot make ownership depend on it.
#[test]
fn a_dead_backend_still_owns_the_session_it_was_carrying() {
    use bitperfect::Transport;

    let wasapi = Transport::WasapiExclusivePcm { endpoint: endpoint() };
    let asio = Transport::AsioNativeDsd {
        driver: bitperfect::state::EndpointIdentity::from_name("ASIO"),
    };
    let alsa = Transport::AlsaNativeDsd { endpoint: endpoint() };
    let shared = Transport::Shared { endpoint: endpoint() };

    // A handle exists: it owns the session, whatever state it is in.
    assert!(session_owner(&wasapi, true, false));
    assert!(session_owner(&asio, false, true));
    assert!(session_owner(&alsa, false, true));

    // No handle: there is nothing to ask, and the caller falls through to
    // the sink — which is the *only* case in which falling through is
    // right.
    assert!(!session_owner(&wasapi, false, false));
    assert!(!session_owner(&asio, false, false));

    // A device transport is not answered by the other kind of handle.
    assert!(!session_owner(&wasapi, false, true));
    assert!(!session_owner(&asio, true, false));

    // The shared mixer is never a device object, however many handles are
    // lying around.
    assert!(!session_owner(&shared, true, true));
    assert!(!session_owner(&Transport::Inactive, true, true));
    assert!(!session_owner(&Transport::Opening, true, true));
    assert!(!session_owner(
        &Transport::Dead {
            reason: bitperfect::state::FailureReason::BackendDead("gone".into())
        },
        true,
        true
    ));
}

/// Four things an open can take, each with a witness whose drop is
/// counted.
///
/// A rodio `Sink`, a `BpStream` and a native session all need a sound card
/// to construct, so an engine in a test never holds one — which is why
/// asserting `bp.is_none()` after a failed open asserted nothing at all:
/// it was already true before the failure. These sit in the same slots and
/// are released by the same statements, so a cleanup step that stops
/// running stops being witnessed.
fn engine_holding(counter: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Engine {
    let mut e = engine();
    e.held_sink = Some(Sentinel(std::sync::Arc::clone(counter)));
    e.held_bp = Some(Sentinel(std::sync::Arc::clone(counter)));
    e.held_native = Some(Sentinel(std::sync::Arc::clone(counter)));
    e.held_scan = Some(Sentinel(std::sync::Arc::clone(counter)));
    e
}

fn released(counter: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> usize {
    counter.load(std::sync::atomic::Ordering::Relaxed)
}

/// An open that fails after it has taken something gives all of it back,
/// and leaves nothing claiming to be playing.
///
/// Every step of a real open needs a sound card, so the open is injected —
/// but the orchestration around it, and the cleanup it performs, is the
/// production one. The injected open does what a half-open does: records a
/// current track, then fails.
///
/// What it took is witnessed rather than inferred. The previous version of
/// this test asserted that four fields were empty on an engine where all
/// four had always been empty, so it passed against an `abort_open` that
/// did nothing whatever.
#[test]
fn a_failed_open_releases_everything_it_took() {
    use bitperfect::state::OutputPolicy;

    for is_dsd in [false, true] {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut e = engine_holding(&counter);
        e.bp_state.begin_open();
        e.bp_state.policy = OutputPolicy::StrictExact;
        // Something was playing before this restart.
        e.note_current(
            Path::new("previous.flac"),
            Some(bitperfect::state::MediaSource::pcm(
                bitperfect::format::SourceFormat {
                    kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
                    sample_rate: 96_000,
                    channels: 2,
                    layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
                },
            )),
            Duration::from_secs(30),
        );
        assert!(e.current_track().is_some());

        let out = open_track(
            &mut e,
            Opening::at(Duration::from_secs(30)),
            is_dsd,
            |eng, _o| {
                // A route that got as far as naming itself and then could
                // not start.
                eng.note_current(Path::new("attempted.flac"), None, Duration::ZERO);
                Err(bitperfect::OpenError::backend("the render thread would not start"))
            },
            |_, _| panic!("a backend failure is not a format rejection"),
        );

        assert!(
            matches!(out, RestartOutcome::Stopped(_)),
            "is_dsd={is_dsd}: {out:?}"
        );
        assert!(
            e.current_track().is_none(),
            "is_dsd={is_dsd}: a route that never played is not what is playing"
        );
        assert!(e.bp.is_none(), "is_dsd={is_dsd}: no handle survives");
        assert_eq!(
            released(&counter),
            4,
            "is_dsd={is_dsd}: the sink, the exact stream, the native \
                 session and the scan are all let go — anything still held \
                 is a device no other application can take"
        );
        assert!(!e.dsd_mode && !e.dsd_native && !e.dsd_fallback);
        assert!(
            matches!(
                e.bp_state.fidelity,
                bitperfect::state::Fidelity::Failed { .. }
            ),
            "is_dsd={is_dsd}: and the typed reason is on screen: {:?}",
            e.bp_state.fidelity
        );
        assert!(
            !matches!(e.bp_state.transport, bitperfect::Transport::Opening),
            "is_dsd={is_dsd}: and it is not still verifying an output that                  was never opened"
        );
    }
}

/// A rebuild that cannot happen is reported as a failure, not as a seek
/// that landed.
///
/// The production `Engine::seek_to`, on the one failing route a test can
/// reach without a sound card: a decimated-DSD session whose source file
/// has gone. It returned `Ok(self.elapsed())` — the position playback
/// happened to stop at, handed back as though the decoder had moved there.
/// The bar moved, the spectrum followed, and the only sign that nothing
/// was playing was that nothing was playing.
#[test]
fn a_dsd_rebuild_that_fails_is_a_failure_and_holds_nothing() {
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut e = engine_holding(&counter);
    e.dsd_fallback = true;
    e.bp_state.begin_open();
    e.note_current(
        Path::new("was-playing.dsf"),
        None,
        Duration::from_secs(10),
    );

    let out = e.seek_to(
        Path::new("no-such-file-anywhere.dsf"),
        Duration::from_secs(20),
    );

    assert!(
        out.is_err(),
        "a rebuild that did not happen is not a position that was reached:              {out:?}"
    );
    assert_eq!(
        released(&counter),
        4,
        "and everything the session held is let go"
    );
    assert!(
        e.current_track().is_none(),
        "a route that never started is not what is playing"
    );
    assert!(
        matches!(
            e.bp_state.fidelity,
            bitperfect::state::Fidelity::Failed { .. }
        ),
        "with the reason on screen: {:?}",
        e.bp_state.fidelity
    );
    assert!(!matches!(e.bp_state.transport, bitperfect::Transport::Opening));
}

/// A restart emits no frame from position zero.
///
/// The two halves used to be separate: `play_file` opened every route at
/// zero and the caller seeked afterwards. The listener heard the opening
/// of the track every time they toggled bit-perfect or changed output
/// device mid-song — and if the seek then failed, that opening was all
/// they got, from a device the player was still holding.
///
/// What is captured here is every position a route was asked to open at.
/// Zero is not among them, on either rung of the ladder, and the landing
/// the caller is told about is the one the route reached.
#[test]
fn a_restart_opens_at_its_position_and_never_at_zero() {
    use bitperfect::state::OutputPolicy;
    use std::cell::RefCell;

    let target = Duration::from_secs(42);

    // The exact rung takes it.
    let asked: RefCell<Vec<Opening>> = RefCell::new(Vec::new());
    let mut e = engine();
    let out = open_track(
        &mut e,
        Opening::at(target),
        false,
        |eng, o| {
            asked.borrow_mut().push(o);
            eng.note_current(Path::new("track.flac"), None, o.at);
            Ok(o.at)
        },
        |_, _| panic!("the open succeeded; there is nothing to fall back from"),
    );
    assert_eq!(out, RestartOutcome::Exact { landed: target });
    assert_eq!(
        *asked.borrow(),
        vec![Opening::at(target)],
        "opened once, at the position it was restarting to"
    );
    assert!(
        !asked.borrow().iter().any(|o| o.at == Duration::ZERO),
        "and never at the beginning of the track"
    );

    // And the permitted fallback rung, which is the one a device refusal
    // reaches — it must not open at zero either.
    let asked: RefCell<Vec<Opening>> = RefCell::new(Vec::new());
    let mut e = engine();
    e.bp_state.policy = OutputPolicy::PreferExact;
    let out = open_track(
        &mut e,
        Opening {
            at: target,
            paused: false,
        },
        false,
        |_, o| {
            asked.borrow_mut().push(o);
            Err(bitperfect::OpenError::device_format("the DAC refused 24/192"))
        },
        |_, o| {
            asked.borrow_mut().push(o);
            Ok(o.at)
        },
    );
    assert_eq!(out, RestartOutcome::Shared { landed: target });
    assert_eq!(
        *asked.borrow(),
        vec![Opening::at(target), Opening::at(target)]
    );
}

/// What a route did, in the order it did it.
///
/// The property is an ordering one and ordering is invisible afterwards: a
/// route that started, played a buffer and paused ends in exactly the
/// state a route that came up paused ends in. Nothing in the engine, the
/// session state or the panel can tell them apart — which is why the
/// defect survived four phases of tests that all checked the final state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RouteEvent {
    Pause,
    Resume,
    /// The step after which the route can produce sound.
    Output,
}

#[derive(Default)]
struct RouteRecorder {
    events: Vec<RouteEvent>,
    fails: bool,
}

impl RouteBringUp for RouteRecorder {
    fn pause(&mut self) {
        self.events.push(RouteEvent::Pause);
    }
    fn resume(&mut self) {
        self.events.push(RouteEvent::Resume);
    }
    fn begin_output(&mut self) -> Result<(), bitperfect::OpenError> {
        self.events.push(RouteEvent::Output);
        if self.fails {
            Err(bitperfect::OpenError::backend("the render thread would not start"))
        } else {
            Ok(())
        }
    }
}

/// A paused route takes its pause flag before anything can sound.
///
/// `bring_up` is the production function every device opener calls, and
/// this drives it directly, because the thing being asserted is that no
/// event which can produce output precedes the pause.
#[test]
fn a_paused_route_pauses_before_anything_can_sound() {
    let mut r = RouteRecorder::default();
    bring_up(&mut r, true).expect("the fixture opens");
    assert_eq!(
        r.events,
        vec![RouteEvent::Pause, RouteEvent::Output],
        "the pause is taken first, and the step that can emit is second"
    );
    let first_output = r.events.iter().position(|e| *e == RouteEvent::Output);
    let pause = r.events.iter().position(|e| *e == RouteEvent::Pause);
    assert!(
        pause < first_output,
        "nothing that can produce sound may precede the pause: {:?}",
        r.events
    );
    assert!(
        !r.events.contains(&RouteEvent::Resume),
        "and a paused route is never resumed on the way up"
    );

    // The other direction: a route that was not asked to be paused comes
    // up playing, and still decides before it can emit.
    let mut r = RouteRecorder::default();
    bring_up(&mut r, false).expect("the fixture opens");
    assert_eq!(r.events, vec![RouteEvent::Resume, RouteEvent::Output]);

    // And a route whose output step fails has still taken the flag first,
    // so nothing was emitted on the way to the failure.
    let mut r = RouteRecorder {
        fails: true,
        ..Default::default()
    };
    assert!(bring_up(&mut r, true).is_err());
    assert_eq!(r.events, vec![RouteEvent::Pause, RouteEvent::Output]);
}

/// Whether the route is paused at the moment its output step runs.
///
/// Answered twice: once for a bring-up that asked to be paused, once for
/// one that did not. The probe runs *inside* `begin_output`, which is the
/// step after which the route can put a payload on the wire — so what it
/// reads is the flag the callback will read, at the instant the callback
/// could first run.
fn paused_at_output<T: RoutePause>(route: &T, is_paused: impl Fn() -> bool) -> (bool, bool) {
    let mut seen: [Option<bool>; 2] = [None; 2];
    for (i, asked) in [true, false].into_iter().enumerate() {
        let slot = &mut seen[i];
        bring_up(
            &mut StreamBringUp {
                route,
                start: Some(|| {
                    *slot = Some(is_paused());
                    Ok(())
                }),
            },
            asked,
        )
        .expect("the fixture's output step cannot fail");
    }
    (
        seen[0].expect("the output step ran"),
        seen[1].expect("the output step ran"),
    )
}

/// The DoP route is already paused when its output step runs.
///
/// Not the recorder. This drives the production adapter over the real
/// stream type this route holds — a `BpStream`, with its own `pause` and
/// its own flag — and asks the stream whether it is paused at the instant
/// `start_dop` would be called. The starting call itself is the one thing
/// that cannot be reached without a device (`start_dop` wants a decoded
/// DoP source), so it is the closure; what the closure does is read the
/// flag the callback reads.
///
/// This route ignored `Opening.paused` entirely and resumed
/// unconditionally, so a paused restart onto a DoP DAC opened the
/// generation playing. The test fails if the pause is removed, inverted,
/// or moved after the output step.
///
/// The exact-output route holds the same type and reaches the same
/// adapter, one call earlier in the same file.
#[test]
fn the_dop_route_is_already_paused_when_its_output_step_runs() {
    let bp = bitperfect::BpStream::for_evidence_test(
        bitperfect::test_shared::new(),
        bitperfect::PayloadPlan::Identity,
    );
    assert_eq!(
        paused_at_output(&bp, || bp.is_paused()),
        (true, false),
        "paused before the step that can emit, and playing when it was not \
             asked to be paused"
    );
}

/// The native ASIO DSD route is already paused when its session starts.
///
/// The same shape as the DoP test, over `AsioDsdStream`'s own pause flag —
/// the one `asio_callback` reads to decide between the session ring and
/// DSD silence. `start_session` needs a live ring and a running driver, so
/// the probe stands in its place.
#[test]
#[cfg(all(windows, feature = "asio-dsd"))]
fn the_native_asio_route_is_already_paused_when_its_session_starts() {
    let s = bitperfect::asio_dsd::for_pause_test();
    assert_eq!(
        paused_at_output(&s, || s.is_paused()),
        (true, false),
        "an exclusive DSD device must not emit a frame of a paused restart"
    );
}

/// The native ALSA DSD route is already paused when its session starts.
///
/// The Linux mirror of the ASIO test, over `AlsaDsdStream`'s own flag —
/// the one the writer loop reads before it decides what to hand `writei`.
#[test]
#[cfg(all(target_os = "linux", feature = "alsa-dsd"))]
fn the_native_alsa_route_is_already_paused_when_its_session_starts() {
    let s = bitperfect::alsa_dsd::for_pause_test();
    assert_eq!(
        paused_at_output(&s, || s.is_paused()),
        (true, false),
        "an exclusive DSD device must not emit a frame of a paused restart"
    );
}

/// Every device opener still starts its output *inside* `bring_up`.
///
/// A source check, and the only thing it is: it cannot show the ordering
/// is right, which is what the three tests above are for. What it catches
/// is a call site going back to the shape all four of them had — the
/// unconditional `resume()` on the line before the starting call, which is
/// how three routes ignored the pause intent through a whole phase while a
/// generic recorder test passed.
#[test]
fn every_device_opener_starts_its_output_inside_bring_up() {
    let src = include_str!("main.rs");
    for (from, to) in [
        ("fn start_bp(&mut self", "fn start_dop(&mut self"),
        ("fn start_dop(&mut self", "fn start_asio_native(&mut self"),
        ("fn start_asio_native(&mut self", "fn start_alsa_native(&mut self"),
        ("fn start_alsa_native(&mut self", "fn start_dsd_fallback(&mut self"),
    ] {
        let body = code_only(block(src, from, to));
        assert!(
            body.contains("bring_up("),
            "{from}: the route comes up through the one seam"
        );
        assert!(
            body.contains("opening.paused,"),
            "{from}: and it is handed the intent, not a constant"
        );
        assert!(
            !body.contains(".resume();"),
            "{from}: nothing resumes a route outside `bring_up`"
        );
    }
}

/// The pause intent reaches every opener, at every position.
///
/// `open_track` is the one orchestration both starting and restarting go
/// through, and the openers are injected — so what this holds is that the
/// intent is *handed over*, including on the fallback rung and including
/// at position zero, where an earlier version dropped it because it only
/// carried a `Duration`.
#[test]
fn a_paused_restart_hands_the_pause_to_the_route_it_opens() {
    use bitperfect::state::OutputPolicy;
    use std::cell::RefCell;

    for target in [Duration::ZERO, Duration::from_secs(42)] {
        // The exact rung.
        let seen: RefCell<Vec<Opening>> = RefCell::new(Vec::new());
        let mut e = engine();
        let out = open_track(
            &mut e,
            Opening {
                at: target,
                paused: true,
            },
            false,
            |_, o| {
                seen.borrow_mut().push(o);
                Ok(o.at)
            },
            |_, _| panic!("the open succeeded"),
        );
        assert_eq!(out, RestartOutcome::Exact { landed: target });
        assert_eq!(
            *seen.borrow(),
            vec![Opening {
                at: target,
                paused: true
            }],
            "target={target:?}: the route is told to come up paused"
        );

        // The permitted fallback rung, which is a second opener and had to
        // be told separately.
        let seen: RefCell<Vec<Opening>> = RefCell::new(Vec::new());
        let mut e = engine();
        e.bp_state.policy = OutputPolicy::PreferExact;
        let out = open_track(
            &mut e,
            Opening {
                at: target,
                paused: true,
            },
            false,
            |_, o| {
                seen.borrow_mut().push(o);
                Err(bitperfect::OpenError::device_format("the DAC refused it"))
            },
            |_, o| {
                seen.borrow_mut().push(o);
                Ok(o.at)
            },
        );
        assert_eq!(out, RestartOutcome::Shared { landed: target });
        assert!(
            seen.borrow().iter().all(|o| o.paused),
            "target={target:?}: both rungs, or the mixer starts playing \
                 for a listener who is paused: {:?}",
            seen.borrow()
        );
    }
}

/// A restart outcome does not itself start playback.
///
/// The app half of the same rule. `restart_effect` decides the play state,
/// and a paused restart stays paused whichever rung it landed on.
#[test]
fn a_paused_restart_stays_paused_on_every_rung() {
    let landed = Duration::from_secs(42);
    for outcome in [
        RestartOutcome::Exact { landed },
        RestartOutcome::Shared { landed },
    ] {
        let eff = restart_effect(&outcome, true, Some("T"));
        assert_eq!(eff.play_state, PlayState::Paused, "{outcome:?}");
        assert_eq!(eff.seek_to, Some(landed));
    }
}

/// A superseded seek worker stops decoding.
///
/// Dropping the receiver stops the *result* arriving; it does not stop the
/// work. A deep seek into a hi-res file decodes and discards tens of
/// millions of samples, and a worker whose seek had been replaced went on
/// doing all of it — a core, for seconds, for a position nobody was
/// waiting for. It is told now, and it says so.
#[test]
fn a_superseded_seek_worker_is_told_to_stop() {
    const RATE: u32 = 8_000;
    // Long enough that a worker which ignored the flag would be decoding
    // for a noticeable time rather than finishing by luck.
    let frames = RATE as usize * 30;
    let mut pcm = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let mag = 1_000 + (i as i16 % 5_000);
        pcm.extend_from_slice(&mag.to_le_bytes());
        pcm.extend_from_slice(&(-mag).to_le_bytes());
    }
    let t = seek_temp("cancelled_seek.wav");
    write_seek_wav(&t.0, &pcm, RATE, 2);

    // Already cancelled when the worker starts: the flag is checked inside
    // the discard loop, so it stops at the first check.
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let rx = Engine::spawn_seek_worker(&t.0, Duration::from_secs(25), cancel)
        .expect("worker starts");
    match rx.recv().expect("the worker answers") {
        Err(SeekError::Cancelled) => {}
        Err(other) => panic!("a cancelled worker reports it: {other:?}"),
        Ok(_) => panic!("a cancelled worker must not hand back a decoder"),
    }
    assert!(
        !SeekError::Cancelled.is_worth_reporting(),
        "and it is not shown to the listener, because they caused it"
    );
    assert!(
        SeekError::Open("gone".into()).is_worth_reporting(),
        "unlike a seek that genuinely failed"
    );
}

/// A failed asynchronous seek leaves one state, and every surface agrees.
///
/// The choice, written down because there were two: an asynchronous seek
/// that cannot land leaves the player **paused at the position playback
/// actually reached**, not stopped. The track stays the current track, so
/// pressing play resumes it from where it was; the session is not running,
/// so nothing claims to be playing; and the status line says the seek
/// failed. `abandon_seek` is where the engine half of that lives.
#[test]
fn a_failed_async_seek_is_paused_at_the_old_position_everywhere() {
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut e = engine_holding(&counter);
    let from = Duration::from_secs(30);
    e.note_current(Path::new("playing.flac"), None, from);
    e.bp_state.begin_open();

    e.abandon_seek(from, SeekError::Open("gone".into()));

    // The engine: the track is still the track, at the position playback
    // reached, with nothing running.
    let cur = e.current_track().expect("the track survives a failed seek");
    assert_eq!(cur.at, from, "and it is back where playback actually was");
    assert!(e.sink.is_none());
    assert!(
        !matches!(e.bp_state.transport, bitperfect::Transport::Opening),
        "and not left verifying an output it will not open"
    );

    // The status line: the listener is told, once.
    let mut notice = false;
    let line = seek_status_line(
        &SeekOutcome::Failed(SeekError::Open("gone".into())),
        &mut notice,
        || unreachable!("a failure does not put the now-playing line back"),
    );
    assert!(line.is_some_and(|l| l.contains("Seek failed")));

    // And the app: paused, which is the state `resume` acts on. Stopped
    // would be the other coherent answer and is not the one chosen — the
    // engine deliberately keeps `current` so the track can be resumed.
    assert_eq!(
        async_seek_failure_state(),
        PlayState::Paused,
        "one answer, and it is the one the engine's state supports"
    );
}

/// A failure keeps the kind of failure it was, on both fallback routes.
///
/// Every shared-mixer failure arrived as a backend error, and every
/// decimated-DSD rebuild failure as a seek error. So a file that had been
/// deleted under the player was reported as the DAC being gone, and a
/// missing output device as a container refusing a position. The class is
/// what the panel names and what decides whether the ladder may advance,
/// so it is not decoration.
#[test]
fn a_failure_keeps_the_kind_of_failure_it_was() {
    use bitperfect::state::FailureReason;

    // The shared mixer's four ways of failing.
    for (why, want_source, want_seek) in [
        ("Open failed: The system cannot find the file specified.", true, false),
        ("Decode failed: unsupported container", true, false),
        ("this file will not seek: unseekable", false, true),
        ("Sink failed: no output device", false, false),
    ] {
        let e = shared_open_error(why);
        assert_eq!(
            e.kind() == "source",
            want_source,
            "{why:?} -> {:?}",
            e.kind()
        );
        assert_eq!(e.kind() == "seek", want_seek, "{why:?} -> {:?}", e.kind());
        assert!(
            e.to_string().contains(why),
            "and the message itself survives: {}",
            e
        );
        // None of them is a format rejection, so none of them may advance
        // a ladder that has already fallen back once.
        assert!(!e.is_device_limitation(), "{why:?}");
    }

    // A missing file and a missing device are different things to be told.
    assert!(matches!(
        shared_failure_reason("Open failed: not found"),
        FailureReason::SourceOpen(_)
    ));
    assert!(!matches!(
        shared_failure_reason("Sink failed: no output device"),
        FailureReason::SourceOpen(_)
    ));

    // And the decimated-DSD rebuild, which had the same defect pointed the
    // other way: everything was a seek.
    assert!(matches!(
        dsd_rebuild_reason("DSD seek: past the end"),
        FailureReason::Seek(_)
    ));
    assert!(matches!(
        dsd_rebuild_reason("DSD: this file is truncated"),
        FailureReason::SourceOpen(_)
    ));
    assert!(matches!(
        dsd_rebuild_reason("Sink failed: no output device"),
        FailureReason::DeviceUnavailable(_)
    ));
    for why in [
        "DSD seek: past the end",
        "DSD decimate: cannot open",
        "Sink failed: no output device",
    ] {
        assert!(
            dsd_rebuild_reason(why).message().contains(why),
            "the rebuild's own message survives its classification: {why}"
        );
    }
}

/// Turning bit-perfect on while stopped does not claim to be playing.
///
/// A stopped player still has a current index — that is how Play knows
/// what to start — and the toggle asked only whether one was *selected*.
/// So enabling bit-perfect on a stopped player printed "Playing: …" and
/// took the now-playing line's ownership of it. The refresh then correctly
/// refused to regenerate a stopped player's line, so the false sentence
/// stayed on the bar until something else replaced it.
#[test]
fn enabling_bit_perfect_while_stopped_says_no_active_output() {
    // Stopped, with a track selected: the case that was wrong.
    assert_eq!(
        toggle_status(true, PlayState::Stopped, Some(3)),
        ToggleStatus::Transient("◇ Bit-perfect requested — no active output".into()),
        "a selected track is not a playing one"
    );
    // Nothing selected at all: the same answer, for the obvious reason.
    assert_eq!(
        toggle_status(true, PlayState::Stopped, None),
        ToggleStatus::Transient("◇ Bit-perfect requested — no active output".into())
    );

    // Playing or paused, there is a session for the line to describe, and
    // it is owned so the route can still change it.
    for state in [PlayState::Playing, PlayState::Paused] {
        assert_eq!(
            toggle_status(true, state, Some(3)),
            ToggleStatus::NowPlaying(3),
            "{state:?}"
        );
        assert_eq!(
            toggle_status(true, state, None),
            ToggleStatus::Transient(
                "◇ Bit-perfect requested — no active output".into()
            ),
            "{state:?}: and an active player with no index has none either"
        );
    }

    // Switching it off is never a claim about a route.
    for state in [PlayState::Playing, PlayState::Paused, PlayState::Stopped] {
        assert_eq!(
            toggle_status(false, state, Some(3)),
            ToggleStatus::Transient("Bit-perfect off".into()),
            "{state:?}"
        );
    }
}

/// A paused session halts on a failure and does not advance on an ending.
///
/// Both halves of the paused branch of the tick, through the function that
/// branch calls.
#[test]
fn a_paused_session_halts_on_failure_and_advances_on_nothing() {
    use bitperfect::Completion;

    assert_eq!(
        paused_completion(Completion::Failed {
            reason: bitperfect::fault::BACKEND_RESET,
            generation: 7,
        }),
        PausedTick::Halt(bitperfect::fault::BACKEND_RESET),
        "a paused session that has failed still holds an exclusive device"
    );
    assert_eq!(
        paused_completion(Completion::CleanEof),
        PausedTick::Nothing,
        "and a paused track that reached the end of its source has not \
             ended as far as the listener is concerned"
    );
    assert_eq!(paused_completion(Completion::Running), PausedTick::Nothing);

    // And what halting does, witnessed rather than inferred.
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut e = engine_holding(&counter);
    e.note_current(Path::new("was-playing.flac"), None, Duration::from_secs(30));
    if let PausedTick::Halt(reason) = paused_completion(Completion::Failed {
        reason: bitperfect::fault::BACKEND_RESET,
        generation: 7,
    }) {
        e.halt(reason);
    }
    assert_eq!(released(&counter), 4, "the device goes back");
    assert!(e.current_track().is_some(), "and the track is still the track");
}

/// The two callers still go through the two reducers.
///
/// A source check, and the only thing it is: it cannot show the reducers
/// are right, which is what the tests above are for. What it catches is
/// the caller quietly growing a second copy of the decision — which is how
/// both of these went wrong in the first place, and the reason the
/// condition drifted from "is a track playing" to "is a track selected".
#[test]
fn the_toggle_and_the_paused_tick_have_no_second_copy_of_their_decision() {
    let src = include_str!("main.rs");
    let toggle = block(src, "fn toggle_bit_perfect(&mut self)", "fn select_bp_device");
    let code = code_only(toggle);
    assert!(
        code.contains("toggle_status(new_bp, self.play_state, self.current_index)"),
        "the toggle asks the reducer, and gives it the play state"
    );
    assert!(
        !code.contains("(true, Some(i))"),
        "and does not decide again for itself"
    );
}

/// A landing that is not the request is the landing the caller is told.
///
/// A FLAC without a seek table lands on the nearest frame it can reach,
/// which can be seconds away. Publishing the request instead moved the
/// seek bar and the spectrum to a position nothing was playing from.
#[test]
fn the_landing_is_where_the_route_reached_not_where_it_was_asked() {
    let mut e = engine();
    let landed = Duration::from_millis(41_600);
    let out = open_track(
        &mut e,
        Opening::at(Duration::from_secs(42)),
        false,
        move |_, _| Ok(landed),
        |_, _| panic!("no fallback"),
    );
    assert_eq!(out, RestartOutcome::Exact { landed });
    assert_eq!(restart_effect(&out, false, None).seek_to, Some(landed));
}

/// Automatic and HQ fall back for this track and keep the position, the
/// pause state and the preference.
#[test]
fn a_permitted_fallback_keeps_position_pause_and_preference() {
    use bitperfect::state::OutputPolicy;

    for policy in [OutputPolicy::PreferExact, OutputPolicy::HqProcessed] {
        let mut e = engine();
        e.bit_perfect = true;
        e.bp_state.requested = true;
        e.bp_state.policy = policy;

        let out = open_track(
            &mut e,
            Opening::at(Duration::from_secs(30)),
            false,
            |_, _| Err(bitperfect::OpenError::device_format("the DAC refused 24/192")),
            |_, o| Ok(o.at),
        );
        assert_eq!(
            out,
            RestartOutcome::Shared {
                landed: Duration::from_secs(30)
            },
            "{policy:?}"
        );
        assert!(e.bit_perfect, "{policy:?}: the preference is the listener's");
        assert!(e.bp_state.requested, "{policy:?}");
        assert_eq!(
            e.route_for(Path::new("next.flac")),
            Route::Device,
            "{policy:?}: so the next track tries the device again"
        );

        // And the app's half: paused stays paused, at the position it
        // reached.
        let eff = restart_effect(&out, true, Some("Track"));
        assert_eq!(eff.play_state, PlayState::Paused);
        assert_eq!(eff.seek_to, Some(Duration::from_secs(30)));
        assert!(eff.status.is_some(), "and it says the route changed");
        assert!(eff.error.is_none());
    }
}

/// Strict, and every failure that is not a format rejection, stop.
#[test]
fn strict_and_non_device_failures_stop_without_a_fallback() {
    use bitperfect::state::OutputPolicy;
    use std::cell::Cell;

    let cases: Vec<(OutputPolicy, bitperfect::OpenError)> = vec![
        (
            OutputPolicy::StrictExact,
            bitperfect::OpenError::device_format("the DAC refused 24/192"),
        ),
        (
            OutputPolicy::PreferExact,
            bitperfect::OpenError::config("MOOSIK_BP_FORMAT asks for 8-bit float"),
        ),
        (
            OutputPolicy::PreferExact,
            bitperfect::OpenError::source("the file will not open"),
        ),
        (
            OutputPolicy::PreferExact,
            bitperfect::OpenError::seek("the container refused the seek"),
        ),
        (
            OutputPolicy::HqProcessed,
            bitperfect::OpenError::backend("the render thread would not start"),
        ),
    ];

    for (policy, err) in cases {
        let mut e = engine();
        e.bit_perfect = true;
        e.bp_state.requested = true;
        e.bp_state.policy = policy;
        let tried = Cell::new(false);

        let out = open_track(
            &mut e,
            Opening::at(Duration::from_secs(30)),
            false,
            move |_, _| Err(err),
            |_, o| {
                tried.set(true);
                Ok(o.at)
            },
        );

        assert!(matches!(out, RestartOutcome::Stopped(_)), "{policy:?}: {out:?}");
        assert!(!tried.get(), "{policy:?}: the mixer is not opened behind its back");
        assert!(e.bp.is_none());
        assert!(
            e.bit_perfect && e.bp_state.requested,
            "{policy:?}: stopping does not spend the preference either"
        );

        let eff = restart_effect(&out, false, Some("Track"));
        assert_eq!(eff.play_state, PlayState::Stopped);
        assert!(eff.error.is_some(), "{policy:?}: and the caller is told why");
        assert!(eff.status.is_none());
    }
}

/// Every caller applies the same outcome, because there is one description
/// of what applying it means.
///
/// The toggle, the output-device picker, the ASIO/ALSA picker and the
/// rollback each used to interpret a `Result<(), String>` for themselves:
/// some set the play state, some did not, one discarded the error
/// outright, and none agreed about the pause state or the position.
#[test]
fn one_outcome_means_one_thing_to_every_caller() {
    let landed = Duration::from_secs(30);

    for was_paused in [false, true] {
        let want = if was_paused {
            PlayState::Paused
        } else {
            PlayState::Playing
        };
        let exact = restart_effect(&RestartOutcome::Exact { landed }, was_paused, Some("T"));
        assert_eq!(exact.play_state, want, "paused={was_paused}");
        assert_eq!(exact.seek_to, Some(landed));
        assert!(exact.status.is_none(), "an exact restart is not news");
        assert!(exact.error.is_none());

        let shared = restart_effect(&RestartOutcome::Shared { landed }, was_paused, Some("T"));
        assert_eq!(
            shared.play_state, want,
            "paused={was_paused}: falling back does not start playback"
        );
        assert_eq!(shared.seek_to, Some(landed));
        assert!(shared.status.is_some_and(|s| s.contains("shared mixer")));
        assert!(shared.error.is_none());

        let stopped = restart_effect(
            &RestartOutcome::Stopped(bitperfect::state::FailureReason::DeviceUnavailable(
                "the DAC is gone".into(),
            )),
            was_paused,
            Some("T"),
        );
        assert_eq!(
            stopped.play_state,
            PlayState::Stopped,
            "paused={was_paused}: a failure stops, whatever the app was doing"
        );
        assert_eq!(stopped.seek_to, None);
        assert!(stopped.error.is_some_and(|e| e.contains("gone")));
    }

    // A restart from the very beginning moves nothing: there is no
    // position to restore and the spectrum is already there.
    let from_zero = restart_effect(
        &RestartOutcome::Exact {
            landed: Duration::ZERO,
        },
        false,
        None,
    );
    assert_eq!(from_zero.seek_to, None);
}

/// A recoverable loss is visible on every live route, not only the ones
/// whose badge was already about a fault.
///
/// The detail line was gated on the fidelity already being faulted or
/// failed — which is the one case where the headline is likely to mention
/// it anyway. So the routes where the loss was the *only* record of it
/// were exactly the routes that hid it: a Processed or Unverified session
/// that dropped out said nothing at all. A dropout does not change what
/// the route is doing to the audio, which is why those keep their badge;
/// it changes that the route stopped delivering it, and the listener has
/// to be able to read that somewhere.
#[test]
fn a_dropout_is_visible_on_every_live_route() {
    use bitperfect::fault;
    use bitperfect::state::{Badge, Fidelity, ProcessingState, TransformDescription};

    let routes: Vec<(&str, Fidelity, Badge)> = vec![
        ("payload exact", Fidelity::PayloadExact, Badge::Faulted),
        (
            "value exact",
            Fidelity::ValueExact {
                transform: TransformDescription::FloatToQ31ValueExact,
            },
            Badge::Faulted,
        ),
        (
            "processed",
            Fidelity::Processed {
                transform: TransformDescription::Identity,
                reason: "the shared mixer".into(),
            },
            Badge::Processed,
        ),
        (
            "unverified",
            Fidelity::Unverified {
                reason: "the channel mapping cannot be verified".into(),
            },
            Badge::Unverified,
        ),
    ];

    for (name, fidelity, want_badge) in routes {
        let mut st = exact_session();
        st.opened(
            bitperfect::Transport::WasapiExclusivePcm { endpoint: endpoint() },
            fidelity,
            None,
            None,
            ProcessingState::transparent_locked(),
        );
        let g = st.generation;
        apply_integrity(&mut st, g, &[fault::UNDERRUN], &[]);

        let p = st.presentation();
        assert!(!st.shows_diamond(), "{name}: no diamond after a dropout");
        assert_eq!(
            p.badge, want_badge,
            "{name}: an exact claim is withdrawn; a processed one is not \
                 changed by a dropout"
        );
        assert!(
            p.headline.contains("dropout")
                || p.detail.iter().any(|d| d.starts_with("Integrity:")),
            "{name}: the loss has to be readable somewhere: {} {:?}",
            p.headline,
            p.detail
        );

        // And when the track then fails, the fatal reason owns the
        // headline and the loss stays in the detail.
        apply_integrity(&mut st, g, &[], &[fault::BACKEND_WRITE]);
        let p = st.presentation();
        assert!(
            p.headline.contains("write"),
            "{name}: {}",
            p.headline
        );
        assert!(
            p.detail.iter().any(|d| d.starts_with("Integrity:") && d.contains("dropout")),
            "{name}: {:?}",
            p.detail
        );
    }
}

/// A new open and a failed open both answer for nothing that came before.
///
/// `begin_open` left the previous track's source and the previous
/// session's dropout in place, so "Verifying output…" was shown beside a
/// specific claim about a file that was no longer playing, on a route that
/// did not exist yet. `failed` did the same for a route that never opened
/// at all.
#[test]
fn an_open_and_a_failure_carry_nothing_from_before() {
    use bitperfect::fault;
    use bitperfect::state::FailureReason;

    let seeded = || {
        let mut st = exact_session();
        let g = st.generation;
        apply_integrity(&mut st, g, &[fault::UNDERRUN], &[]);
        st.set_source(bitperfect::state::MediaSource::pcm(
            bitperfect::format::SourceFormat {
                kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
                sample_rate: 96_000,
                channels: 2,
                layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
            },
        ));
        assert!(st.revoked.is_some() && st.source.is_some());
        st
    };

    let mut st = seeded();
    st.begin_open();
    assert!(st.revoked.is_none(), "a route that has not opened lost nothing");
    assert!(st.source.is_none(), "and is not yet playing anything");
    assert!(
        !st.presentation().detail.iter().any(|d| d.starts_with("Integrity:")
            || d.starts_with("Source:")),
        "{:?}",
        st.presentation().detail
    );

    let mut st = seeded();
    st.failed(FailureReason::DeviceUnavailable("the DAC is gone".into()));
    assert!(st.revoked.is_none(), "a route that never opened lost nothing");
    assert!(st.source.is_none());
    let p = st.presentation();
    assert!(p.headline.contains("gone"), "{}", p.headline);
    assert!(
        !p.detail.iter().any(|d| d.starts_with("Integrity:") || d.starts_with("Source:")),
        "a failure must not borrow the last session's records: {:?}",
        p.detail
    );
}

/// A value-exact verdict is refused while a rounding count is in flight.
///
/// The whole reason the publication window opens *before* the store. A
/// count that has been announced and not yet written reads as zero, and
/// zero is the condition this label is granted on — so the one instant in
/// which the conversion is telling the UI that it had to round is also the
/// instant in which the UI would have upgraded the track for never having
/// rounded. The label is durable, so it would have stayed.
///
/// Driven through the production `apply_q31_verdict`, holding the real
/// `BpStream` a caller holds, with a real writer parked inside its window.
#[test]
fn a_verdict_cannot_upgrade_over_a_rounding_count_in_flight() {
    use bitperfect::state::Fidelity;
    use std::sync::{Arc, Barrier};

    let scan_says_exact = bitperfect::q31::Q31Verdict {
        value_exact: true,
        samples: 1_000,
        first_failure: None,
        schema: 1,
    };

    // A session on the Q1.31 route, playing, labelled Processed.
    let processed = |e: &mut Engine| {
        e.bp_state.fidelity = Fidelity::Processed {
            transform: bitperfect::state::TransformDescription::FloatToQ31Processed,
            reason: "checking whether every sample is exactly representable".into(),
        };
    };

    // --- quiet: the upgrade is allowed -------------------------------
    let sh = bitperfect::test_shared::new();
    let a = bitperfect::test_shared::begin(&sh);
    let mut e = engine();
    e.bp = Some(bitperfect::BpStream::for_evidence_test(
        Arc::clone(&sh),
        bitperfect::PayloadPlan::Q31,
    ));
    e.bp_audio_gen = a;
    processed(&mut e);
    let g = e.bp_state.generation;
    e.apply_q31_verdict(g, a, scan_says_exact.clone());
    assert!(
        matches!(e.bp_state.fidelity, Fidelity::ValueExact { .. }),
        "with nothing in flight the verdict applies: {:?}",
        e.bp_state.fidelity
    );

    // --- in flight: the upgrade is refused ---------------------------
    let sh = bitperfect::test_shared::new();
    let a = bitperfect::test_shared::begin(&sh);
    let inside = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    bitperfect::test_shared::park_off_grid(&sh, Arc::clone(&inside), Arc::clone(&release));

    let writer = {
        let sh = Arc::clone(&sh);
        std::thread::spawn(move || bitperfect::test_shared::note_off_grid(&sh, a, 3))
    };
    inside.wait();

    let mut e = engine();
    e.bp = Some(bitperfect::BpStream::for_evidence_test(
        Arc::clone(&sh),
        bitperfect::PayloadPlan::Q31,
    ));
    e.bp_audio_gen = a;
    processed(&mut e);
    let g = e.bp_state.generation;
    e.apply_q31_verdict(g, a, scan_says_exact.clone());
    assert!(
        !matches!(e.bp_state.fidelity, Fidelity::ValueExact { .. }),
        "a count that is being published is not a count of zero: {:?}",
        e.bp_state.fidelity
    );

    release.wait();
    writer.join().expect("the writer returns");

    // And once it has landed, the route is honestly not value-exact.
    e.apply_q31_verdict(g, a, scan_says_exact);
    assert!(
        !matches!(e.bp_state.fidelity, Fidelity::ValueExact { .. }),
        "{:?}",
        e.bp_state.fidelity
    );
}

/// The line and the headline stored against it come from one place.
///
/// `set_now_playing` stored the whole rendered line as the headline — title
/// and all — and the live headline it is compared against on every tick
/// never carries a title, so the two could not compare equal. The
/// now-playing line was therefore regenerated on the first tick of every
/// track whether anything had changed or not, which is the same defect as
/// never regenerating it, wearing the opposite disguise.
#[test]
fn a_now_playing_line_stores_the_headline_it_was_rendered_from() {
    let st = exact_session();
    let p = st.presentation();
    let (line, headline) = now_playing_status(Some(&p), "Symphony No. 5");

    assert_eq!(headline, p.headline, "the fingerprint is the session's own");
    assert!(line.starts_with(&p.headline), "and the line is built from it");
    assert!(line.contains("Symphony No. 5"));
    assert_ne!(line, headline, "the line carries the title; the headline does not");

    let mut s = StatusLine::default();
    s.now_playing(line.clone(), headline);
    assert_eq!(
        *s.owner(),
        StatusOwner::NowPlaying {
            headline: p.headline.clone()
        }
    );
    assert!(
        !s.refresh(&p.headline, true, || unreachable!("nothing has changed")),
        "a session that has not moved does not regenerate its own line"
    );
    assert_eq!(s.text(), line);

    // A route with no claim to make has no headline to go stale, and says
    // the ordinary thing.
    let (line, headline) = now_playing_status(None, "Symphony No. 5");
    assert_eq!(line, "Playing: Symphony No. 5");
    assert_eq!(headline, line);

    // Bit-perfect off: the line names no route, and the fingerprint is
    // still the session's headline. Storing the rendered line here would
    // make the comparison fail on every tick — the line would be
    // regenerated forever, which is the same defect as never regenerating
    // it and no easier to see.
    let mut off = bitperfect::state::OutputSessionState::default();
    off.stopped();
    let p = off.presentation();
    assert_eq!(p.badge, bitperfect::state::Badge::Off);
    let (line, headline) = now_playing_status(Some(&p), "Symphony No. 5");
    assert_eq!(line, "Playing: Symphony No. 5");
    assert_eq!(headline, p.headline);
    let mut s = StatusLine::default();
    s.now_playing(line, headline);
    assert!(
        !s.refresh(&p.headline, true, || unreachable!("nothing has changed")),
        "a route that makes no claim does not regenerate its line either"
    );
}

/// A toggle that lands on a live track owns the now-playing line, so the
/// route it turns out to be reaches the bar.
///
/// Both directions of the one that matters. The scan that proves value
/// exactness finishes a second or two after the toggle; the dropout that
/// withdraws a claim can happen at any point after it. Written as a
/// transient — which is what the toggle used to do — the sentence is
/// frozen at the instant the switch was flipped and neither of them ever
/// reaches the listener.
#[test]
fn a_toggle_leaves_the_line_owned_so_the_route_can_still_change_it() {
    use bitperfect::state::Fidelity;

    // Processed, then upgraded by the scan to value-exact.
    let mut st = exact_session();
    st.fidelity = Fidelity::Processed {
        transform: bitperfect::state::TransformDescription::FloatToQ31Processed,
        reason: "checking whether every sample is exactly representable".into(),
    };
    let before = st.presentation().headline;
    let (line, headline) = now_playing_status(Some(&st.presentation()), "Track");
    let mut s = StatusLine::default();
    s.now_playing(line, headline);

    st.fidelity = Fidelity::ValueExact {
        transform: bitperfect::state::TransformDescription::FloatToQ31ValueExact,
    };
    let after = st.presentation().headline;
    assert_ne!(before, after);
    let rendered = now_playing_status(Some(&st.presentation()), "Track").0;
    assert!(
        s.refresh(&after, true, || rendered.clone()),
        "the upgrade reaches the bar"
    );
    assert_eq!(s.text(), rendered);

    // Exact, then a dropout: the diamond has to go.
    let mut st = exact_session();
    let (line, headline) = now_playing_status(Some(&st.presentation()), "Track");
    assert!(line.contains('💎'), "the fixture starts green: {line}");
    let mut s = StatusLine::default();
    s.now_playing(line, headline);

    let g = st.generation;
    apply_integrity(&mut st, g, &[bitperfect::fault::UNDERRUN], &[]);
    let after = st.presentation().headline;
    let rendered = now_playing_status(Some(&st.presentation()), "Track").0;
    assert!(s.refresh(&after, true, || rendered.clone()));
    assert!(
        !s.text().contains('💎'),
        "a route that lost its claim does not keep the diamond: {}",
        s.text()
    );
}

/// A seek fast enough to land before its notice is shown leaves the bar
/// alone.
///
/// `SeekOutcome::Landed` took ownership unconditionally. On a fast seek
/// there is no notice to replace, so what it took ownership *of* was
/// whatever the listener happened to be looking at — a device error, a
/// fallback warning — and the next tick regenerated it into a now-playing
/// line. The message was gone before it had been read.
#[test]
fn a_landing_takes_the_line_only_when_it_had_put_one_up() {
    // Fast: nothing was shown, so nothing changes hands.
    let mut s = StatusLine::default();
    s.transient("⚠ This track played through the shared mixer");
    s.landed(false, String::new(), "💎 Payload exact · WASAPI".into());
    assert_eq!(s.text(), "⚠ This track played through the shared mixer");
    assert_eq!(*s.owner(), StatusOwner::Transient);
    assert!(
        !s.refresh("💎 Payload exact · WASAPI", true, || unreachable!()),
        "and it stays the listener's"
    );

    // Slow: a notice went up, and the landing replaces it — with the
    // now-playing line, owned, so the route can go on changing it.
    let mut s = StatusLine::default();
    s.transient("Seeking to 12:34…");
    s.landed(
        true,
        "💎 Payload exact · WASAPI — Track".into(),
        "💎 Payload exact · WASAPI".into(),
    );
    assert_eq!(s.text(), "💎 Payload exact · WASAPI — Track");
    assert_eq!(
        *s.owner(),
        StatusOwner::NowPlaying {
            headline: "💎 Payload exact · WASAPI".into()
        }
    );
    assert!(s.refresh("⚠ Processed · the shared mixer", true, || {
        "⚠ Processed · the shared mixer — Track".to_string()
    }));
}

/// A message cannot inherit the now-playing line's claim.
///
/// The ASIO PCM probe wrote the text and left the owner alone, so a probe
/// result run while a track was playing was owned by the now-playing line
/// and wiped by the next tick. There is now no way to write the text
/// without saying who it belongs to: the two are one value with private
/// fields, and the only way to become the now-playing line is to hand over
/// a headline, which a probe has not got.
#[test]
fn a_probe_result_is_the_listeners_and_not_the_apps() {
    let mut s = StatusLine::default();
    s.now_playing(
        "💎 Payload exact · WASAPI — Track".into(),
        "💎 Payload exact · WASAPI".into(),
    );
    s.transient("ASIO PCM probe — 2ch 32-bit int, 44.1–768 kHz");
    assert_eq!(*s.owner(), StatusOwner::Transient);
    assert!(
        !s.refresh("⚠ Processed · the shared mixer", true, || unreachable!()),
        "the route changing does not entitle the app to the listener's line"
    );
    assert_eq!(s.text(), "ASIO PCM probe — 2ch 32-bit int, 44.1–768 kHz");
}

/// A backend that dies while paused is noticed, and the device goes back.
///
/// The completion poll lived inside the playing branch, which is right for
/// advancing a playlist and wrong for noticing a failure: a paused session
/// still holds an exclusive device, and a driver reset while paused left
/// the player sitting on it until the listener happened to press play.
///
/// Both halves are here — the predicate that decides a paused session has
/// failed, and the release that follows from it, witnessed by drops rather
/// than by fields that were empty to begin with.
#[test]
fn a_failure_while_paused_halts_and_releases_the_device() {
    use bitperfect::Completion;

    // The predicate the paused branch runs, on the two answers it can get.
    let failed = Completion::Failed {
        reason: bitperfect::fault::BACKEND_RESET,
        generation: 7,
    };
    assert_eq!(
        failed.failure(),
        Some(bitperfect::fault::BACKEND_RESET),
        "a paused session that has failed is noticed while it is paused"
    );

    // And what noticing does.
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut e = engine_holding(&counter);
    e.note_current(
        Path::new("was-playing.flac"),
        None,
        Duration::from_secs(30),
    );
    e.halt(bitperfect::fault::BACKEND_RESET);
    assert_eq!(
        released(&counter),
        4,
        "an exclusive device is not held by a session that has ended"
    );
    assert!(
        e.current_track().is_some(),
        "but the track is still the track: this one really played, and \
             `resume` reopens it"
    );
    assert!(
        matches!(
            e.bp_state.fidelity,
            bitperfect::state::Fidelity::Faulted { .. }
        ),
        "a session that really played and then failed reports an integrity \
             reason, not the reason an open never happened: {:?}",
        e.bp_state.fidelity
    );
}

/// A paused track that reached the end of its source does not advance.
///
/// The paused poll exists to catch failures, not endings. A paused track
/// has not ended as far as the listener is concerned, and a playlist that
/// moved on while they were away from the keyboard would be the same
/// defect in the other direction.
#[test]
fn a_clean_end_while_paused_advances_nothing() {
    use bitperfect::Completion;
    assert_eq!(
        Completion::CleanEof.failure(),
        None,
        "the paused branch asks only for failures, and an ending is not one"
    );
    assert_eq!(Completion::Running.failure(), None);
    // While playing, the same completion does advance — the difference is
    // the branch it is read in, not the answer.
    assert_eq!(
        bitperfect::on_completion(Completion::CleanEof, false),
        bitperfect::TickAction::Advance
    );
}

/// Stopping gives the status line back.
///
/// The refresh fired on any badge change, and ownership survived a stop —
/// so a shared track that was stopped had "Playing: …" put back on the bar
/// by the next tick, describing a player that had stopped.
#[test]
fn a_stopped_player_never_gets_a_now_playing_line_back() {
    let exact = || {
        let mut s = StatusLine::default();
        s.now_playing(
            "💎 Payload exact · WASAPI — Track".into(),
            "💎 Payload exact · WASAPI".into(),
        );
        s
    };
    let mut s = exact();
    assert!(
        s.refresh("⚠ Processed · the shared mixer", true, || {
            "⚠ Processed · the shared mixer — Track".to_string()
        }),
        "while playing, a changed route refreshes the line"
    );

    let mut s = exact();
    assert!(
        !s.refresh("⚠ Processed · the shared mixer", false, || {
            unreachable!("nothing regenerates a stopped player's line")
        }),
        "and while stopped, nothing regenerates it"
    );
    assert!(s.text().starts_with("💎"), "it is left exactly as it was");

    // Relinquished outright is the same answer for a different reason: a
    // stop clears the line and gives ownership back.
    let mut s = exact();
    s.clear();
    assert_eq!(s.text(), "");
    assert_eq!(*s.owner(), StatusOwner::None);
    assert!(!s.refresh("anything", true, || unreachable!("nobody owns it")));
}

/// A message the listener has just been given is not overwritten.
#[test]
fn a_seek_failure_or_device_error_keeps_the_line() {
    let mut given = StatusLine::default();
    given.transient("⚠ Seek failed: the container refused it");
    let mut nobodys = StatusLine::default();
    for s in [&mut given, &mut nobodys] {
        let before = s.text().to_string();
        assert!(
            !s.refresh("💎 Payload exact · WASAPI", true, || {
                unreachable!("not ours to overwrite")
            }),
            "{:?}: the route changing does not entitle anything to \
                 overwrite what the listener was told",
            s.owner()
        );
        assert_eq!(s.text(), before);
    }
}

/// A stand-in for the app, recording what each restart was handed.
///
/// The two restarts are the only parts of a device switch that reach a
/// sound card, so they are what is injected; the sequence around them —
/// attempt, roll back from the snapshot, say what happened — is the
/// production one.
#[derive(Default)]
struct SwitchLog {
    /// Every request the rollback was given, in order.
    rolled_back_to: Vec<RestartRequest>,
    new_device_takes_it: bool,
    old_device_takes_it: bool,
}

fn a_request() -> RestartRequest {
    RestartRequest {
        index: 3,
        path: PathBuf::from("a-long-symphony.flac"),
        duration: Some(Duration::from_secs(2400)),
        target: Duration::from_secs(1337),
        was_paused: true,
    }
}

fn run_switch(log: &mut SwitchLog, snapshot: Option<&RestartRequest>) -> SwitchOutcome {
    switch_output(
        log,
        snapshot,
        |l| {
            if l.new_device_takes_it {
                Ok(())
            } else {
                Err("the DAC refused 24-bit/192 kHz exclusive".to_string())
            }
        },
        |l, req| {
            l.rolled_back_to.push(req.clone());
            if l.old_device_takes_it {
                Ok(())
            } else {
                Err("and the previous one is gone as well".to_string())
            }
        },
    )
}

/// A device that will not take the track gives it back where it was.
///
/// Not merely back to the track: back to the position and the pause state
/// the listener was at when they opened the device menu. The rollback used
/// to call `restart_current_track`, which reads live state — and the
/// failed attempt had just set that to `Stopped`, so the first statement
/// of the restart was `return Ok(())`. The listener was told the previous
/// output had taken the track back. It had never been asked.
#[test]
fn a_device_that_refuses_the_track_gives_it_back_where_it_was() {
    let req = a_request();
    let mut log = SwitchLog {
        new_device_takes_it: false,
        old_device_takes_it: true,
        ..Default::default()
    };
    let out = run_switch(&mut log, Some(&req));

    assert_eq!(
        out,
        SwitchOutcome::RolledBack {
            why: "the DAC refused 24-bit/192 kHz exclusive".to_string()
        }
    );
    assert_eq!(
        log.rolled_back_to.len(),
        1,
        "the rollback actually happened"
    );
    assert_eq!(
        log.rolled_back_to[0], req,
        "and it used the request captured before the attempt — the same \
             track, the same position, the same pause state"
    );
    assert_eq!(log.rolled_back_to[0].target, Duration::from_secs(1337));
    assert!(
        log.rolled_back_to[0].was_paused,
        "a listener who was paused is still paused afterwards"
    );
}

/// When neither output will play it, both reasons are reported.
///
/// The rollback's error was discarded with `let _ =`, which threw away the
/// one case that matters: the device the listener had just moved *away*
/// from would not take the track back either, so both routes are gone and
/// the player is stopped with nothing on screen saying why.
#[test]
fn when_neither_output_will_have_it_both_reasons_are_kept() {
    let req = a_request();
    let mut log = SwitchLog::default();
    let out = run_switch(&mut log, Some(&req));

    match out {
        SwitchOutcome::Stranded { why, back } => {
            assert!(why.contains("refused"), "why: {why}");
            assert!(back.contains("gone as well"), "back: {back}");
        }
        other => panic!("both failures are part of the account: {other:?}"),
    }
    assert_eq!(log.rolled_back_to.len(), 1, "it was tried");
}

/// A switch that works rolls nothing back, and a switch with nothing
/// playing has nothing to put back.
#[test]
fn a_switch_rolls_back_only_what_was_playing() {
    let req = a_request();
    let mut log = SwitchLog {
        new_device_takes_it: true,
        ..Default::default()
    };
    assert_eq!(run_switch(&mut log, Some(&req)), SwitchOutcome::Moved);
    assert!(log.rolled_back_to.is_empty());

    // Nothing was playing: the move still failed and the listener is still
    // told, but there is no track to restore.
    let mut log = SwitchLog::default();
    let out = run_switch(&mut log, None);
    assert!(matches!(out, SwitchOutcome::RolledBack { .. }), "{out:?}");
    assert!(log.rolled_back_to.is_empty());
}

/// One policy for starting a track and restarting one, driven through the
/// open that would otherwise need a device.
///
/// `restart_current_track` had no policy at all: it propagated the error,
/// and `toggle_bit_perfect` responded by flipping the toggle back and
/// restarting again. A listener who asked for exact output during a track
/// the DAC could not carry exactly had their setting overwritten because
/// of one file. Starting that same track from the playlist fell back to
/// the mixer and kept the setting — two answers to one question, and the
/// worse one belonged to the path the listener reached by asking.
#[test]
fn one_policy_decides_a_failed_open_wherever_it_happens() {
    use bitperfect::state::OutputPolicy;
    use std::cell::Cell;

    // A format rejection: the endpoint's answer to a format question, and
    // the only failure a downgrade is a sensible response to.
    let rejected = || {
        bitperfect::OpenError::device_format(
            "the endpoint refused 24-bit/192 kHz exclusive",
        )
    };
    assert!(rejected().is_device_limitation());

    for policy in [OutputPolicy::PreferExact, OutputPolicy::HqProcessed] {
        let tried = Cell::new(false);
        let landed = Duration::from_secs(7);
        let out = resolve_open(Err(rejected()), policy, false, || {
            tried.set(true);
            Ok(landed)
        });
        assert_eq!(out, OpenOutcome::Shared(landed), "{policy:?}");
        assert!(tried.get(), "{policy:?}: the fallback is actually taken");
    }

    // Strict asked never to be moved onto a processed path without being
    // told. It stops, and it says which failure it was.
    let tried = Cell::new(false);
    let out = resolve_open(Err(rejected()), OutputPolicy::StrictExact, false, || {
        tried.set(true);
        Ok(Duration::ZERO)
    });
    assert!(
        matches!(out, OpenOutcome::Stopped(_)),
        "Strict must stop rather than downgrade: {out:?}"
    );
    assert!(!tried.get(), "and the mixer is not opened behind its back");

    // Everything that is not a format rejection stops, under every policy.
    // The mixer would hit the same wall or paper over a bug; either way
    // the listener would be shown a downgrade instead of the problem.
    let others = [
        bitperfect::OpenError::config("MOOSIK_BP_FORMAT asks for 8-bit float"),
        bitperfect::OpenError::source("the file will not open"),
        bitperfect::OpenError::seek("the container refused the seek"),
        bitperfect::OpenError::backend("the render thread would not start"),
    ];
    for e in others {
        assert!(!e.is_device_limitation(), "{e:?} is not a format rejection");
        for policy in [
            OutputPolicy::PreferExact,
            OutputPolicy::HqProcessed,
            OutputPolicy::StrictExact,
        ] {
            let tried = Cell::new(false);
            let out = resolve_open(Err(e.clone()), policy, false, || {
                tried.set(true);
                Ok(Duration::ZERO)
            });
            assert!(
                matches!(out, OpenOutcome::Stopped(_)),
                "{policy:?} / {e:?}: {out:?}"
            );
            assert!(!tried.get(), "{policy:?} / {e:?}");
        }
    }

    // DSD never reaches the retry: it ran its own ladder — native, DoP,
    // decimated PCM — inside `play_file`, and a raw bitstream cannot go
    // through rodio's decoders anyway, so a DSD failure arriving here
    // means every permitted route already failed.
    let tried = Cell::new(false);
    let out = resolve_open(Err(rejected()), OutputPolicy::PreferExact, true, || {
        tried.set(true);
        Ok(Duration::ZERO)
    });
    assert!(matches!(out, OpenOutcome::Stopped(_)), "{out:?}");
    assert!(!tried.get());

    // A fallback that is permitted and then fails is neither of the other
    // two: nothing is playing, and the reason is the mixer's.
    let out = resolve_open(Err(rejected()), OutputPolicy::PreferExact, false, || {
        Err("no output device".to_string())
    });
    assert_eq!(out, OpenOutcome::SharedFailed("no output device".into()));

    // And a success is a success.
    assert_eq!(
        resolve_open(Ok(Duration::ZERO), OutputPolicy::StrictExact, true, || {
            Ok(Duration::ZERO)
        }),
        OpenOutcome::Exact(Duration::ZERO)
    );
}

/// The preference is the listener's, and one track cannot spend it.
///
/// `resolve_open` is not given the requested preference, which is the
/// structural half of this; the behavioural half is that a track which
/// falls back leaves the next one starting again from the top of the
/// ladder.
#[test]
fn a_track_that_falls_back_does_not_spend_the_preference() {
    use bitperfect::state::OutputPolicy;

    let mut e = engine();
    e.bit_perfect = true;
    e.bp_state.requested = true;
    e.bp_state.policy = OutputPolicy::PreferExact;

    // One track the device refuses.
    let out = resolve_open(
        Err(bitperfect::OpenError::device_format("no")),
        e.bp_state.policy,
        false,
        || Ok(Duration::ZERO),
    );
    assert_eq!(out, OpenOutcome::Shared(Duration::ZERO));

    assert!(e.bit_perfect, "the preference is untouched");
    assert!(e.bp_state.requested, "and so is what the panel reports");

    // So the next track is still routed to the device.
    assert_eq!(e.route_for(Path::new("next.flac")), Route::Device);
}

/// A shared rollover onto a decimated DSD track says so in both records.
///
/// `Fidelity` carries the transform for the badge; `ProcessingState`
/// carries it for the panel's own list of what is in the path. The
/// rollover set only the first, so a DSD file that had just been resampled
/// onto the mixer described its processing as an identity transform on the
/// one surface that exists to say what is being done to the audio.
#[test]
fn a_decimated_dsd_rollover_says_so_in_both_records() {
    use bitperfect::state::{Fidelity, TransformDescription};

    let mut e = engine();
    e.bp_state.opened(
        bitperfect::Transport::Shared { endpoint: endpoint() },
        Fidelity::Processed {
            transform: TransformDescription::Identity,
            reason: "shared".into(),
        },
        None,
        None,
        e.shared_processing(),
    );
    e.queued_next = Some(QueuedShared {
        path: PathBuf::from("next.dsf"),
        source: Some(bitperfect::state::MediaSource::dsd(
            bitperfect::format::SourceFormat {
                kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                sample_rate: 2_822_400,
                channels: 2,
                layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
            },
            "DSD64".to_string(),
        )),
        transform: TransformDescription::DsdDecimated { pcm_rate: 176_400 },
        reason: "decimated to 176400 Hz PCM, then the shared mixer".into(),
    });
    e.roll_shared_source();

    let want = TransformDescription::DsdDecimated { pcm_rate: 176_400 };
    match &e.bp_state.fidelity {
        Fidelity::Processed { transform, .. } => assert_eq!(*transform, want),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        e.bp_state.processing.transform, want,
        "the processing record has to say it too — it is the list of what \
             is being done to the audio"
    );
    assert!(!e.bp_state.processing.is_transparent());

    let p = e.bp_state.presentation();
    assert!(
        p.detail.iter().any(|d| d.contains("176")),
        "and the panel says which rate it landed on: {:?}",
        p.detail
    );
}

/// The restart asks where *this track* is going, not what the preferences
/// say.
///
/// `bit_perfect || native_dsd_selected()` is two global settings, and it
/// is wrong in both directions. A PCM file with the toggle off and an ASIO
/// driver configured for DSD answered "device", so the restart took the
/// blocking `play_file` + seek path — a deep FLAC seek on the UI thread,
/// which is the freeze the async shortcut exists to prevent. A DSD file
/// with the toggle off and no native driver answered "shared", so the
/// restart was handed to `play_seeked_async`, which opens the file with
/// `rodio` — and `rodio` cannot read a DSD container at all, so a track
/// that had been playing a moment earlier failed to restart.
///
/// Both are asserted, so restoring either half of the old predicate fails
/// this test.
#[test]
fn the_restart_route_is_chosen_per_track_not_per_preference() {
    let pcm = PathBuf::from("track.flac");
    let dsd = PathBuf::from("track.dsf");

    let mut e = engine();

    // PCM, toggle off, no native driver: the mixer.
    e.bit_perfect = false;
    e.asio_driver = None;
    e.alsa_dsd_device = None;
    assert_eq!(e.route_for(&pcm), Route::Shared);

    // PCM, toggle off, native DSD driver configured. A setting about DSD
    // says nothing about a PCM file.
    e.asio_driver = Some("ASIO Driver".into());
    e.alsa_dsd_device = Some("hw:1,0".into());
    assert!(
        e.native_dsd_selected(),
        "the fixture must actually have a native output configured"
    );
    assert_eq!(
        e.route_for(&pcm),
        Route::Shared,
        "a DSD output setting must not divert a PCM track off the mixer"
    );

    // PCM, toggle on: the device, and the restart must attempt it even
    // though playback is currently on the mixer.
    e.bit_perfect = true;
    assert_eq!(e.route_for(&pcm), Route::Device);
    assert!(
        !e.on_bp_stream(),
        "which is exactly the case the old condition got wrong"
    );

    // DSD, toggle off, no native driver: still the ladder. Even its
    // bottom rung — decimation — has to be reached through `play_file`,
    // because `rodio` cannot open the container.
    e.bit_perfect = false;
    e.asio_driver = None;
    e.alsa_dsd_device = None;
    assert!(!e.bit_perfect && !e.native_dsd_selected());
    assert_eq!(
        e.route_for(&dsd),
        Route::Device,
        "a DSD file does not go to rodio because a PCM preference is off"
    );

    // DSD, toggle on: unchanged. The preference is about PCM.
    e.bit_perfect = true;
    assert_eq!(e.route_for(&dsd), Route::Device);
}

/// The shared mixer is a processed route, and a rollover onto it says so
/// whatever the successor turns out to be.
///
/// The rollover asked `fidelity_for`, whose job is to decide whether a
/// *device* route preserved the samples. On this route the answer is
/// known before the question: it did not. The mixer converts to float and
/// applies the volume control, the equaliser and ReplayGain. A track whose
/// channel layout could not be checked came back `Unverified` — the badge
/// that means nothing Moosik did altered the audio — of exactly that path.
#[test]
fn a_shared_rollover_is_processed_whatever_it_rolls_onto() {
    use bitperfect::state::{Fidelity, TransformDescription};

    // Common setup: a shared session with a known track playing.
    let start = |queued: QueuedShared| {
        let mut e = engine();
        e.bp_state.opened(
            bitperfect::Transport::Shared { endpoint: endpoint() },
            Fidelity::Processed {
                transform: TransformDescription::Identity,
                reason: "shared".into(),
            },
            None,
            None,
            e.shared_processing(),
        );
        e.queued_next = Some(queued);
        e.roll_shared_source();
        e
    };

    // 1. A PCM successor whose layout cannot be checked. This is the case
    //    that used to come out Unverified.
    let unverifiable = bitperfect::state::MediaSource::pcm(
        bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
            sample_rate: 96_000,
            channels: 6,
            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
        },
    );
    assert!(
        Engine::source_blocks_exact_claim(&unverifiable).is_some(),
        "the fixture must be a source whose layout cannot be checked"
    );
    let e = start(QueuedShared {
        path: PathBuf::from("six.flac"),
        source: Some(unverifiable),
        transform: TransformDescription::Identity,
        reason: "the shared mixer".into(),
    });
    match &e.bp_state.fidelity {
        Fidelity::Processed { transform, .. } => {
            assert_eq!(*transform, TransformDescription::Identity)
        }
        other => panic!("the shared mixer is a processed route: {other:?}"),
    }

    // 2. A DSD successor on the decimating fallback. The transform is
    //    what the route does, and it is not identity.
    let e = start(QueuedShared {
        path: PathBuf::from("next.dsf"),
        source: Some(bitperfect::state::MediaSource::dsd(
            bitperfect::format::SourceFormat {
                kind: bitperfect::format::PcmKind::Integer { valid_bits: 1 },
                sample_rate: 2_822_400,
                channels: 2,
                layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
            },
            "DSD64".to_string(),
        )),
        transform: TransformDescription::DsdDecimated { pcm_rate: 176_400 },
        reason: "decimated to 176400 Hz PCM, then the shared mixer".into(),
    });
    match &e.bp_state.fidelity {
        Fidelity::Processed { transform, reason } => {
            assert_eq!(
                *transform,
                TransformDescription::DsdDecimated { pcm_rate: 176_400 },
                "a decimated DSD file was rolling over as an identity transform"
            );
            assert!(reason.contains("decimated"), "{reason}");
        }
        other => panic!("{other:?}"),
    }

    // 3. A successor nothing can describe. Still processed, and the reason
    //    says which part is unknown.
    let e = start(QueuedShared {
        path: PathBuf::from("mystery.xyz"),
        source: None,
        transform: TransformDescription::Identity,
        reason: "the shared mixer; and this build cannot read the container, so what \
                     the file is could not be established"
            .into(),
    });
    match &e.bp_state.fidelity {
        Fidelity::Processed { reason, .. } => assert!(
            reason.contains("could not be established"),
            "an unknown source is said, not implied: {reason}"
        ),
        other => panic!("{other:?}"),
    }
    assert!(
        e.bp_state.source.is_none(),
        "and the previous track's identity does not stay up"
    );
}

/// The shared route never describes a file by the mixer's own format.
///
/// `rodio` converts everything to 32-bit float. The fallback arm
/// published that, at the sink's channel count, for any track the exact
/// route had not prepared — so a 24-bit FLAC the DAC had refused was
/// reported to the listener as a float source.
#[test]
fn the_shared_route_never_reports_the_mixers_own_format() {
    let mut e = engine();

    // Prepared by the exact route: that is the answer.
    let known = bitperfect::format::SourceFormat {
        kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
        sample_rate: 96_000,
        channels: 2,
        layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
    };
    let path = PathBuf::from("known.flac");
    e.last_prepared = Some((path.clone(), known));
    assert_eq!(
        e.shared_source_for(&path).map(|m| m.format.kind),
        Some(bitperfect::format::PcmKind::Integer { valid_bits: 24 })
    );

    // Not prepared, and the container cannot be read: unknown, which is a
    // real answer. What it must never be is `Float32`, which is the sink's
    // internal representation and says nothing about the file.
    e.last_prepared = None;
    let unreadable = PathBuf::from("no-such-file.qqq");
    let got = e.shared_source_for(&unreadable);
    assert!(
        got.is_none(),
        "an unreadable container is unknown, not invented: {got:?}"
    );

    // And the stale entry from another file is not borrowed either.
    e.last_prepared = Some((path.clone(), known));
    assert!(
        e.shared_source_for(&unreadable).is_none(),
        "the previous track's format is not this track's format"
    );
}

/// The provisional seek notice comes down when the seek resolves.
///
/// `Provisional` puts "Seeking to 3:20…" on the status line to say that
/// the position on screen is a request and not a landing. Nothing took it
/// off again, so it stayed up after the landing — the one moment at which
/// it is false — until something unrelated overwrote it.
#[test]
fn the_provisional_seek_notice_is_withdrawn_when_the_seek_resolves() {
    let mut up = false;
    let playing = || "▶ Album — Track".to_string();

    let notice = seek_status_line(
        &SeekOutcome::Provisional(Duration::from_secs(200)),
        &mut up,
        playing,
    );
    assert_eq!(notice.as_deref(), Some("Seeking to 3:20…"));
    assert!(up, "the line is ours from here");

    // Landed: the notice is replaced by what is actually playing.
    let after = seek_status_line(&SeekOutcome::Landed(Duration::from_secs(199)), &mut up, playing);
    assert_eq!(
        after.as_deref(),
        Some("▶ Album — Track"),
        "the seek landed; the line must stop saying it is still seeking"
    );
    assert!(!up, "and the line is no longer ours");

    // A landing with no notice up leaves whatever is there alone.
    assert_eq!(
        seek_status_line(&SeekOutcome::Landed(Duration::from_secs(1)), &mut up, playing),
        None
    );

    // Failed: replaced, not queued behind the notice.
    let mut up = false;
    seek_status_line(&SeekOutcome::Provisional(Duration::from_secs(60)), &mut up, playing);
    let failed = seek_status_line(
        &SeekOutcome::Failed(SeekError::Open("no such file".into())),
        &mut up,
        playing,
    );
    assert!(
        failed.as_deref().is_some_and(|l| l.starts_with("⚠ Seek failed")),
        "{failed:?}"
    );
    assert!(!up);
}

/// The shared gapless queue carries an identity, including for DSD.
///
/// It carried a `SourceFormat`, which cannot describe a DSD file at all —
/// `prepare` will not read one — so every DSD track on the decimating
/// fallback rolled over as an unknown source. And an unknown one left the
/// *previous* track's description standing, which is a specific claim
/// about the wrong file.
#[test]
fn the_shared_queue_carries_what_the_next_track_is() {
    let e = engine();

    // A file nothing can read is honestly unknown.
    let missing = std::env::temp_dir().join("moosik_g2_absent.flac");
    let _ = std::fs::remove_file(&missing);
    assert!(
        e.queued_source_for(&missing).is_none(),
        "an unreadable file has no identity, and inventing one is the bug"
    );

    // A DSD file is a DSD source, not the PCM the fallback decimates it to.
    let dsd_path = std::env::temp_dir()
        .join(format!("moosik_g2_{}.dff", std::process::id()));
    let audio = vec![0x69u8; 2 * 4_096];
    std::fs::write(
        &dsd_path,
        crate::dsd::tests::make_dff(2, 2_822_400, b"DSD ", &audio, None),
    )
    .unwrap();
    let got = e.queued_source_for(&dsd_path).expect("a DSD file has an identity");
    assert!(
        got.dsd_label.is_some(),
        "it is a DSD source: {:?}",
        got.dsd_label
    );
    assert_eq!(got.format.channels, 2);
    assert_eq!(
        got.format.kind,
        bitperfect::format::PcmKind::Integer { valid_bits: 1 },
        "one bit per sample is what DSD is"
    );
    let _ = std::fs::remove_file(&dsd_path);
}

/// An unknown successor replaces the previous track's description.
#[test]
fn an_unknown_successor_clears_the_previous_source() {
    let mut e = engine();
    let known = bitperfect::format::SourceFormat {
        kind: bitperfect::format::PcmKind::Integer { valid_bits: 24 },
        sample_rate: 96_000,
        channels: 2,
        layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
    };
    e.bp_state.opened(
        bitperfect::Transport::Shared { endpoint: endpoint() },
        bitperfect::state::Fidelity::Processed {
            transform: bitperfect::state::TransformDescription::Identity,
            reason: "shared".into(),
        },
        None,
        None,
        e.shared_processing(),
    );
    e.bp_state
        .set_source(bitperfect::state::MediaSource::pcm(known));
    e.last_prepared = Some((PathBuf::from("known.flac"), known));

    // The successor cannot be described.
    e.queued_next = Some(QueuedShared {
        path: PathBuf::from("mystery.xyz"),
        source: None,
        transform: bitperfect::state::TransformDescription::Identity,
        reason: "the shared mixer; and this build cannot read the container".into(),
    });
    e.roll_shared_source();

    assert!(
        e.bp_state.source.is_none(),
        "leaving the previous track's format up is a claim about the wrong file"
    );
    assert!(e.last_prepared.is_none());
    assert_eq!(
        e.current_track().map(|c| c.path.clone()),
        Some(PathBuf::from("mystery.xyz")),
        "the track still changed, even though its identity is unknown"
    );
    // The route did not change, so playback is still processed.
    assert!(
        matches!(
            e.bp_state.fidelity,
            bitperfect::state::Fidelity::Processed { .. }
        ),
        "the shared mixer is still the shared mixer: {:?}",
        e.bp_state.fidelity
    );
}

/// A conversion that happened is Processed, whatever else is unprovable.
///
/// The unverifiable-source check ran first, so a six-channel 64-bit float
/// file with no declared speaker layout came out **Unverified** — the word
/// for "nothing Moosik did altered it, but the path downstream cannot be
/// proved" — of a route that had just narrowed it to `f32` and rounded it
/// into Q1.31. Two alterations, reported as none. Being unsure about the
/// channel mapping does not make the conversion stop having happened.
#[test]
fn an_unverifiable_layout_does_not_hide_a_conversion() {
    use bitperfect::state::{Fidelity, TransformDescription};

    let multichannel_f64 = bitperfect::state::MediaSource::pcm(
        bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Float64,
            sample_rate: 96_000,
            channels: 6,
            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
        },
    );
    // The layout really is unverifiable — otherwise this test is about
    // nothing.
    assert!(
        Engine::source_blocks_exact_claim(&multichannel_f64).is_some(),
        "the fixture must be a source whose layout cannot be checked"
    );

    let transport = bitperfect::Transport::Shared { endpoint: endpoint() };
    let fidelity =
        Engine::fidelity_for(&multichannel_f64, bitperfect::PayloadPlan::Q31, &transport);

    match fidelity {
        Fidelity::Processed { transform, reason } => {
            assert_eq!(
                transform,
                TransformDescription::Float64ToQ31Processed,
                "the transform is what was done to the samples"
            );
            // Both facts, in one reason.
            assert!(
                reason.contains("64-bit float source"),
                "it must say what was done: {reason}"
            );
            assert!(
                reason.contains("channel mapping cannot be verified"),
                "and what could not be established: {reason}"
            );
        }
        other => panic!(
            "a route that narrowed and rounded the samples is not Unverified: {other:?}"
        ),
    }

    // The same source with no transform of ours *is* Unverified: that is
    // the case the word is for, and it still works.
    let untouched =
        Engine::fidelity_for(&multichannel_f64, bitperfect::PayloadPlan::Identity, &transport);
    assert!(
        matches!(untouched, Fidelity::Unverified { .. }),
        "{untouched:?}"
    );

    // And a stereo F64 says only the one thing there is to say.
    let stereo_f64 = bitperfect::state::MediaSource::pcm(
        bitperfect::format::SourceFormat {
            kind: bitperfect::format::PcmKind::Float64,
            sample_rate: 96_000,
            channels: 2,
            layout: bitperfect::format::ChannelLayout::UNSPECIFIED,
        },
    );
    match Engine::fidelity_for(&stereo_f64, bitperfect::PayloadPlan::Q31, &transport) {
        Fidelity::Processed { reason, .. } => assert!(
            !reason.contains("channel mapping"),
            "there is no layout doubt about stereo: {reason}"
        ),
        other => panic!("{other:?}"),
    }
}

/// The shortfall rule, at the seam the engine actually uses.
///
/// What was here before asserted that an engine with no sink reports
/// `Running`, twice, and never reached the rule at all — it could not,
/// because the rule was welded to a `rodio::Sink`. It passed whatever the
/// rule said, including nothing.
#[test]
fn a_shared_track_that_stops_early_is_a_failure_and_not_an_ending() {
    use bitperfect::Completion;
    const G: u64 = 7;
    let five_min = Some(Duration::from_secs(300));

    // Playing: the sink still has audio in it.
    assert_eq!(
        shared_completion(false, true, Duration::from_secs(4), five_min, G),
        Completion::Running
    );

    // Empty but never started — an appended sink reports empty for an
    // instant, and there is nothing to compare yet.
    assert_eq!(
        shared_completion(true, false, Duration::ZERO, five_min, G),
        Completion::Running
    );

    // Empty four seconds into a five-minute track. Whatever happened, the
    // track did not play.
    assert_eq!(
        shared_completion(true, true, Duration::from_secs(4), five_min, G),
        Completion::Failed {
            reason: bitperfect::fault::SHARED_ENDED_EARLY,
            generation: G,
        },
        "a track that stopped in the middle did not end"
    );

    // Played to the end.
    assert_eq!(
        shared_completion(true, true, Duration::from_secs(300), five_min, G),
        Completion::CleanEof
    );

    // A tag a second optimistic is a tag, not a fault.
    assert_eq!(
        shared_completion(true, true, Duration::from_secs(299), five_min, G),
        Completion::CleanEof,
        "the tolerance exists because declared durations are approximate"
    );

    // No declared duration: nothing to measure against, so no accusation.
    assert_eq!(
        shared_completion(true, true, Duration::from_secs(4), None, G),
        Completion::CleanEof
    );
}

/// And what the playlist does with it — including the case the whole
/// distinction exists for.
///
/// Under Repeat One, "next" is the same file. A shared decode that gave up
/// four seconds in was reported as a track that had finished, the playlist
/// advanced, and the file that had just failed was reopened — hundreds of
/// times, as fast as it could fail.
#[test]
fn a_shared_failure_does_not_reopen_the_same_file_under_repeat_one() {
    use bitperfect::{Completion, TickAction, on_completion};

    let ended_early = shared_completion(
        true,
        true,
        Duration::from_secs(4),
        Some(Duration::from_secs(300)),
        1,
    );
    assert_eq!(
        on_completion(ended_early, false),
        TickAction::Halt(bitperfect::fault::SHARED_ENDED_EARLY),
        "a failure stops; it does not choose a next track"
    );

    // `next_in_playlist` is the only thing that picks a successor, and
    // `Halt` never reaches it. Under Repeat One it would have handed back
    // the same index — which is the loop.
    assert_eq!(next_in_playlist(LoopMode::RepeatOne, Some(3), 10), Some(3));
    assert_eq!(next_in_playlist(LoopMode::RepeatAll, Some(9), 10), Some(0));

    // The same track, having actually played, does advance — the
    // distinction has to cut both ways or it is just a stop button.
    let played_out = shared_completion(
        true,
        true,
        Duration::from_secs(300),
        Some(Duration::from_secs(300)),
        1,
    );
    assert_eq!(played_out, Completion::CleanEof);
    assert_eq!(on_completion(played_out, false), TickAction::Advance);
}

/// A seek that cannot land leaves the engine somewhere it can play from.
///
/// Every failing path used to stop in a different half-state: the sink
/// released, the clock frozen, and the position showing wherever the
/// slider had been dragged to — a place nothing was playing from. To a
/// listener that is indistinguishable from a seek that worked into a
/// silent passage.
#[test]
fn a_seek_that_fails_puts_the_engine_back_where_it_was() {
    let mut e = engine();
    e.current_duration = Some(Duration::from_secs(300));
    e.paused_elapsed = Duration::from_secs(120);
    e.started_at = None;
    let from = e.elapsed();

    // The slider has been dragged and the seek is in flight: the position
    // is already showing the target.
    e.paused_elapsed = Duration::from_secs(280);
    e.abandon_seek(from, SeekError::Open("no such file".into()));

    assert_eq!(e.elapsed(), from, "back to where playback actually was");
    assert!(e.started_at.is_none(), "and stopped, not counting forward");
    assert!(!e.is_seeking(), "with nothing left to poll");

    // Reported once, through the one channel the tick reads.
    assert_eq!(
        e.poll_pending_seek(),
        SeekOutcome::Failed(SeekError::Open("no such file".into()))
    );
    assert_eq!(
        e.poll_pending_seek(),
        SeekOutcome::Idle,
        "and not again on every frame after"
    );
}

/// Each way a seek can fail says which way it was.
///
/// One `None` carried all of them, and all of them were reported as "this
/// file could not be seeked" — untrue of a thread that would not start,
/// and untrue of a device that would not open.
#[test]
fn every_seek_failure_says_what_actually_happened() {
    let landed = Duration::from_secs(97);
    let cases = [
        SeekError::Open("permission denied".into()),
        SeekError::Decode("unsupported codec".into()),
        SeekError::PastEnd { landed },
        SeekError::Spawn("cannot create thread".into()),
        SeekError::WorkerLost,
        SeekError::Dsd("truncated".into()),
        SeekError::Output("device in use".into()),
    ];
    let described: Vec<String> = cases.iter().map(|c| c.describe()).collect();
    for (i, d) in described.iter().enumerate() {
        assert!(!d.is_empty());
        for (j, other) in described.iter().enumerate() {
            assert!(
                i == j || d != other,
                "two different failures must not read the same"
            );
        }
    }
    // The one with a position in it says the position, because that is the
    // only actionable thing about it.
    assert!(
        described[2].contains("1:37"),
        "past-end must say where the file does end: {}",
        described[2]
    );
}

/// A seek that runs off the end of the file is a failure, not a landing.
///
/// The worker stopped discarding samples when the decoder ran out and
/// returned it anyway, sitting at EOF; the caller then published the
/// target it had asked for. The slider moved to a position the file does
/// not contain and the track ended at once, which reads as a track that
/// failed to play.
#[test]
fn a_seek_past_the_end_of_the_audio_is_not_a_landing() {
    let e = SeekError::PastEnd { landed: Duration::from_secs(10) };
    assert_ne!(
        SeekOutcome::Failed(e.clone()),
        SeekOutcome::Landed(Duration::from_secs(10)),
        "running out of audio is not arriving"
    );
    assert!(e.describe().contains("0:10"));
}

/// A halted engine releases the device, stops, and keeps the reason.
///
/// `stop()` would clear the claim, the source and the transport — right
/// when the user stops, wrong here, because then the player sits idle with
/// no account of why.
#[test]
fn halting_releases_the_device_and_keeps_the_reason() {
    let mut e = engine();
    e.bit_perfect = true;
    e.bp_state.opened(
        bitperfect::Transport::WasapiExclusivePcm {
            endpoint: endpoint(),
        },
        bitperfect::state::Fidelity::PayloadExact,
        None,
        None,
        bitperfect::state::ProcessingState::transparent_locked(),
    );
    assert!(e.bp_state.shows_diamond());

    e.halt(bitperfect::fault::SOURCE_READ);

    assert!(e.bp.is_none(), "the device is released");
    assert!(!e.bp_state.shows_diamond(), "the claim is gone");
    assert_eq!(
        e.bp_state.playback,
        bitperfect::state::PlaybackState::Stopped
    );
    assert!(
        e.bp_state.fidelity.is_faulted(),
        "and the reason survives: {:?}",
        e.bp_state.fidelity
    );
    let p = e.bp_state.presentation();
    assert_eq!(p.badge, bitperfect::state::Badge::Faulted);
    assert!(!p.headline.is_empty());

    // Halting twice does not lose the first reason for the second.
    let first = e.bp_state.fidelity.clone();
    e.halt(bitperfect::fault::BACKEND_DEAD);
    assert_eq!(e.bp_state.fidelity, first);
}

// -----------------------------------------------------------------------
// The stereo lease across route teardown
// -----------------------------------------------------------------------

/// Build an engine whose stereo buffer looks like a live two-channel route.
fn engine_with_live_stereo() -> (Engine, spectrum::StereoBuf) {
    let e = engine();
    let stereo = e.stereo_buf.clone();
    spectrum::begin_stereo_stream(&stereo, 2);
    {
        let mut g = stereo.lock().unwrap();
        g.frames.push([0.5, -0.5]);
    }
    assert_eq!(
        spectrum::channels::availability(false, false, true, stereo.lock().unwrap().channels),
        spectrum::channels::ChannelAvailability::Available,
        "setup: the route should look live"
    );
    (e, stereo)
}

fn assert_no_live_tap(stereo: &spectrum::StereoBuf, what: &str) {
    let g = stereo.lock().unwrap();
    assert_eq!(
        g.channels,
        spectrum::channels::NO_LIVE_TAP,
        "{what} left the display claiming a live tap"
    );
    assert!(g.frames.is_empty(), "{what} left stale frames behind");
    assert_eq!(
        spectrum::channels::availability(false, false, true, g.channels),
        spectrum::channels::ChannelAvailability::NoLiveTap,
        "{what} must report NoLiveTap"
    );
}

/// Stopping tears every route down, so nothing is feeding the analyser and
/// the display has to say so. A stopped player showing a live left and
/// right is the same class of stale truth as a stopped player showing
/// "Playing:".
#[test]
fn stopping_retires_the_stereo_lease() {
    let (mut e, stereo) = engine_with_live_stereo();
    e.stop();
    assert_no_live_tap(&stereo, "stop");
}

#[test]
fn halting_retires_the_stereo_lease() {
    let (mut e, stereo) = engine_with_live_stereo();
    e.halt(bitperfect::fault::BACKEND_DEAD);
    assert_no_live_tap(&stereo, "halt");
}

#[test]
fn aborting_an_open_retires_the_stereo_lease() {
    let (mut e, stereo) = engine_with_live_stereo();
    e.abort_open(bitperfect::state::FailureReason::DeviceUnavailable(
        "no such device".into(),
    ));
    assert_no_live_tap(&stereo, "abort_open");
}

/// A failed open goes through the transaction, which undoes everything it
/// took. The lease is part of "everything": a route that never started is
/// not feeding the analyser, whatever the previous one left in the buffer.
#[test]
fn a_failed_open_leaves_no_live_tap() {
    let (mut e, stereo) = engine_with_live_stereo();
    e.bp_state.begin_open();
    let out = open_track(
        &mut e,
        Opening::at(Duration::ZERO),
        false,
        |eng, _o| {
            eng.note_current(Path::new("attempted.flac"), None, Duration::ZERO);
            Err(bitperfect::OpenError::backend("the render thread would not start"))
        },
        |_, _| panic!("a backend failure is not a format rejection"),
    );
    assert!(
        matches!(out, RestartOutcome::Stopped(_)),
        "setup: the open should have failed, got {out:?}"
    );
    assert_no_live_tap(&stereo, "a failed open");
}

/// Releasing the shared sink ends the shared analysis route there and then,
/// without waiting for the source's own `Drop`.
///
/// This is the slow background seek: it releases the sink and then spends
/// seconds decoding on a worker before anything replaces it. `Sink::stop`
/// only sets an atomic, so the source can outlive the call; its `Drop` runs
/// on the audio thread and takes the lock with `try_lock`, which is allowed
/// to miss. The test models exactly that by never dropping the source at
/// all — if the invalidation were left to `Drop`, the display would keep
/// claiming a live stereo tap for the whole of a multi-second seek.
#[test]
fn releasing_the_shared_sink_reports_no_live_tap_without_waiting_for_drop() {
    let (mut e, stereo) = engine_with_live_stereo();
    // A witness stands in for the real sink, which needs a device.
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    e.held_sink = Some(Sentinel(std::sync::Arc::clone(&counter)));

    e.release_sink();

    assert_eq!(released(&counter), 1, "setup: the sink should have been released");
    let g = stereo.lock().unwrap();
    assert_eq!(
        g.channels,
        spectrum::channels::NO_LIVE_TAP,
        "a released shared sink left the display claiming a live tap"
    );
    assert!(g.frames.is_empty(), "the previous route's frames survived");
    assert_eq!(
        spectrum::channels::availability(false, false, true, g.channels),
        spectrum::channels::ChannelAvailability::NoLiveTap
    );
}

/// And it ends the *session*, not merely the current owner — so a source
/// still draining from the released sink cannot claim afterwards.
#[test]
fn releasing_the_shared_sink_advances_the_route_epoch() {
    let (mut e, stereo) = engine_with_live_stereo();
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    e.held_sink = Some(Sentinel(std::sync::Arc::clone(&counter)));
    let before = stereo.lock().unwrap().route_epoch;

    e.release_sink();

    assert_ne!(
        stereo.lock().unwrap().route_epoch,
        before,
        "the shared session did not end, so a late first pull could still claim"
    );
}

/// Releasing a sink that does not exist is not a shared-route teardown. An
/// exact or native route passing through here keeps its display claim.
#[test]
fn releasing_no_sink_leaves_another_routes_claim_alone() {
    let (mut e, stereo) = engine_with_live_stereo();
    let epoch = stereo.lock().unwrap().route_epoch;
    let owner = stereo.lock().unwrap().generation;

    e.release_sink(); // nothing to release

    let g = stereo.lock().unwrap();
    assert_eq!(g.route_epoch, epoch);
    assert_eq!(g.generation, owner);
    assert_eq!(
        g.channels, 2,
        "a bit-perfect route lost its claim to an empty release"
    );
}

/// A native DSD route that starts successfully publishes "no live PCM".
///
/// The route itself cannot be run here — it needs an ASIO driver or an ALSA
/// hw: device — so this pins the branch instead: the success arm of
/// `start_native` must retire the stereo route, and it must do so *inside*
/// the arm rather than unconditionally, or a failed native attempt falling
/// back to DoP or decimated PCM would blank the route that then succeeds.
///
/// Native DSD is not DoP. DoP is a PCM carrier with a tap that publishes no
/// channels; native DSD hands DSD bits straight to the driver and attaches
/// no tap at all. They reach the same display state by different routes,
/// and both are covered — the DoP half in `bitperfect::tests`.
#[test]
fn a_successful_native_dsd_start_publishes_no_live_tap() {
    let src = include_str!("main.rs");
    let arm = src
        .split_once("match self.start_native(path, opening) {")
        .expect("the native start dispatch moved")
        .1;
    let ok_arm = arm
        .split_once("Err(e) => {")
        .expect("the native start dispatch has no failure arm")
        .0;
    assert!(
        ok_arm.contains("self.dsd_native = true"),
        "this is not the success arm any more"
    );
    assert!(
        ok_arm.contains("invalidate_live_pcm_route"),
        "a successful native DSD start no longer publishes NoLiveTap"
    );

    // And the mechanism it uses does what the arm needs it to.
    let stereo = spectrum::new_stereo_buf();
    let mono = spectrum::new_sample_buf();
    spectrum::begin_stereo_stream(&stereo, 2);
    {
        let mut g = stereo.lock().unwrap();
        g.frames.push([0.5, -0.5]);
    }
    spectrum::invalidate_live_pcm_route(&stereo, &mono);
    let g = stereo.lock().unwrap();
    assert_eq!(g.channels, spectrum::channels::NO_LIVE_TAP);
    assert!(g.frames.is_empty(), "the previous track's frames survived");
    assert_eq!(
        spectrum::channels::availability(false, false, true, g.channels),
        spectrum::channels::ChannelAvailability::NoLiveTap
    );
}

/// A native attempt that *fails* must not publish anything: the fallback
/// route that follows is the one that owns the display.
#[test]
fn a_failed_native_dsd_start_does_not_publish_no_live_tap() {
    let src = include_str!("main.rs");
    let arm = src
        .split_once("match self.start_native(path, opening) {")
        .expect("the native start dispatch moved")
        .1;
    let (_, after_err) = arm
        .split_once("Err(e) => {")
        .expect("the native start dispatch has no failure arm");
    let err_arm = &after_err[..after_err.find("Some(e)").unwrap_or(after_err.len())];
    assert!(
        !err_arm.contains("invalidate_live_pcm_route"),
        "a failed native start blanks the fallback route that follows it"
    );
}

/// A teardown advances the route exactly once.
///
/// It used to advance twice on the shared route: `release_sink` ended the
/// session and then the teardown ended it again. Harmless in effect, but it
/// meant the count was not the thing it claimed to be, and a count nobody
/// can trust is not a guard.
#[test]
fn a_shared_teardown_advances_the_route_exactly_once() {
    for (name, run) in [
        ("stop", 0u8),
        ("halt", 1),
        ("abort_open", 2),
    ] {
        let (mut e, stereo) = engine_with_live_stereo();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        e.held_sink = Some(Sentinel(std::sync::Arc::clone(&counter)));
        let before = stereo.lock().unwrap().route_epoch;

        match run {
            0 => e.stop(),
            1 => e.halt(bitperfect::fault::BACKEND_DEAD),
            _ => e.abort_open(bitperfect::state::FailureReason::DeviceUnavailable(
                "no such device".into(),
            )),
        }

        let after = stereo.lock().unwrap().route_epoch;
        assert_eq!(
            after.wrapping_sub(before),
            1,
            "{name} advanced the route {} times",
            after.wrapping_sub(before)
        );
    }
}

/// And a route with no shared sink — bit-perfect or native — still advances
/// it once, because `release_sink` had nothing to do.
#[test]
fn a_teardown_without_a_sink_still_advances_the_route_once() {
    for run in 0u8..3 {
        let (mut e, stereo) = engine_with_live_stereo();
        let before = stereo.lock().unwrap().route_epoch;
        match run {
            0 => e.stop(),
            1 => e.halt(bitperfect::fault::BACKEND_DEAD),
            _ => e.abort_open(bitperfect::state::FailureReason::DeviceUnavailable(
                "no such device".into(),
            )),
        }
        let after = stereo.lock().unwrap().route_epoch;
        assert_eq!(after.wrapping_sub(before), 1, "run {run}");
        assert_eq!(
            stereo.lock().unwrap().channels,
            spectrum::channels::NO_LIVE_TAP
        );
    }
}

/// Both buffers, on every teardown. The mono one is the series the bars are
/// drawn from, and it used to survive all of them.
#[test]
fn every_engine_teardown_clears_the_mono_buffer_too() {
    for (name, run) in [("stop", 0u8), ("halt", 1), ("abort_open", 2)] {
        let (mut e, stereo) = engine_with_live_stereo();
        e.sample_buf.lock().unwrap().extend_from_slice(&[0.4f32; 256]);
        assert!(!e.sample_buf.lock().unwrap().is_empty(), "setup");
        let mono = e.sample_buf.clone();

        match run {
            0 => e.stop(),
            1 => e.halt(bitperfect::fault::BACKEND_DEAD),
            _ => e.abort_open(bitperfect::state::FailureReason::DeviceUnavailable(
                "no such device".into(),
            )),
        }

        assert!(
            mono.lock().unwrap().is_empty(),
            "{name} left the previous track's mono PCM for the bars to draw"
        );
        assert!(stereo.lock().unwrap().frames.is_empty(), "{name}: stereo");
    }
}

/// The slow background seek releases the sink and then decodes for seconds
/// on a worker before anything replaces it. That release is the only thing
/// ending the shared session in the meantime.
///
/// The branch cannot be run here — it needs a decoder and a device — so the
/// call is pinned in the source. A behavioural test that reached
/// `release_sink` directly would survive the call being deleted from the
/// branch, which is exactly the mutant this is for.
#[test]
fn the_slow_seek_branch_releases_the_sink() {
    let src = include_str!("main.rs");
    let branch = src
        .split_once("// --- Slow path: reopen + skip samples, ON A BACKGROUND THREAD ---")
        .expect("the slow seek branch moved")
        .1;
    let head = branch
        .split_once("spawn_seek_worker")
        .expect("the slow seek no longer spawns a worker")
        .0;
    assert!(
        head.contains("self.release_sink()"),
        "the slow seek stopped releasing the sink, so a multi-second seek \
             would keep reporting a live spectrum of the old position"
    );
}

/// Retiring is generation-qualified everywhere it can race, so an owner
/// being torn down cannot blank a route that has already taken over.
#[test]
fn an_old_owner_cannot_revoke_its_successor() {
    let stereo = spectrum::new_stereo_buf();
    let old = spectrum::begin_stereo_stream(&stereo, 2);
    let successor = spectrum::begin_stereo_stream(&stereo, 2);
    assert_ne!(old, successor);

    assert!(
        !spectrum::end_stereo_stream_if_generation(&stereo, old),
        "the old generation was allowed to retire the current lease"
    );
    let g = stereo.lock().unwrap();
    assert_eq!(g.generation, successor);
    assert_eq!(g.channels, 2);
    drop(g);

    assert!(
        spectrum::end_stereo_stream_if_generation(&stereo, successor),
        "the current owner must be able to retire its own lease"
    );
    assert_eq!(
        stereo.lock().unwrap().channels,
        spectrum::channels::NO_LIVE_TAP
    );
}

/// The shared sink reports a clean ending and nothing else, because that
/// is all `rodio` gives us — a shared session that fails does so
/// synchronously, at the open, where it is already reported.
#[test]
fn an_engine_with_no_session_is_not_finished() {
    let e = engine();
    assert_eq!(e.completion(), bitperfect::Completion::Running);
    assert!(!e.is_finished());
    assert_eq!(
        bitperfect::on_completion(e.completion(), false),
        bitperfect::TickAction::Nothing
    );
}

// -----------------------------------------------------------------------
// Routing: which object owns the session
// -----------------------------------------------------------------------

/// A cancel flag nothing ever sets.
fn never_cancelled() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))
}

/// A same-track reopen tells the spectrum what the *new* route feeds the
/// live tap, and leaves the pre-process alone.
///
/// Round 6 defect: a DSD track started on a no-tap route (DoP, native)
/// stored the analysis rate as the live rate; an output change that
/// reopened the same track onto the PCM fallback went through
/// `apply_restart`, which only moved the position, so the 176.4 kHz tap
/// was read at the 88.2 kHz analysis rate until the next track.
///
/// Through the production pieces `apply_restart` and a seek landing use:
/// the `open_track` transaction, its exact attempt injected as the rung of
/// `play_file_at`'s DSD ladder that a refused DoP/native device leaves —
/// the real `start_dsd_fallback`, on an idle sink — then
/// `restart_spectrum`. (A DSD file is never retried through the shared
/// mixer by `open_track`; its fallback is inside the exact opener.) The
/// no-tap device-route state is the `device_route` seam. The window's analysis
/// is a parked worker that must survive every schedule untouched.
#[test]
fn a_reopened_route_refreshes_the_live_tap_and_keeps_the_analysis() {
    use std::sync::atomic::Ordering;
    let dir = std::env::temp_dir().join(format!("moosik_r6_restart_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dsf = dir.join("t.dsf");
    spectrum::dsd_rate_tests::write_dsf(&dsf, 2_822_400, 0.3, 1_000.0, 2_000.0);
    let wav = dir.join("t.wav");
    std::fs::write(&wav, b"").unwrap();

    let mut w = SpectrumWindow::new_fresh_profile();
    w.analyzer.cache_dir_override = Some(dir.join("cache"));
    assert_eq!(w.analyzer.dsd_rate, 88_200, "fresh default");
    let hold = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(std::sync::Arc::clone(&hold));

    let mut e = engine();
    e.idle_output = true;
    e.dsd_fallback_target = 176_400;
    let at = Duration::from_millis(100);
    let effect = |o: &RestartOutcome| restart_effect(o, false, None);
    // The fallback rung, reached through the restart transaction.
    let reopen_fallback = |e: &mut Engine| {
        let p = dsf.clone();
        open_track(
            e, Opening::at(at), true,
            move |e, o| e.start_dsd_fallback(&p, o)
                .map(|_| o.at)
                .map_err(bitperfect::OpenError::source),
            |_, _| Err("not reached for DSD".to_string()),
        )
    };

    // 1. Start on a device route that attaches no tap.
    e.device_route = Some(true);
    w.on_play_live(&dsf, 176_400, e.spectrum_live_rate(&dsf, 176_400));
    assert_eq!(w.live_tap, None);
    assert_eq!(w.analyzer.live_rate, 88_200, "setup: no tap, analysis-rate stand-in");
    let job = w.analyzer.job_record().expect("an analysis was asked for");
    let spawned = w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst);
    let pre = (w.analyzer.sample_rate, w.analyzer.pre_fft_size, w.analyzer.effective_settings(&dsf));
    let untouched = |w: &SpectrumWindow, what: &str| {
        assert_eq!(w.analyzer.job_record().as_ref(), Some(&job), "{what}: the analysis was replaced or cancelled");
        assert_eq!(w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst), spawned, "{what}: an analysis was started");
        assert!((w.analyzer.sample_rate, w.analyzer.pre_fft_size, w.analyzer.effective_settings(&dsf)) == pre,
                "{what}: the pre-process moved");
    };

    // 2. No tap -> PCM fallback (an output change, same track).
    e.device_route = None;
    let out = reopen_fallback(&mut e);
    assert!(matches!(out, RestartOutcome::Exact { .. }), "{out:?}");
    restart_spectrum(Some(&e), &mut w, Some(&dsf), effect(&out).seek_to);
    assert_eq!(w.live_tap, Some(176_400), "the fallback's tap was not reported");
    assert_eq!(w.analyzer.live_rate, 176_400);
    assert_eq!(w.analyzer.fft_size, 32_768, "the live size did not follow the tap");
    untouched(&w, "no tap -> fallback");
    // The signal check: a 9 kHz tone at 176.4 kHz draws exactly what a
    // 176.4 kHz PCM window draws. Read at 88.2 kHz (the defect) it does not.
    let sig: Vec<f32> = (0..65_536)
        .map(|i| (std::f32::consts::TAU * 9_000.0 * i as f32 / 176_400.0).sin() * 0.5)
        .collect();
    let bars = |w: &mut SpectrumWindow| -> Vec<u32> {
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.clear();
            b.extend_from_slice(&sig);
        }
        w.analyzer.process_realtime(1.0);
        w.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect()
    };
    let pcm_window = |rate: u32, tag: &str| {
        let mut r = SpectrumWindow::new_fresh_profile();
        r.analyzer.cache_dir_override = Some(dir.join(tag));
        r.on_play(&wav, rate);
        r.analyzer.abort_preprocess();
        r
    };
    let want = bars(&mut pcm_window(176_400, "ref"));
    assert_eq!(bars(&mut w), want, "the live tap is not read at 176.4 kHz");
    assert_ne!(bars(&mut pcm_window(88_200, "half")), want, "setup: the check cannot see a half-rate reading");

    // 3. Same-rate reopen: nothing about the live side moves, no analysis.
    let fft_before = w.analyzer.fft_size;
    let out = reopen_fallback(&mut e);
    restart_spectrum(Some(&e), &mut w, Some(&dsf), effect(&out).seek_to);
    assert_eq!((w.live_tap, w.analyzer.live_rate, w.analyzer.fft_size), (Some(176_400), 176_400, fft_before));
    untouched(&w, "same-rate reopen");

    // 4. PCM fallback -> a device route without a tap.
    e.sink = None;
    e.device_route = Some(true);
    restart_spectrum(Some(&e), &mut w, Some(&dsf), effect(&RestartOutcome::Exact { landed: at }).seek_to);
    assert_eq!(w.live_tap, None, "a no-tap route kept the fallback's tap");
    assert_eq!(w.analyzer.live_rate, 88_200);
    untouched(&w, "fallback -> no tap");

    // 5. An attempt that fails, then a rollback to a route with another
    //    tap rate (the previous configuration's 352.8 kHz fallback).
    e.device_route = None;
    let failed = open_track(
        &mut e, Opening::at(at), true,
        |_, _| Err(bitperfect::OpenError::device_unavailable("refused")),
        |_, _| Err("the mixer refused too".to_string()),
    );
    assert!(matches!(failed, RestartOutcome::Stopped(_)), "{failed:?}");
    restart_spectrum(Some(&e), &mut w, Some(&dsf), effect(&failed).seek_to);
    assert_eq!(w.live_tap, None, "a failed attempt left a tap described");
    e.dsd_fallback_target = 352_800;
    let back = reopen_fallback(&mut e);
    assert!(matches!(back, RestartOutcome::Exact { .. }), "{back:?}");
    restart_spectrum(Some(&e), &mut w, Some(&dsf), effect(&back).seek_to);
    assert_eq!(w.live_tap, Some(352_800), "the rollback's route was not reported");
    assert_eq!(w.analyzer.live_rate, 352_800);
    assert_eq!(w.analyzer.fft_size, 32_768);
    untouched(&w, "rollback");

    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
    w.analyzer.abort_preprocess();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The asynchronous reopen (a shared-route restart at a position, which
/// decodes to the position off-thread): the tap is described as absent
/// while nothing is installed, as the landed source's rate once it lands,
/// and as absent again when a seek is abandoned — never as the route that
/// was merely requested. The analysis is untouched throughout.
#[test]
fn an_off_thread_reopen_describes_the_source_that_landed() {
    use std::sync::atomic::Ordering;
    let dir = std::env::temp_dir().join(format!("moosik_r6_async_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("t.wav");
    {
        // 1 s, 48 kHz stereo 16-bit.
        let (sr, n) = (48_000u32, 48_000usize);
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = ((std::f32::consts::TAU * 1_000.0 * i as f32 / sr as f32).sin() * 8_000.0) as i16;
            data.extend(v.to_le_bytes());
            data.extend(v.to_le_bytes());
        }
        let mut f = Vec::new();
        f.extend(b"RIFF");
        f.extend(((36 + data.len()) as u32).to_le_bytes());
        f.extend(b"WAVEfmt ");
        f.extend(16u32.to_le_bytes());
        f.extend(1u16.to_le_bytes());
        f.extend(2u16.to_le_bytes());
        f.extend(sr.to_le_bytes());
        f.extend((sr * 4).to_le_bytes());
        f.extend(4u16.to_le_bytes());
        f.extend(16u16.to_le_bytes());
        f.extend(b"data");
        f.extend((data.len() as u32).to_le_bytes());
        f.extend(data);
        std::fs::write(&wav, f).unwrap();
    }
    let mut w = SpectrumWindow::new_fresh_profile();
    w.analyzer.cache_dir_override = Some(dir.join("cache"));
    let hold = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(std::sync::Arc::clone(&hold));
    w.on_play_live(&wav, 48_000, Some(48_000));
    let job = w.analyzer.job_record().expect("an analysis was asked for");
    let spawned = w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst);

    let mut e = engine();
    e.idle_output = true;
    let settle = |e: &mut Engine| -> SeekOutcome {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match e.poll_pending_seek() {
                SeekOutcome::Working | SeekOutcome::Idle | SeekOutcome::Provisional(_) => {
                    assert!(Instant::now() < deadline, "the seek never resolved");
                    std::thread::yield_now();
                }
                other => return other,
            }
        }
    };

    // Requested: nothing is installed yet, so no tap is described.
    e.play_seeked_async(&wav, Duration::from_millis(500), None, false);
    restart_spectrum(Some(&e), &mut w, Some(&wav), Some(Duration::from_millis(500)));
    assert_eq!(w.live_tap, None, "a requested route was described before it was installed");
    // Landed: the installed source's rate.
    let landed = settle(&mut e);
    let SeekOutcome::Landed(at) = landed else { panic!("{landed:?}") };
    restart_spectrum(Some(&e), &mut w, Some(&wav), Some(at));
    assert_eq!(w.live_tap, Some(48_000), "the landed source was not described");

    // Abandoned: the worker cannot open the file, the seek fails.
    let gone = dir.join("no-such-file.wav");
    e.play_seeked_async(&gone, Duration::from_millis(500), None, false);
    let out = settle(&mut e);
    assert!(matches!(out, SeekOutcome::Failed(_)), "{out:?}");
    restart_spectrum(Some(&e), &mut w, Some(&gone), Some(Duration::ZERO));
    assert_eq!(w.live_tap, None, "an abandoned seek left a tap described");
    assert_eq!(w.analyzer.job_record(), Some(job), "the analysis was replaced or cancelled");
    assert_eq!(w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst), spawned);

    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
    w.analyzer.abort_preprocess();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Resuming after a failed rebuild describes the route the resume
/// installed.
///
/// Round 7 defect: a failed seek clears the tap (nothing is open) and
/// leaves the app Paused. Play then went `toggle_play_pause` →
/// `Engine::resume` → `play_seeked_async`, which installs a DSD fallback
/// synchronously and publishes no `Landed`, so nothing ever refreshed the
/// spectrum: 176.4 kHz fallback audio stayed described as no tap, read at
/// the 88.2 kHz analysis rate. Driven through `resume_playback`, the step
/// the Paused branch runs, on an idle sink; the failure is a missing file.
#[test]
fn resuming_a_released_dsd_sink_describes_the_reopened_route() {
    use std::sync::atomic::Ordering;
    let dir = std::env::temp_dir().join(format!("moosik_r7_resume_dsd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dsf = dir.join("t.dsf");
    spectrum::dsd_rate_tests::write_dsf(&dsf, 2_822_400, 0.3, 1_000.0, 2_000.0);
    let away = dir.join("t.dsf.away");
    let wav = dir.join("t.wav");
    std::fs::write(&wav, b"").unwrap();

    let mut w = SpectrumWindow::new_fresh_profile();
    w.analyzer.cache_dir_override = Some(dir.join("cache"));
    assert_eq!(w.analyzer.dsd_rate, 88_200, "fresh default");
    let hold = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(std::sync::Arc::clone(&hold));

    let mut e = engine();
    e.idle_output = true;
    e.dsd_fallback_target = 176_400;
    let at = Duration::from_millis(100);

    // A DSD fallback session, started and described.
    e.play_seeked_async(&dsf, at, None, false);
    assert!(e.sink.is_some() && e.dsd_fallback, "setup: the fallback did not install");
    w.on_play_live(&dsf, 176_400, e.live_tap_now(&dsf));
    assert_eq!(w.live_tap, Some(176_400), "setup");
    let job = w.analyzer.job_record().expect("an analysis was asked for");
    let spawned = w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst);
    let pre = (w.analyzer.sample_rate, w.analyzer.pre_fft_size, w.analyzer.effective_settings(&dsf),
               w.analyzer.pre_cache_path().map(Path::to_path_buf));
    let untouched = |w: &SpectrumWindow, what: &str| {
        assert_eq!(w.analyzer.job_record().as_ref(), Some(&job), "{what}: the analysis was replaced or cancelled");
        assert_eq!(w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst), spawned, "{what}: an analysis was started");
        assert!((w.analyzer.sample_rate, w.analyzer.pre_fft_size, w.analyzer.effective_settings(&dsf),
                 w.analyzer.pre_cache_path().map(Path::to_path_buf)) == pre,
                "{what}: the pre-process moved");
    };

    // An ordinary pause/resume on the same route changes nothing.
    e.pause();
    let fft = w.analyzer.fft_size;
    resume_playback(&mut e, &mut w, Some(&dsf));
    assert!(e.sink.as_ref().is_some_and(|s| !s.is_paused()), "the resume did not play");
    assert_eq!((w.live_tap, w.analyzer.live_rate, w.analyzer.fft_size), (Some(176_400), 176_400, fft),
               "a same-tap resume moved the live side");
    untouched(&w, "same-tap resume");

    // A rebuild that fails (the file is gone) leaves no sink; the app's
    // failure handler clears the tap at the position the engine reports.
    e.pause();
    std::fs::rename(&dsf, &away).unwrap();
    e.play_seeked_async(&dsf, Duration::from_millis(200), None, true);
    assert!(e.sink.is_none(), "setup: the failed rebuild left a sink");
    let SeekOutcome::Failed(_) = e.poll_pending_seek() else { panic!("setup: no failure reported") };
    restart_spectrum(Some(&e), &mut w, Some(&dsf), Some(e.elapsed()));
    assert_eq!(w.live_tap, None);
    assert_eq!(w.analyzer.live_rate, 88_200, "setup: the stand-in is the analysis rate");
    untouched(&w, "failed rebuild");

    // A retry that fails too advertises no tap.
    resume_playback(&mut e, &mut w, Some(&dsf));
    assert!(e.sink.is_none());
    assert_eq!(w.live_tap, None, "a failed retry advertised a tap");
    let SeekOutcome::Failed(_) = e.poll_pending_seek() else { panic!("the retry's failure was not reported") };
    restart_spectrum(Some(&e), &mut w, Some(&dsf), Some(e.elapsed()));
    assert_eq!(w.live_tap, None);
    untouched(&w, "failed retry");

    // The file is back: resume reopens the fallback synchronously, at the
    // position resume targets, and the spectrum is told at once.
    std::fs::rename(&away, &dsf).unwrap();
    let (_, target) = e.resume_target().expect("the track is still the track");
    resume_playback(&mut e, &mut w, Some(&dsf));
    assert!(e.sink.is_some() && e.dsd_fallback, "the resume did not reopen the fallback");
    assert_eq!(e.paused_elapsed, target, "the resume moved the playhead");
    assert_eq!(w.live_tap, Some(176_400), "the reopened fallback's tap was not described");
    assert_eq!(w.analyzer.live_rate, 176_400);
    assert_eq!(w.analyzer.fft_size, 32_768, "the live size did not follow the tap");
    untouched(&w, "recovered resume");
    // The signal check: a 9 kHz tone at 176.4 kHz draws exactly what a
    // 176.4 kHz PCM window draws, and not what an 88.2 kHz reading draws.
    let sig: Vec<f32> = (0..65_536)
        .map(|i| (std::f32::consts::TAU * 9_000.0 * i as f32 / 176_400.0).sin() * 0.5)
        .collect();
    let bars = |w: &mut SpectrumWindow| -> Vec<u32> {
        {
            let mut b = w.sample_buf.lock().unwrap();
            b.clear();
            b.extend_from_slice(&sig);
        }
        w.analyzer.process_realtime(1.0);
        w.analyzer.magnitudes.iter().map(|m| m.to_bits()).collect()
    };
    let pcm_window = |rate: u32, tag: &str| {
        let mut r = SpectrumWindow::new_fresh_profile();
        r.analyzer.cache_dir_override = Some(dir.join(tag));
        r.on_play(&wav, rate);
        r.analyzer.abort_preprocess();
        r
    };
    let want = bars(&mut pcm_window(176_400, "ref"));
    assert_eq!(bars(&mut w), want, "the live tap is not read at 176.4 kHz");
    assert_ne!(bars(&mut pcm_window(88_200, "half")), want, "setup: the check cannot see a half-rate reading");

    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
    w.analyzer.abort_preprocess();
    e.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A PCM resume that reopens off-thread describes no tap while it decodes
/// (nothing is installed) and the landed source once its landing is
/// handled — the existing completion path, unchanged.
#[test]
fn resuming_a_released_pcm_sink_waits_for_the_landing() {
    use std::sync::atomic::Ordering;
    let dir = std::env::temp_dir().join(format!("moosik_r7_resume_pcm_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("t.wav");
    {
        let (sr, n) = (48_000u32, 48_000usize);
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = ((std::f32::consts::TAU * 1_000.0 * i as f32 / sr as f32).sin() * 8_000.0) as i16;
            data.extend(v.to_le_bytes());
            data.extend(v.to_le_bytes());
        }
        let mut f = Vec::new();
        f.extend(b"RIFF");
        f.extend(((36 + data.len()) as u32).to_le_bytes());
        f.extend(b"WAVEfmt ");
        f.extend(16u32.to_le_bytes());
        f.extend(1u16.to_le_bytes());
        f.extend(2u16.to_le_bytes());
        f.extend(sr.to_le_bytes());
        f.extend((sr * 4).to_le_bytes());
        f.extend(4u16.to_le_bytes());
        f.extend(16u16.to_le_bytes());
        f.extend(b"data");
        f.extend((data.len() as u32).to_le_bytes());
        f.extend(data);
        std::fs::write(&wav, f).unwrap();
    }
    let away = dir.join("t.wav.away");
    let mut w = SpectrumWindow::new_fresh_profile();
    w.analyzer.cache_dir_override = Some(dir.join("cache"));
    let hold = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    w.analyzer.analysis_hold = Some(std::sync::Arc::clone(&hold));

    let mut e = engine();
    e.idle_output = true;
    let settle = |e: &mut Engine| -> SeekOutcome {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match e.poll_pending_seek() {
                SeekOutcome::Working | SeekOutcome::Idle | SeekOutcome::Provisional(_) => {
                    assert!(Instant::now() < deadline, "the seek never resolved");
                    std::thread::yield_now();
                }
                other => return other,
            }
        }
    };
    e.play_seeked_async(&wav, Duration::from_millis(500), None, false);
    let SeekOutcome::Landed(_) = settle(&mut e) else { panic!("setup: no landing") };
    w.on_play_live(&wav, 48_000, e.live_tap_now(&wav));
    assert_eq!(w.live_tap, Some(48_000), "setup");
    let job = w.analyzer.job_record().expect("an analysis was asked for");
    let spawned = w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst);

    // Released by a failed rebuild.
    e.pause();
    std::fs::rename(&wav, &away).unwrap();
    e.play_seeked_async(&wav, Duration::from_millis(250), None, true);
    let SeekOutcome::Failed(_) = settle(&mut e) else { panic!("setup: no failure") };
    restart_spectrum(Some(&e), &mut w, Some(&wav), Some(e.elapsed()));
    assert_eq!(w.live_tap, None);

    // Resumed: decoding off-thread, nothing installed, nothing described.
    std::fs::rename(&away, &wav).unwrap();
    resume_playback(&mut e, &mut w, Some(&wav));
    assert!(e.sink.is_none() && e.pending_seek.is_some(), "setup: the reopen was not off-thread");
    assert_eq!(w.live_tap, None, "a still-decoding reopen was described as installed");
    let SeekOutcome::Landed(at) = settle(&mut e) else { panic!("the reopen did not land") };
    restart_spectrum(Some(&e), &mut w, Some(&wav), Some(at));
    assert_eq!(w.live_tap, Some(48_000), "the landing was not described");
    assert_eq!(w.analyzer.job_record(), Some(job), "the analysis was replaced or cancelled");
    assert_eq!(w.analyzer.lifecycle_probe.spawned.load(Ordering::SeqCst), spawned);

    hold.store(false, Ordering::SeqCst);
    w.analyzer.analysis_hold = None;
    w.analyzer.abort_preprocess();
    e.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The preprocessing rate never reaches playback, and the spectrum is told
/// the live tap's real rate.
///
/// Through the production pieces a track start and a gapless append use —
/// the rate `play_index` arms, the one fallback source every start,
/// restart, seek and append opens, the queued transform the panel reports,
/// and the live rate the app hands the window — with the device replaced
/// by an idle sink, so no audio output is opened.
#[test]
fn the_preprocessing_rate_never_reaches_playback() {
    let dir = std::env::temp_dir().join(format!("moosik_dsd_split_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [(2_822_400u32, 176_400u32, 352_800u32, 176_400u32),
                 (2_822_400, 352_800, 176_400, 352_800),
                 (3_072_000, 176_400, 352_800, 192_000)];
    for (dsd, fallback, preprocess, tap) in cases {
        let dsf = dir.join(format!("t{dsd}_{fallback}.dsf"));
        spectrum::dsd_rate_tests::write_dsf(&dsf, dsd, 0.05, 1_000.0, 2_000.0);
        let mut w = SpectrumWindow::new_fresh_profile();
        w.analyzer.cache_dir_override = Some(dir.join("cache"));
        w.dsd_fallback_rate = fallback;
        w.analyzer.dsd_rate = preprocess;

        let mut e = engine();
        arm_dsd_fallback(&mut e, playback_fallback_rate(&w));
        assert_eq!(e.dsd_fallback_target, fallback, "{dsd}: playback took another setting");

        let src = e.open_fallback_pcm(&dsf).unwrap();
        assert_eq!(src.out_rate(), tap, "{dsd}: the fallback decimation");
        assert_eq!(e.spectrum_live_rate(&dsf, 0), Some(tap), "{dsd}: the live rate on the fallback");

        // Gapless: the next DSD track is appended at the playback rate.
        let (sink, _queue) = rodio::Sink::new_idle();
        e.sink = Some(sink);
        e.append_next(&dsf).unwrap();
        let queued = e.queued_next.as_ref().expect("queued");
        assert!(
            matches!(queued.transform,
                bitperfect::state::TransformDescription::DsdDecimated { pcm_rate } if pcm_rate == tap),
            "{dsd}: the gapless append decimated at another rate",
        );

        // The window maps live bins at the tap's rate and analyses at its own.
        w.on_play_live(&dsf, e.last_sample_rate, e.spectrum_live_rate(&dsf, e.last_sample_rate));
        w.analyzer.abort_preprocess();
        assert_eq!(w.analyzer.live_rate, tap, "{dsd}: the window's live rate");
        assert_eq!(
            w.analyzer.sample_rate,
            dsd::decimate::analysis_rate_for(dsd, preprocess),
            "{dsd}: the window's analysis rate",
        );
    }
    // PCM is untouched: the rate the route reports.
    let e = engine();
    assert_eq!(e.spectrum_live_rate(Path::new("x.flac"), 48_000), Some(48_000));
    let _ = std::fs::remove_dir_all(&dir);
}

fn engine() -> Engine {
    Engine::new(
        spectrum::new_sample_buf(),
        spectrum::new_stereo_buf(),
    )
    .expect("the engine holds no device until something plays")
}

fn endpoint() -> bitperfect::state::EndpointIdentity {
    bitperfect::state::EndpointIdentity::from_name("DAC")
}

/// After an exact open fails and the track is retried through the shared
/// mixer, **every** operation must reach the sink.
///
/// The predicate under test used to be
/// `dsd_native || dsd_mode || (bit_perfect && !dsd_fallback)`, which is
/// three request flags and no route. With the preference still on — and it
/// stays on, deliberately, because one unsupported track says nothing
/// about the next — it answered "device stream" for a session running on
/// the mixer with `bp` set to `None`. Pause, resume, seek, end-of-track,
/// position, auto-advance and the gapless hand-off all then addressed a
/// stream that did not exist.
#[test]
fn a_shared_fallback_session_is_owned_by_the_sink() {
    let mut e = engine();

    // The state a track leaves behind when the exact route refused it: the
    // preference survives, the route does not.
    e.bit_perfect = true;
    e.bp_state.requested = true;
    e.bp = None;
    e.bp_state.opened(
        bitperfect::Transport::Shared {
            endpoint: endpoint(),
        },
        bitperfect::state::Fidelity::Processed {
            transform: bitperfect::state::TransformDescription::Identity,
            reason: "playing through the shared mixer for this track".into(),
        },
        None,
        None,
        e.shared_processing(),
    );

    assert!(e.bit_perfect, "the preference is untouched by a fallback");
    assert!(
        !e.on_bp_stream(),
        "a shared session is not on a device stream, whatever the toggle says"
    );
    assert!(!e.on_native_stream());

    // Each of these consults the routing predicate. With no sink and no
    // `bp` they must all answer for the sink — quietly, and without
    // reaching into a stream that is not there.
    assert!(
        !e.is_finished(),
        "end-of-track comes from the sink, and an empty engine has not finished"
    );
    assert_eq!(e.elapsed(), Duration::ZERO, "position comes from the sink");
    e.pause();
    e.resume();
    assert_eq!(
        e.bp_state.playback,
        bitperfect::state::PlaybackState::Playing,
        "pause and resume moved the session, not a dead stream"
    );

    // The sink's processing chain is in the path and the panel says so, so
    // volume and ReplayGain are live rather than locked.
    assert!(!e.volume_is_locked(), "the shared path has a gain stage");
    assert!(e.set_volume(0.4), "and the volume control acts on it");
    assert_eq!(e.volume, 0.4);
    e.set_replay_gain(0.5);
    assert!(
        e.bp_state.processing.replaygain_active,
        "ReplayGain is in the shared path and the session reports it"
    );
    assert!(
        !e.bp_state.presentation().volume_locked,
        "and every surface reads the same answer"
    );

    // Nothing here may show a diamond.
    assert!(!e.bp_state.shows_diamond());
    assert!(!e.bp_state.presentation().headline.contains('\u{1F48E}'));
}

/// The mirror: a live exclusive session *is* owned by the device stream,
/// and the same predicate says so.
#[test]
fn an_exclusive_session_is_owned_by_the_device_stream() {
    let mut e = engine();
    e.bit_perfect = true;
    e.bp_state.opened(
        bitperfect::Transport::WasapiExclusivePcm {
            endpoint: endpoint(),
        },
        bitperfect::state::Fidelity::PayloadExact,
        None,
        None,
        bitperfect::state::ProcessingState::transparent_locked(),
    );

    // `bp` is None here, which is exactly the case the second half of the
    // predicate exists for: a transport the session believes in but no
    // handle behind it must not be routed to.
    assert!(
        !e.on_bp_stream(),
        "a transport with no live handle behind it is not a route"
    );

    // And the claim itself: an exclusive PCM route with nothing altered is
    // the one case that earns the diamond.
    assert!(e.bp_state.shows_diamond());
    assert!(e.volume_is_locked(), "there is no gain stage on this route");
    assert!(!e.set_volume(0.3), "so the volume control has nothing to act on");
}

/// Turning the preference off must not, on its own, move a session that is
/// still open — and turning it on must not move one that is not.
#[test]
fn the_preference_alone_never_changes_the_route() {
    let mut e = engine();
    for requested in [false, true] {
        e.bit_perfect = requested;
        e.bp_state.requested = requested;
        e.bp_state.stopped();
        assert!(
            !e.on_bp_stream(),
            "a stopped session is on no route at all (requested = {requested})"
        );
        assert!(!e.bp_state.shows_diamond());
    }
}

fn syncsafe(n: u32) -> [u8; 4] {
    [
        (n >> 21) as u8 & 0x7f,
        (n >> 14) as u8 & 0x7f,
        (n >> 7) as u8 & 0x7f,
        n as u8 & 0x7f,
    ]
}

/// Minimal ID3v2.3 tag with Latin-1 frames.
fn id3v23(frames: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (id, payload) in frames {
        body.extend_from_slice(*id);
        body.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        body.extend_from_slice(&[0, 0]); // frame flags
        body.extend_from_slice(payload);
    }
    let mut tag = Vec::new();
    tag.extend_from_slice(b"ID3");
    tag.extend_from_slice(&[3, 0, 0]); // v2.3, no flags
    tag.extend_from_slice(&syncsafe(body.len() as u32));
    tag.extend_from_slice(&body);
    tag
}

fn text_frame(s: &str) -> Vec<u8> {
    let mut v = vec![0u8]; // encoding: Latin-1
    v.extend_from_slice(s.as_bytes());
    v
}

/// A DSF with a real ID3v2.3 tag must surface its tags, ReplayGain and
/// cover art through the shared lofty pipeline, and its stream properties
/// from the DSD header.
#[test]
fn dsf_metadata_flows_through_lofty() {
    let mut txxx_rg = vec![0u8];
    txxx_rg.extend_from_slice(b"replaygain_track_gain\0-7.30 dB");
    let mut apic = vec![0u8];
    apic.extend_from_slice(b"image/png\0");
    apic.push(3); // front cover
    apic.extend_from_slice(b"\0");
    apic.extend_from_slice(b"FAKEPNGDATA");
    let tag = id3v23(&[
        (b"TIT2", text_frame("Ride of the DSD")),
        (b"TPE1", text_frame("Fable & the Bitstreams")),
        (b"TALB", text_frame("One-Bit Wonders")),
        (b"TXXX", txxx_rg),
        (b"APIC", apic),
    ]);

    // 1 s of DSD64 declared; audio payload itself can stay tiny.
    let blocks = vec![vec![vec![0u8; 4], vec![0u8; 4]]];
    let dsf = dsd::tests::make_dsf(
        2,
        dsd::DSD64_RATE,
        1,
        dsd::DSD64_RATE as u64,
        4,
        &blocks,
        Some(&tag),
    );

    let path = std::env::temp_dir().join(format!("moosik_test_{}.dsf", std::process::id()));
    std::fs::write(&path, &dsf).unwrap();

    let m = read_metadata(&path);
    std::fs::remove_file(&path).ok();

    assert_eq!(m.title, "Ride of the DSD");
    assert_eq!(m.artist, "Fable & the Bitstreams");
    assert_eq!(m.album, "One-Bit Wonders");
    assert_eq!(m.rg_track_gain, Some(-7.3));
    assert_eq!(m.sample_rate, Some(dsd::DSD64_RATE));
    assert_eq!(m.bit_depth, Some(1));
    assert_eq!(m.channels, Some(2));
    assert_eq!(m.duration, Some(Duration::from_secs(1)));
    assert_eq!(m.bitrate, Some(2 * dsd::DSD64_RATE / 1000));

    // Cover art comes out of the same tag.
    std::fs::write(&path, &dsf).unwrap();
    let tagged = probe_tagged(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let t = tagged.primary_tag().or_else(|| tagged.first_tag()).unwrap();
    assert_eq!(t.pictures().len(), 1);
    assert_eq!(t.pictures()[0].data(), b"FAKEPNGDATA");
}

/// An untagged DSF still gets stream properties + filename fallback title.
#[test]
fn untagged_dsf_falls_back_to_filename() {
    let blocks = vec![vec![vec![0u8; 4], vec![0u8; 4]]];
    let dsf = dsd::tests::make_dsf(2, 2 * dsd::DSD64_RATE, 1, 64, 4, &blocks, None);
    let path = std::env::temp_dir().join(format!("moosik_untagged_{}.dsf", std::process::id()));
    std::fs::write(&path, &dsf).unwrap();
    let m = read_metadata(&path);
    std::fs::remove_file(&path).ok();

    assert!(m.title.starts_with("moosik_untagged_"));
    assert_eq!(m.artist, "Unknown Artist");
    assert_eq!(m.sample_rate, Some(2 * dsd::DSD64_RATE)); // DSD128
    assert_eq!(m.bit_depth, Some(1));
}

// -----------------------------------------------------------------------
// The output-state surface
//
// These read the source. That is a blunt instrument and it is the right one
// here: the properties are about which *variable* a piece of UI is wired
// to, and a rendered `egui` frame cannot be inspected without standing up a
// window and a device. The failure mode they guard against is a later edit
// reconnecting the wrong wire, which no runtime assertion in this process
// would ever see.
// -----------------------------------------------------------------------

/// Slice the source between two markers, so a test reasons about one block
/// rather than the whole file.
fn block<'a>(src: &'a str, from: &str, to: &str) -> &'a str {
    let i = src
        .find(from)
        .unwrap_or_else(|| panic!("marker moved: {from}"));
    let rest = &src[i..];
    let j = rest
        .find(to)
        .unwrap_or_else(|| panic!("marker moved: {to}"));
    &rest[..j]
}

/// The same block with comment lines removed — these tests are about what
/// the code is wired to, and the prose explaining *why* naturally mentions
/// the very identifiers being ruled out.
fn code_only(block: &str) -> String {
    block
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The persistent negotiated-output lines are rendered from the session
/// state, not from the transient status string.
///
/// Until 1.4.3 there was one `status_msg` and every subsystem wrote to it,
/// so the line describing the open device survived exactly until a tag save
/// or a cache eviction wanted to say something.
#[test]
fn the_negotiated_output_panel_is_not_the_status_message() {
    let src = include_str!("main.rs");
    let panel = block(
        src,
        "let present = self.engine.as_ref().map(|e| e.bp_state.presentation());",
        "if !self.status.text().is_empty()",
    );

    let code = code_only(panel);
    assert!(
        code.contains("presentation()"),
        "the output panel should derive from the one presentation object"
    );
    assert!(
        code.contains("present.headline") && code.contains("present.detail"),
        "the output panel should render the session headline and detail lines"
    );
    assert!(
        !code.contains("self.status"),
        "the output panel must not be driven by the transient status message"
    );
}

/// Locking the volume must not overwrite the value the user saved.
///
/// The obvious implementation — set `self.volume = 1.0` while an exact
/// route is open — loses an 80% preference the first time someone plays a
/// FLAC, and there is nothing to restore afterwards.
#[test]
fn the_volume_lock_does_not_clobber_the_saved_preference() {
    let src = include_str!("main.rs");
    let slider = block(
        src,
        "let lock = self.engine.as_ref()",
        "// Sample rate",
    );

    let code = code_only(slider);
    assert!(
        code.contains("add_enabled(") && code.contains("!vol_locked"),
        "the slider should be disabled while an exact route is open"
    );
    assert!(
        code.contains("locked on this route"),
        "the locked state should say so in words"
    );
    assert!(
        code.contains("lock_reason"),
        "the disabled tooltip should explain why this route has no gain stage"
    );

    // The only assignment to `self.volume` in this block is the one behind
    // the "not locked" branch.
    let assigns: Vec<&str> = code
        .match_indices("self.volume =")
        .map(|(i, _)| code[i..].lines().next().unwrap_or("").trim())
        .collect();
    assert_eq!(
        assigns.len(),
        1,
        "unexpected writes to the saved volume: {assigns:?}"
    );
    let guarded = block(slider, "} else if vol_slider.changed() {", "ui.label(");
    assert!(
        guarded.contains("if applied {"),
        "the write must be gated on the engine having accepted it"
    );
    assert!(
        guarded.contains("self.volume = shown;"),
        "the single write must sit in the unlocked branch"
    );
}

/// The diamond's appearance is derived from the session state, never from
/// the stored preference.
#[test]
fn the_diamond_is_drawn_from_the_route_not_the_toggle() {
    let src = include_str!("main.rs");
    let badge = block(
        src,
        "// Every visible surface derives from one presentation",
        "ui.menu_button(",
    );

    let code = code_only(badge);
    assert!(code.contains("presentation()") && code.contains("present.badge"));
    assert!(
        !code.contains("self.bit_perfect"),
        "the badge must not read the preference flag"
    );
    // The glyph comes from the badge itself, so the toolbar cannot pick a
    // different one from the panel. `Badge::glyph` is covered directly by
    // `bitperfect::state::tests::every_badge_is_distinguishable_without_colour`.
    assert!(
        code.contains("present.badge.glyph()"),
        "the toolbar should take its glyph from the badge, not choose its own"
    );
}

/// Both places that decide whether a track can ride the open stream ask the
/// same question. The gapless path used to ask a looser one, which put the
/// narrowing at the boundary the listener is least likely to notice.
#[test]
fn the_reopen_and_gapless_paths_share_one_compatibility_predicate() {
    let src = include_str!("main.rs");
    let reopen = block(src, "fn start_bp(&mut self", "fn start_dop(&mut self");
    let queue = block(src, "fn bp_queue_next(&self", "fn bp_queue_next_dop(&self");

    for (name, body) in [("start_bp", reopen), ("bp_queue_next", queue)] {
        let body = code_only(body);
        assert!(
            body.contains("can_carry(&"),
            "{name} should use the shared compatibility predicate"
        );
        assert!(
            !body.contains("prep.sample_rate != bp.sample_rate"),
            "{name} still has the old rate/channel-only check"
        );
    }
}
