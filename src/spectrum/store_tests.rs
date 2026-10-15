//! Store protocol tests. Every test owns a private directory under the test
//! process's TEMP (the custody wrapper gives each invocation its own), and
//! never touches any other path. Interleavings are forced with the store's
//! named seams, not with sleeps.
use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

struct Fx {
    root: PathBuf,
    store: Arc<Store>,
    src: PathBuf,
}

impl Fx {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "moosik_store_{tag}_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let src = root.join("track.flac");
        fs::write(&src, pattern(3 << 20, 1)).unwrap();
        let store = Arc::new(Store::open(&root.join("v2")).unwrap());
        Self { root, store, src }
    }

    fn unvalidated(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("moosik_store_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        test_scope::force_unvalidated(&root.join("v2"));
        let src = root.join("track.flac");
        fs::write(&src, pattern(1 << 20, 1)).unwrap();
        let store = Arc::new(Store::open(&root.join("v2")).unwrap());
        Self { root, store, src }
    }

    fn desc(&self) -> Descriptor {
        Descriptor { record: "moosik-analysis/1;bars=8;sr=44100;hop=245".into(), source: SourceSnapshot::of_path(&self.src).unwrap() }
    }

    fn stem(&self) -> String {
        stem(&self.src, "settings-A")
    }

    fn publish(&self, mix: &[u8], lr: Option<&[u8]>) -> Result<Published, PublishError> {
        let mut src = OpenSource::open(&self.src).unwrap();
        let desc = Descriptor { record: self.desc().record, source: src.snapshot.clone() };
        self.store.publish(&desc, "settings-A", [7; 32], mix, lr, &mut src, &|| false)
    }

    fn installed(&self, p: Published) -> Installed {
        Installed {
            store: self.store.clone(),
            stem: p.stem,
            generation: p.generation,
            envelope: p.envelope,
            has_lr: p.has_lr,
            pin: p.pin.unwrap(),
        }
    }

    /// What a reader sees now: the Mix payload and, if present, the L/R
    /// payload of the head generation, read under its pin.
    fn read_pair(&self) -> Result<(Vec<u8>, Option<Vec<u8>>), Miss> {
        let (pin, env, has_lr, mix) = self.store.lookup(&self.stem(), &self.desc())?;
        let inst = Installed { store: self.store.clone(), stem: self.stem(), generation: pin.generation, envelope: env, has_lr, pin };
        let lr = if has_lr { Some(self.store.read_lr(&inst)?.to_vec()) } else { None };
        Ok((mix.to_vec(), lr))
    }

    /// Change the source in place, keeping its size: size alone cannot catch it.
    fn edit_in_place(&self) {
        let mut b = fs::read(&self.src).unwrap();
        b[0] ^= 0xff;
        fs::write(&self.src, b).unwrap();
    }
}

impl Drop for Fx {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n).map(|i| (i as u32).wrapping_mul(2654435761).rotate_left(seed as u32) as u8 ^ seed).collect()
}

fn coord_free(store: &Store, stem: &str) -> bool {
    let f = OpenOptions::new().read(true).write(true).open(store.dir().join(format!("{stem}.coord"))).unwrap();
    let ok = f.try_lock().is_ok();
    let _ = f.unlock();
    ok
}

// ── identity ──

#[test]
fn sha256_is_the_published_function() {
    assert_eq!(hex(&sha256(&[b"abc"])), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(hex(&sha256(&[b"a", b"bc"])), hex(&sha256(&[b"abc"])));
}

#[test]
fn the_sample_policy_reads_the_whole_small_file_and_sixteen_windows_of_a_large_one() {
    let fx = Fx::new("sample");
    let a = SourceSnapshot::of_path(&fx.src).unwrap();
    // A byte in a sampled window (the very first) changes the digest.
    fx.edit_in_place();
    let b = SourceSnapshot::of_path(&fx.src).unwrap();
    assert_ne!(a.sample, b.sample);
    // A byte outside every window, with size and time put back, does not:
    // the declared limitation of a sampled key (D3).
    let mtime = fs::metadata(&fx.src).unwrap().modified().unwrap();
    let mut bytes = fs::read(&fx.src).unwrap();
    let gap = (SAMPLE_WINDOW + 7) as usize; // between window 0 and window 1
    assert!(gap < ((bytes.len() as u64 - SAMPLE_WINDOW) / (SAMPLE_WINDOWS - 1)) as usize);
    bytes[gap] ^= 0x55;
    fs::write(&fx.src, &bytes).unwrap();
    File::options().write(true).open(&fx.src).unwrap().set_modified(mtime).unwrap();
    let c = SourceSnapshot::of_path(&fx.src).unwrap();
    assert_eq!((c.size, c.mtime_ns, c.sample), (b.size, b.mtime_ns, b.sample), "unsampled edit with size and time kept: undetected");
    // A small file is digested whole: any byte counts.
    let small = fx.root.join("small.wav");
    fs::write(&small, pattern(5000, 3)).unwrap();
    let s1 = SourceSnapshot::of_path(&small).unwrap();
    let mut sb = fs::read(&small).unwrap();
    sb[2500] ^= 1;
    fs::write(&small, &sb).unwrap();
    let t = fs::metadata(&small).unwrap().modified().unwrap();
    File::options().write(true).open(&small).unwrap().set_modified(t).unwrap();
    assert_ne!(SourceSnapshot::of_path(&small).unwrap().sample, s1.sample);
}

#[test]
fn descriptors_are_deterministic_and_carry_every_field() {
    let fx = Fx::new("desc");
    let d = fx.desc();
    assert_eq!(d.canonical(), fx.desc().canonical());
    assert_eq!(Descriptor::record_in(&d.canonical()), Some(d.record.as_str()));
    let mut e = d.clone();
    e.source.mtime_ns += 1;
    assert_ne!(d.digest(), e.digest());
    let mut e = d.clone();
    e.record.push_str(";x");
    assert_ne!(d.digest(), e.digest());
    // The stem names path and settings, not content.
    assert_eq!(stem(&fx.src, "s"), stem(&fx.src, "s"));
    assert_ne!(stem(&fx.src, "s"), stem(&fx.src, "t"));
    assert!(stem(&fx.src, "s").starts_with(&path_prefix(&fx.src)));
}

#[test]
fn malformed_envelopes_are_refused_whole() {
    let e = Envelope { kind: Kind::Mix, generation: 5, descriptor: sha256(&[b"d"]), decoded: [1; 32], mix: sha256(&[b"pay"]), payload: sha256(&[b"pay"]) };
    let good = encode_envelope(&e, "d", b"pay");
    assert!(decode_envelope(&good).is_ok());
    assert_eq!(decode_envelope(&good[..good.len() - 1]).unwrap_err(), EnvelopeError::Length { declared: Some(good.len()), actual: good.len() - 1 });
    let mut longer = good.clone();
    longer.push(0);
    assert!(matches!(decode_envelope(&longer), Err(EnvelopeError::Length { .. })));
    let mut flip = good.clone();
    *flip.last_mut().unwrap() ^= 1;
    assert_eq!(decode_envelope(&flip).unwrap_err(), EnvelopeError::PayloadDigest);
    let mut d = good.clone();
    d[HEADER] ^= 1;
    assert_eq!(decode_envelope(&d).unwrap_err(), EnvelopeError::DescriptorDigest);
    let mut v = good.clone();
    v[8] = 2;
    assert_eq!(decode_envelope(&v).unwrap_err(), EnvelopeError::Version(2));
    let mut r = good.clone();
    r[11] = 1;
    assert_eq!(decode_envelope(&r).unwrap_err(), EnvelopeError::Reserved);
    // An overflowing declared payload length is a length error, not a panic.
    let mut big = good.clone();
    big[160..168].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(decode_envelope(&big), Err(EnvelopeError::Length { .. })));
    // A Mix whose binding digest is not its own payload is refused.
    let bad = encode_envelope(&Envelope { mix: [9; 32], ..e }, "d", b"pay");
    assert_eq!(decode_envelope(&bad).unwrap_err(), EnvelopeError::MixBinding);
}

/// The payload is decoded while its digest is checked. A decode of a payload
/// whose digest does not match is never handed out, even when it succeeded.
#[test]
fn a_decode_beside_the_digest_is_dropped_when_the_digest_fails() {
    let fx = Fx::new("overlap");
    let inst = fx.installed(fx.publish(b"mix-1", Some(b"lr-1")).unwrap());
    let (.., mix) = fx.store.lookup_with(&fx.stem(), &fx.desc(), |b| b.to_vec()).unwrap();
    assert_eq!(mix, b"mix-1");
    assert_eq!(fx.store.read_lr_with(&inst, |b| b.to_vec()).unwrap().1, b"lr-1");
    // One payload byte changed in each file; the decoders here accept anything.
    for ext in ["mix", "lr"] {
        let path = fx.store.gen_path(&fx.stem(), inst.generation, ext);
        let mut raw = fs::read(&path).unwrap();
        *raw.last_mut().unwrap() ^= 1;
        fs::write(&path, raw).unwrap();
    }
    let ran = AtomicBool::new(false);
    let r = fx.store.lookup_with(&fx.stem(), &fx.desc(), |b| {
        ran.store(true, Ordering::SeqCst);
        b.to_vec()
    });
    assert!(ran.swap(false, Ordering::SeqCst), "the Mix decode did not run beside the digest");
    assert!(matches!(r.as_ref().err(), Some(Miss::Damaged(m)) if m.contains("PayloadDigest")), "{:?}", r.err());
    let r = fx.store.read_lr_with(&inst, |b| {
        ran.store(true, Ordering::SeqCst);
        b.to_vec()
    });
    assert!(ran.load(Ordering::SeqCst), "the channel decode did not run beside the digest");
    assert!(matches!(r.as_ref().err(), Some(Miss::Damaged(m)) if m.contains("PayloadDigest")), "{:?}", r.err());
}

// ── publication ──

#[test]
fn publication_round_trips_and_hands_its_pin_to_installation() {
    let fx = Fx::new("roundtrip");
    let p = fx.publish(b"mix-1", Some(b"lr-1")).unwrap();
    assert!(p.has_lr);
    let inst = fx.installed(p);
    assert_eq!(fx.read_pair().unwrap(), (b"mix-1".to_vec(), Some(b"lr-1".to_vec())));
    assert_eq!(&*fx.store.read_lr(&inst).unwrap(), b"lr-1");
    // The installed generation stays pinned: neither collection nor
    // eviction deletes it.
    let r = fx.store.evict_stem(&inst.stem);
    assert_eq!(r.skipped_pinned, vec![inst.generation]);
    assert!(inst.mix_path().exists() && inst.lr_path().exists());
    drop(inst);
    let r = fx.store.evict_stem(&fx.stem());
    assert_eq!(r.deleted.len(), 1);
    assert_eq!(fx.store.bytes(), 0);
    // Coordination files stay (D5).
    assert!(fx.store.dir().join(format!("{}.coord", fx.stem())).exists());
    assert!(fx.store.dir().join(format!("{}.pub", fx.stem())).exists());
}

#[test]
fn a_changed_source_misses_and_keeps_its_stem() {
    let fx = Fx::new("srcchange");
    drop(fx.publish(b"m", None).unwrap());
    fx.edit_in_place();
    assert_eq!(fx.store.lookup(&fx.stem(), &fx.desc()).unwrap_err(), Miss::DescriptorMismatch);
}

#[test]
fn the_source_guard_rejects_an_in_place_edit_and_keeps_the_old_pair() {
    let fx = Fx::new("guard");
    drop(fx.publish(b"old-mix", Some(b"old-lr")).unwrap());
    let free_at_rejection = Arc::new(Mutex::new(None));
    let (store, stem, src) = (fx.store.clone(), fx.stem(), fx.src.clone());
    let seen = free_at_rejection.clone();
    let _h = seam::hook(fx.store.dir(), move |point, s| {
        if point == "pub.before_head" {
            let mut b = fs::read(&src).unwrap();
            b[0] ^= 0xff;
            fs::write(&src, b).unwrap();
        }
        if point == "pub.guard_rejected" {
            *seen.lock().unwrap() = Some(coord_free(&store, s));
        }
        let _ = &stem;
    });
    let mut src = OpenSource::open(&fx.src).unwrap();
    let desc = Descriptor { record: fx.desc().record, source: src.snapshot.clone() };
    let before = fx.store.head_state(&fx.stem());
    let r = fx.store.publish(&desc, "settings-A", [1; 32], b"new-mix", Some(b"new-lr"), &mut src, &|| false);
    assert_eq!(r.unwrap_err(), PublishError::SourceChanged { cleanup_pending: false });
    assert_eq!(*free_at_rejection.lock().unwrap(), Some(true), "S.coord released before abandon");
    assert_eq!(fx.store.head_state(&fx.stem()), before, "nothing committed");
    assert_eq!(fx.store.generations(&fx.stem()).len(), 1, "the rejected generation is gone");
    // The old pair is still exactly the old pair (read with the old snapshot's
    // record: the source changed, so a fresh lookup misses).
    let HeadState::Valid(g) = before else { panic!() };
    let raw = fs::read(fx.store.gen_path(&fx.stem(), g, "mix")).unwrap();
    assert_eq!(decode_envelope(&raw).unwrap().2, b"old-mix");
}

#[test]
fn a_checker_failure_commits_nothing() {
    let fx = Fx::new("checkfail");
    let _f = seam::fault(fx.store.dir(), |point, _| (point == "source.check").then_some(io::ErrorKind::Other));
    let r = fx.publish(b"m", None).unwrap_err();
    assert!(matches!(r, PublishError::SourceCheck { cleanup_pending: false, .. }), "{r:?}");
    assert_eq!(fx.store.head_state(&fx.stem()), HeadState::Absent);
    assert!(fx.store.generations(&fx.stem()).is_empty());
}

#[test]
fn a_checker_failure_with_payload_removal_failure_leaves_cleanup_pending_for_collection() {
    let fx = Fx::new("checkfail2");
    let blocked = Arc::new(AtomicBool::new(true));
    let b = blocked.clone();
    let _f = seam::fault(fx.store.dir(), move |point, _| match point {
        "source.check" => Some(io::ErrorKind::Other),
        "payload.remove" if b.load(Ordering::SeqCst) => Some(io::ErrorKind::PermissionDenied),
        _ => None,
    });
    let r = fx.publish(b"m", Some(b"l")).unwrap_err();
    assert!(matches!(r, PublishError::SourceCheck { cleanup_pending: true, .. }), "{r:?}");
    assert_eq!(fx.store.head_state(&fx.stem()), HeadState::Absent);
    assert_eq!(fx.store.generations(&fx.stem()).len(), 1, "kept with its pin for a later collection");
    blocked.store(false, Ordering::SeqCst);
    let c = fx.store.collect(&fx.stem());
    assert_eq!(c.deleted.len(), 1, "{c:?}");
    assert!(fx.store.generations(&fx.stem()).is_empty());
}

#[test]
fn cancellation_at_the_commit_decision_commits_nothing() {
    let fx = Fx::new("cancel");
    drop(fx.publish(b"old", None).unwrap());
    let flag = Arc::new(AtomicBool::new(false));
    let f2 = flag.clone();
    let _h = seam::hook(fx.store.dir(), move |p, _| {
        if p == "pub.before_head" {
            f2.store(true, Ordering::SeqCst);
        }
    });
    let mut src = OpenSource::open(&fx.src).unwrap();
    let desc = Descriptor { record: fx.desc().record, source: src.snapshot.clone() };
    let r = fx.store.publish(&desc, "settings-A", [0; 32], b"new", None, &mut src, &|| flag.load(Ordering::SeqCst));
    assert_eq!(r.unwrap_err(), PublishError::Cancelled { cleanup_pending: false });
    assert_eq!(fx.read_pair().unwrap().0, b"old");
    assert_eq!(fx.store.generations(&fx.stem()).len(), 1);
}

#[test]
fn a_reader_between_payload_writes_and_the_head_commit_sees_the_old_pair() {
    let fx = Fx::new("midpub");
    drop(fx.publish(b"old-mix", Some(b"old-lr")).unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (s2, st, src, stem) = (seen.clone(), fx.store.clone(), fx.src.clone(), fx.stem());
    let _h = seam::hook(fx.store.dir(), move |p, _| {
        // Not at pub.after_check: that point is inside the publisher's own
        // S.coord section, so a reader there waits for the commit.
        if matches!(p, "pub.lr_written" | "pub.mix_written" | "pub.before_head") {
            let d = Descriptor { record: "moosik-analysis/1;bars=8;sr=44100;hop=245".into(), source: SourceSnapshot::of_path(&src).unwrap() };
            let (pin, env, has_lr, mix) = st.lookup(&stem, &d).unwrap();
            let inst = Installed { store: st.clone(), stem: stem.clone(), generation: pin.generation, envelope: env, has_lr, pin };
            s2.lock().unwrap().push((p.to_string(), mix.to_vec(), st.read_lr(&inst).unwrap().to_vec()));
        }
    });
    drop(fx.publish(b"new-mix", Some(b"new-lr")).unwrap());
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3, "{:?}", seen.iter().map(|s| &s.0).collect::<Vec<_>>());
    for (p, mix, lr) in seen.iter() {
        assert_eq!((mix.as_slice(), lr.as_slice()), (&b"old-mix"[..], &b"old-lr"[..]), "at {p}");
    }
    assert_eq!(fx.read_pair().unwrap(), (b"new-mix".to_vec(), Some(b"new-lr".to_vec())));
}

#[test]
fn two_publishers_for_one_stem_never_interleave() {
    let fx = Fx::new("twopub");
    let second = Arc::new(Mutex::new(None));
    let (s2, st, src) = (second.clone(), fx.store.clone(), fx.src.clone());
    let fired = Arc::new(AtomicBool::new(false));
    let _h = seam::hook(fx.store.dir(), move |p, _| {
        if p == "pub.mix_written" && !fired.swap(true, Ordering::SeqCst) {
            let mut o = OpenSource::open(&src).unwrap();
            let d = Descriptor { record: "moosik-analysis/1;bars=8;sr=44100;hop=245".into(), source: o.snapshot.clone() };
            *s2.lock().unwrap() = Some(st.publish(&d, "settings-A", [2; 32], b"second", None, &mut o, &|| false).map(|_| ()));
        }
    });
    drop(fx.publish(b"first", None).unwrap());
    assert_eq!(*second.lock().unwrap(), Some(Err(PublishError::Busy)));
    assert_eq!(fx.read_pair().unwrap().0, b"first");
}

// ── collection ──

#[test]
fn collection_revalidates_the_head_under_the_lock() {
    let fx = Fx::new("reval");
    drop(fx.publish(b"a", None).unwrap());
    // Generation b is written but uncommitted when the collector selects, so
    // it is a candidate. Before the collector's decision, b is committed as
    // the head and its publisher's pin released. Only re-reading the head
    // under the lock keeps the collector off it.
    let (written_tx, written_rx) = std::sync::mpsc::channel::<()>();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let go_rx = Mutex::new(go_rx);
    let written_tx = Mutex::new(written_tx);
    let done_rx = Mutex::new(done_rx);
    let go_tx = Mutex::new(Some(go_tx));
    let _h = seam::hook(fx.store.dir(), move |p, _| match p {
        "pub.before_head" => {
            written_tx.lock().unwrap().send(()).unwrap();
            go_rx.lock().unwrap().recv().unwrap();
        }
        "col.selected" => {
            if let Some(go) = go_tx.lock().unwrap().take() {
                go.send(()).unwrap();
                done_rx.lock().unwrap().recv().unwrap();
            }
        }
        _ => {}
    });
    let (st, src) = (fx.store.clone(), fx.src.clone());
    let publisher = std::thread::spawn(move || {
        let mut o = OpenSource::open(&src).unwrap();
        let d = Descriptor { record: "moosik-analysis/1;bars=8;sr=44100;hop=245".into(), source: o.snapshot.clone() };
        drop(st.publish(&d, "settings-A", [3; 32], b"b", None, &mut o, &|| false).unwrap()); // pin released here
        done_tx.send(()).unwrap();
    });
    written_rx.recv().unwrap();
    let r = fx.store.collect(&fx.stem());
    publisher.join().unwrap();
    drop(_h);
    assert!(r.errors.is_empty(), "{r:?}");
    assert_eq!(fx.read_pair().unwrap().0, b"b", "the head named at the decision survives");
    assert!(r.deleted.is_empty(), "a was the head at selection, b is the head now: {r:?}");
    assert_eq!(fx.store.collect(&fx.stem()).deleted.len(), 1, "a later pass collects the superseded a");
    assert_eq!(fx.read_pair().unwrap().0, b"b");
}

#[test]
fn unvalidated_scope_deletes_nothing_and_says_so() {
    let fx = Fx::unvalidated("scope");
    assert!(!fx.store.capability.deletes());
    drop(fx.publish(b"a", None).unwrap());
    drop(fx.publish(b"b", None).unwrap());
    let before = fx.store.bytes();
    for r in [fx.store.collect(&fx.stem()), fx.store.evict_stem(&fx.stem()), fx.store.trim(0).0] {
        assert!(r.unsupported.is_some() && r.deleted.is_empty() && r.deleted_bytes == 0, "{r:?}");
    }
    assert_eq!(fx.store.bytes(), before);
    assert_eq!(fx.store.generations(&fx.stem()).len(), 2);
}

// ── D6 on unpublished cleanup ──
//
// The policy is "no deletion on an unvalidated scope", including a publisher's
// own uncommitted files. Each refusal below runs on both scopes: the validated
// one removes the rejected generation, the unvalidated one keeps every file,
// says cleanup is pending and still releases its pin handle.

#[derive(Clone, Copy, Debug)]
enum Reject {
    Cancel,
    Changed,
    CheckFail,
    WriteFail,
}

fn reject(fx: &Fx, how: Reject) -> PublishError {
    let flag = Arc::new(AtomicBool::new(false));
    let (f2, src_path) = (flag.clone(), fx.src.clone());
    let _h = seam::hook(fx.store.dir(), move |p, _| {
        if p == "pub.before_head" {
            match how {
                Reject::Cancel => f2.store(true, Ordering::SeqCst),
                Reject::Changed => {
                    let mut b = fs::read(&src_path).unwrap();
                    b[0] ^= 0xff;
                    fs::write(&src_path, b).unwrap();
                }
                _ => {}
            }
        }
    });
    let _f = seam::fault(fx.store.dir(), move |p, path| match (how, p) {
        (Reject::CheckFail, "source.check") => Some(io::ErrorKind::Other),
        (Reject::WriteFail, "rename") if path.extension().is_some_and(|e| e == "mix") => Some(io::ErrorKind::PermissionDenied),
        _ => None,
    });
    let mut src = OpenSource::open(&fx.src).unwrap();
    let desc = Descriptor { record: fx.desc().record, source: src.snapshot.clone() };
    fx.store.publish(&desc, "settings-A", [3; 32], b"mix", Some(b"lr"), &mut src, &|| flag.load(Ordering::SeqCst)).unwrap_err()
}

/// The stem's files other than its stable coordination files.
fn stem_files(fx: &Fx) -> Vec<String> {
    let stem = fx.stem();
    let mut v: Vec<String> = fs::read_dir(fx.store.dir())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().into_string().unwrap())
        .filter(|n| n.starts_with(&stem) && !n.ends_with(".coord") && !n.ends_with(".pub"))
        .collect();
    v.sort();
    v
}

fn pending(e: &PublishError) -> bool {
    match e {
        PublishError::Cancelled { cleanup_pending }
        | PublishError::SourceChanged { cleanup_pending }
        | PublishError::SourceCheck { cleanup_pending, .. }
        | PublishError::Verify { cleanup_pending } => *cleanup_pending,
        PublishError::Io(m) => m.contains("cleanup pending"),
        PublishError::Busy => false,
    }
}

#[test]
fn an_unvalidated_scope_keeps_every_rejected_publication_and_says_cleanup_pending() {
    for how in [Reject::Cancel, Reject::Changed, Reject::CheckFail, Reject::WriteFail] {
        let fx = Fx::unvalidated(&format!("d6keep{how:?}"));
        assert!(!fx.store.capability.deletes());
        let e = reject(&fx, how);
        assert!(pending(&e), "{how:?}: {e:?}");
        assert_eq!(fx.store.head_state(&fx.stem()), HeadState::Absent, "{how:?}");
        let gens = fx.store.generations(&fx.stem());
        assert_eq!(gens.len(), 1, "{how:?}");
        let g = gens[0];
        let files = stem_files(&fx);
        let name = |ext: &str| fx.store.gen_path(&fx.stem(), g, ext).file_name().unwrap().to_string_lossy().to_string();
        assert!(files.contains(&name("pin")) && files.contains(&name("lr")), "{how:?}: {files:?}");
        match how {
            // The Mix was never renamed into place; its temporary file stays.
            Reject::WriteFail => assert!(files.iter().any(|n| n.starts_with(&format!("{}.tmp", name("mix")))), "{files:?}"),
            _ => assert!(files.contains(&name("mix")), "{how:?}: {files:?}"),
        }
        // The handle is released, not the file: nothing holds the pin.
        let f = OpenOptions::new().read(true).write(true).open(fx.store.gen_path(&fx.stem(), g, "pin")).unwrap();
        assert!(f.try_lock().is_ok(), "{how:?}: the pin handle was kept");
    }
}

#[test]
fn a_validated_scope_still_removes_its_own_rejected_publication() {
    for how in [Reject::Cancel, Reject::Changed, Reject::CheckFail, Reject::WriteFail] {
        let fx = Fx::new(&format!("d6clean{how:?}"));
        assert!(fx.store.capability.deletes(), "{:?}", fx.store.capability);
        let e = reject(&fx, how);
        assert!(!pending(&e), "{how:?}: {e:?}");
        assert_eq!(fx.store.head_state(&fx.stem()), HeadState::Absent, "{how:?}");
        assert_eq!(stem_files(&fx), Vec::<String>::new(), "{how:?}");
    }
}

/// D6 in `write_atomic`: on an unvalidated scope a failed write never removes
/// its temporary entry, even when an existence check cannot see it. The
/// fixture turns the temporary Mix into a dangling junction just before the
/// rename fails: the directory entry is still there, but `exists()` is false.
#[cfg(windows)]
#[test]
fn an_unvalidated_scope_keeps_a_temporary_entry_whose_existence_is_unknown() {
    let fx = Fx::unvalidated("d6unknown");
    assert!(!fx.store.capability.deletes());
    let (dir, root) = (fx.store.dir().to_path_buf(), fx.root.clone());
    let seen: Arc<Mutex<Option<PathBuf>>> = Arc::default();
    let seen2 = seen.clone();
    let _f = seam::fault(fx.store.dir(), move |p, path| {
        if p != "rename" || path.extension().is_none_or(|e| e != "mix") {
            return None;
        }
        let prefix = format!("{}.tmp", path.file_name().unwrap().to_string_lossy());
        let tmp = fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).find(|q| q.file_name().unwrap().to_string_lossy().starts_with(&prefix)).unwrap();
        fs::rename(&tmp, root.join("moved_tmp")).unwrap();
        let cmd = std::env::var_os("ComSpec").unwrap_or_else(|| r"C:\Windows\System32\cmd.exe".into());
        let ok = std::process::Command::new(cmd)
            .args(["/c", "mklink", "/J"])
            .arg(&tmp)
            .arg(root.join("absent_target"))
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap()
            .success();
        assert!(ok, "setup: mklink /J failed");
        assert!(!tmp.exists() && fs::symlink_metadata(&tmp).is_ok(), "setup: the entry must exist but not be seen");
        *seen2.lock().unwrap() = Some(tmp);
        Some(io::ErrorKind::PermissionDenied)
    });
    // The junction (only the link) is removed even when an assertion fails.
    struct Unlink(Arc<Mutex<Option<PathBuf>>>);
    impl Drop for Unlink {
        fn drop(&mut self) {
            if let Some(p) = self.0.lock().unwrap_or_else(|e| e.into_inner()).take() {
                let _ = fs::remove_dir(p);
            }
        }
    }
    let _unlink = Unlink(seen.clone());
    let e = fx.publish(b"mix", Some(b"lr")).unwrap_err();
    let tmp = seen.lock().unwrap().clone().expect("the fault ran");
    assert!(matches!(&e, PublishError::Io(m) if m.contains("temporary file left (cleanup pending)")), "{e:?}");
    assert!(fs::symlink_metadata(&tmp).is_ok(), "the unseen temporary entry was removed");
    assert!(fx.root.join("moved_tmp").is_file());
}

/// An unvalidated scope never gets a probe file; a probe that fails leaves
/// its file and is not deletion permission; one that passes leaves nothing.
#[test]
fn the_lock_probe_runs_only_on_a_validated_scope_and_deletes_only_when_it_passed() {
    let probes = |dir: &Path| {
        fs::read_dir(dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(".probe")).count()
    };
    let root = std::env::temp_dir().join(format!("moosik_store_probe_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (un, bad, good) = (root.join("un"), root.join("bad"), root.join("good"));
    test_scope::force_unvalidated(&un);
    let ran = Arc::new(AtomicBool::new(false));
    let r2 = ran.clone();
    let _h = seam::hook(&un, move |p, _| {
        if p == "probe" {
            r2.store(true, Ordering::SeqCst);
        }
    });
    let s = Store::open(&un).unwrap();
    assert!(!s.capability.deletes());
    assert!(!ran.load(Ordering::SeqCst), "a probe ran on an unvalidated scope");
    assert_eq!(probes(&un), 0);

    let _f = seam::fault(&bad, |p, _| (p == "probe.lock").then_some(io::ErrorKind::Other));
    let s = Store::open(&bad).unwrap();
    assert!(matches!(&s.capability, Capability::Unsupported(w) if w.contains("lock probe failed")), "{:?}", s.capability);
    assert_eq!(probes(&bad), 1, "a failed probe deleted its file");

    let s = Store::open(&good).unwrap();
    assert!(s.capability.deletes(), "{:?}", s.capability);
    assert_eq!(probes(&good), 0);
    drop(s);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_failed_deletion_is_never_counted_as_reclaimed() {
    let fx = Fx::new("faildel");
    drop(fx.publish(b"aaaa", None).unwrap());
    drop(fx.publish(b"bbbb", None).unwrap());
    let fail = Arc::new(AtomicBool::new(true));
    let f2 = fail.clone();
    let _f = seam::fault(fx.store.dir(), move |p, _| (p == "payload.remove" && f2.load(Ordering::SeqCst)).then_some(io::ErrorKind::PermissionDenied));
    let r = fx.store.collect(&fx.stem());
    assert_eq!((r.deleted.len(), r.partial.len(), r.deleted_bytes), (0, 1, 0), "{r:?}");
    assert_eq!(fx.store.generations(&fx.stem()).len(), 2, "kept, pin and all, for a retry");
    fail.store(false, Ordering::SeqCst);
    let r = fx.store.collect(&fx.stem());
    assert_eq!(r.deleted.len(), 1);
    assert!(r.deleted_bytes > 0);
}

#[test]
fn a_pin_file_that_cannot_be_removed_is_cleanup_pending_not_deleted() {
    let fx = Fx::new("pinpend");
    drop(fx.publish(b"aaaa", None).unwrap());
    drop(fx.publish(b"bbbb", None).unwrap());
    let fail = Arc::new(AtomicBool::new(true));
    let f2 = fail.clone();
    let _f = seam::fault(fx.store.dir(), move |p, _| (p == "pin.remove" && f2.load(Ordering::SeqCst)).then_some(io::ErrorKind::PermissionDenied));
    let r = fx.store.collect(&fx.stem());
    assert_eq!((r.deleted.len(), r.pin_pending.len()), (0, 1), "{r:?}");
    fail.store(false, Ordering::SeqCst);
    let r = fx.store.collect(&fx.stem());
    assert_eq!(r.deleted.len(), 1);
    assert_eq!(r.deleted_bytes, 0, "the payload bytes were already counted once");
}

#[cfg(windows)]
#[test]
fn a_native_os_refused_deletion_keeps_the_generation_and_counts_nothing() {
    use std::os::windows::fs::OpenOptionsExt;
    let fx = Fx::new("osrefuse");
    let a = fx.publish(b"aaaa", None).unwrap();
    let (stem, g) = (a.stem.clone(), a.generation);
    drop(a);
    drop(fx.publish(b"bbbb", None).unwrap());
    // Read and write sharing, but no FILE_SHARE_DELETE: Windows refuses the unlink.
    let hold = OpenOptions::new().read(true).share_mode(0x1 | 0x2).open(fx.store.gen_path(&stem, g, "mix")).unwrap();
    let r = fx.store.collect(&stem);
    assert_eq!((r.deleted.len(), r.partial.len(), r.deleted_bytes), (0, 1, 0), "{r:?}");
    assert!(fx.store.gen_path(&stem, g, "pin").exists());
    drop(hold);
    assert_eq!(fx.store.collect(&stem).deleted, vec![g]);
}

#[test]
fn an_unreadable_head_stops_deletion_for_its_stem() {
    let fx = Fx::new("badhead");
    drop(fx.publish(b"a", None).unwrap());
    drop(fx.publish(b"b", None).unwrap());
    fs::write(fx.store.head_path(&fx.stem()), b"garbage").unwrap();
    assert!(matches!(fx.store.lookup(&fx.stem(), &fx.desc()), Err(Miss::Io(_))));
    let r = fx.store.collect(&fx.stem());
    assert!(r.deleted.is_empty() && !r.errors.is_empty(), "{r:?}");
    let r = fx.store.evict_stem(&fx.stem());
    assert!(r.deleted.is_empty() && !r.errors.is_empty(), "{r:?}");
    assert_eq!(fx.store.generations(&fx.stem()).len(), 2);
}

#[test]
fn trim_reports_pinned_bytes_and_the_remaining_excess() {
    let fx = Fx::new("trim");
    let shown = fx.installed(fx.publish(&[1u8; 4000], Some(&[2u8; 4000])).unwrap());
    let (r, left) = fx.store.trim(0);
    assert_eq!(r.skipped_pinned, vec![shown.generation], "{r:?}");
    assert!(r.pinned_bytes > 8000 && left > 0, "D8: the budget is exceeded and reported");
    drop(shown);
    let (r, left) = fx.store.trim(0);
    assert_eq!((r.deleted.len(), left), (1, 0));
}

#[test]
fn channels_from_another_generation_are_refused() {
    let fx = Fx::new("lrbind");
    let a = fx.installed(fx.publish(b"mix", Some(b"lr-a")).unwrap());
    let b = fx.installed(fx.publish(b"mix", Some(b"lr-b")).unwrap());
    // Same descriptor and same Mix: only the generation tells them apart.
    fs::copy(b.lr_path(), a.lr_path()).unwrap();
    assert!(matches!(fx.store.read_lr(&a), Err(Miss::Damaged(_))));
    assert_eq!(&*fx.store.read_lr(&b).unwrap(), b"lr-b");
}

// ── publisher process termination at fixed boundaries ──

const CHILD_DIR: &str = "MOOSIK_STORE_CHILD_DIR";
const CHILD_EXIT_AT: &str = "MOOSIK_STORE_CHILD_EXIT_AT";

/// Run only as a child of the test below: publish one generation into the
/// given directory and exit the process at the named point.
#[test]
#[ignore = "child process of a_terminated_publisher_leaves_the_old_pair_or_the_complete_new_one"]
fn store_child_publisher() {
    let (Ok(root), Ok(at)) = (std::env::var(CHILD_DIR), std::env::var(CHILD_EXIT_AT)) else { return };
    let root = PathBuf::from(root);
    let store = Store::open(&root.join("v2")).unwrap();
    let src_path = root.join("track.flac");
    let _h = seam::hook(store.dir(), move |p, _| {
        if p == at {
            std::process::exit(77);
        }
    });
    let mut o = OpenSource::open(&src_path).unwrap();
    let d = Descriptor { record: "moosik-analysis/1;bars=8;sr=44100;hop=245".into(), source: o.snapshot.clone() };
    let r = store.publish(&d, "settings-A", [4; 32], b"new-mix", Some(b"new-lr"), &mut o, &|| false);
    std::process::exit(if r.is_ok() { 0 } else { 3 });
}

#[test]
fn a_terminated_publisher_leaves_the_old_pair_or_the_complete_new_one() {
    let exe = std::env::current_exe().unwrap();
    for at in ["pub.authority", "pub.created", "pub.lr_written", "pub.mix_written", "pub.before_head", "pub.after_check", "pub.after_head"] {
        let fx = Fx::new(&format!("kill_{}", at.replace('.', "_")));
        drop(fx.publish(b"old-mix", Some(b"old-lr")).unwrap());
        let mut child = std::process::Command::new(&exe)
            .args(["--exact", "spectrum::store::tests::store_child_publisher", "--ignored", "--nocapture", "--test-threads=1"])
            .env(CHILD_DIR, &fx.root)
            .env(CHILD_EXIT_AT, at)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Owned child: waited for by its own handle, killed by it if it hangs.
        let t0 = std::time::Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break s;
            }
            if t0.elapsed() > std::time::Duration::from_secs(120) {
                child.kill().unwrap();
                panic!("child publisher hung at {at}");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        assert_eq!(status.code(), Some(77), "the child must die at {at}");
        let (mix, lr) = fx.read_pair().unwrap();
        let committed = at == "pub.after_head";
        let want: (&[u8], &[u8]) = if committed { (b"new-mix", b"new-lr") } else { (b"old-mix", b"old-lr") };
        assert_eq!((mix.as_slice(), lr.unwrap().as_slice()), want, "after termination at {at}: old or complete new, never a hybrid");
        // Whatever the dead publisher left uncommitted is collectable debris,
        // and the head's pair is untouched by collecting it.
        let r = fx.store.collect(&fx.stem());
        assert!(r.errors.is_empty(), "{at}: {r:?}");
        assert_eq!(fx.store.generations(&fx.stem()).len(), 1, "{at}");
        assert_eq!(fx.read_pair().unwrap().0, want.0);
    }
}
