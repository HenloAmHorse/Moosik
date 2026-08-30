//! A frame-atomic SPSC ring over `rtrb`.
//!
//! The bit-perfect path used a raw `rtrb` ring of individual samples, drained
//! with `while len < want { pop() }`. That loop has no idea what a frame is.
//! When the producer was mid-frame, or the ring simply ran dry between a left
//! and a right sample, the render thread consumed half a frame, padded the
//! rest of the device buffer with silence, and left the odd sample in the
//! ring. The next callback then read that leftover as the *first* channel of
//! its frame, so every channel shifted by one for the rest of the session —
//! stereo with the channels swapped, and worse on multichannel — until some
//! later odd-sized read happened to shift it back.
//!
//! This wrapper makes that unrepresentable. Capacity is a multiple of the
//! channel count, the producer only ever publishes whole frames, and the
//! consumer only ever commits whole frames. The number of samples in the ring
//! is therefore always a multiple of the channel count, so a short read leaves
//! a remainder that is still frame-aligned rather than a torn frame.
//!
//! Both sides use the chunk APIs, so a transfer is one or two `memcpy`s and
//! never an allocation — the consumer half runs inside a realtime callback.

use std::mem::MaybeUninit;

/// Create a frame-aware ring holding at least `frames` frames of `channels`
/// samples.
///
/// Capacity is rounded up to a whole number of frames. `channels` must be
/// non-zero; a zero-channel stream is a caller bug, not a runtime condition.
pub fn channel_ring<T>(channels: usize, frames: usize) -> (FrameProducer<T>, FrameConsumer<T>) {
    assert!(channels > 0, "a ring needs at least one channel");
    let frames = frames.max(1);
    let (prod, cons) = rtrb::RingBuffer::<T>::new(frames * channels);
    (
        FrameProducer { prod, channels },
        FrameConsumer { cons, channels },
    )
}

/// The writing half. Lives on a decode thread, so it may block; it never
/// allocates regardless.
pub struct FrameProducer<T> {
    prod: rtrb::Producer<T>,
    channels: usize,
}

impl<T: Copy> FrameProducer<T> {
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Frames that would fit right now.
    pub fn free_frames(&self) -> usize {
        self.prod.slots() / self.channels
    }

    /// Publish as many whole frames from `interleaved` as fit, and return how
    /// many.
    ///
    /// Partial publication is normal — the caller retries with the remainder.
    /// What never happens is publishing part of a frame.
    ///
    /// A sub-frame input is an error in every build, not a `debug_assert`. The
    /// whole-frame invariant is what keeps channels in their own lanes, so a
    /// release build silently accepting a partial frame would defeat the whole
    /// wrapper exactly where it is hardest to observe.
    pub fn push_frames(&mut self, interleaved: &[T]) -> Result<usize, PartialFrame> {
        if !interleaved.len().is_multiple_of(self.channels) {
            return Err(PartialFrame { samples: interleaved.len(), channels: self.channels });
        }
        let want = interleaved.len() / self.channels;
        let n = want.min(self.free_frames());
        if n == 0 {
            return Ok(0);
        }
        let samples = n * self.channels;
        let Ok(mut chunk) = self.prod.write_chunk_uninit(samples) else {
            return Ok(0);
        };
        let (a, b) = chunk.as_mut_slices();
        let (src_a, src_b) = interleaved[..samples].split_at(a.len());
        copy_into_uninit(a, src_a);
        copy_into_uninit(b, src_b);
        // SAFETY: every slot in the chunk was just initialised above — the two
        // destination slices together are exactly `samples` long, and so are
        // the two source slices.
        unsafe { chunk.commit_all() };
        Ok(n)
    }
}

/// A push whose length was not a whole number of frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialFrame {
    pub samples: usize,
    pub channels: usize,
}

impl std::fmt::Display for PartialFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} samples is not a whole number of {}-channel frames",
            self.samples, self.channels
        )
    }
}

/// The reading half. Runs inside realtime callbacks: no allocation, no
/// locking, no partial frames.
pub struct FrameConsumer<T> {
    cons: rtrb::Consumer<T>,
    channels: usize,
}

impl<T: Copy> FrameConsumer<T> {
    /// Whole frames available right now.
    pub fn available_frames(&self) -> usize {
        self.cons.slots() / self.channels
    }

    pub fn is_empty(&self) -> bool {
        self.cons.is_empty()
    }

    /// The frame invariant, checkable at runtime.
    ///
    /// If this is ever false the ring has been torn by something outside this
    /// module and the session's exactness claim is void — callers escalate it
    /// to an integrity fault rather than trying to resynchronise, because
    /// there is no way to know which channel the stray samples belong to.
    pub fn invariant_holds(&self) -> bool {
        self.cons.slots().is_multiple_of(self.channels)
    }

    /// Copy up to `max_frames` whole frames into `out`, returning the frame
    /// count. Never copies a partial frame, and never consumes one.
    ///
    /// `out` is truncated to whole frames as well, so an awkward device buffer
    /// cannot induce a tear from the other direction.
    pub fn pop_frames(&mut self, out: &mut [T], max_frames: usize) -> usize {
        let room = out.len() / self.channels;
        let n = max_frames.min(room).min(self.available_frames());
        if n == 0 {
            return 0;
        }
        let samples = n * self.channels;
        let Ok(chunk) = self.cons.read_chunk(samples) else {
            return 0;
        };
        let (a, b) = chunk.as_slices();
        out[..a.len()].copy_from_slice(a);
        out[a.len()..a.len() + b.len()].copy_from_slice(b);
        chunk.commit_all();
        n
    }
}

#[inline]
fn copy_into_uninit<T: Copy>(dst: &mut [MaybeUninit<T>], src: &[T]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        d.write(*s);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample tagged with the frame and channel it was created for, so a
    /// misrouted sample is identifiable rather than merely different.
    fn tag(frame: u32, ch: usize) -> u32 {
        (frame << 4) | (ch as u32 & 0xF)
    }
    fn frame_of(t: u32) -> u32 {
        t >> 4
    }
    fn ch_of(t: u32) -> usize {
        (t & 0xF) as usize
    }

    fn frames(first: u32, count: usize, channels: usize) -> Vec<u32> {
        let mut v = Vec::with_capacity(count * channels);
        for f in 0..count as u32 {
            for c in 0..channels {
                v.push(tag(first + f, c));
            }
        }
        v
    }

    /// The invariant that makes the whole wrapper worth having: whatever the
    /// producer chunking, the consumer request size, or the wrap offset, every
    /// sample arrives in its own channel slot and no sample is lost.
    ///
    /// Would have caught the 1.4.2 `while len < want { pop() }` drain, which
    /// consumed a partial frame whenever the ring ran dry mid-frame.
    #[test]
    fn every_channel_keeps_its_own_samples_through_any_chunking() {
        for channels in 1..=8usize {
            // Odd ring size in frames so wraps land at every phase relative to
            // the producer and consumer chunk sizes.
            for ring_frames in [3usize, 7, 16, 23] {
                for push_frames in [1usize, 2, 5, 9] {
                    for pull_frames in [1usize, 3, 4, 11] {
                        let (mut p, mut c) = channel_ring::<u32>(channels, ring_frames);
                        let total = 97usize; // prime: never aligns with anything
                        let src = frames(0, total, channels);

                        let mut sent = 0usize;
                        let mut got: Vec<u32> = Vec::new();
                        let mut out = vec![0u32; pull_frames * channels];

                        let mut spins = 0;
                        while got.len() < total * channels {
                            if sent < total {
                                let hi = (sent + push_frames).min(total);
                                let n =
                                    p.push_frames(&src[sent * channels..hi * channels]).unwrap();
                                sent += n;
                            }
                            let n = c.pop_frames(&mut out, pull_frames);
                            got.extend_from_slice(&out[..n * channels]);
                            assert!(
                                c.invariant_holds(),
                                "ring left a partial frame: {channels}ch"
                            );
                            spins += 1;
                            assert!(spins < 10_000, "no progress");
                        }

                        assert_eq!(
                            got, src,
                            "ch={channels} ring={ring_frames} \
                                    push={push_frames} pull={pull_frames}"
                        );
                        for (i, &t) in got.iter().enumerate() {
                            assert_eq!(ch_of(t), i % channels, "sample changed channel");
                            assert_eq!(frame_of(t), (i / channels) as u32, "frame reordered");
                        }
                    }
                }
            }
        }
    }

    /// A consumer that asks for more than the ring holds takes the whole
    /// frames available and leaves nothing behind but frame-aligned data.
    #[test]
    fn a_short_ring_yields_whole_frames_and_keeps_the_rest() {
        for channels in 1..=8usize {
            let (mut p, mut c) = channel_ring::<u32>(channels, 64);
            assert_eq!(p.push_frames(&frames(0, 5, channels)).unwrap(), 5);

            // Ask for far more than is there.
            let mut out = vec![0u32; 40 * channels];
            let n = c.pop_frames(&mut out, 40);
            assert_eq!(n, 5, "should take exactly the whole frames present");
            assert!(c.is_empty());
            assert!(c.invariant_holds());

            // Ask for less than is there: the remainder stays, frame-aligned.
            assert_eq!(p.push_frames(&frames(10, 9, channels)).unwrap(), 9);
            let n = c.pop_frames(&mut out, 4);
            assert_eq!(n, 4);
            assert_eq!(c.available_frames(), 5, "remainder must survive intact");
            assert!(c.invariant_holds());
            let n = c.pop_frames(&mut out, 100);
            assert_eq!(n, 5);
            for (i, &t) in out[..5 * channels].iter().enumerate() {
                assert_eq!(frame_of(t), 14 + (i / channels) as u32);
                assert_eq!(ch_of(t), i % channels);
            }
        }
    }

    /// An output buffer that is not a whole number of frames long cannot make
    /// the consumer tear one — it fills what whole frames fit and stops.
    #[test]
    fn an_awkward_output_buffer_cannot_induce_a_tear() {
        for channels in 2..=8usize {
            let (mut p, mut c) = channel_ring::<u32>(channels, 64);
            p.push_frames(&frames(0, 20, channels)).unwrap();

            // One sample short of three frames.
            let mut out = vec![0u32; 3 * channels - 1];
            let n = c.pop_frames(&mut out, 100);
            assert_eq!(n, 2, "{channels}ch: only two whole frames fit");
            assert_eq!(c.available_frames(), 18);
            assert!(c.invariant_holds());
        }
    }

    /// A ring cannot accept a partial frame even when there is room for one
    /// more sample.
    #[test]
    fn the_producer_never_publishes_a_partial_frame() {
        for channels in 2..=8usize {
            // Room for exactly four frames.
            let (mut p, mut c) = channel_ring::<u32>(channels, 4);
            assert_eq!(p.push_frames(&frames(0, 4, channels)).unwrap(), 4);
            assert_eq!(p.free_frames(), 0);
            assert_eq!(p.push_frames(&frames(4, 1, channels)).unwrap(), 0);

            // Free exactly one frame; a two-frame push publishes one.
            let mut out = vec![0u32; 8 * channels];
            assert_eq!(c.pop_frames(&mut out, 1), 1);
            assert_eq!(p.push_frames(&frames(4, 2, channels)).unwrap(), 1);
            assert!(c.invariant_holds());
        }
    }

    /// A sub-frame push is refused in every build, not only under
    /// `debug_assert`.
    ///
    /// The whole-frame invariant is what keeps channels in their own lanes. A
    /// release build that accepted a partial frame would rotate every channel
    /// for the rest of the session, in the one configuration where nobody is
    /// watching an assertion.
    #[test]
    fn a_partial_frame_push_is_an_error_in_every_build() {
        for channels in 2..=8usize {
            let (mut p, c) = channel_ring::<u32>(channels, 16);
            let partial = vec![0u32; channels + 1];
            let err = p
                .push_frames(&partial)
                .expect_err("a partial frame must be refused");
            assert_eq!(err.samples, channels + 1);
            assert_eq!(err.channels, channels);
            assert!(!err.to_string().is_empty());

            // Nothing was published, and the ring is still usable afterwards.
            assert_eq!(c.available_frames(), 0);
            assert!(c.invariant_holds());
            assert_eq!(p.push_frames(&vec![7u32; channels * 3]).unwrap(), 3);
            assert_eq!(c.available_frames(), 3);
        }
    }

    /// Zero frames in, zero frames out, no panic — the idle case the render
    /// loop hits on every buffer while paused.
    #[test]
    fn empty_transfers_are_not_special_cases() {
        for channels in 1..=8usize {
            let (mut p, mut c) = channel_ring::<u32>(channels, 8);
            let mut out = vec![0u32; 4 * channels];
            assert_eq!(c.pop_frames(&mut out, 4), 0);
            assert_eq!(p.push_frames(&[]).unwrap(), 0);
            assert_eq!(c.pop_frames(&mut out, 0), 0);
            assert_eq!(c.pop_frames(&mut [], 4), 0);
            assert!(c.invariant_holds());
        }
    }
}
