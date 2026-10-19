//! Which local model, and which route, to try on this computer — in three
//! separate senses, never merged into one ranking:
//!
//! * **Quality first**: the best-judged pack that this computer can run at
//!   all, on its fastest route that fits.
//! * **Balanced**: the best-judged pack whose route fits with room to spare
//!   and answers a sheet in a practical time.
//! * **Largest that fits**: the biggest model this computer can hold. Size
//!   is not quality; it is labelled as what it is.
//!
//! From a cheap inventory (total and free RAM, commit headroom, the graphics
//! card's dedicated memory) and each pack's recorded costs. Nothing here
//! downloads, switches the selected model, measures speed or starts a model;
//! the panel shows the advice and the choice stays the user's.
//!
//! The budget is conservative: dedicated graphics memory minus a reserve (the
//! desktop, Moosik's own spectrum textures, the driver), never shared memory;
//! total RAM minus a reserve for Windows and Moosik. One computer's
//! measurements cannot set thresholds for every computer: a cost not measured
//! here says "estimated".

use super::Pack;

/// One route a pack can run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// The pack's configured graphics-card route (may be hybrid).
    Gpu,
    /// The CPU only.
    Cpu,
    GpuLowRam,
}

impl Path {
    pub fn label(self) -> &'static str {
        match self {
            Path::Gpu => "graphics card",
            Path::Cpu => "CPU only",
            Path::GpuLowRam => "graphics card, lower RAM (more graphics memory and commit)",
        }
    }
}

impl Machine {
    pub fn headroom(&self) -> super::memory::Headroom {
        // A successful Windows inventory never reports zero total physical
        // memory. Zero free on a known machine is pressure, not "unknown".
        if self.ram_total == 0 { return super::memory::Headroom::default() }
        super::memory::Headroom { phys: Some(self.ram_free), commit: Some(self.commit_free) }
    }
}

/// What a route costs this computer for one request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    /// Dedicated graphics memory the route adds, bytes.
    pub vram: u64,
    /// Working set (RAM) the model process peaks at, bytes.
    pub ram: u64,
    /// Seconds for one 10-row sheet.
    pub secs_per_sheet: f32,
    /// Measured on the reference computer (otherwise estimated).
    pub measured: bool,
}

/// This computer, read cheaply (no model run, no benchmark).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Machine {
    pub ram_total: u64,
    pub ram_free: u64,
    pub commit_free: u64,
    /// The graphics card with the most dedicated memory, if any is known.
    pub gpu: Option<Gpu>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub dedicated: u64,
}

/// Graphics memory kept back: the desktop, Moosik's spectrum, the driver.
pub const VRAM_RESERVE: u64 = 1536 << 20;
/// RAM kept back for Windows, the desktop and Moosik itself.
pub const RAM_RESERVE: u64 = 4 << 30;
/// A sheet answered within this is called practical.
pub const PRACTICAL_SECS: f32 = 15.0;
/// "With room to spare": at most this share of the RAM budget. (Graphics
/// memory's room to spare is `VRAM_RESERVE` itself.)
pub const HEADROOM: f64 = 0.85;

/// What a route may use here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub vram: u64,
    pub ram: u64,
}

pub fn budget(m: &Machine) -> Budget {
    Budget {
        vram: m.gpu.as_ref().map_or(0, |g| g.dedicated.saturating_sub(VRAM_RESERVE)),
        ram: m.ram_total.saturating_sub(RAM_RESERVE),
    }
}

/// One suggestion.
#[derive(Clone, Copy, Debug)]
pub struct Pick {
    pub pack: &'static Pack,
    pub path: Path,
    pub cost: Cost,
}

impl Pick {
    pub fn device(self) -> super::Device {
        match self.path { Path::Gpu => super::Device::Gpu, Path::Cpu => super::Device::Cpu, Path::GpuLowRam => super::Device::GpuLowRam }
    }
    pub fn need(self) -> super::memory::Need {
        super::memory_need(self.pack, self.device()).expect("a declared model route")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Advice {
    pub quality: Option<Pick>,
    pub balanced: Option<Pick>,
    pub largest: Option<Pick>,
    pub budget: Budget,
}

fn fits(c: &Cost, b: &Budget, share: f64) -> bool {
    c.vram <= b.vram && c.ram as f64 <= b.ram as f64 * share
}

/// The model's own size: the pack's model files.
fn weights(p: &Pack) -> u64 {
    p.files.iter().filter(|f| f.kind == super::Kind::Model).map(|f| f.bytes).sum()
}

/// Each pack's fitting routes in its own order (fastest first).
fn candidates<'a>(packs: &'a [&'static Pack], b: &'a Budget, share: f64) -> impl Iterator<Item = Pick> + 'a {
    packs.iter().flat_map(move |&pack| {
        pack.costs.iter().filter(move |(_, c)| fits(c, b, share)).map(move |&(path, cost)| Pick { pack, path, cost })
    })
}

pub fn advise(packs: &[&'static Pack], m: &Machine) -> Advice {
    let b = budget(m);
    // Best judged first; for one pack, its fastest fitting route.
    let best = |it: &mut dyn Iterator<Item = Pick>| {
        it.fold(None::<Pick>, |acc, p| match acc {
            Some(a) if (a.pack.quality, super::memory::room(a.need(), m.headroom()) == super::memory::Room::Fits, -a.cost.secs_per_sheet)
                >= (p.pack.quality, super::memory::room(p.need(), m.headroom()) == super::memory::Room::Fits, -p.cost.secs_per_sheet) => Some(a),
            _ => Some(p),
        })
    };
    let quality = best(&mut candidates(packs, &b, 1.0));
    let balanced = best(&mut candidates(packs, &b, HEADROOM).filter(|p| p.cost.secs_per_sheet <= PRACTICAL_SECS));
    let largest = candidates(packs, &b, 1.0).fold(None::<Pick>, |acc, p| match acc {
        Some(a) if (weights(a.pack), super::memory::room(a.need(), m.headroom()) == super::memory::Room::Fits, -a.cost.secs_per_sheet)
            >= (weights(p.pack), super::memory::room(p.need(), m.headroom()) == super::memory::Room::Fits, -p.cost.secs_per_sheet) => Some(a),
        _ => Some(p),
    });
    Advice { quality, balanced, largest, budget: b }
}

/// The inventory, read now.
pub fn inventory() -> Machine {
    let (ram_total, ram_free, commit_free) = ram();
    Machine { ram_total, ram_free, commit_free, gpu: gpu() }
}

#[cfg(windows)]
fn ram() -> (u64, u64, u64) {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0,
        total_page_file: 0, avail_page_file: 0, total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0,
    };
    // SAFETY: `m` is a correctly sized, writable MEMORYSTATUSEX with `length` set.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return (0, 0, 0);
    }
    (m.total_phys, m.avail_phys, m.avail_page_file)
}

#[cfg(not(windows))]
fn ram() -> (u64, u64, u64) {
    (0, 0, 0)
}

/// The adapter with the most dedicated video memory, through DXGI
/// (`IDXGIFactory1::EnumAdapters1`, `IDXGIAdapter1::GetDesc1`). Software
/// adapters are skipped. Shared system memory is never counted.
#[cfg(windows)]
fn gpu() -> Option<Gpu> {
    use std::ffi::c_void;
    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);
    #[repr(C)]
    struct Desc1 {
        description: [u16; 128],
        vendor: u32,
        device: u32,
        subsys: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid: (u32, i32),
        flags: u32,
    }
    // IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    const IID_FACTORY1: Guid = Guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);
    const SOFTWARE: u32 = 2; // DXGI_ADAPTER_FLAG_SOFTWARE
    #[link(name = "dxgi")]
    unsafe extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> i32;
    }
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    type EnumAdapters1 = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
    type GetDesc1 = unsafe extern "system" fn(*mut c_void, *mut Desc1) -> i32;
    // SAFETY: COM calls through the documented vtable slots of
    // IDXGIFactory1 (Release 2, EnumAdapters1 12) and IDXGIAdapter1
    // (Release 2, GetDesc1 10); every interface obtained is released once.
    unsafe {
        let mut f: *mut c_void = std::ptr::null_mut();
        if CreateDXGIFactory1(&IID_FACTORY1, &mut f) < 0 || f.is_null() {
            return None;
        }
        let fv = *(f as *const *const usize);
        let enum1: EnumAdapters1 = std::mem::transmute(*fv.add(12));
        let frel: Release = std::mem::transmute(*fv.add(2));
        let mut best: Option<Gpu> = None;
        for i in 0..16 {
            let mut a: *mut c_void = std::ptr::null_mut();
            if enum1(f, i, &mut a) < 0 || a.is_null() {
                break;
            }
            let av = *(a as *const *const usize);
            let desc1: GetDesc1 = std::mem::transmute(*av.add(10));
            let arel: Release = std::mem::transmute(*av.add(2));
            let mut d: Desc1 = std::mem::zeroed();
            if desc1(a, &mut d) >= 0 && d.flags & SOFTWARE == 0 && d.dedicated_video > 0 {
                let n = d.description.iter().position(|&c| c == 0).unwrap_or(128);
                let g = Gpu { name: String::from_utf16_lossy(&d.description[..n]), dedicated: d.dedicated_video as u64 };
                if best.as_ref().is_none_or(|b| g.dedicated > b.dedicated) {
                    best = Some(g);
                }
            }
            arel(a);
        }
        frel(f);
        best
    }
}

#[cfg(not(windows))]
fn gpu() -> Option<Gpu> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lyrics::local_ai::{HYMT2_1_8B, HYMT2_7B, PACKS, QWEN25_7B};

    const GIB: u64 = 1 << 30;

    fn machine(ram: u64, vram: Option<u64>) -> Machine {
        Machine { ram_total: ram * GIB, ram_free: ram * GIB / 2, commit_free: ram * GIB,
                  // A card sold as N GB reports a little under N GiB (the
                  // reference card: 8,394,899,456 bytes).
                  gpu: vram.map(|v| Gpu { name: "test".into(), dedicated: v * GIB * 977 / 1000 }) }
    }

    /// The reference computer (32 GB RAM, 8 GB card): Hy-MT2 7B on the card
    /// is both the quality pick and the balanced one; the largest is the 7B
    /// that ranks higher among equals in size.
    #[test]
    fn the_reference_computer() {
        let a = advise(PACKS, &machine(32, Some(8)));
        let q = a.quality.unwrap();
        assert_eq!((q.pack.id, q.path), (HYMT2_7B.id, Path::Gpu));
        let b = a.balanced.unwrap();
        assert_eq!((b.pack.id, b.path), (HYMT2_7B.id, Path::Gpu));
        assert!(a.largest.is_some());
    }

    /// A 4 GB card cannot hold a 7B on the card: quality falls back to the
    /// 7B on the CPU (slow, so not "balanced"), and balanced is the 1.8B on
    /// the card. Shared memory is never counted.
    #[test]
    fn a_small_card() {
        let a = advise(PACKS, &machine(16, Some(4)));
        let q = a.quality.unwrap();
        assert_eq!((q.pack.id, q.path), (HYMT2_7B.id, Path::Cpu));
        let b = a.balanced.unwrap();
        assert_eq!((b.pack.id, b.path), (HYMT2_1_8B.id, Path::Gpu));
    }

    /// No graphics card known: CPU routes only.
    #[test]
    fn no_card() {
        let a = advise(PACKS, &machine(16, None));
        for p in [a.quality, a.balanced, a.largest].into_iter().flatten() {
            assert_eq!(p.path, Path::Cpu, "{}", p.pack.id);
        }
    }

    /// Too little RAM for anything: no advice rather than a bad one.
    #[test]
    fn too_small() {
        let a = advise(PACKS, &machine(4, None));
        assert!(a.quality.is_none() && a.balanced.is_none() && a.largest.is_none());
    }

    #[test]
    fn advice_uses_the_route_startup_need_and_preserves_unknown_readings() {
        use super::super::{memory, HYMT2_30B};
        let p = Pick { pack: &HYMT2_30B, path: Path::Gpu, cost: HYMT2_30B.costs[0].1 };
        let mut m = machine(32, Some(8));
        m.ram_free = 18 * GIB;
        assert_eq!(memory::room(p.need(), m.headroom()), memory::Room::Short { phys: GIB, commit: 0 });
        m.ram_free = 19 * GIB;
        assert_eq!(memory::room(p.need(), m.headroom()), memory::Room::Fits);
        m.commit_free = 5 * GIB;
        assert_eq!(memory::room(p.need(), m.headroom()), memory::Room::Short { phys: 0, commit: GIB });
        assert_eq!(Machine::default().headroom(), memory::Headroom::default());
        m.ram_free = 0;
        assert_eq!(m.headroom().phys, Some(0));
    }

    #[test]
    fn largest_selects_a_lower_ram_placement_only_when_it_can_start() {
        let mut m = machine(32, Some(8));
        m.ram_free = 16 * GIB;
        m.commit_free = 24 * GIB;
        let p = advise(PACKS, &m).largest.unwrap();
        assert_eq!(p.path, Path::GpuLowRam);
        assert_eq!(super::super::memory::room(p.need(), m.headroom()), super::super::memory::Room::Fits);
        m.ram_free = 19 * GIB;
        assert_eq!(advise(PACKS, &m).largest.unwrap().path, Path::Gpu);
        m.ram_free = 16 * GIB;
        m.commit_free = 6 * GIB;
        assert_ne!(advise(PACKS, &m).largest.unwrap().path, Path::GpuLowRam);
    }

    /// The ranking basis: Hy-MT2 7B above Qwen2.5 7B above Hy-MT2 1.8B
    /// (the owner's trial in batch 155; none of them approved yet).
    #[test]
    fn quality_order_is_the_owners_trial() {
        let rank = |id: &str| PACKS.iter().find(|p| p.id == id).map(|p| p.quality).unwrap();
        assert!(rank(HYMT2_7B.id) > rank(QWEN25_7B.id) && rank(QWEN25_7B.id) > rank(HYMT2_1_8B.id));
    }

    /// The inventory reads something on Windows (cheaply; no model is run).
    #[cfg(windows)]
    #[test]
    fn the_inventory_reads_this_computer() {
        let m = inventory();
        assert!(m.ram_total > 0 && m.ram_free <= m.ram_total, "{m:?}");
    }

    /// Prints this computer's inventory and advice (run by hand).
    #[test]
    #[ignore]
    fn show_this_computer() {
        let m = inventory();
        eprintln!("{m:?}");
        eprintln!("{:#?}", advise(PACKS, &m).budget);
        let a = advise(PACKS, &m);
        for (k, p) in [("quality", a.quality), ("balanced", a.balanced), ("largest", a.largest)] {
            eprintln!("{k}: {:?}", p.map(|p| (p.pack.id, p.path)));
        }
    }
}
