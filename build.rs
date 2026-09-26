/// The product version, derived once from the Cargo package version.
///
/// Cargo cannot hold a four-part version, so a release like 1.5.2.1 is written
/// in `Cargo.toml` as `1.5.2+1`: the build-metadata number is the fourth part.
/// Everything a person sees — the log banner, the HTTP User-Agent, the
/// Windows file properties — uses `MOOSIK_VERSION`, which this computes, so
/// no surface can say `1.5.2+1` while another says `1.5.2.1`.
///
/// A plain `x.y.z` maps to `x.y.z`; anything else fails the build rather
/// than being guessed at.
fn product_version() -> (String, [u16; 4]) {
    let pkg = std::env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");
    let (core, build) = match pkg.split_once('+') {
        Some((c, b)) => (c, Some(b)),
        None => (pkg.as_str(), None),
    };
    let num = |s: &str| -> u16 {
        s.parse().unwrap_or_else(|_| panic!("version part {s:?} of {pkg:?} is not a number"))
    };
    let parts: Vec<u16> = core.split('.').map(num).collect();
    assert!(parts.len() == 3 && !core.contains('-'),
            "package version {pkg:?} must be x.y.z or x.y.z+n");
    let fourth = build.map(num).unwrap_or(0);
    let text = match build {
        Some(_) => format!("{core}.{fourth}"),
        None => core.to_string(),
    };
    (text, [parts[0], parts[1], parts[2], fourth])
}

fn main() {
    let (version, numeric) = product_version();
    println!("cargo:rustc-env=MOOSIK_VERSION={version}");
    let _ = numeric;

    #[cfg(target_os = "windows")]
    {
        let packed = numeric.iter().fold(0u64, |acc, &p| (acc << 16) | p as u64);
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        // Set, not inferred: winres would take `1.5.2+1` for both strings and
        // drop the build number from the numeric version.
        res.set("FileVersion", &version);
        res.set("ProductVersion", &version);
        res.set_version_info(winres::VersionInfo::FILEVERSION, packed);
        res.set_version_info(winres::VersionInfo::PRODUCTVERSION, packed);
        res.compile().expect("failed to compile Windows resources");
    }
}
