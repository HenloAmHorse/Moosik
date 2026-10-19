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

/// An optional label for a build that is not a release, such as
/// `prereview a4fa75f`, from `MOOSIK_BUILD_LABEL`. Letters, digits, spaces,
/// dots and hyphens only; anything else fails the build.
fn build_label() -> String {
    println!("cargo:rerun-if-env-changed=MOOSIK_BUILD_LABEL");
    let label = std::env::var("MOOSIK_BUILD_LABEL").unwrap_or_default();
    assert!(label.len() <= 64 && label.chars().all(|c| c.is_ascii_alphanumeric() || " .-".contains(c)),
            "MOOSIK_BUILD_LABEL {label:?} must be at most 64 letters, digits, spaces, dots or hyphens");
    label.trim().to_string()
}

fn main() {
    let (version, numeric) = product_version();
    let label = build_label();
    println!("cargo:rustc-env=MOOSIK_VERSION={version}");
    println!("cargo:rustc-env=MOOSIK_BUILD_LABEL={label}");
    let _ = numeric;

    #[cfg(target_os = "windows")]
    {
        let packed = numeric.iter().fold(0u64, |acc, &p| (acc << 16) | p as u64);
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        // Set, not inferred: winres would take `1.5.2+1` for both strings and
        // drop the build number from the numeric version.
        res.set("FileVersion", &version);
        if label.is_empty() {
            res.set("ProductVersion", &version);
        } else {
            // Shown in the file's properties: "1.5.4 prereview a4fa75f".
            res.set("ProductVersion", &format!("{version} {label}"));
            res.set("SpecialBuild", &label);
        }
        res.set_version_info(winres::VersionInfo::FILEVERSION, packed);
        res.set_version_info(winres::VersionInfo::PRODUCTVERSION, packed);
        res.compile().expect("failed to compile Windows resources");
    }
}
