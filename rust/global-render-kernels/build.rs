//! Inject the source revision the release bundle checks.
//!
//! `tools/build_door_bundle.py pin` reads `GPUWM_BRIDGE_SOURCE_REV=<commit>`
//! out of every door as plain bytes and refuses a door whose stamp names a
//! different commit than the cut declares.  This crate is built from this
//! package's repository, not the engine's, so its stamp names this
//! repository's commit and the cut checks it against `--package-rev`.
//!
//! Resolution order: `GPUWM_BRIDGE_SOURCE_REV` in the environment when it is
//! a full commit; otherwise this checkout's HEAD when the crate's tree is
//! clean (a binary built from modified sources is not that commit); otherwise
//! `unknown`, which the cut refuses by name.  Local builds are unaffected.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    println!("cargo:rerun-if-changed=src");
    let rev = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .filter(|value| is_commit(value))
        .or_else(head_of_clean_checkout)
        .unwrap_or_else(|| String::from("unknown"));
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={rev}");
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn git(args: &[&str]) -> Option<String> {
    let dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git").arg("-C").arg(&dir).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn head_of_clean_checkout() -> Option<String> {
    let dirty = git(&["status", "--porcelain", "-uno", "--", "."])?;
    if !dirty.is_empty() {
        return None;
    }
    let rev = git(&["rev-parse", "HEAD"])?.to_ascii_lowercase();
    is_commit(&rev).then_some(rev)
}
