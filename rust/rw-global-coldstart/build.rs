//! Inject the source revision the library embeds.
//!
//! The door bundle's pin step refuses a member that does not carry
//! `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` (tools/build_door_bundle.py),
//! read out of the built bytes and never by executing them.  Resolution:
//! the environment variable when it is shaped like a full commit, else the
//! checkout's HEAD when the workspace's tracked sources are clean, else
//! `unknown`, which the pin step refuses.  A local development build is
//! unaffected: nothing outside the release cut reads the stamp.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
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

fn workspace_dir() -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let root = std::path::Path::new(&manifest).parent()?;
    Some(root.to_string_lossy().into_owned())
}

fn git(args: &[&str]) -> Option<String> {
    let dir = workspace_dir()?;
    let output = Command::new("git").arg("-C").arg(&dir).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn head_of_clean_checkout() -> Option<String> {
    for tracked in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", tracked]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &reference]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let dirty = git(&["status", "--porcelain", "-uno", "--", "."])?;
    if !dirty.is_empty() {
        return None;
    }
    let rev = git(&["rev-parse", "HEAD"])?.to_ascii_lowercase();
    is_commit(&rev).then_some(rev)
}
