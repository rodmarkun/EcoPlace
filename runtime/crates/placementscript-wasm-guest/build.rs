//! Embeds the policy source into the module (TODO 1.4: "module layout = interpreter +
//! embedded policy source"). The bundle packaging step (Phase 7 promotion pipeline)
//! points `PS_EMBED_POLICY` at the promoted PolicyBundle source; the default embeds the
//! `least_allocated` seed so a bare build is a runnable module.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Reserve a 4 MiB guest stack: the spec §8.1 structural limits bound worst-case
    // interpreter recursion, and this stack comfortably fits the worst *legal* program
    // on the wasm target too (native/wasm parity for the boundary conformance cases).
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        println!("cargo:rustc-link-arg=-zstack-size=4194304");
    }
    println!("cargo:rerun-if-env-changed=PS_EMBED_POLICY");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let default = manifest.join("../../policies/least_allocated.ps");
    let src_path = env::var("PS_EMBED_POLICY")
        .map(PathBuf::from)
        .unwrap_or(default);
    println!("cargo:rerun-if-changed={}", src_path.display());
    let source = fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("PS_EMBED_POLICY {}: {e}", src_path.display()));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("embedded_policy.ps");
    fs::write(&out, source).unwrap();
}
