//! Conformance suite, wasm32-wasip1 target (TODO 1.5, C18 regression guard).
//!
//! Runs every case through the guest module's `ps_call` under the wasmtime harness
//! (fuel + memory limits, zero WASI capabilities) and requires the outcome to match
//! BOTH the expectation and the native outcome exactly — value bits, error class,
//! fuel consumed, memory charged (spec §6.4 determinism across build targets).
//!
//! Requires the guest module (CI builds it first):
//!   cargo build -p placementscript-wasm-guest --target wasm32-wasip1
//! Set `PS_SKIP_WASM_CONFORMANCE=1` to skip explicitly (never set in CI).

use placementscript_conformance::{cases_dir, check, load_cases, run_native, Outcome};
use placementscript_core::wire::{CallRequest, WireValue};
use placementscript_wasm_host::{locate_guest_module, HostConfig, PolicyModule, WasmCallError};

#[test]
fn wasm_conformance_matches_native() {
    if std::env::var("PS_SKIP_WASM_CONFORMANCE").as_deref() == Ok("1") {
        eprintln!("skipped: PS_SKIP_WASM_CONFORMANCE=1");
        return;
    }
    let module_path = locate_guest_module().expect("guest module (see test docs)");
    let mut host = PolicyModule::from_file(&module_path, HostConfig::default()).unwrap();

    let cases = load_cases();
    let mut failures = Vec::new();
    for (file, case) in &cases {
        let req = CallRequest {
            source: case.resolve_source(&cases_dir()),
            func: case.func.clone().unwrap_or_default(),
            args: case.args.clone(),
            limits: case.limits(),
            load_limits: case.load_limits(),
        };
        let wasm_outcome = match host.call(&req) {
            Ok(resp) => match resp.result {
                Ok(value) => Outcome::Value {
                    value,
                    fuel_used: resp.fuel_used,
                    mem_used: resp.mem_used,
                },
                // Compile-only cases send func="" — an unknown-function host error
                // after successful compile means "compiled fine", mirroring run_native.
                Err(e) if case.func.is_none() && e.class() == "host" => Outcome::Value {
                    value: WireValue::B(true),
                    fuel_used: 0,
                    mem_used: 0,
                },
                Err(e) => Outcome::Error {
                    class: e.class().to_string(),
                    message: format!("{e:?}"),
                    fuel_used: resp.fuel_used,
                    mem_used: resp.mem_used,
                },
            },
            Err(WasmCallError::Guest(e)) => Outcome::from_wire_error(&e),
            Err(WasmCallError::Sandbox(e)) => {
                failures.push(format!("{}: sandbox error: {e:#}", case.name));
                continue;
            }
        };
        if let Err(e) = check(case, &wasm_outcome, "wasm") {
            failures.push(format!(
                "{}: {e}",
                file.file_name().unwrap().to_string_lossy()
            ));
            continue;
        }
        // C18: identical outcome across build targets, including resource accounting.
        let native = run_native(case);
        match (&native, &wasm_outcome) {
            (
                Outcome::Value {
                    value: nv,
                    fuel_used: nf,
                    mem_used: nm,
                },
                Outcome::Value {
                    value: wv,
                    fuel_used: wf,
                    mem_used: wm,
                },
            ) => {
                let (nv, wv) = (nv.to_value().unwrap(), wv.to_value().unwrap());
                if !nv.bit_eq(&wv) || (nf, nm) != (wf, wm) {
                    failures.push(format!(
                        "{}: native/wasm divergence: value_eq={} fuel {nf}/{wf} mem {nm}/{wm}",
                        case.name,
                        nv.bit_eq(&wv)
                    ));
                }
            }
            (
                Outcome::Error {
                    class: nc,
                    fuel_used: nf,
                    mem_used: nm,
                    ..
                },
                Outcome::Error {
                    class: wc,
                    fuel_used: wf,
                    mem_used: wm,
                    ..
                },
            ) => {
                // Error class AND error-path resource accounting must agree (§6.4).
                if nc != wc || (nf, nm) != (wf, wm) {
                    failures.push(format!(
                        "{}: native/wasm error divergence: {nc}/{wc} fuel {nf}/{wf} mem {nm}/{wm}",
                        case.name
                    ));
                }
            }
            _ => failures.push(format!(
                "{}: native/wasm outcome-shape divergence",
                case.name
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} wasm conformance failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!(
        "wasm conformance: {} cases passed (bit-identical to native)",
        cases.len()
    );
}
