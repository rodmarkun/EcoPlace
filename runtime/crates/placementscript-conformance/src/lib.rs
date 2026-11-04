//! Conformance-case model + native runner (TODO 1.5).
//!
//! Cases live in `runtime/conformance/cases/*.json` (JSON arrays of [`Case`]). The suite
//! is a **regression guard**, not an equivalence proof (C18: there is exactly one
//! interpreter implementation; the suite pins its behavior across the two build
//! targets): the native runner executes each case in-process; the wasm runner (in
//! `tests/wasm.rs`) sends the same case through the guest module's `ps_call` and
//! requires the *identical* outcome — value bits, error class, fuel, memory.
//!
//! Expected values are compared **bit-exactly** (`Value::bit_eq`): `-0.0 != 0.0` here,
//! and float results must match to the bit, per spec §6.3/§6.4.

use placementscript_core as core;
use placementscript_core::wire::{WireError, WireLimits, WireValue};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Expected outcome of a case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    /// Expected value (bit-exact) for success cases.
    #[serde(default)]
    pub value: Option<WireValue>,
    /// Expected error class (`parse`/`type`/`trap`/`fuel`/`oom`) for failure cases.
    #[serde(default)]
    pub error: Option<String>,
    /// Substring the error message must contain (optional tightening).
    #[serde(default)]
    pub message_contains: Option<String>,
    /// Golden fuel consumption (spec §8.1 accounting tables). Checked on success and
    /// failure outcomes, on both build targets, when present.
    #[serde(default)]
    pub fuel_used: Option<u64>,
    /// Golden memory charge in bytes (spec §8.2). Checked like `fuel_used`.
    #[serde(default)]
    pub mem_used: Option<u64>,
}

/// One conformance case.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    /// Unique name (grep handle).
    pub name: String,
    /// Program source. Alternatively `source_file` relative to the cases dir.
    #[serde(default)]
    pub source: Option<String>,
    /// Path to a source file (e.g. `../../policies/least_allocated.ps`).
    #[serde(default)]
    pub source_file: Option<String>,
    /// Function to call; omit for compile-only cases.
    #[serde(default)]
    pub func: Option<String>,
    /// Arguments (wire encoding).
    #[serde(default)]
    pub args: Vec<WireValue>,
    /// Fuel budget (default: spec §8 score default).
    #[serde(default)]
    pub fuel: Option<u64>,
    /// Memory budget in bytes (default: spec §8 score default).
    #[serde(default)]
    pub memory_bytes: Option<u64>,
    /// Module-load fuel budget (default: spec §8.1 load default).
    #[serde(default)]
    pub load_fuel: Option<u64>,
    /// Module-load memory budget (default: spec §8.1 load default).
    #[serde(default)]
    pub load_memory_bytes: Option<u64>,
    /// Expected outcome.
    pub expect: Expect,
}

impl Case {
    /// Effective limits of the case.
    pub fn limits(&self) -> WireLimits {
        WireLimits {
            fuel: self.fuel.unwrap_or(core::Limits::default_score().fuel),
            memory_bytes: self
                .memory_bytes
                .unwrap_or(core::Limits::default_score().memory_bytes),
        }
    }

    /// Effective module-load limits of the case.
    pub fn load_limits(&self) -> WireLimits {
        WireLimits {
            fuel: self.load_fuel.unwrap_or(core::Limits::default_load().fuel),
            memory_bytes: self
                .load_memory_bytes
                .unwrap_or(core::Limits::default_load().memory_bytes),
        }
    }

    /// Resolve the source text (inline or from `source_file`).
    pub fn resolve_source(&self, cases_dir: &Path) -> String {
        match (&self.source, &self.source_file) {
            (Some(s), None) => s.clone(),
            (None, Some(f)) => {
                let p = cases_dir.join(f);
                std::fs::read_to_string(&p)
                    .unwrap_or_else(|e| panic!("{}: read {}: {e}", self.name, p.display()))
            }
            _ => panic!(
                "{}: exactly one of source / source_file is required",
                self.name
            ),
        }
    }
}

/// Outcome of running a case, normalized for native/wasm comparison. Both variants
/// carry resource accounting (spec §6.4: fuel/memory consumed are part of the
/// deterministic outcome, on success *and* failure).
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Success: value + resource accounting.
    Value {
        /// Returned value (wire form).
        value: WireValue,
        /// Fuel consumed.
        fuel_used: u64,
        /// Bytes charged.
        mem_used: u64,
    },
    /// Failure: spec §9 class + message + resource accounting (zero when the
    /// invocation never ran: compile-stage and host-boundary errors).
    Error {
        /// `parse`/`type`/`trap`/`fuel`/`oom`/`host`.
        class: String,
        /// Error message.
        message: String,
        /// Fuel consumed before the failure.
        fuel_used: u64,
        /// Bytes charged before the failure.
        mem_used: u64,
    },
}

impl Outcome {
    fn from_ps(e: core::PsError, fuel_used: u64, mem_used: u64) -> Outcome {
        Outcome::Error {
            class: e.class().to_string(),
            message: e.to_string(),
            fuel_used,
            mem_used,
        }
    }

    /// Build from a guest `WireError` (request-level failure: zero cost).
    pub fn from_wire_error(e: &WireError) -> Outcome {
        Outcome::Error {
            class: e.class().to_string(),
            message: format!("{e:?}"),
            fuel_used: 0,
            mem_used: 0,
        }
    }

    /// The accounting pair of either variant.
    pub fn cost(&self) -> (u64, u64) {
        match self {
            Outcome::Value {
                fuel_used,
                mem_used,
                ..
            }
            | Outcome::Error {
                fuel_used,
                mem_used,
                ..
            } => (*fuel_used, *mem_used),
        }
    }
}

/// The directory holding the case files (workspace-relative).
pub fn cases_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cases")
}

/// Load all cases from `runtime/conformance/cases/*.json`, sorted by file then order.
pub fn load_cases() -> Vec<(PathBuf, Case)> {
    let dir = cases_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cases dir {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no conformance case files in {dir:?}");
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let cases: Vec<Case> =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        for c in cases {
            out.push((f.clone(), c));
        }
    }
    let mut names: Vec<&str> = out.iter().map(|(_, c)| c.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), out.len(), "duplicate case names");
    out
}

/// Run a case against the in-process (native) interpreter.
pub fn run_native(case: &Case) -> Outcome {
    let source = case.resolve_source(&cases_dir());
    let prog = match core::compile_with_limits(&source, case.load_limits().into()) {
        Ok(p) => p,
        Err(e) => return Outcome::from_ps(e, 0, 0),
    };
    let Some(func) = &case.func else {
        // Compile-only case: success is an empty marker value.
        return Outcome::Value {
            value: WireValue::B(true),
            fuel_used: 0,
            mem_used: 0,
        };
    };
    let args: Vec<core::Value> = case
        .args
        .iter()
        .map(|a| a.to_value().expect("case args must be well-formed"))
        .collect();
    let (result, fuel_used, mem_used) = core::call_costed(&prog, func, &args, case.limits().into());
    match result {
        Ok(value) => Outcome::Value {
            value: WireValue::from_value(&value),
            fuel_used,
            mem_used,
        },
        Err(core::CallError::Ps(e)) => Outcome::from_ps(e, fuel_used, mem_used),
        Err(core::CallError::Host(e)) => Outcome::Error {
            class: "host".into(),
            message: e.to_string(),
            fuel_used,
            mem_used,
        },
    }
}

/// Check an outcome against the case's expectation. Returns an error description.
pub fn check(case: &Case, outcome: &Outcome, target: &str) -> Result<(), String> {
    // Golden resource accounting (spec §8 tables), when the case pins it.
    let (fuel_used, mem_used) = outcome.cost();
    if let Some(want) = case.expect.fuel_used {
        if fuel_used != want {
            return Err(format!(
                "{} [{target}]: fuel_used {fuel_used} != golden {want}",
                case.name
            ));
        }
    }
    if let Some(want) = case.expect.mem_used {
        if mem_used != want {
            return Err(format!(
                "{} [{target}]: mem_used {mem_used} != golden {want}",
                case.name
            ));
        }
    }
    match (&case.expect.value, &case.expect.error) {
        (Some(want), None) => match outcome {
            Outcome::Value { value, .. } => {
                let want_v = want.to_value().expect("expected value well-formed");
                let got_v = value.to_value().map_err(|e| {
                    format!("{} [{target}]: malformed result value: {e}", case.name)
                })?;
                if want_v.bit_eq(&got_v) {
                    Ok(())
                } else {
                    Err(format!(
                        "{} [{target}]: expected {want:?}, got {value:?}",
                        case.name
                    ))
                }
            }
            Outcome::Error { class, message, .. } => Err(format!(
                "{} [{target}]: expected a value, got {class} error: {message}",
                case.name
            )),
        },
        (None, Some(want_class)) => match outcome {
            Outcome::Error { class, message, .. } => {
                if class != want_class {
                    return Err(format!(
                        "{} [{target}]: expected {want_class} error, got {class}: {message}",
                        case.name
                    ));
                }
                if let Some(sub) = &case.expect.message_contains {
                    if !message.contains(sub.as_str()) {
                        return Err(format!(
                            "{} [{target}]: error message {message:?} does not contain {sub:?}",
                            case.name
                        ));
                    }
                }
                Ok(())
            }
            Outcome::Value { value, .. } => Err(format!(
                "{} [{target}]: expected {want_class} error, got value {value:?}",
                case.name
            )),
        },
        _ => panic!(
            "{}: exactly one of expect.value / expect.error is required",
            case.name
        ),
    }
}
