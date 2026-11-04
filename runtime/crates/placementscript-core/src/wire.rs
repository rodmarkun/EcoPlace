//! Wire mirror types (feature `wire`): serde-encodable forms of values, errors, and the
//! host-call requests/responses shared by the WASM ABI (postcard) and the conformance
//! suite (JSON).
//!
//! Why these exist: `Value` uses `Rc`/`RefCell` (interpreter-internal), and the guest ↔
//! host boundary needs a validated, ownership-free encoding. Conversion into `Value`
//! re-validates records and float finiteness — a malformed wire payload is a host error,
//! never undefined behavior.

use serde::{Deserialize, Serialize};

use crate::ast::RecordType;
use crate::interp::{CallError, HostError};
use crate::value::Value;
use crate::PsError;

/// Serde-encodable value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WireValue {
    /// Integer.
    I(i64),
    /// Float (must be finite; validated on conversion).
    F(f64),
    /// Boolean.
    B(bool),
    /// String.
    S(String),
    /// List.
    L(Vec<WireValue>),
    /// Record: type name + fields in schema order.
    R {
        /// Record type name (spec §4).
        ty: String,
        /// Field values in schema order.
        fields: Vec<WireValue>,
    },
}

impl WireValue {
    /// Convert to a runtime `Value`, validating records and float finiteness.
    pub fn to_value(&self) -> Result<Value, String> {
        Ok(match self {
            WireValue::I(v) => Value::Int(*v),
            WireValue::F(v) => {
                if !v.is_finite() {
                    return Err("non-finite float in wire value".into());
                }
                Value::Float(*v)
            }
            WireValue::B(v) => Value::Bool(*v),
            WireValue::S(s) => Value::str(s),
            // Host boundary: lists arriving from the wire are frozen (spec §5.2) so
            // host state can never be mutated through an alias inside the policy.
            WireValue::L(xs) => Value::frozen_list(
                xs.iter()
                    .map(|x| x.to_value())
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            WireValue::R { ty, fields } => {
                let rt =
                    RecordType::by_name(ty).ok_or_else(|| format!("unknown record type '{ty}'"))?;
                let fields = fields
                    .iter()
                    .map(|x| x.to_value())
                    .collect::<Result<Vec<_>, _>>()?;
                Value::record(rt, fields)?
            }
        })
    }

    /// Convert from a runtime `Value`.
    pub fn from_value(v: &Value) -> WireValue {
        match v {
            Value::Int(v) => WireValue::I(*v),
            Value::Float(v) => WireValue::F(*v),
            Value::Bool(v) => WireValue::B(*v),
            Value::Str(s) => WireValue::S(s.to_string()),
            Value::List(xs) => WireValue::L(xs.borrow().iter().map(Self::from_value).collect()),
            Value::Record(r) => WireValue::R {
                ty: r.rt.name().to_string(),
                fields: r.fields.iter().map(Self::from_value).collect(),
            },
        }
    }
}

/// Serde-encodable error, mirroring `PsError` plus the host class (spec §9).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WireError {
    /// Parse error.
    Parse {
        /// 1-based line.
        line: u32,
        /// Message.
        message: String,
    },
    /// Type / bundle-contract error.
    Type {
        /// 1-based line.
        line: u32,
        /// Message.
        message: String,
    },
    /// Runtime trap.
    Trap {
        /// Message.
        message: String,
    },
    /// Fuel exhausted.
    Fuel,
    /// Memory budget exceeded.
    Oom,
    /// Host-boundary error (bad arguments, malformed payload, unknown function).
    Host {
        /// Message.
        message: String,
    },
}

impl WireError {
    /// Stable short class name (`parse`/`type`/`trap`/`fuel`/`oom`/`host`).
    pub fn class(&self) -> &'static str {
        match self {
            WireError::Parse { .. } => "parse",
            WireError::Type { .. } => "type",
            WireError::Trap { .. } => "trap",
            WireError::Fuel => "fuel",
            WireError::Oom => "oom",
            WireError::Host { .. } => "host",
        }
    }
}

impl From<PsError> for WireError {
    fn from(e: PsError) -> Self {
        match e {
            PsError::Parse { line, message } => WireError::Parse { line, message },
            PsError::Type { line, message } => WireError::Type { line, message },
            PsError::Trap { message } => WireError::Trap { message },
            PsError::FuelExhausted => WireError::Fuel,
            PsError::OutOfMemory => WireError::Oom,
        }
    }
}

impl From<HostError> for WireError {
    fn from(e: HostError) -> Self {
        WireError::Host {
            message: e.to_string(),
        }
    }
}

impl From<CallError> for WireError {
    fn from(e: CallError) -> Self {
        match e {
            CallError::Ps(e) => e.into(),
            CallError::Host(e) => e.into(),
        }
    }
}

/// Execution limits on the wire.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct WireLimits {
    /// Fuel budget.
    pub fuel: u64,
    /// Memory budget (bytes).
    pub memory_bytes: u64,
}

impl From<WireLimits> for crate::Limits {
    fn from(w: WireLimits) -> Self {
        crate::Limits {
            fuel: w.fuel,
            memory_bytes: w.memory_bytes,
        }
    }
}

/// `ps_call` request: compile `source` and invoke one function (conformance path —
/// stateless so every case is hermetic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRequest {
    /// Program source.
    pub source: String,
    /// Function to invoke.
    pub func: String,
    /// Arguments.
    pub args: Vec<WireValue>,
    /// Invocation limits.
    pub limits: WireLimits,
    /// Module-load limits (spec §8.1 load budget; conformance tests these explicitly).
    pub load_limits: WireLimits,
}

/// `ps_call` outcome envelope: the result **plus resource accounting for success and
/// failure alike** (spec §6.4 — fuel/memory consumed are part of the deterministic,
/// cross-target-comparable outcome). Compile-stage and host-boundary errors carry zero
/// cost (the invocation never ran); a fuel/oom failure reports the full budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallResponse {
    /// Returned value or structured error.
    pub result: Result<WireValue, WireError>,
    /// Fuel consumed by the invocation.
    pub fuel_used: u64,
    /// Bytes charged to the invocation.
    pub mem_used: u64,
}

/// `ps_load_policy` success payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadResponse {
    /// `POLICY_NAME`.
    pub name: String,
    /// `POLICY_VERSION`.
    pub version: String,
}

/// `ps_score_batch` request against the *loaded* policy (deployment path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBatchRequest {
    /// The pod being scheduled.
    pub pod: WireValue,
    /// Candidate (feasible) nodes.
    pub nodes: Vec<WireValue>,
    /// Scoring context.
    pub ctx: WireValue,
    /// Per-node invocation limits.
    pub limits: WireLimits,
}

/// `ps_score_batch` success payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBatchResponse {
    /// One (finite, `-0.0`-normalized) score per candidate node.
    pub scores: Vec<f64>,
    /// Winning node id under the normative spec §5.5 tie-break (highest score,
    /// bitwise ties → lowest node id), computed by `bundle::select_node` — the same
    /// function every host uses (C18). `-1` for an empty batch.
    pub best_node: i64,
    /// Total fuel consumed across the batch.
    pub fuel_used: u64,
    /// Total bytes charged across the batch.
    pub mem_used: u64,
}

/// `ps_propose_moves` request against the loaded policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposeRequest {
    /// Cluster state.
    pub state: WireValue,
    /// Call graph.
    pub graph: WireValue,
    /// Move budget for this window.
    pub budget: i64,
    /// Invocation limits.
    pub limits: WireLimits,
}

/// A move on the wire.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct WireMove {
    /// 0 = move pod, 1 = wake node.
    pub kind: i64,
    /// Pod id (`-1` for wake).
    pub pod: i64,
    /// Destination / target node id.
    pub dest: i64,
}

/// `ps_propose_moves` success payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposeResponse {
    /// Proposed actions, in order.
    pub moves: Vec<WireMove>,
    /// Fuel consumed.
    pub fuel_used: u64,
    /// Bytes charged.
    pub mem_used: u64,
}
