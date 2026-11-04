//! `placementscript-core` — the single interpreter for the PlacementScript DSL (C18).
//!
//! PlacementScript is the statically-typed, capability-free Python-subset DSL in which all
//! evolved placement policies are written. The normative language definition is
//! `docs/placementscript-spec.md` (TODO 1.1); this crate is its one implementation:
//! lexer → parser → typed AST → tree-walking interpreter with fuel metering, memory
//! accounting, and a structured error taxonomy (parse/type/trap/fuel/oom — evolution
//! feedback branches on the distinction).
//!
//! The same crate compiles natively (PyO3/FFI into the simulator, TODO 1.3) and to
//! `wasm32-wasip1` (embedded in the scheduler-plugin module, TODO 1.4) — one
//! implementation of the semantics, no cross-executor equivalence problem (C18, A.5).
//! Determinism is a language guarantee: no I/O, no clocks, no OS capabilities, no NaN.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ast;
pub mod astapi;
pub mod bundle;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod typecheck;
pub mod value;
#[cfg(feature = "wire")]
pub mod wire;

pub use interp::{call, call_costed, CallError, CallOutcome, HostError};
pub use typecheck::Program;
pub use value::Value;

/// Why a PlacementScript program failed. The variants are load-bearing for evolution:
/// the text-feedback generator (TODO 5.4) and the risk gates (TODO 1.6) branch on them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsError {
    /// Source text is not a well-formed PlacementScript program.
    Parse {
        /// 1-based line of the first offending token.
        line: u32,
        /// Human-readable description, stable enough to feed back to the mutation prompt.
        message: String,
    },
    /// The program is well-formed but ill-typed (PlacementScript is statically typed),
    /// or violates the PolicyBundle contract (spec §5.5).
    Type {
        /// 1-based line of the type error.
        line: u32,
        /// Description of the mismatch.
        message: String,
    },
    /// A checked runtime trap: integer overflow, division by zero, non-finite float
    /// result, out-of-bounds index, list mutation during iteration.
    /// (NaN never traps at runtime — it is rejected at parse time per C18.)
    Trap {
        /// Description of the trap condition.
        message: String,
    },
    /// The fuel budget was exhausted (bounded-execution guarantee).
    FuelExhausted,
    /// The memory budget was exceeded.
    OutOfMemory,
}

impl PsError {
    /// Stable short class name: `parse` / `type` / `trap` / `fuel` / `oom`.
    pub fn class(&self) -> &'static str {
        match self {
            PsError::Parse { .. } => "parse",
            PsError::Type { .. } => "type",
            PsError::Trap { .. } => "trap",
            PsError::FuelExhausted => "fuel",
            PsError::OutOfMemory => "oom",
        }
    }
}

impl core::fmt::Display for PsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PsError::Parse { line, message } => write!(f, "parse error (line {line}): {message}"),
            PsError::Type { line, message } => write!(f, "type error (line {line}): {message}"),
            PsError::Trap { message } => write!(f, "trap: {message}"),
            PsError::FuelExhausted => write!(f, "fuel exhausted"),
            PsError::OutOfMemory => write!(f, "memory budget exceeded"),
        }
    }
}

impl std::error::Error for PsError {}

/// Execution limits for one policy invocation. Both budgets are part of the language
/// specification (spec §8), not incidental runtime behavior: the same numbers apply in
/// the simulator, the conformance suite, and the wasmtime host of the scheduler plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Abstract interpreter steps available to one invocation (spec §8.1).
    pub fuel: u64,
    /// Cumulative allocation budget in bytes for one invocation (spec §8.2).
    pub memory_bytes: u64,
}

impl Limits {
    /// Default budget for one `score` invocation (spec §8.1).
    pub const fn default_score() -> Limits {
        Limits {
            fuel: 200_000,
            memory_bytes: 1 << 20, // 1 MiB
        }
    }

    /// Default budget for one `propose_moves` invocation (spec §8.1).
    pub const fn default_propose() -> Limits {
        Limits {
            fuel: 20_000_000,
            memory_bytes: 16 << 20, // 16 MiB
        }
    }

    /// Default budget for module load (constants + interned strings; spec §8.1).
    pub const fn default_load() -> Limits {
        Limits {
            fuel: 100_000,
            memory_bytes: 64 << 10, // 64 KiB
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Limits::default_score()
    }
}

/// Parse and type-check a PlacementScript program under the default load limits.
pub fn compile(source: &str) -> Result<Program, PsError> {
    compile_with_limits(source, Limits::default_load())
}

/// Parse and type-check a PlacementScript program under explicit load limits.
/// Load fuel is charged one per source byte (during lexing, so a token-dense
/// over-budget source is a structured `fuel` outcome with bounded compiler
/// allocation — Round-3 review), plus one per AST node and one per constant.
pub fn compile_with_limits(source: &str, load_limits: Limits) -> Result<Program, PsError> {
    let module = parser::parse(source, load_limits.fuel)?;
    typecheck::check(module, source.to_string(), load_limits)
}

/// Parse and type-check only; discard the program (cheap validity probe).
pub fn check_program(source: &str) -> Result<(), PsError> {
    compile(source).map(|_| ())
}
