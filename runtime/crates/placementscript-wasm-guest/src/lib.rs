//! WASM guest (TODO 1.4): `placementscript-core` compiled to `wasm32-wasip1` plus an
//! embedded policy source, exporting the host-call ABI.
//!
//! ## ABI (postcard)
//!
//! Serialization is **postcard** (documented choice, see `runtime/README.md`): compact,
//! `no_std`-friendly, deterministic byte layout, serde-native — versus flatbuffers,
//! whose zero-copy advantage is irrelevant at our message sizes (a few KB of cluster
//! state) and whose schema compiler would be a second source of truth for spec §4.
//!
//! Memory protocol (all pointers are guest linear-memory offsets):
//! 1. Host calls `ps_alloc(len)` and writes the postcard-encoded request.
//! 2. Host calls an entry point `f(ptr, len) -> u64`; the return packs
//!    `(response_ptr << 32) | response_len`.
//! 3. Host reads the response — always a postcard-encoded
//!    `Result<Response, WireError>` — then calls `ps_free(ptr, len)` on both buffers.
//!
//! Entry points:
//! - `ps_load_embedded_policy() -> u64` — compile + bundle-validate the embedded source
//!   (deployment path; response `LoadResponse`).
//! - `ps_load_policy(ptr, len) -> u64` — same for a host-provided source (conformance
//!   and promotion-screening path).
//! - `ps_score_batch(ptr, len) -> u64` — `ScoreBatchRequest` against the loaded policy.
//! - `ps_propose_moves(ptr, len) -> u64` — `ProposeRequest` against the loaded policy.
//! - `ps_call(ptr, len) -> u64` — `CallRequest` (source + function + args), stateless;
//!   the conformance suite's entry point (`CallResponse`).
//!
//! The module never *uses* WASI capabilities (no I/O, clock, or randomness — language
//! guarantee); std/wasi-libc machinery may still *import* `wasi_snapshot_preview1`
//! symbols (e.g. `fd_write` in the panic path), which the host resolves as trapping
//! stubs — zero capabilities granted (see `placementscript-wasm-host`).

// Guest-side pointer handling requires `unsafe` (reconstructing Vec/slice from raw
// linear-memory offsets). This is the only crate in the workspace allowed to use it,
// and only in the three helpers below.

use std::cell::RefCell;

use placementscript_core as core;
use placementscript_core::wire::*;
use serde::Serialize;

/// The policy source embedded at build time (see `build.rs`).
pub const EMBEDDED_POLICY: &str = include_str!(concat!(env!("OUT_DIR"), "/embedded_policy.ps"));

thread_local! {
    // WASM is single-threaded; thread_local is the zero-unsafe way to hold the loaded
    // program (which is deliberately !Send: the interpreter uses Rc internally).
    static POLICY: RefCell<Option<core::Program>> = const { RefCell::new(None) };
}

/// Allocate `len` bytes of guest memory for a request buffer.
///
/// # Safety
/// Standard WASM allocator-export pattern; the host must pass the same `(ptr, len)` to
/// `ps_free` exactly once.
#[no_mangle]
pub extern "C" fn ps_alloc(len: u32) -> u32 {
    let buf = vec![0u8; len as usize];
    let ptr = Box::into_raw(buf.into_boxed_slice()) as *mut u8;
    ptr as u32
}

/// Free a buffer previously returned by `ps_alloc` or an entry point.
#[no_mangle]
pub extern "C" fn ps_free(ptr: u32, len: u32) {
    if ptr == 0 || len == 0 {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr as *mut u8, len as usize);
        drop(Box::from_raw(slice));
    }
}

fn read_input(ptr: u32, len: u32) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

fn respond<T: Serialize>(result: Result<T, WireError>) -> u64 {
    let bytes = postcard::to_allocvec(&result).unwrap_or_default();
    let len = bytes.len() as u32;
    let ptr = Box::into_raw(bytes.into_boxed_slice()) as *mut u8 as u32;
    ((ptr as u64) << 32) | len as u64
}

fn load(source: &str) -> Result<LoadResponse, WireError> {
    let prog = core::compile(source)?;
    let meta = core::bundle::validate(&prog)?;
    POLICY.with(|p| *p.borrow_mut() = Some(prog));
    Ok(LoadResponse {
        name: meta.name,
        version: meta.version,
    })
}

/// Compile + bundle-validate the embedded policy source (deployment path).
#[no_mangle]
pub extern "C" fn ps_load_embedded_policy() -> u64 {
    respond(load(EMBEDDED_POLICY))
}

/// Compile + bundle-validate a host-provided policy source.
#[no_mangle]
pub extern "C" fn ps_load_policy(ptr: u32, len: u32) -> u64 {
    let bytes = read_input(ptr, len);
    let result = match std::str::from_utf8(bytes) {
        Ok(source) => load(source),
        Err(_) => Err(WireError::Host {
            message: "policy source is not valid UTF-8".into(),
        }),
    };
    respond(result)
}

fn host_err(message: impl Into<String>) -> WireError {
    WireError::Host {
        message: message.into(),
    }
}

fn decode<'a, T: serde::Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, WireError> {
    postcard::from_bytes(bytes).map_err(|e| host_err(format!("malformed request: {e}")))
}

/// Score one pod against N candidate nodes with the loaded policy.
#[no_mangle]
pub extern "C" fn ps_score_batch(ptr: u32, len: u32) -> u64 {
    let result = (|| -> Result<ScoreBatchResponse, WireError> {
        let req: ScoreBatchRequest = decode(read_input(ptr, len))?;
        let pod = req.pod.to_value().map_err(host_err)?;
        let ctx = req.ctx.to_value().map_err(host_err)?;
        let nodes = req
            .nodes
            .iter()
            .map(|n| n.to_value())
            .collect::<Result<Vec<_>, _>>()
            .map_err(host_err)?;
        let (scores, cost) = POLICY.with(|p| {
            let guard = p.borrow();
            let prog = guard.as_ref().ok_or_else(|| host_err("no policy loaded"))?;
            core::bundle::score_batch(prog, &pod, &nodes, &ctx, req.limits.into()).map_err(
                |(i, e)| match e {
                    core::CallError::Ps(e) => e.into(),
                    core::CallError::Host(e) => host_err(format!("node index {i}: {e}")),
                },
            )
        })?;
        // The normative host tie-break (spec §5.5), via the single shared function.
        let best_node = core::bundle::select_node(&scores, &nodes).unwrap_or(-1);
        Ok(ScoreBatchResponse {
            scores,
            best_node,
            fuel_used: cost.fuel_used,
            mem_used: cost.mem_used,
        })
    })();
    respond(result)
}

/// Run `propose_moves` with the loaded policy.
#[no_mangle]
pub extern "C" fn ps_propose_moves(ptr: u32, len: u32) -> u64 {
    let result = (|| -> Result<ProposeResponse, WireError> {
        let req: ProposeRequest = decode(read_input(ptr, len))?;
        let state = req.state.to_value().map_err(host_err)?;
        let graph = req.graph.to_value().map_err(host_err)?;
        let (moves, cost) = POLICY.with(|p| {
            let guard = p.borrow();
            let prog = guard.as_ref().ok_or_else(|| host_err("no policy loaded"))?;
            core::bundle::propose_moves(prog, &state, &graph, req.budget, req.limits.into())
                .map_err(WireError::from)
        })?;
        Ok(ProposeResponse {
            moves: moves
                .iter()
                .map(|m| WireMove {
                    kind: m.kind,
                    pod: m.pod,
                    dest: m.dest,
                })
                .collect(),
            fuel_used: cost.fuel_used,
            mem_used: cost.mem_used,
        })
    })();
    respond(result)
}

/// Stateless compile-and-call (conformance suite entry point). The response is an
/// outcome *envelope*: result plus fuel/memory accounting for success and failure
/// alike (spec §6.4); only a malformed request itself yields an outer error.
#[no_mangle]
pub extern "C" fn ps_call(ptr: u32, len: u32) -> u64 {
    let result = (|| -> Result<CallResponse, WireError> {
        let req: CallRequest = decode(read_input(ptr, len))?;
        let prog = match core::compile_with_limits(&req.source, req.load_limits.into()) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallResponse {
                    result: Err(e.into()),
                    fuel_used: 0,
                    mem_used: 0,
                })
            }
        };
        let args = req
            .args
            .iter()
            .map(|a| a.to_value())
            .collect::<Result<Vec<_>, _>>()
            .map_err(host_err)?;
        let (result, fuel_used, mem_used) =
            core::call_costed(&prog, &req.func, &args, req.limits.into());
        Ok(CallResponse {
            result: result
                .map(|v| WireValue::from_value(&v))
                .map_err(WireError::from),
            fuel_used,
            mem_used,
        })
    })();
    respond(result)
}
