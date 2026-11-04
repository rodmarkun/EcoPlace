//! wasmtime host harness for the PlacementScript WASM module (TODO 1.4).
//!
//! Sandbox configuration (the four A.1 properties, enforced here):
//! - **Zero WASI capabilities:** no WASI context is provided at all. Any
//!   `wasi_snapshot_preview1` import the module carries (std/wasi-libc machinery, e.g.
//!   the panic path's `fd_write`) is resolved as a **trapping stub**
//!   (`define_unknown_imports_as_traps`), and instantiation *fails* if the module
//!   imports from any other namespace.
//! - **wasmtime fuel** (outer, defense-in-depth): bounds total instructions per host
//!   call. The *language* fuel budget (spec §8) is enforced inside the interpreter and
//!   travels with each request; wasmtime fuel only catches host-ABI bugs or interpreter
//!   escapes. It is **derived per call** from the request (base + per-byte decode cost +
//!   per-step margin × language fuel × invocations), so batch size and requested
//!   language limits can never make the sandbox kill a legal computation; requests
//!   beyond the ABI ceilings (`max_batch`, `max_language_fuel`) are rejected as
//!   structured host errors instead.
//! - **Memory cap:** a `StoreLimits` ceiling on linear memory growth.
//! - **Hot-swap:** a policy is a module + `ps_load_*` call; promotion/rollback is
//!   an atomic instance swap (Phase 7 consumes this).
//!
//! This crate is the reference host for the conformance suite (1.5) and the prototype
//! for the Phase-7 kube-scheduler plugin host (Go/wasmtime-go with the same ABI).

use anyhow::{anyhow, bail, Context, Result};
use placementscript_core::wire::*;
use serde::de::DeserializeOwned;
use wasmtime::{
    Config, Engine, Instance, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc,
};

/// Host sandbox configuration.
///
/// Outer (wasmtime) fuel is **derived per host call** from the work the call can
/// legally perform: `base + per_byte × request_len + per_step × language_fuel ×
/// invocations` (saturating). A batch of N candidates is N interpreter invocations, so
/// its outer budget scales with N — outer fuel can therefore never turn a language-level
/// outcome (value / trap / fuel / oom) into a sandbox kill for requests within the ABI
/// ceilings below. Requests *outside* the ceilings are rejected up front as structured
/// `host`-class errors (never sent to the guest).
#[derive(Debug, Clone, Copy)]
pub struct HostConfig {
    /// Fixed outer-fuel allowance per host call (instantiation/ABI overhead).
    pub wasm_fuel_base: u64,
    /// Outer fuel per request byte (postcard decode + value building).
    pub wasm_fuel_per_byte: u64,
    /// Outer fuel per unit of language fuel per invocation (generous over-estimate of
    /// wasm instructions per abstract interpreter step; the language budget binds).
    pub wasm_fuel_per_step: u64,
    /// Linear-memory ceiling in bytes.
    pub max_memory_bytes: usize,
    /// ABI ceiling: maximum candidates per `ps_score_batch` (spec §4.1 `MAX_NODES`).
    pub max_batch: usize,
    /// ABI ceiling: maximum language fuel a request may ask for.
    pub max_language_fuel: u64,
    /// ABI ceiling: maximum encoded request size in bytes. Defense in depth for the
    /// R2.1 domain/memory alignment: a domain-maximum request is proven to fit (see
    /// `bundle::domain`), and anything larger than this ceiling is rejected as a
    /// structured host error *before* it is written into guest memory.
    pub max_request_bytes: usize,
    /// ABI ceiling: maximum language *memory* budget a request may ask for
    /// (invocation or load). Bounds the guest's language-level allocation so a legal
    /// policy hits its deterministic language OOM before the linear-memory cap can
    /// sandbox-kill it (post-Round-2 self-review of R2.1/R1.7): a domain-maximum
    /// request plus a policy allocating up to this ceiling is proven to fit the
    /// default sandbox in the conformance suite. Over-ceiling requests are structured
    /// host errors.
    pub max_language_memory: u64,
    /// ABI ceiling: maximum *load* fuel a `ps_call` request may ask for. Load fuel is
    /// charged per source byte / AST node / constant (spec §8.1), so it bounds
    /// compiler token and AST allocation inside the guest (Round-3 review: lexing is
    /// budgeted, and this ceiling keeps the worst legal compile's allocation well
    /// under the linear-memory cap — an over-budget token-dense source is a guest
    /// `fuel` outcome, never a sandbox kill). Much lower than `max_language_fuel`:
    /// load fuel scales with *program size*, not execution steps.
    pub max_load_fuel: u64,
}

impl Default for HostConfig {
    fn default() -> Self {
        HostConfig {
            wasm_fuel_base: 50_000_000,
            wasm_fuel_per_byte: 10_000,
            wasm_fuel_per_step: 10_000,
            max_memory_bytes: 256 << 20, // 256 MiB linear-memory ceiling
            max_batch: placementscript_core::bundle::domain::MAX_NODES,
            max_language_fuel: 1 << 32,
            max_request_bytes: 64 << 20, // 64 MiB: > any encoded domain-max request
            max_language_memory: 32 << 20, // 32 MiB: 2x the default propose budget
            max_load_fuel: 1 << 20,      // 1 Mi source bytes/nodes: ~10x the default load budget
        }
    }
}

impl HostConfig {
    /// Derived outer fuel for one host call (see struct docs).
    fn outer_fuel(&self, request_len: usize, language_fuel: u64, invocations: u64) -> u64 {
        self.wasm_fuel_base
            .saturating_add(self.wasm_fuel_per_byte.saturating_mul(request_len as u64))
            .saturating_add(
                self.wasm_fuel_per_step
                    .saturating_mul(language_fuel)
                    .saturating_mul(invocations),
            )
    }
}

struct StoreState {
    limits: StoreLimits,
}

/// A loaded PlacementScript WASM module instance.
pub struct PolicyModule {
    store: Store<StoreState>,
    #[allow(dead_code)]
    instance: Instance,
    memory: wasmtime::Memory,
    alloc: TypedFunc<u32, u32>,
    free: TypedFunc<(u32, u32), ()>,
    config: HostConfig,
}

/// Failures crossing the host boundary: either the guest reported a structured error
/// (`WireError`, incl. the spec §9 taxonomy) or the sandbox itself failed/trapped.
#[derive(Debug)]
pub enum WasmCallError {
    /// Structured error from the guest (parse/type/trap/fuel/oom/host).
    Guest(WireError),
    /// wasmtime-level failure: instantiation, outer-fuel exhaustion, memory ceiling,
    /// trapped stub import, ABI violation.
    Sandbox(anyhow::Error),
}

impl std::fmt::Display for WasmCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmCallError::Guest(e) => write!(f, "guest error ({}): {e:?}", e.class()),
            WasmCallError::Sandbox(e) => write!(f, "sandbox error: {e:#}"),
        }
    }
}

impl std::error::Error for WasmCallError {}

impl PolicyModule {
    /// Instantiate a module from its bytes under `config`.
    ///
    /// Fails if the module imports from any namespace other than
    /// `wasi_snapshot_preview1` (those are resolved as trapping stubs — zero
    /// capabilities are ever granted).
    pub fn new(module_bytes: &[u8], config: HostConfig) -> Result<Self> {
        let mut cfg = Config::new();
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg)?;
        let module = Module::new(&engine, module_bytes)?;

        for import in module.imports() {
            if import.module() != "wasi_snapshot_preview1" {
                bail!(
                    "module imports from '{}::{}' — only trap-stubbed wasi_snapshot_preview1 \
                     imports are tolerated",
                    import.module(),
                    import.name()
                );
            }
        }

        let mut linker: Linker<StoreState> = Linker::new(&engine);
        linker.define_unknown_imports_as_traps(&module)?;

        let mut store = Store::new(
            &engine,
            StoreState {
                limits: StoreLimitsBuilder::new()
                    .memory_size(config.max_memory_bytes)
                    .memories(1)
                    .instances(1)
                    .build(),
            },
        );
        store.limiter(|s| &mut s.limits);
        // Instantiation allowance (module start + allocator setup): the base budget.
        store.set_fuel(config.wasm_fuel_base)?;

        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .context("module exports no linear memory")?;
        let alloc = instance.get_typed_func::<u32, u32>(&mut store, "ps_alloc")?;
        let free = instance.get_typed_func::<(u32, u32), ()>(&mut store, "ps_free")?;
        Ok(PolicyModule {
            store,
            instance,
            memory,
            alloc,
            free,
            config,
        })
    }

    /// Instantiate from a `.wasm` file path.
    pub fn from_file(path: &std::path::Path, config: HostConfig) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::new(&bytes, config)
    }

    fn raw_call(&mut self, export: &str, request: &[u8], outer_fuel: u64) -> Result<Vec<u8>> {
        self.store.set_fuel(outer_fuel)?;
        let func = self
            .instance
            .get_typed_func::<(u32, u32), u64>(&mut self.store, export)?;
        let req_ptr = self.alloc.call(&mut self.store, request.len() as u32)?;
        self.memory
            .write(&mut self.store, req_ptr as usize, request)?;
        let packed = func.call(&mut self.store, (req_ptr, request.len() as u32))?;
        self.free
            .call(&mut self.store, (req_ptr, request.len() as u32))?;
        let (resp_ptr, resp_len) = ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32);
        if resp_ptr == 0 || resp_len == 0 {
            bail!("guest returned an empty response");
        }
        let mut out = vec![0u8; resp_len as usize];
        self.memory.read(&self.store, resp_ptr as usize, &mut out)?;
        self.free.call(&mut self.store, (resp_ptr, resp_len))?;
        Ok(out)
    }

    fn raw_call_noarg(&mut self, export: &str, outer_fuel: u64) -> Result<Vec<u8>> {
        self.store.set_fuel(outer_fuel)?;
        let func = self
            .instance
            .get_typed_func::<(), u64>(&mut self.store, export)?;
        let packed = func.call(&mut self.store, ())?;
        let (resp_ptr, resp_len) = ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32);
        let mut out = vec![0u8; resp_len as usize];
        self.memory.read(&self.store, resp_ptr as usize, &mut out)?;
        self.free.call(&mut self.store, (resp_ptr, resp_len))?;
        Ok(out)
    }

    /// Reject a request at the host boundary before it reaches the guest. Reported in
    /// the structured `host` error class (spec §9: host-boundary faults are never the
    /// policy's fault), wrapped in the same `Guest` envelope callers already match on.
    fn reject(message: String) -> WasmCallError {
        WasmCallError::Guest(WireError::Host { message })
    }

    /// Validate a requested language-limits pair against the ABI ceilings.
    fn check_language_limits(&self, limits: &WireLimits) -> Result<(), WasmCallError> {
        if limits.fuel > self.config.max_language_fuel {
            return Err(Self::reject(format!(
                "requested language fuel {} exceeds the ABI ceiling {}",
                limits.fuel, self.config.max_language_fuel
            )));
        }
        if limits.memory_bytes > self.config.max_language_memory {
            return Err(Self::reject(format!(
                "requested language memory {} exceeds the ABI ceiling {}",
                limits.memory_bytes, self.config.max_language_memory
            )));
        }
        Ok(())
    }

    fn typed_call<Req: serde::Serialize, Resp: DeserializeOwned>(
        &mut self,
        export: &str,
        req: &Req,
        language_fuel: u64,
        invocations: u64,
    ) -> Result<Resp, WasmCallError> {
        let bytes = postcard::to_allocvec(req)
            .map_err(|e| WasmCallError::Sandbox(anyhow!("encode request: {e}")))?;
        if bytes.len() > self.config.max_request_bytes {
            return Err(Self::reject(format!(
                "encoded request of {} bytes exceeds the ABI ceiling {}",
                bytes.len(),
                self.config.max_request_bytes
            )));
        }
        let outer = self
            .config
            .outer_fuel(bytes.len(), language_fuel, invocations);
        let resp = self
            .raw_call(export, &bytes, outer)
            .map_err(WasmCallError::Sandbox)?;
        let result: Result<Resp, WireError> = postcard::from_bytes(&resp)
            .map_err(|e| WasmCallError::Sandbox(anyhow!("decode response: {e}")))?;
        result.map_err(WasmCallError::Guest)
    }

    /// Load the policy source embedded in the module at build time.
    pub fn load_embedded_policy(&mut self) -> Result<LoadResponse, WasmCallError> {
        let outer = self
            .config
            .outer_fuel(0, placementscript_core::Limits::default_load().fuel, 1);
        let resp = self
            .raw_call_noarg("ps_load_embedded_policy", outer)
            .map_err(WasmCallError::Sandbox)?;
        let result: Result<LoadResponse, WireError> = postcard::from_bytes(&resp)
            .map_err(|e| WasmCallError::Sandbox(anyhow!("decode response: {e}")))?;
        result.map_err(WasmCallError::Guest)
    }

    /// Load a host-provided policy source (conformance / screening path).
    pub fn load_policy(&mut self, source: &str) -> Result<LoadResponse, WasmCallError> {
        if source.len() > self.config.max_request_bytes {
            return Err(Self::reject(format!(
                "policy source of {} bytes exceeds the ABI ceiling {}",
                source.len(),
                self.config.max_request_bytes
            )));
        }
        let outer = self.config.outer_fuel(
            source.len(),
            placementscript_core::Limits::default_load().fuel,
            1,
        );
        let resp = self
            .raw_call("ps_load_policy", source.as_bytes(), outer)
            .map_err(WasmCallError::Sandbox)?;
        let result: Result<LoadResponse, WireError> = postcard::from_bytes(&resp)
            .map_err(|e| WasmCallError::Sandbox(anyhow!("decode response: {e}")))?;
        result.map_err(WasmCallError::Guest)
    }

    /// Score one pod against N nodes with the loaded policy. Outer fuel scales with N
    /// (one interpreter invocation per candidate); `N > max_batch` or an over-ceiling
    /// language fuel is rejected as a structured host error.
    pub fn score_batch(
        &mut self,
        req: &ScoreBatchRequest,
    ) -> Result<ScoreBatchResponse, WasmCallError> {
        if req.nodes.len() > self.config.max_batch {
            return Err(Self::reject(format!(
                "batch of {} candidates exceeds the ABI ceiling {}",
                req.nodes.len(),
                self.config.max_batch
            )));
        }
        self.check_language_limits(&req.limits)?;
        self.typed_call(
            "ps_score_batch",
            req,
            req.limits.fuel,
            req.nodes.len().max(1) as u64,
        )
    }

    /// Run `propose_moves` with the loaded policy.
    pub fn propose_moves(
        &mut self,
        req: &ProposeRequest,
    ) -> Result<ProposeResponse, WasmCallError> {
        self.check_language_limits(&req.limits)?;
        self.typed_call("ps_propose_moves", req, req.limits.fuel, 1)
    }

    /// Stateless compile-and-call (conformance suite entry point).
    pub fn call(&mut self, req: &CallRequest) -> Result<CallResponse, WasmCallError> {
        self.check_language_limits(&req.limits)?;
        // The *requested* load budget bounds compile work, so it is both ceiling-checked
        // and included in the outer-fuel derivation (a raised load budget must never let
        // the sandbox kill a legal compile; R2 review of R1.7). Load fuel has its own,
        // much lower ceiling (`max_load_fuel`): it bounds compiler token/AST allocation
        // (Round-3 review). Load memory (interned strings) is ceiling-checked for the
        // same reason as invocation memory.
        if req.load_limits.fuel > self.config.max_load_fuel {
            return Err(Self::reject(format!(
                "requested load fuel {} exceeds the ABI ceiling {}",
                req.load_limits.fuel, self.config.max_load_fuel
            )));
        }
        self.check_language_limits(&WireLimits {
            fuel: 0, // load fuel already checked against its own ceiling above
            memory_bytes: req.load_limits.memory_bytes,
        })?;
        let fuel = req.limits.fuel.saturating_add(req.load_limits.fuel);
        self.typed_call("ps_call", req, fuel, 1)
    }
}

/// Locate the built guest module.
///
/// `PS_WASM_GUEST` (an exact path) always wins — CI pins it so the tested artifact is
/// exactly the built one (debug and release are each tested explicitly). Without it,
/// local convenience discovery picks the **most recently built** of the workspace
/// `target/wasm32-wasip1/{release,debug}` artifacts, so a stale binary of the other
/// profile is never silently preferred.
pub fn locate_guest_module() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("PS_WASM_GUEST") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        bail!("PS_WASM_GUEST={} does not exist", p.display());
    }
    let ws = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for profile in ["release", "debug"] {
        let p = ws.join(format!(
            "target/wasm32-wasip1/{profile}/placementscript_wasm_guest.wasm"
        ));
        if let Ok(meta) = std::fs::metadata(&p) {
            let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if !matches!(&best, Some((t, _)) if mtime <= *t) {
                best = Some((mtime, p));
            }
        }
    }
    if let Some((_, p)) = best {
        return Ok(p);
    }
    bail!(
        "guest module not found; build it first:\n  cargo build -p placementscript-wasm-guest \
         --target wasm32-wasip1 [--release]\nor set PS_WASM_GUEST=/path/to/module.wasm"
    )
}
