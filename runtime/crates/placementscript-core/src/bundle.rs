//! PolicyBundle contract (C22, spec §5.5): bundle validation and the typed entry points
//! every host uses (`score`, `score_batch`, `propose_moves`).
//!
//! All hosts go through this module so the contract lives in exactly one place:
//! signature validation, `-0.0 → 0.0` normalization on the score boundary (spec §6.3),
//! and Move extraction.

use crate::ast::{RecordType, Type};
use crate::interp::{CallError, HostError};
use crate::typecheck::Program;
use crate::value::{RecordVal, Value};
use crate::{Limits, PsError};

/// Host input domain (spec §4.1). These bounds are what hosts guarantee and what the
/// boundary validators enforce — the risk gates (TODO 1.6) quantify totality over
/// exactly this domain.
///
/// The shape bounds are **jointly representable in the WASM sandbox** (Round-2 review
/// R2.1): per-node latency vectors make cluster state quadratic in `MAX_NODES`, so the
/// domain maximum must fit the default 256 MiB linear-memory ceiling with the request
/// bytes, the decoded wire tree, and the runtime values all live at once. A
/// domain-maximum request is executed end-to-end against the wasm guest in the
/// conformance suite (`stateful.rs::domain_ceiling_request_fits_the_sandbox_r2`).
/// Experiment scales (continuum testbeds ~10 nodes, sim topologies of a few hundred)
/// sit far below these caps.
pub mod domain {
    /// Maximum number of nodes in `ClusterCtx`/`ClusterState` (and per score batch).
    pub const MAX_NODES: usize = 512;
    /// Maximum number of pods.
    pub const MAX_PODS: usize = 16_384;
    /// Maximum number of call-graph edges.
    pub const MAX_EDGES: usize = 65_536;
    /// Maximum `propose_moves` budget.
    pub const MAX_BUDGET: i64 = 16_384;
    /// Maximum CPU/memory capacity/allocation/request and image size (integer fields).
    /// Small enough that summing over `MAX_PODS` of them cannot overflow `i64`.
    pub const MAX_RESOURCE: i64 = 1 << 40;
    /// Maximum wall power, watts.
    pub const MAX_WATTS: f64 = 1e6;
    /// Maximum inter-node latency, ms.
    pub const MAX_LATENCY_MS: f64 = 1e7;
    /// Maximum call rate, calls/s.
    pub const MAX_RATE: f64 = 1e9;
    /// Maximum payload, KB.
    pub const MAX_PAYLOAD_KB: f64 = 1e9;
    /// Maximum scenario clock, ms (2^53: exactly representable in f64).
    pub const MAX_NOW_MS: i64 = 1 << 53;
    /// Maximum tier index.
    pub const MAX_TIER: i64 = 15;
    /// Maximum service/app id.
    pub const MAX_GROUP_ID: i64 = u32::MAX as i64;
}

/// Required bundle metadata (spec §5.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleMeta {
    /// `POLICY_NAME` module constant.
    pub name: String,
    /// `POLICY_VERSION` module constant.
    pub version: String,
}

/// A proposed action, host-friendly form of the `Move` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveOut {
    /// 0 = move pod, 1 = wake node.
    pub kind: i64,
    /// Pod id (`-1` for wake).
    pub pod: i64,
    /// Destination / target node id.
    pub dest: i64,
}

fn bundle_err(message: impl Into<String>) -> PsError {
    PsError::Type {
        line: 1,
        message: message.into(),
    }
}

/// Validate the PolicyBundle contract on a compiled program.
///
/// Checks: `POLICY_NAME`/`POLICY_VERSION` string constants; `score` and `propose_moves`
/// present with the exact spec §5.5 signatures. Violations are `PsError::Type` (the
/// bundle-contract class of type errors).
pub fn validate(prog: &Program) -> Result<BundleMeta, PsError> {
    let name = match prog.const_value("POLICY_NAME") {
        Some(Value::Str(s)) => s.to_string(),
        Some(_) => return Err(bundle_err("POLICY_NAME must be a str constant")),
        None => return Err(bundle_err("missing required constant POLICY_NAME: str")),
    };
    let version = match prog.const_value("POLICY_VERSION") {
        Some(Value::Str(s)) => s.to_string(),
        Some(_) => return Err(bundle_err("POLICY_VERSION must be a str constant")),
        None => return Err(bundle_err("missing required constant POLICY_VERSION: str")),
    };

    let score_sig = (
        vec![
            Type::Record(RecordType::Pod),
            Type::Record(RecordType::NodeState),
            Type::Record(RecordType::ClusterCtx),
        ],
        Type::Float,
    );
    match prog.signature("score") {
        Some(sig) if sig == score_sig => {}
        Some(_) => {
            return Err(bundle_err(
                "score must have signature (pod: Pod, node: NodeState, ctx: ClusterCtx) -> float",
            ))
        }
        None => return Err(bundle_err("missing required function 'score'")),
    }

    let propose_sig = (
        vec![
            Type::Record(RecordType::ClusterState),
            Type::Record(RecordType::CallGraphCtx),
            Type::Int,
        ],
        Type::List(Box::new(Type::Record(RecordType::Move))),
    );
    match prog.signature("propose_moves") {
        Some(sig) if sig == propose_sig => {}
        Some(_) => {
            return Err(bundle_err(
                "propose_moves must have signature (state: ClusterState, graph: CallGraphCtx, \
                 budget: int) -> list[Move]",
            ))
        }
        None => return Err(bundle_err("missing required function 'propose_moves'")),
    }

    Ok(BundleMeta { name, version })
}

/// Resource-use diagnostics of one invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cost {
    /// Fuel consumed.
    pub fuel_used: u64,
    /// Bytes charged.
    pub mem_used: u64,
}

// ---------------------------------------------------------------------------
// Boundary validation (spec §4 index conventions + §4.1 host input domain).
// Shared by every embedding: the PyO3 module and the wasm guest both enter
// through this module, so the checks exist exactly once (C18).
// ---------------------------------------------------------------------------

fn rec<'a>(v: &'a Value, what: &str) -> Result<&'a RecordVal, String> {
    match v {
        Value::Record(r) => Ok(r),
        _ => Err(format!("{what}: expected a record, got {}", v.type_name())),
    }
}

fn f_int(r: &RecordVal, i: usize) -> i64 {
    match &r.fields[i] {
        Value::Int(v) => *v,
        _ => unreachable!("field type checked by conforms"),
    }
}

fn f_float(r: &RecordVal, i: usize) -> f64 {
    match &r.fields[i] {
        Value::Float(v) => *v,
        _ => unreachable!("field type checked by conforms"),
    }
}

/// Lazy-message range check: the error string is only built on failure, so the happy
/// path is allocation-free (validation runs per batch on the scheduling hot path).
macro_rules! chk {
    ($cond:expr, $($msg:tt)*) => {
        if !$cond {
            return Err(format!($($msg)*));
        }
    };
}

/// Validate a `NodeState`. `expect_id` enforces the dense-index convention
/// (`nodes[i].id == i`); candidates in a score batch instead only require
/// `0 <= id < n_nodes`.
fn validate_node(
    r: &RecordVal,
    expect_id: Option<i64>,
    n_nodes: usize,
    what: &str,
) -> Result<(), String> {
    let id = f_int(r, 0);
    match expect_id {
        Some(i) => chk!(
            id == i,
            "{what}[{i}].id == {id} (ids must be dense, spec §4)"
        ),
        None => chk!(
            (0..n_nodes as i64).contains(&id),
            "{what}.id = {id} not a valid node id (0..{n_nodes}) (spec §4.1)"
        ),
    }
    let tier = f_int(r, 1);
    chk!(
        (0..=domain::MAX_TIER).contains(&tier),
        "{what}.tier = {tier} outside host domain (spec §4.1)"
    );
    for (fi, name) in [
        (2, "cpu_capacity"),
        (3, "cpu_allocated"),
        (4, "mem_capacity"),
        (5, "mem_allocated"),
    ] {
        let v = f_int(r, fi);
        chk!(
            (0..=domain::MAX_RESOURCE).contains(&v),
            "{what}.{name} = {v} outside host domain (spec §4.1)"
        );
    }
    for (fi, name) in [(6, "power_idle_w"), (7, "power_max_w")] {
        let v = f_float(r, fi);
        chk!(
            (0.0..=domain::MAX_WATTS).contains(&v),
            "{what}.{name} = {v} outside host domain (spec §4.1)"
        );
    }
    let Value::List(lat) = &r.fields[10] else {
        unreachable!("latency_to_ms checked by conforms")
    };
    let lat = lat.borrow();
    chk!(
        lat.len() == n_nodes,
        "{what}.latency_to_ms has {} entries, expected one per node ({n_nodes}) (spec §4)",
        lat.len()
    );
    for (j, v) in lat.iter().enumerate() {
        let Value::Float(ms) = v else { unreachable!() };
        chk!(
            (0.0..=domain::MAX_LATENCY_MS).contains(ms),
            "{what}.latency_to_ms[{j}] = {ms} outside host domain (spec §4.1)"
        );
    }
    Ok(())
}

fn validate_pod(
    r: &RecordVal,
    expect_id: Option<i64>,
    n_nodes: usize,
    what: &str,
) -> Result<(), String> {
    let id = f_int(r, 0);
    match expect_id {
        Some(i) => chk!(
            id == i,
            "{what}[{i}].id == {id} (ids must be dense, spec §4)"
        ),
        None => chk!(
            (0..domain::MAX_PODS as i64).contains(&id),
            "{what}.id = {id} outside host domain (spec §4.1)"
        ),
    }
    for (fi, name) in [(1, "service"), (2, "app")] {
        let v = f_int(r, fi);
        chk!(
            (0..=domain::MAX_GROUP_ID).contains(&v),
            "{what}.{name} = {v} outside host domain (spec §4.1)"
        );
    }
    for (fi, name) in [(3, "cpu_request"), (4, "mem_request"), (7, "image_mb")] {
        let v = f_int(r, fi);
        chk!(
            (0..=domain::MAX_RESOURCE).contains(&v),
            "{what}.{name} = {v} outside host domain (spec §4.1)"
        );
    }
    let cn = f_int(r, 8);
    chk!(
        (-1..n_nodes as i64).contains(&cn),
        "{what}.current_node = {cn} not -1 or a valid node id (spec §4)"
    );
    Ok(())
}

fn validate_graph(r: &RecordVal, n_pods: usize, what: &str) -> Result<(), String> {
    let col_len = |i: usize| match &r.fields[i] {
        Value::List(xs) => xs.borrow().len(),
        _ => unreachable!("graph columns checked by conforms"),
    };
    let n = col_len(0);
    chk!(
        col_len(1) == n && col_len(2) == n && col_len(3) == n,
        "{what}: parallel columns have unequal lengths [{}, {}, {}, {}] (spec §4)",
        n,
        col_len(1),
        col_len(2),
        col_len(3)
    );
    chk!(
        n <= domain::MAX_EDGES,
        "{what}: {n} edges exceed host domain max {} (spec §4.1)",
        domain::MAX_EDGES
    );
    for (fi, name) in [(0usize, "src"), (1, "dst")] {
        let Value::List(xs) = &r.fields[fi] else {
            unreachable!()
        };
        for (j, v) in xs.borrow().iter().enumerate() {
            let Value::Int(p) = v else { unreachable!() };
            chk!(
                (0..n_pods as i64).contains(p),
                "{what}.{name}[{j}] = {p} not a valid pod id (spec §4)"
            );
        }
    }
    for (fi, name, hi) in [
        (2usize, "rate", domain::MAX_RATE),
        (3, "payload_kb", domain::MAX_PAYLOAD_KB),
    ] {
        let Value::List(xs) = &r.fields[fi] else {
            unreachable!()
        };
        for (j, v) in xs.borrow().iter().enumerate() {
            let Value::Float(x) = v else { unreachable!() };
            chk!(
                (0.0..=hi).contains(x),
                "{what}.{name}[{j}] = {x} outside host domain (spec §4.1)"
            );
        }
    }
    Ok(())
}

/// Validate nodes+pods lists shared by `ClusterCtx` and `ClusterState`.
fn validate_nodes_pods(r: &RecordVal, what: &str) -> Result<(usize, usize), String> {
    let Value::List(nodes) = &r.fields[0] else {
        unreachable!()
    };
    let Value::List(pods) = &r.fields[1] else {
        unreachable!()
    };
    let (nodes, pods) = (nodes.borrow(), pods.borrow());
    if nodes.is_empty() || nodes.len() > domain::MAX_NODES {
        return Err(format!(
            "{what}.nodes has {} entries, host domain is [1, {}] (spec §4.1)",
            nodes.len(),
            domain::MAX_NODES
        ));
    }
    if pods.len() > domain::MAX_PODS {
        return Err(format!(
            "{what}.pods has {} entries, host domain max is {} (spec §4.1)",
            pods.len(),
            domain::MAX_PODS
        ));
    }
    let n_nodes = nodes.len();
    for (i, n) in nodes.iter().enumerate() {
        let Value::Record(nr) = n else { unreachable!() };
        validate_node(nr, Some(i as i64), n_nodes, what)?;
    }
    for (i, p) in pods.iter().enumerate() {
        let Value::Record(pr) = p else { unreachable!() };
        validate_pod(pr, Some(i as i64), n_nodes, what)?;
    }
    Ok((n_nodes, pods.len()))
}

/// Validate the full `score` input triple (shape via `conforms`, then the spec §4
/// index conventions and §4.1 domain bounds). Returns `(n_nodes, n_pods)` of the ctx.
pub fn validate_score_inputs(
    pod: &Value,
    nodes: &[Value],
    ctx: &Value,
) -> Result<(usize, usize), String> {
    if !ctx.conforms(&Type::Record(RecordType::ClusterCtx)) {
        return Err("ctx does not conform to ClusterCtx (finite floats required)".into());
    }
    let ctx_r = rec(ctx, "ctx")?;
    let (n_nodes, n_pods) = validate_nodes_pods(ctx_r, "ctx")?;
    let Value::Record(graph) = &ctx_r.fields[2] else {
        unreachable!()
    };
    validate_graph(graph, n_pods, "ctx.graph")?;
    let now = f_int(ctx_r, 3);
    chk!(
        (0..=domain::MAX_NOW_MS).contains(&now),
        "ctx.now_ms = {now} outside host domain (spec §4.1)"
    );

    if !pod.conforms(&Type::Record(RecordType::Pod)) {
        return Err("pod does not conform to Pod (finite floats required)".into());
    }
    validate_pod(rec(pod, "pod")?, None, n_nodes, "pod")?;

    if nodes.len() > domain::MAX_NODES {
        return Err(format!(
            "batch of {} candidates exceeds host domain max {} (spec §4.1)",
            nodes.len(),
            domain::MAX_NODES
        ));
    }
    for (i, n) in nodes.iter().enumerate() {
        if !n.conforms(&Type::Record(RecordType::NodeState)) {
            return Err(format!(
                "candidate node [{i}] does not conform to NodeState (finite floats required)"
            ));
        }
        validate_node(rec(n, "candidate")?, None, n_nodes, "candidate")?;
    }
    Ok((n_nodes, n_pods))
}

/// Validate the full `propose_moves` input triple. Returns `(n_nodes, n_pods)`.
pub fn validate_propose_inputs(
    state: &Value,
    graph: &Value,
    budget: i64,
) -> Result<(usize, usize), String> {
    if !state.conforms(&Type::Record(RecordType::ClusterState)) {
        return Err("state does not conform to ClusterState (finite floats required)".into());
    }
    let (n_nodes, n_pods) = validate_nodes_pods(rec(state, "state")?, "state")?;
    if !graph.conforms(&Type::Record(RecordType::CallGraphCtx)) {
        return Err("graph does not conform to CallGraphCtx (finite floats required)".into());
    }
    validate_graph(rec(graph, "graph")?, n_pods, "graph")?;
    chk!(
        (0..=domain::MAX_BUDGET).contains(&budget),
        "budget = {budget} outside host domain [0, {}] (spec §4.1)",
        domain::MAX_BUDGET
    );
    Ok((n_nodes, n_pods))
}

/// The normative host tie-break (spec §5.5): highest score wins; on a bitwise-equal
/// tie (`score_batch` scores are already `-0.0`-normalized) the **lowest node id**
/// wins, regardless of candidate order. Returns the winning node id, or `None` for an
/// empty batch. Every host (simulator, wasm plugin host, conformance) selects through
/// this one function (C18 parity).
pub fn select_node(scores: &[f64], nodes: &[Value]) -> Option<i64> {
    debug_assert_eq!(scores.len(), nodes.len());
    let mut best: Option<(f64, i64)> = None;
    for (s, n) in scores.iter().zip(nodes.iter()) {
        let Value::Record(r) = n else {
            unreachable!("candidates validated as NodeState records")
        };
        let id = f_int(r, 0);
        let better = match best {
            None => true,
            Some((bs, bid)) => *s > bs || (s.to_bits() == bs.to_bits() && id < bid),
        };
        if better {
            best = Some((*s, id));
        }
    }
    best.map(|(_, id)| id)
}

/// Run `score(pod, node, ctx)`. Output is guaranteed finite; `-0.0` is normalized to
/// `0.0` on this boundary (spec §6.3) so host bitwise tie-breaking agrees with IEEE
/// equality.
pub fn score(
    prog: &Program,
    pod: &Value,
    node: &Value,
    ctx: &Value,
    limits: Limits,
) -> Result<(f64, Cost), CallError> {
    validate_score_inputs(pod, std::slice::from_ref(node), ctx)
        .map_err(|m| CallError::Host(HostError::BadArguments(m)))?;
    // Defense in depth (spec §5.2): freeze host state at this choke point no matter
    // how the embedding built the values (PyO3/wire already construct frozen lists;
    // deep_frozen is then a cheap shared clone).
    let out = crate::interp::call_prevalidated(
        prog,
        "score",
        &[pod.deep_frozen(), node.deep_frozen(), ctx.deep_frozen()],
        limits,
    )?;
    let Value::Float(v) = out.value else {
        return Err(CallError::Host(HostError::BadArguments(
            "score did not return a float (bundle not validated?)".into(),
        )));
    };
    let v = if v == 0.0 { 0.0 } else { v }; // -0.0 → 0.0
    Ok((
        v,
        Cost {
            fuel_used: out.fuel_used,
            mem_used: out.mem_used,
        },
    ))
}

/// Score one pod against N candidate nodes (one interpreter invocation per node, each
/// under its own `limits`). This is the batch API the embeddings expose: the expensive
/// host boundary is crossed once per *batch*, not once per node — and the shared
/// `pod`/`ctx` arguments are validated (shape + spec §4/§4.1 semantics) **once** for
/// the whole batch (validating them per node would make a scheduling cycle O(N²) in
/// cluster size).
///
/// The first failing node aborts the batch (index reported; validation errors report
/// index 0).
pub fn score_batch(
    prog: &Program,
    pod: &Value,
    nodes: &[Value],
    ctx: &Value,
    limits: Limits,
) -> Result<(Vec<f64>, Cost), (usize, CallError)> {
    validate_score_inputs(pod, nodes, ctx)
        .map_err(|m| (0usize, CallError::Host(HostError::BadArguments(m))))?;
    let mut scores = Vec::with_capacity(nodes.len());
    let mut total = Cost {
        fuel_used: 0,
        mem_used: 0,
    };
    let (pod, ctx) = (pod.deep_frozen(), ctx.deep_frozen()); // see `score`
    for (i, node) in nodes.iter().enumerate() {
        let out = crate::interp::call_prevalidated(
            prog,
            "score",
            &[pod.clone(), node.deep_frozen(), ctx.clone()],
            limits,
        )
        .map_err(|e| (i, e))?;
        let Value::Float(v) = out.value else {
            return Err((
                i,
                CallError::Host(HostError::BadArguments(
                    "score did not return a float (bundle not validated?)".into(),
                )),
            ));
        };
        scores.push(if v == 0.0 { 0.0 } else { v });
        total.fuel_used += out.fuel_used;
        total.mem_used += out.mem_used;
    }
    Ok((scores, total))
}

/// Run `propose_moves(state, graph, budget)` and extract the moves.
pub fn propose_moves(
    prog: &Program,
    state: &Value,
    graph: &Value,
    budget: i64,
    limits: Limits,
) -> Result<(Vec<MoveOut>, Cost), CallError> {
    validate_propose_inputs(state, graph, budget)
        .map_err(|m| CallError::Host(HostError::BadArguments(m)))?;
    let out = crate::interp::call_prevalidated(
        prog,
        "propose_moves",
        &[state.deep_frozen(), graph.deep_frozen(), Value::Int(budget)],
        limits,
    )?;
    let Value::List(xs) = out.value else {
        return Err(CallError::Host(HostError::BadArguments(
            "propose_moves did not return a list (bundle not validated?)".into(),
        )));
    };
    let mut moves = Vec::with_capacity(xs.borrow().len());
    for v in xs.borrow().iter() {
        let Value::Record(r) = v else {
            return Err(CallError::Host(HostError::BadArguments(
                "propose_moves returned a non-Move element".into(),
            )));
        };
        let (Value::Int(kind), Value::Int(pod), Value::Int(dest)) =
            (&r.fields[0], &r.fields[1], &r.fields[2])
        else {
            unreachable!("Move fields are ints by construction")
        };
        moves.push(MoveOut {
            kind: *kind,
            pod: *pod,
            dest: *dest,
        });
    }
    Ok((
        moves,
        Cost {
            fuel_used: out.fuel_used,
            mem_used: out.mem_used,
        },
    ))
}
