//! Stateful-ABI parity cases (Round-1 findings R1.4/R1.6/R1.7): host tie-breaking,
//! signed-zero normalization, unordered candidates, invalid bundles, malformed states,
//! and the ABI fuel/batch ceilings — each exercised through BOTH the native embedding
//! (`bundle::*`) and the wasm stateful ABI, requiring identical outcomes.

use placementscript_core::wire::*;
use placementscript_core::{bundle, compile, CallError, HostError, Limits};
use placementscript_wasm_host::{locate_guest_module, HostConfig, PolicyModule, WasmCallError};

fn skip() -> bool {
    if std::env::var("PS_SKIP_WASM_CONFORMANCE").as_deref() == Ok("1") {
        eprintln!("skipped: PS_SKIP_WASM_CONFORMANCE=1");
        return true;
    }
    false
}

fn host() -> PolicyModule {
    let module_path = locate_guest_module().expect("guest module (see wasm.rs docs)");
    PolicyModule::from_file(&module_path, HostConfig::default()).unwrap()
}

const N_NODES: i64 = 5;

fn node(id: i64) -> WireValue {
    WireValue::R {
        ty: "NodeState".into(),
        fields: vec![
            WireValue::I(id),
            WireValue::I(id % 3),
            WireValue::I(4000),
            WireValue::I(500 * (id % 3)),
            WireValue::I(8192),
            WireValue::I(1024),
            WireValue::F(3.0),
            WireValue::F(9.0),
            WireValue::B(false),
            WireValue::B(true),
            WireValue::L(
                (0..N_NODES)
                    .map(|j| WireValue::F((id - j).abs() as f64))
                    .collect(),
            ),
        ],
    }
}

fn pod(id: i64) -> WireValue {
    WireValue::R {
        ty: "Pod".into(),
        fields: vec![
            WireValue::I(id),
            WireValue::I(0),
            WireValue::I(0),
            WireValue::I(100),
            WireValue::I(128),
            WireValue::B(false),
            WireValue::B(false),
            WireValue::I(50),
            WireValue::I(-1),
        ],
    }
}

fn graph() -> WireValue {
    WireValue::R {
        ty: "CallGraphCtx".into(),
        fields: vec![
            WireValue::L(vec![]),
            WireValue::L(vec![]),
            WireValue::L(vec![]),
            WireValue::L(vec![]),
        ],
    }
}

fn ctx() -> WireValue {
    WireValue::R {
        ty: "ClusterCtx".into(),
        fields: vec![
            WireValue::L((0..N_NODES).map(node).collect()),
            WireValue::L(vec![pod(0), pod(1)]),
            graph(),
            WireValue::I(0),
        ],
    }
}

fn bundle_src(score_body: &str) -> String {
    format!(
        "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    {score_body}\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n"
    )
}

fn bundle_src_propose(propose_body: &str) -> String {
    format!(
        "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return 1.0\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    {propose_body}\n"
    )
}

fn limits() -> WireLimits {
    WireLimits {
        fuel: Limits::default_score().fuel,
        memory_bytes: Limits::default_score().memory_bytes,
    }
}

/// Run one score batch through native + wasm; return (native_best, wasm_best, scores).
fn both(src: &str, candidates: Vec<WireValue>) -> (i64, i64, Vec<f64>) {
    let prog = compile(src).unwrap();
    bundle::validate(&prog).unwrap();
    let n_pod = pod(0).to_value().unwrap();
    let n_ctx = ctx().to_value().unwrap();
    let n_cands: Vec<_> = candidates.iter().map(|c| c.to_value().unwrap()).collect();
    let (scores, _) =
        bundle::score_batch(&prog, &n_pod, &n_cands, &n_ctx, Limits::default_score()).unwrap();
    let native_best = bundle::select_node(&scores, &n_cands).unwrap_or(-1);

    let mut h = host();
    h.load_policy(src).unwrap();
    let resp = h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: candidates,
            ctx: ctx(),
            limits: limits(),
        })
        .unwrap();
    assert_eq!(
        scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        resp.scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "score divergence"
    );
    (native_best, resp.best_node, scores)
}

#[test]
fn equal_scores_tie_break_lowest_id_even_unordered() {
    if skip() {
        return;
    }
    // Constant scorer; candidates deliberately unordered: lowest id must win.
    let src = bundle_src("return 1.0");
    let (nb, wb, _) = both(&src, vec![node(3), node(1), node(4), node(2)]);
    assert_eq!(nb, 1);
    assert_eq!(wb, 1);
}

#[test]
fn signed_zero_normalized_before_tie_break() {
    if skip() {
        return;
    }
    // -0.0 for even ids, 0.0 for odd ids: normalization makes them bitwise ties,
    // so the lowest id in the batch wins on both targets.
    let src = bundle_src("x = node.id % 2\n    if x == 0:\n        return -0.0\n    return 0.0");
    let (nb, wb, scores) = both(&src, vec![node(4), node(3), node(2)]);
    assert!(scores.iter().all(|s| s.to_bits() == 0.0f64.to_bits()));
    assert_eq!(nb, 2);
    assert_eq!(wb, 2);
}

#[test]
fn distinct_scores_pick_max_regardless_of_order() {
    if skip() {
        return;
    }
    let src = bundle_src("return float(node.cpu_allocated)");
    let (nb, wb, _) = both(&src, vec![node(0), node(2), node(1)]);
    // cpu_allocated = 500 * (id % 3): id 2 has the max (1000).
    assert_eq!(nb, 2);
    assert_eq!(wb, 2);
}

#[test]
fn invalid_bundle_rejected_identically() {
    if skip() {
        return;
    }
    // Missing propose_moves: native and wasm both report the bundle-contract (type)
    // error class.
    let src = "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return 1.0\n";
    let prog = compile(src).unwrap();
    let native = bundle::validate(&prog).unwrap_err();
    assert_eq!(native.class(), "type");
    let mut h = host();
    match h.load_policy(src).unwrap_err() {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "type"),
        WasmCallError::Sandbox(e) => panic!("sandbox error instead of guest type error: {e:#}"),
    }
}

#[test]
fn malformed_states_rejected_identically() {
    if skip() {
        return;
    }
    let src = bundle_src("return 1.0");
    let prog = compile(&src).unwrap();
    let mut h = host();
    h.load_policy(&src).unwrap();

    // (a) Non-dense node ids in ctx.
    let mut bad_ctx = ctx();
    if let WireValue::R { fields, .. } = &mut bad_ctx {
        if let WireValue::L(nodes) = &mut fields[0] {
            nodes[0] = node(7); // nodes[0].id != 0
        }
    }
    let native = bundle::score_batch(
        &prog,
        &pod(0).to_value().unwrap(),
        &[node(0).to_value().unwrap()],
        &bad_ctx.to_value().unwrap(),
        Limits::default_score(),
    )
    .unwrap_err();
    assert!(matches!(
        native.1,
        CallError::Host(HostError::BadArguments(_))
    ));
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: vec![node(0)],
            ctx: bad_ctx,
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // (b) Ragged call-graph columns.
    let mut bad_ctx = ctx();
    if let WireValue::R { fields, .. } = &mut bad_ctx {
        fields[2] = WireValue::R {
            ty: "CallGraphCtx".into(),
            fields: vec![
                WireValue::L(vec![WireValue::I(0)]),
                WireValue::L(vec![]),
                WireValue::L(vec![]),
                WireValue::L(vec![]),
            ],
        };
    }
    let native = bundle::score_batch(
        &prog,
        &pod(0).to_value().unwrap(),
        &[node(0).to_value().unwrap()],
        &bad_ctx.to_value().unwrap(),
        Limits::default_score(),
    )
    .unwrap_err();
    assert!(matches!(
        native.1,
        CallError::Host(HostError::BadArguments(_))
    ));
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: vec![node(0)],
            ctx: bad_ctx,
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // (c) Wrong record arity (malformed ABI payload) — rejected at deserialization.
    let malformed = WireValue::R {
        ty: "Pod".into(),
        fields: vec![WireValue::I(0)],
    };
    assert!(malformed.to_value().is_err());
    match h
        .score_batch(&ScoreBatchRequest {
            pod: malformed,
            nodes: vec![node(0)],
            ctx: ctx(),
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // (d) Negative propose budget.
    let native = bundle::propose_moves(
        &prog,
        &WireValue::R {
            ty: "ClusterState".into(),
            fields: vec![
                WireValue::L((0..N_NODES).map(node).collect()),
                WireValue::L(vec![pod(0), pod(1)]),
            ],
        }
        .to_value()
        .unwrap(),
        &graph().to_value().unwrap(),
        -1,
        Limits::default_propose(),
    )
    .unwrap_err();
    assert!(matches!(
        native,
        CallError::Host(HostError::BadArguments(_))
    ));
    match h
        .propose_moves(&ProposeRequest {
            state: WireValue::R {
                ty: "ClusterState".into(),
                fields: vec![
                    WireValue::L((0..N_NODES).map(node).collect()),
                    WireValue::L(vec![pod(0), pod(1)]),
                ],
            },
            graph: graph(),
            budget: -1,
            limits: WireLimits {
                fuel: Limits::default_propose().fuel,
                memory_bytes: Limits::default_propose().memory_bytes,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }
}

#[test]
fn abi_ceilings_rejected_as_host_errors_and_language_fuel_still_binds() {
    if skip() {
        return;
    }
    let src = bundle_src("return 1.0");
    let mut h = host();
    h.load_policy(&src).unwrap();

    // Over-ceiling language fuel: structured host error, not a sandbox kill.
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: vec![node(0)],
            ctx: ctx(),
            limits: WireLimits {
                fuel: u64::MAX,
                memory_bytes: 1 << 20,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // Over-ceiling batch size (max_batch + 1 candidates): structured host error.
    let over = HostConfig::default().max_batch + 1;
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: (0..over)
                .map(|i| node((i % N_NODES as usize) as i64))
                .collect(),
            ctx: ctx(),
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // Over-ceiling *load* fuel on ps_call (R2 review of R1.7: the requested load
    // budget must be ceiling-checked and drive the outer-fuel derivation).
    match h
        .call(&CallRequest {
            source: src.clone(),
            func: "score".into(),
            args: vec![pod(0), node(0), ctx()],
            limits: limits(),
            load_limits: WireLimits {
                fuel: u64::MAX,
                memory_bytes: 64 << 10,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // Over-ceiling encoded request size: structured host error before guest memory
    // is touched (R2.1 defense in depth).
    let tiny = HostConfig {
        max_request_bytes: 64,
        ..HostConfig::default()
    };
    let module_path = locate_guest_module().unwrap();
    let mut h_tiny = PolicyModule::from_file(&module_path, tiny).unwrap();
    match h_tiny.load_policy(&src).unwrap_err() {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"), // source > 64 bytes
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }
    match h_tiny
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: vec![node(0)],
            ctx: ctx(),
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // A fuel-heavy (but legal) batch completes as a *guest* outcome: every candidate
    // exhausts its language fuel, and the derived outer fuel still covers the batch.
    let burn = bundle_src(
        "acc = 0\n    for i in range(1000000000):\n        acc += 1\n    return float(acc)",
    );
    h.load_policy(&burn).unwrap();
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: (0..N_NODES).map(node).collect(),
            ctx: ctx(),
            limits: limits(),
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "fuel", "{e:?}"),
        WasmCallError::Sandbox(e) => panic!("outer fuel killed a legal batch: {e:#}"),
    }
}

/// R2 review of R1.7: the previous "at the ceiling" test used 5 candidates and default
/// fuel. The *default* ceilings (512 candidates × 2³² language fuel each ≈ 10¹² steps)
/// are not executable in any test budget; the outer-fuel derivation is linear in
/// (request bytes, language_fuel × invocations), so the at-ceiling property is verified
/// exactly on a scaled config: a request at BOTH ceilings, where every candidate
/// actually consumes close to its full requested language fuel and the whole batch
/// completes as a *guest* success — the derived outer budget must cover all of it.
#[test]
fn at_ceiling_full_fuel_batch_is_guest_outcome_r2() {
    if skip() {
        return;
    }
    const CEIL_FUEL: u64 = 100_000;
    const CEIL_BATCH: usize = 16;
    let cfg = HostConfig {
        max_batch: CEIL_BATCH,
        max_language_fuel: CEIL_FUEL,
        ..HostConfig::default()
    };
    // Burn most of the per-candidate budget, then return (success, not fuel error).
    let burn = bundle_src(
        "acc = 0\n    for i in range(30000):\n        acc += 1\n    return float(acc + node.id)",
    );
    // Calibrate natively: the policy must consume >50% of the ceiling per invocation
    // (otherwise this test would not stress the batch-scaled outer budget) and fit it.
    let prog = compile(&burn).unwrap();
    bundle::validate(&prog).unwrap();
    let (_, cost) = bundle::score(
        &prog,
        &pod(0).to_value().unwrap(),
        &node(0).to_value().unwrap(),
        &ctx().to_value().unwrap(),
        placementscript_core::Limits {
            fuel: CEIL_FUEL,
            memory_bytes: 1 << 20,
        },
    )
    .expect("calibration run must fit the ceiling budget");
    assert!(
        cost.fuel_used > CEIL_FUEL / 2,
        "calibration: burn policy uses only {} of {CEIL_FUEL} fuel — raise the loop count",
        cost.fuel_used
    );

    let module_path = locate_guest_module().unwrap();
    let mut h = PolicyModule::from_file(&module_path, cfg).unwrap();
    h.load_policy(&burn).unwrap();
    let resp = h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: (0..CEIL_BATCH)
                .map(|i| node((i as i64) % N_NODES))
                .collect(),
            ctx: ctx(),
            limits: WireLimits {
                fuel: CEIL_FUEL, // exactly the language-fuel ceiling
                memory_bytes: 1 << 20,
            },
        })
        .expect("at-ceiling full-fuel batch must be a guest success, not a sandbox kill");
    assert_eq!(resp.scores.len(), CEIL_BATCH);
    assert!(resp.fuel_used >= cost.fuel_used * CEIL_BATCH as u64);
}

/// Domain-maximum wire values (every spec §4.1 shape and numeric field at its bound),
/// shared by the R2.1 representability test and the post-R2 language-memory ceiling
/// test.
mod domain_max {
    use super::*;
    use placementscript_core::bundle::domain::*;

    fn big_node(id: i64) -> WireValue {
        let nn = MAX_NODES as i64;
        WireValue::R {
            ty: "NodeState".into(),
            fields: vec![
                WireValue::I(id),
                WireValue::I(MAX_TIER),
                WireValue::I(MAX_RESOURCE),
                WireValue::I(MAX_RESOURCE),
                WireValue::I(MAX_RESOURCE),
                WireValue::I(MAX_RESOURCE),
                WireValue::F(MAX_WATTS),
                WireValue::F(MAX_WATTS),
                WireValue::B(false),
                WireValue::B(true),
                WireValue::L((0..nn).map(|_| WireValue::F(MAX_LATENCY_MS)).collect()),
            ],
        }
    }

    fn big_pod(id: i64) -> WireValue {
        let nn = MAX_NODES as i64;
        WireValue::R {
            ty: "Pod".into(),
            fields: vec![
                WireValue::I(id),
                WireValue::I(MAX_GROUP_ID),
                WireValue::I(MAX_GROUP_ID),
                WireValue::I(MAX_RESOURCE),
                WireValue::I(MAX_RESOURCE),
                WireValue::B(true),
                WireValue::B(true),
                WireValue::I(MAX_RESOURCE),
                WireValue::I(nn - 1),
            ],
        }
    }

    pub fn nodes() -> Vec<WireValue> {
        (0..MAX_NODES as i64).map(big_node).collect()
    }

    pub fn pods() -> Vec<WireValue> {
        (0..MAX_PODS as i64).map(big_pod).collect()
    }

    pub fn graph() -> WireValue {
        WireValue::R {
            ty: "CallGraphCtx".into(),
            fields: vec![
                WireValue::L(
                    (0..MAX_EDGES)
                        .map(|e| WireValue::I((e % MAX_PODS) as i64))
                        .collect(),
                ),
                WireValue::L(
                    (0..MAX_EDGES)
                        .map(|e| WireValue::I(((e * 7 + 1) % MAX_PODS) as i64))
                        .collect(),
                ),
                WireValue::L((0..MAX_EDGES).map(|_| WireValue::F(MAX_RATE)).collect()),
                WireValue::L(
                    (0..MAX_EDGES)
                        .map(|_| WireValue::F(MAX_PAYLOAD_KB))
                        .collect(),
                ),
            ],
        }
    }

    pub fn state(nodes: Vec<WireValue>, pods: Vec<WireValue>) -> WireValue {
        WireValue::R {
            ty: "ClusterState".into(),
            fields: vec![WireValue::L(nodes), WireValue::L(pods)],
        }
    }
}

/// R2.1: a **domain-maximum** request (spec §4.1 shape bounds all at their ceilings:
/// 512 nodes with 512-entry latency vectors, 512 candidates, 16 384 pods, 65 536
/// call-graph edges, every numeric field at its bound) must be representable and
/// executable in the default sandbox (256 MiB linear memory), with native/wasm parity.
/// This is the test that makes the §4.1 domain an honest guarantee.
#[test]
fn domain_ceiling_request_fits_the_sandbox_r2() {
    if skip() {
        return;
    }
    use placementscript_core::bundle::domain::*;
    let nn = MAX_NODES as i64;
    let nodes = domain_max::nodes();
    let pods = domain_max::pods();
    let big_graph = domain_max::graph();
    let big_ctx = WireValue::R {
        ty: "ClusterCtx".into(),
        fields: vec![
            WireValue::L(nodes.clone()),
            WireValue::L(pods.clone()),
            big_graph.clone(),
            WireValue::I(MAX_NOW_MS),
        ],
    };

    let src = bundle_src("return float(node.id)");
    let prog = compile(&src).unwrap();
    bundle::validate(&prog).unwrap();

    // Native reference over the full domain-max batch.
    let n_pod = pods[0].to_value().unwrap();
    let n_ctx = big_ctx.to_value().unwrap();
    let n_cands: Vec<_> = nodes.iter().map(|c| c.to_value().unwrap()).collect();
    let (native_scores, _) =
        bundle::score_batch(&prog, &n_pod, &n_cands, &n_ctx, Limits::default_score()).unwrap();
    let native_best = bundle::select_node(&native_scores, &n_cands).unwrap_or(-1);
    assert_eq!(native_best, nn - 1);

    // Same request through the wasm guest under the DEFAULT sandbox config.
    let mut h = host();
    h.load_policy(&src).unwrap();
    let resp = h
        .score_batch(&ScoreBatchRequest {
            pod: pods[0].clone(),
            nodes: nodes.clone(),
            ctx: big_ctx,
            limits: limits(),
        })
        .expect("domain-maximum request must fit the default sandbox (R2.1)");
    assert_eq!(
        native_scores
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        resp.scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "score divergence on the domain-maximum request"
    );
    assert_eq!(resp.best_node, native_best);

    // The propose path at its own domain maximum (ClusterState + graph + max budget).
    let big_state = domain_max::state(nodes, pods);
    let n_moves = bundle::propose_moves(
        &prog,
        &big_state.to_value().unwrap(),
        &big_graph.to_value().unwrap(),
        MAX_BUDGET,
        Limits::default_propose(),
    )
    .unwrap()
    .0;
    let w_resp = h
        .propose_moves(&ProposeRequest {
            state: big_state,
            graph: big_graph,
            budget: MAX_BUDGET,
            limits: WireLimits {
                fuel: Limits::default_propose().fuel,
                memory_bytes: Limits::default_propose().memory_bytes,
            },
        })
        .expect("domain-maximum propose request must fit the default sandbox (R2.1)");
    assert_eq!(n_moves.len(), w_resp.moves.len());
}

/// Round-3 review: lexing is load-fuel-budgeted (one fuel per source byte, spec §8.1),
/// so a token-dense source — well under the request-size ceiling — with an at-ceiling
/// load-fuel request is a *guest* `fuel` outcome with bounded compiler allocation,
/// never a linear-memory sandbox kill; native returns the identical class. (Over-
/// ceiling load fuel stays a structured host error, covered by the ABI-ceilings test.)
#[test]
fn token_dense_source_is_guest_fuel_outcome_r3() {
    if skip() {
        return;
    }
    let ceiling = HostConfig::default().max_load_fuel;
    let dense = "x ".repeat(4 << 20); // 8 MiB of Name tokens >> max_load_fuel bytes
    let e = placementscript_core::compile_with_limits(
        &dense,
        placementscript_core::Limits {
            fuel: ceiling,
            memory_bytes: 64 << 10,
        },
    )
    .unwrap_err();
    assert!(
        matches!(e, placementscript_core::PsError::FuelExhausted),
        "{e}"
    );
    let mut h = host();
    let resp = h
        .call(&CallRequest {
            source: dense,
            func: "f".into(),
            args: vec![],
            limits: limits(),
            load_limits: WireLimits {
                fuel: ceiling, // exactly the ABI load-fuel ceiling: a legal request
                memory_bytes: 64 << 10,
            },
        })
        .expect("token-dense source must be a structured guest outcome, not a sandbox kill");
    match resp.result {
        Err(e) => assert_eq!(e.class(), "fuel", "{e:?}"),
        Ok(v) => panic!("expected a load-fuel error, got a value: {v:?}"),
    }
}

/// Round-4 review: one long token (identifier/number/string/comment) could scan and
/// allocate far past the between-token fuel checks, and non-ASCII string decoding was
/// quadratic — either could exhaust the linearly-derived outer fuel or guest memory on
/// a legal-size request (a sandbox kill, not a structured outcome). Over-length
/// sources are now rejected before scanning, and Unicode decoding is linear: an 8 MiB
/// single-identifier source at the load-fuel ceiling is a guest `fuel` outcome, and a
/// large in-budget Unicode string constant compiles and round-trips bitwise through
/// the guest.
#[test]
fn long_token_and_unicode_sources_bounded_r4() {
    if skip() {
        return;
    }
    let ceiling = HostConfig::default().max_load_fuel;
    let mut h = host();

    // One 8 MiB identifier: structured guest fuel outcome, never a sandbox trap.
    let long_ident = format!("def f() -> int:\n    return {}\n", "x".repeat(8 << 20));
    let e = placementscript_core::compile_with_limits(
        &long_ident,
        placementscript_core::Limits {
            fuel: ceiling,
            memory_bytes: 64 << 10,
        },
    )
    .unwrap_err();
    assert!(
        matches!(e, placementscript_core::PsError::FuelExhausted),
        "{e}"
    );
    let resp = h
        .call(&CallRequest {
            source: long_ident,
            func: "f".into(),
            args: vec![],
            limits: limits(),
            load_limits: WireLimits {
                fuel: ceiling,
                memory_bytes: 64 << 10,
            },
        })
        .expect("long single-token source must be a structured guest outcome");
    match resp.result {
        Err(e) => assert_eq!(e.class(), "fuel", "{e:?}"),
        Ok(v) => panic!("expected a load-fuel error, got a value: {v:?}"),
    }

    // A large in-budget Unicode string constant (400k 2-byte chars ≈ 800 KB source,
    // under the 1 Mi load-fuel ceiling): linear decoding must complete inside the
    // derived outer fuel, and the value round-trips bitwise against native.
    let n = 400_000usize;
    let uni = format!(
        "S: str = \"{}\"\ndef f() -> str:\n    return S\n",
        "é".repeat(n)
    );
    let load = placementscript_core::Limits {
        fuel: ceiling,
        memory_bytes: 2 << 20, // interned string: 32 + 800k bytes
    };
    let prog = placementscript_core::compile_with_limits(&uni, load).unwrap();
    let native = placementscript_core::call(
        &prog,
        "f",
        &[],
        placementscript_core::Limits::default_score(),
    )
    .unwrap();
    let resp = h
        .call(&CallRequest {
            source: uni,
            func: "f".into(),
            args: vec![],
            limits: limits(),
            load_limits: WireLimits {
                fuel: ceiling,
                memory_bytes: 2 << 20,
            },
        })
        .expect("in-budget Unicode source must compile and run in the guest");
    match (resp.result, native.value) {
        (Ok(WireValue::S(w)), placementscript_core::value::Value::Str(nv)) => {
            assert_eq!(w.chars().count(), n);
            assert_eq!(w, nv.as_ref());
        }
        (w, nv) => panic!("expected matching strings, got wasm {w:?} vs native {nv:?}"),
    }
}

/// Post-Round-2 self-review (R2.1/R1.7 residual): language *memory* budgets must be
/// ABI-ceilinged like fuel, and the ceiling must be proven safe — a domain-maximum
/// request plus a policy allocating up to `max_language_memory` must reach its
/// deterministic language OOM as a *guest* outcome (native parity), never a
/// linear-memory sandbox kill; over-ceiling memory requests (invocation or load) are
/// structured host errors.
#[test]
fn language_memory_ceiling_post_r2() {
    if skip() {
        return;
    }
    use placementscript_core::bundle::domain::MAX_BUDGET;
    let ceiling = HostConfig::default().max_language_memory;

    // Over-ceiling invocation memory: structured host error, not a sandbox kill.
    let src = bundle_src("return 1.0");
    let mut h = host();
    h.load_policy(&src).unwrap();
    match h
        .score_batch(&ScoreBatchRequest {
            pod: pod(0),
            nodes: vec![node(0)],
            ctx: ctx(),
            limits: WireLimits {
                fuel: Limits::default_score().fuel,
                memory_bytes: ceiling + 1,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }
    // Over-ceiling *load* memory on ps_call: same structured rejection.
    match h
        .call(&CallRequest {
            source: src.clone(),
            func: "score".into(),
            args: vec![pod(0), node(0), ctx()],
            limits: limits(),
            load_limits: WireLimits {
                fuel: Limits::default_load().fuel,
                memory_bytes: ceiling + 1,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "host"),
        WasmCallError::Sandbox(e) => panic!("sandbox error: {e:#}"),
    }

    // At the ceiling, with the WHOLE domain-maximum state resident in guest memory:
    // an append-heavy legal policy must hit its deterministic language OOM (guest
    // outcome, identical class natively), never the linear-memory cap.
    let hungry = bundle_src_propose(
        "out: list[Move] = []\n    xs: list[int] = []\n    for i in range(2000000000):\n        xs.append(i)\n    return out",
    );
    let prog = compile(&hungry).unwrap();
    bundle::validate(&prog).unwrap();
    let nodes = domain_max::nodes();
    let pods = domain_max::pods();
    let big_graph = domain_max::graph();
    let big_state = domain_max::state(nodes, pods);
    let lim = placementscript_core::Limits {
        fuel: 1 << 30, // plenty: memory must bind first
        memory_bytes: ceiling,
    };
    let native = bundle::propose_moves(
        &prog,
        &big_state.to_value().unwrap(),
        &big_graph.to_value().unwrap(),
        MAX_BUDGET,
        lim,
    )
    .unwrap_err();
    assert!(
        matches!(
            native,
            CallError::Ps(placementscript_core::PsError::OutOfMemory)
        ),
        "{native:?}"
    );
    let mut h = host();
    h.load_policy(&hungry).unwrap();
    match h
        .propose_moves(&ProposeRequest {
            state: big_state,
            graph: big_graph,
            budget: MAX_BUDGET,
            limits: WireLimits {
                fuel: 1 << 30,
                memory_bytes: ceiling,
            },
        })
        .unwrap_err()
    {
        WasmCallError::Guest(e) => assert_eq!(e.class(), "oom", "{e:?}"),
        WasmCallError::Sandbox(e) => {
            panic!("linear-memory cap killed a legal at-ceiling allocation: {e:#}")
        }
    }
}
