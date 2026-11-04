//! Seed policies under the conformance harness (TODO 1.7): each seed bundle runs
//! through the *stateful* ABI (`ps_load_policy` → `ps_score_batch`/`ps_propose_moves`)
//! on the wasm target and must match the native embedding bit-for-bit.

use placementscript_core::wire::*;
use placementscript_core::{bundle, compile, Limits};
use placementscript_wasm_host::{locate_guest_module, HostConfig, PolicyModule};
use std::path::PathBuf;

const SEEDS: [&str; 4] = [
    "least_allocated",
    "most_allocated",
    "latency_greedy",
    "energy_greedy",
];

fn policies_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies")
}

fn node(i: i64, n: i64) -> WireValue {
    WireValue::R {
        ty: "NodeState".into(),
        fields: vec![
            WireValue::I(i),
            WireValue::I(i % 3),
            WireValue::I(4000 + 1000 * (i % 4)),
            WireValue::I(300 * (i % 5)),
            WireValue::I(8192),
            WireValue::I(512 * (i % 6)),
            WireValue::F(2.0 + i as f64),
            WireValue::F(8.0 + 2.0 * i as f64),
            WireValue::B(i == n - 1),
            WireValue::B(i % 2 == 0),
            WireValue::L(
                (0..n)
                    .map(|j| WireValue::F(((i - j).abs() as f64) * 2.5))
                    .collect(),
            ),
        ],
    }
}

fn pod(p: i64, n_nodes: i64) -> WireValue {
    WireValue::R {
        ty: "Pod".into(),
        fields: vec![
            WireValue::I(p),
            WireValue::I(p % 4),
            WireValue::I(0),
            WireValue::I(100 + 50 * (p % 3)),
            WireValue::I(128),
            WireValue::B(p % 5 == 0),
            WireValue::B(p % 7 == 0),
            WireValue::I(50),
            WireValue::I(if p % 4 == 3 { -1 } else { p % (n_nodes - 1) }),
        ],
    }
}

fn graph(n_pods: i64) -> WireValue {
    let m = n_pods - 1;
    WireValue::R {
        ty: "CallGraphCtx".into(),
        fields: vec![
            WireValue::L((0..m).map(WireValue::I).collect()),
            WireValue::L((0..m).map(|e| WireValue::I(e + 1)).collect()),
            WireValue::L((0..m).map(|e| WireValue::F(0.5 + e as f64)).collect()),
            WireValue::L((0..m).map(|_| WireValue::F(1.0)).collect()),
        ],
    }
}

#[test]
fn seed_policies_native_wasm_parity() {
    if std::env::var("PS_SKIP_WASM_CONFORMANCE").as_deref() == Ok("1") {
        eprintln!("skipped: PS_SKIP_WASM_CONFORMANCE=1");
        return;
    }
    let module_path = locate_guest_module().expect("guest module (see wasm.rs docs)");

    let (n_nodes, n_pods) = (8i64, 18i64);
    let nodes: Vec<WireValue> = (0..n_nodes).map(|i| node(i, n_nodes)).collect();
    let pods: Vec<WireValue> = (0..n_pods).map(|p| pod(p, n_nodes)).collect();
    let ctx = WireValue::R {
        ty: "ClusterCtx".into(),
        fields: vec![
            WireValue::L(nodes.clone()),
            WireValue::L(pods.clone()),
            graph(n_pods),
            WireValue::I(0),
        ],
    };
    let state = WireValue::R {
        ty: "ClusterState".into(),
        fields: vec![WireValue::L(nodes.clone()), WireValue::L(pods.clone())],
    };
    let score_limits = WireLimits {
        fuel: Limits::default_score().fuel,
        memory_bytes: Limits::default_score().memory_bytes,
    };
    let propose_limits = WireLimits {
        fuel: Limits::default_propose().fuel,
        memory_bytes: Limits::default_propose().memory_bytes,
    };

    for seed in SEEDS {
        let source = std::fs::read_to_string(policies_dir().join(format!("{seed}.ps"))).unwrap();

        // Native reference.
        let prog = compile(&source).unwrap();
        bundle::validate(&prog).unwrap();
        let n_pod = pods[3].to_value().unwrap();
        let n_nodes_v: Vec<_> = nodes.iter().map(|n| n.to_value().unwrap()).collect();
        let n_ctx = ctx.to_value().unwrap();
        let (native_scores, native_cost) =
            bundle::score_batch(&prog, &n_pod, &n_nodes_v, &n_ctx, Limits::default_score())
                .unwrap();
        let (native_moves, _) = bundle::propose_moves(
            &prog,
            &state.to_value().unwrap(),
            &graph(n_pods).to_value().unwrap(),
            4,
            Limits::default_propose(),
        )
        .unwrap();

        // Wasm via the stateful ABI. Fresh instance per seed (hot-swap = new module/instance).
        let mut host = PolicyModule::from_file(&module_path, HostConfig::default()).unwrap();
        let meta = host
            .load_policy(&source)
            .unwrap_or_else(|e| panic!("{seed}: {e}"));
        assert_eq!(meta.name, seed);
        let resp = host
            .score_batch(&ScoreBatchRequest {
                pod: pods[3].clone(),
                nodes: nodes.clone(),
                ctx: ctx.clone(),
                limits: score_limits,
            })
            .unwrap_or_else(|e| panic!("{seed}: {e}"));
        assert_eq!(
            native_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            resp.scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{seed}: score divergence native vs wasm"
        );
        assert_eq!(
            (native_cost.fuel_used, native_cost.mem_used),
            (resp.fuel_used, resp.mem_used),
            "{seed}: resource-accounting divergence"
        );
        // Host tie-break parity (spec §5.5): the shared selection function agrees.
        let native_best = bundle::select_node(&native_scores, &n_nodes_v).unwrap_or(-1);
        assert_eq!(native_best, resp.best_node, "{seed}: best-node divergence");
        let presp = host
            .propose_moves(&ProposeRequest {
                state: state.clone(),
                graph: graph(n_pods),
                budget: 4,
                limits: propose_limits,
            })
            .unwrap_or_else(|e| panic!("{seed}: {e}"));
        let native_wire: Vec<WireMove> = native_moves
            .iter()
            .map(|m| WireMove {
                kind: m.kind,
                pod: m.pod,
                dest: m.dest,
            })
            .collect();
        assert_eq!(native_wire, presp.moves, "{seed}: move divergence");
        assert!(presp.moves.len() <= 4, "{seed}: budget violation");
    }
}

#[test]
fn embedded_policy_loads() {
    if std::env::var("PS_SKIP_WASM_CONFORMANCE").as_deref() == Ok("1") {
        eprintln!("skipped: PS_SKIP_WASM_CONFORMANCE=1");
        return;
    }
    let module_path = locate_guest_module().expect("guest module");
    let mut host = PolicyModule::from_file(&module_path, HostConfig::default()).unwrap();
    let meta = host.load_embedded_policy().unwrap();
    assert_eq!(meta.name, "least_allocated"); // default embed (build.rs)
}
