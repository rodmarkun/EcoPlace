//! Seed policies (TODO 1.7): every shipped policy must compile, validate as a
//! PolicyBundle, and execute score/propose_moves within the default budgets on a
//! synthetic cluster. The conformance and property harnesses exercise them further.

use placementscript_core::ast::RecordType;
use placementscript_core::value::Value;
use placementscript_core::{bundle, compile, Limits};

const SEEDS: [&str; 4] = [
    "least_allocated",
    "most_allocated",
    "latency_greedy",
    "energy_greedy",
];

fn policy_source(name: &str) -> String {
    let path = format!("{}/../../policies/{name}.ps", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Deterministic synthetic cluster: `n_nodes` nodes (last one suspended), `n_pods` pods
/// on a chain call graph, some pods stateful/PDB-protected.
fn cluster(n_nodes: i64, n_pods: i64) -> (Vec<Value>, Vec<Value>, Value, Value, Value) {
    let mut nodes = Vec::new();
    for i in 0..n_nodes {
        let lat: Vec<Value> = (0..n_nodes)
            .map(|j| Value::Float(((i - j).unsigned_abs() as f64) * 2.5))
            .collect();
        nodes.push(
            Value::record(
                RecordType::NodeState,
                vec![
                    Value::Int(i),
                    Value::Int(i % 3),
                    Value::Int(4000 + 1000 * (i % 4)),
                    Value::Int(300 * (i % 5)),
                    Value::Int(8192),
                    Value::Int(512 * (i % 6)),
                    Value::Float(2.0 + i as f64),
                    Value::Float(8.0 + 2.0 * i as f64),
                    Value::Bool(i == n_nodes - 1),
                    Value::Bool(i % 2 == 0),
                    Value::list(lat),
                ],
            )
            .unwrap(),
        );
    }
    let mut pods = Vec::new();
    for p in 0..n_pods {
        pods.push(
            Value::record(
                RecordType::Pod,
                vec![
                    Value::Int(p),
                    Value::Int(p % 4),
                    Value::Int(0),
                    Value::Int(100 + 50 * (p % 3)),
                    Value::Int(128),
                    Value::Bool(p % 5 == 0),
                    Value::Bool(p % 7 == 0),
                    Value::Int(50),
                    Value::Int(if p % 4 == 3 { -1 } else { p % (n_nodes - 1) }),
                ],
            )
            .unwrap(),
        );
    }
    let m = n_pods - 1;
    let graph = Value::record(
        RecordType::CallGraphCtx,
        vec![
            Value::list((0..m).map(Value::Int).collect()),
            Value::list((0..m).map(|e| Value::Int(e + 1)).collect()),
            Value::list((0..m).map(|e| Value::Float(0.5 + e as f64)).collect()),
            Value::list((0..m).map(|_| Value::Float(1.0)).collect()),
        ],
    )
    .unwrap();
    let ctx = Value::record(
        RecordType::ClusterCtx,
        vec![
            Value::list(nodes.clone()),
            Value::list(pods.clone()),
            graph.clone(),
            Value::Int(1_000),
        ],
    )
    .unwrap();
    let state = Value::record(
        RecordType::ClusterState,
        vec![Value::list(nodes.clone()), Value::list(pods.clone())],
    )
    .unwrap();
    (nodes, pods, graph, ctx, state)
}

#[test]
fn all_seed_policies_compile_validate_and_run() {
    let (nodes, pods, graph, ctx, state) = cluster(6, 14);
    for seed in SEEDS {
        let src = policy_source(seed);
        let prog = compile(&src).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let meta = bundle::validate(&prog).unwrap_or_else(|e| panic!("{seed}: {e}"));
        assert_eq!(meta.name, seed);

        // Score the pending pod against every node — the batch path.
        let (scores, cost) =
            bundle::score_batch(&prog, &pods[3], &nodes, &ctx, Limits::default_score())
                .unwrap_or_else(|(i, e)| panic!("{seed}: score node {i}: {e}"));
        assert_eq!(scores.len(), nodes.len());
        for s in &scores {
            assert!(s.is_finite(), "{seed}: non-finite score");
        }
        assert!(
            cost.fuel_used < Limits::default_score().fuel * nodes.len() as u64,
            "{seed}: fuel"
        );

        // Propose under several budgets; C8 exclusions and budget compliance.
        for budget in [0i64, 1, 3, 100] {
            let (moves, _) =
                bundle::propose_moves(&prog, &state, &graph, budget, Limits::default_propose())
                    .unwrap_or_else(|e| panic!("{seed}: propose: {e}"));
            assert!(
                (moves.len() as i64) <= budget,
                "{seed}: budget {budget} exceeded with {} actions",
                moves.len()
            );
            for m in &moves {
                assert!(m.kind == 0 || m.kind == 1, "{seed}: bad kind");
                if m.kind == 0 {
                    let p = m.pod as usize;
                    assert!(p < 14, "{seed}: unknown pod");
                    // stateful (p%5==0) and PDB (p%7==0) pods are never moved (C8).
                    assert!(p % 5 != 0 && p % 7 != 0, "{seed}: moved excluded pod {p}");
                    assert!((0..6).contains(&m.dest), "{seed}: unknown dest");
                }
            }
        }
    }
}

#[test]
fn seed_policies_are_deterministic() {
    let (nodes, pods, graph, ctx, state) = cluster(5, 9);
    for seed in SEEDS {
        let prog = compile(&policy_source(seed)).unwrap();
        let (s1, _) =
            bundle::score_batch(&prog, &pods[0], &nodes, &ctx, Limits::default_score()).unwrap();
        let (s2, _) =
            bundle::score_batch(&prog, &pods[0], &nodes, &ctx, Limits::default_score()).unwrap();
        assert_eq!(
            s1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            s2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{seed}"
        );
        let (m1, _) =
            bundle::propose_moves(&prog, &state, &graph, 4, Limits::default_propose()).unwrap();
        let (m2, _) =
            bundle::propose_moves(&prog, &state, &graph, 4, Limits::default_propose()).unwrap();
        assert_eq!(m1, m2, "{seed}");
    }
}
