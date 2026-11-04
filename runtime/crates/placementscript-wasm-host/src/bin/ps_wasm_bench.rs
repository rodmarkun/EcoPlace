//! Decision-latency measurement for the WASM embedding (E6 feeds from this; per C9 we
//! measure and report, never claim). Times `ps_score_batch` (one pod × N nodes per host
//! call) and `ps_propose_moves` against the embedded policy.
//!
//! Usage: ps-wasm-bench [nodes] [pods] [iters]
//! (module located via PS_WASM_GUEST or the workspace target dir; build with --release
//! for meaningful numbers)

use placementscript_core::wire::*;
use placementscript_wasm_host::{locate_guest_module, HostConfig, PolicyModule};

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
            WireValue::B(false),
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_nodes: i64 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(200);
    let n_pods: i64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(60);
    let iters: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(50);

    let path = locate_guest_module()?;
    println!("module: {}", path.display());
    let mut host = PolicyModule::from_file(&path, HostConfig::default())?;
    let meta = host
        .load_embedded_policy()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("policy: {} v{}", meta.name, meta.version);

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
    let limits = WireLimits {
        fuel: 200_000,
        memory_bytes: 1 << 20,
    };
    let req = ScoreBatchRequest {
        pod: pods[3].clone(),
        nodes: nodes.clone(),
        ctx,
        limits,
    };

    // Warm-up + measure: one host call scores all N nodes (the scheduling-cycle shape).
    host.score_batch(&req).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let resp = host.score_batch(&req).map_err(|e| anyhow::anyhow!("{e}"))?;
        samples.push(t0.elapsed());
        assert_eq!(resp.scores.len(), n_nodes as usize);
    }
    samples.sort();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    println!(
        "score_batch: {n_nodes} nodes/call, {iters} calls: p50={:?} p99={:?} (per-node p50 ~{:?})",
        p50,
        p99,
        p50 / n_nodes as u32
    );

    let state = WireValue::R {
        ty: "ClusterState".into(),
        fields: vec![WireValue::L(nodes), WireValue::L(pods)],
    };
    let preq = ProposeRequest {
        state,
        graph: graph(n_pods),
        budget: 10,
        limits: WireLimits {
            fuel: 20_000_000,
            memory_bytes: 16 << 20,
        },
    };
    host.propose_moves(&preq)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        host.propose_moves(&preq)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        samples.push(t0.elapsed());
    }
    samples.sort();
    println!(
        "propose_moves: {n_pods} pods, {n_nodes} nodes: p50={:?} p99={:?}",
        samples[samples.len() / 2],
        samples[(samples.len() * 99) / 100]
    );
    Ok(())
}
