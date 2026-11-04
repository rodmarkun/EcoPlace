//! Core semantics tests. The exhaustive per-construct matrix lives in the conformance
//! suite (`placementscript-conformance`, run against native AND wasm builds); these are
//! the interpreter-development tests.

use placementscript_core::ast::RecordType;
use placementscript_core::value::Value;
use placementscript_core::{bundle, call, compile, CallError, Limits, PsError};

fn run(src: &str, func: &str, args: &[Value]) -> Result<Value, PsError> {
    let prog = compile(src).expect("compile");
    match call(&prog, func, args, Limits::default_score()) {
        Ok(o) => Ok(o.value),
        Err(CallError::Ps(e)) => Err(e),
        Err(CallError::Host(e)) => panic!("host error: {e}"),
    }
}

fn run_i(src: &str, func: &str, args: &[Value]) -> i64 {
    match run(src, func, args).unwrap() {
        Value::Int(v) => v,
        other => panic!("expected int, got {other:?}"),
    }
}

fn run_f(src: &str, func: &str, args: &[Value]) -> f64 {
    match run(src, func, args).unwrap() {
        Value::Float(v) => v,
        other => panic!("expected float, got {other:?}"),
    }
}

#[test]
fn arithmetic_and_precedence() {
    let src = "def f(a: int, b: int) -> int:\n    return a + b * 2 - (a - b) // 3\n";
    assert_eq!(run_i(src, "f", &[Value::Int(10), Value::Int(4)]), 16);
}

#[test]
fn python_floor_div_and_mod_signs() {
    let src = "def f(a: int, b: int) -> int:\n    return a // b\n";
    assert_eq!(run_i(src, "f", &[Value::Int(-7), Value::Int(2)]), -4);
    assert_eq!(run_i(src, "f", &[Value::Int(7), Value::Int(-2)]), -4);
    let src = "def f(a: int, b: int) -> int:\n    return a % b\n";
    assert_eq!(run_i(src, "f", &[Value::Int(-7), Value::Int(2)]), 1);
    assert_eq!(run_i(src, "f", &[Value::Int(7), Value::Int(-2)]), -1);
}

#[test]
fn int_overflow_traps() {
    let src = "def f(a: int, b: int) -> int:\n    return a + b\n";
    let e = run(src, "f", &[Value::Int(i64::MAX), Value::Int(1)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }), "{e}");
    let src = "def f(a: int) -> int:\n    return -a\n";
    let e = run(src, "f", &[Value::Int(i64::MIN)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let src = "def f(a: int, b: int) -> int:\n    return a // b\n";
    let e = run(src, "f", &[Value::Int(i64::MIN), Value::Int(-1)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let e = run(src, "f", &[Value::Int(1), Value::Int(0)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
}

#[test]
fn float_nonfinite_traps() {
    let src = "def f(a: float, b: float) -> float:\n    return a / b\n";
    let e = run(src, "f", &[Value::Float(1.0), Value::Float(0.0)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let src = "def f(a: float) -> float:\n    return a * 1e308\n";
    let e = run(src, "f", &[Value::Float(10.0)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let src = "def f(a: float) -> float:\n    return sqrt(a)\n";
    let e = run(src, "f", &[Value::Float(-1.0)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
}

#[test]
fn subnormals_are_ordinary() {
    let src = "def f(a: float) -> float:\n    return a / 2.0\n";
    let v = run_f(src, "f", &[Value::Float(f64::MIN_POSITIVE)]);
    assert!(v > 0.0 && v < f64::MIN_POSITIVE); // gradual underflow, no trap
    let v2 = run_f(src, "f", &[Value::Float(5e-324)]);
    assert_eq!(v2, 0.0); // underflow to zero does not trap
}

#[test]
fn nan_literal_rejected_at_parse() {
    let src = "def f() -> float:\n    return 1e999\n";
    let e = compile(src).unwrap_err();
    assert!(matches!(e, PsError::Parse { .. }), "{e}");
}

#[test]
fn int_literal_out_of_range_rejected() {
    let src = "def f() -> int:\n    return 9223372036854775808\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Parse { .. }));
}

#[test]
fn loops_and_lists() {
    let src = "def f(n: int) -> int:\n    acc = 0\n    for i in range(n):\n        acc += i\n    return acc\n";
    assert_eq!(run_i(src, "f", &[Value::Int(10)]), 45);
    let src = "def f(xs: list[int]) -> int:\n    acc = 0\n    for x in xs:\n        acc += x\n    return acc\n";
    let xs = Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(run_i(src, "f", &[xs]), 6);
}

#[test]
fn list_mutation_during_iteration_traps() {
    let src = "def f(xs: list[int]) -> int:\n    for x in xs:\n        xs.append(x)\n    return len(xs)\n";
    let xs = Value::list(vec![Value::Int(1)]);
    let e = run(src, "f", &[xs]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
}

#[test]
fn index_out_of_bounds_traps_including_negative() {
    let src = "def f(xs: list[int], i: int) -> int:\n    return xs[i]\n";
    let xs = Value::list(vec![Value::Int(7)]);
    let e = run(src, "f", &[xs.clone(), Value::Int(1)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let e = run(src, "f", &[xs, Value::Int(-1)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
}

#[test]
fn fuel_exhaustion() {
    let src = "def f(n: int) -> int:\n    acc = 0\n    for i in range(n):\n        acc += 1\n    return acc\n";
    let prog = compile(src).unwrap();
    let e = call(
        &prog,
        "f",
        &[Value::Int(1_000_000)],
        Limits {
            fuel: 1000,
            memory_bytes: 1 << 20,
        },
    )
    .unwrap_err();
    assert!(matches!(e, CallError::Ps(PsError::FuelExhausted)));
}

#[test]
fn memory_exhaustion() {
    let src = "def f(n: int) -> int:\n    xs: list[int] = []\n    for i in range(n):\n        xs.append(i)\n    return len(xs)\n";
    let prog = compile(src).unwrap();
    let e = call(
        &prog,
        "f",
        &[Value::Int(1_000_000)],
        Limits {
            fuel: 100_000_000,
            memory_bytes: 4096,
        },
    )
    .unwrap_err();
    assert!(matches!(e, CallError::Ps(PsError::OutOfMemory)));
}

#[test]
fn no_recursion_by_construction() {
    let src = "def f(n: int) -> int:\n    return f(n)\n";
    let e = compile(src).unwrap_err();
    assert!(matches!(e, PsError::Type { .. }), "{e}");
    // Forward reference (mutual recursion) also impossible.
    let src = "def f(n: int) -> int:\n    return g(n)\ndef g(n: int) -> int:\n    return f(n)\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Type { .. }));
}

#[test]
fn definite_assignment_and_return_paths() {
    let src = "def f(c: bool) -> int:\n    if c:\n        x = 1\n    return x\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Type { .. }));
    let src = "def f(c: bool) -> int:\n    if c:\n        return 1\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Type { .. }));
    let src = "def f(c: bool) -> int:\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n";
    assert_eq!(run_i(src, "f", &[Value::Bool(true)]), 1);
}

#[test]
fn no_mixed_arithmetic_or_chained_comparison() {
    let src = "def f(a: int, b: float) -> float:\n    return float(a) + b\n";
    assert!(compile(src).is_ok());
    let src = "def f(a: int, b: float) -> float:\n    return a + b\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Type { .. }));
    let src = "def f(a: int) -> bool:\n    return 0 < a < 10\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Parse { .. }));
}

#[test]
fn short_circuit_prevents_trap() {
    let src = "def f(a: int, b: int) -> bool:\n    return b != 0 and a // b > 0\n";
    assert!(matches!(
        run(src, "f", &[Value::Int(4), Value::Int(0)]).unwrap(),
        Value::Bool(false)
    ));
}

#[test]
fn min_max_return_first_on_tie() {
    let src = "def f(a: float, b: float) -> float:\n    return max(a, b)\n";
    // -0.0 vs 0.0: equal per IEEE, so `a` is returned bit-exactly.
    let v = run_f(src, "f", &[Value::Float(-0.0), Value::Float(0.0)]);
    assert_eq!(v.to_bits(), (-0.0f64).to_bits());
}

#[test]
fn casts() {
    let src = "def f(x: float) -> int:\n    return int(x)\n";
    assert_eq!(run_i(src, "f", &[Value::Float(-2.9)]), -2); // truncation toward zero
    let e = run(src, "f", &[Value::Float(1e30)]).unwrap_err();
    assert!(matches!(e, PsError::Trap { .. }));
    let src = "def f(x: int) -> float:\n    return float(x)\n";
    assert_eq!(run_f(src, "f", &[Value::Int(3)]), 3.0);
}

#[test]
fn loop_var_scoped_to_body() {
    let src = "def f(n: int) -> int:\n    for i in range(n):\n        pass\n    return i\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Type { .. }));
}

#[test]
fn tab_indentation_rejected() {
    let src = "def f() -> int:\n\treturn 1\n";
    assert!(matches!(compile(src).unwrap_err(), PsError::Parse { .. }));
}

#[test]
fn host_arg_mismatch_is_host_error_not_policy_error() {
    let src = "def f(a: int) -> int:\n    return a\n";
    let prog = compile(src).unwrap();
    let e = call(&prog, "f", &[Value::Float(1.0)], Limits::default()).unwrap_err();
    assert!(matches!(e, CallError::Host(_)));
    let e = call(&prog, "g", &[], Limits::default()).unwrap_err();
    assert!(matches!(e, CallError::Host(_)));
    // Non-finite float input is a host error too.
    let src = "def f(a: float) -> float:\n    return a\n";
    let prog = compile(src).unwrap();
    let e = call(&prog, "f", &[Value::Float(f64::NAN)], Limits::default()).unwrap_err();
    assert!(matches!(e, CallError::Host(_)));
}

#[test]
fn bundle_validation() {
    let good = "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return 0.0\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n";
    let prog = compile(good).unwrap();
    let meta = bundle::validate(&prog).unwrap();
    assert_eq!(meta.name, "t");
    // Missing metadata.
    let bad = good.replace("POLICY_NAME: str = \"t\"\n", "");
    let prog = compile(&bad).unwrap();
    assert!(matches!(
        bundle::validate(&prog).unwrap_err(),
        PsError::Type { .. }
    ));
    // Wrong score signature.
    let bad = good.replace("-> float:\n    return 0.0", "-> int:\n    return 0");
    let prog = compile(&bad).unwrap();
    assert!(matches!(
        bundle::validate(&prog).unwrap_err(),
        PsError::Type { .. }
    ));
}

#[test]
fn score_normalizes_negative_zero() {
    let src = "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return -0.0\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n";
    let prog = compile(src).unwrap();
    bundle::validate(&prog).unwrap();
    let (pod, node, ctx) = tiny_cluster();
    let (v, _) = bundle::score(&prog, &pod, &node, &ctx, Limits::default_score()).unwrap();
    assert_eq!(v.to_bits(), 0.0f64.to_bits());
}

#[test]
fn move_builtins_and_propose() {
    let src = "POLICY_NAME: str = \"t\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return 0.0\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    if budget > 0:\n        out.append(wake(1))\n        out.append(move_pod(0, 1))\n    return out\n";
    let prog = compile(src).unwrap();
    let (pod, node, _ctx) = tiny_cluster();
    let state = Value::record(
        RecordType::ClusterState,
        vec![Value::list(vec![node]), Value::list(vec![pod])],
    )
    .unwrap();
    let graph = empty_graph();
    let (moves, _) =
        bundle::propose_moves(&prog, &state, &graph, 5, Limits::default_propose()).unwrap();
    assert_eq!(moves.len(), 2);
    assert_eq!((moves[0].kind, moves[0].pod, moves[0].dest), (1, -1, 1));
    assert_eq!((moves[1].kind, moves[1].pod, moves[1].dest), (0, 0, 1));
}

#[test]
fn determinism_bitwise() {
    let src = "def f(a: float, b: float) -> float:\n    return (a / b + sqrt(b)) * a - 0.1\n";
    let args = [Value::Float(1.7e-3), Value::Float(3.9)];
    let v1 = run_f(src, "f", &args);
    let v2 = run_f(src, "f", &args);
    assert_eq!(v1.to_bits(), v2.to_bits());
}

#[test]
fn fuel_and_mem_accounting_deterministic() {
    let src = "def f(n: int) -> int:\n    xs: list[int] = []\n    for i in range(n):\n        xs.append(i * 2)\n    return len(xs)\n";
    let prog = compile(src).unwrap();
    let o1 = call(&prog, "f", &[Value::Int(100)], Limits::default_propose()).unwrap();
    let o2 = call(&prog, "f", &[Value::Int(100)], Limits::default_propose()).unwrap();
    assert_eq!(o1.fuel_used, o2.fuel_used);
    assert_eq!(o1.mem_used, o2.mem_used);
    assert!(o1.fuel_used > 0 && o1.mem_used > 0);
}

// --- fixtures ---

fn empty_graph() -> Value {
    Value::record(
        RecordType::CallGraphCtx,
        vec![
            Value::list(vec![]),
            Value::list(vec![]),
            Value::list(vec![]),
            Value::list(vec![]),
        ],
    )
    .unwrap()
}

fn tiny_cluster() -> (Value, Value, Value) {
    let pod = Value::record(
        RecordType::Pod,
        vec![
            Value::Int(0),
            Value::Int(0),
            Value::Int(0),
            Value::Int(100),
            Value::Int(128),
            Value::Bool(false),
            Value::Bool(false),
            Value::Int(50),
            Value::Int(-1),
        ],
    )
    .unwrap();
    let node = Value::record(
        RecordType::NodeState,
        vec![
            Value::Int(0),
            Value::Int(2),
            Value::Int(4000),
            Value::Int(1000),
            Value::Int(8192),
            Value::Int(2048),
            Value::Float(3.0),
            Value::Float(8.0),
            Value::Bool(false),
            Value::Bool(true),
            Value::list(vec![Value::Float(0.0)]),
        ],
    )
    .unwrap();
    let ctx = Value::record(
        RecordType::ClusterCtx,
        vec![
            Value::list(vec![node.clone()]),
            Value::list(vec![pod.clone()]),
            empty_graph(),
            Value::Int(0),
        ],
    )
    .unwrap();
    (pod, node, ctx)
}

// --- Round-1 review regressions (docs/PHASE1_REVIEW.md) ---

#[test]
fn frozen_host_lists_reject_append() {
    // R1.2: host-provided lists are frozen; appending through an alias traps.
    let src = "def f(xs: list[int]) -> int:\n    ys = xs\n    ys.append(1)\n    return len(ys)\n";
    let xs = Value::frozen_list(vec![Value::Int(1)]);
    let e = run(src, "f", &[xs]).unwrap_err();
    assert!(
        matches!(&e, PsError::Trap { message } if message.contains("host-provided")),
        "{e}"
    );
    // Policy-created lists remain append-mutable.
    let src = "def f() -> int:\n    xs: list[int] = []\n    ys = xs\n    ys.append(4)\n    return xs[0]\n";
    assert_eq!(run_i(src, "f", &[]), 4);
}

#[test]
fn scorer_cannot_contaminate_batch_via_ctx_mutation() {
    // R1.2 repro: `xs = ctx.pods; xs.append(...)` used to make batch scores
    // candidate-order-dependent. With frozen host state it traps deterministically,
    // and a non-mutating scorer is batch/scalar equivalent.
    let mutator = "POLICY_NAME: str = \"m\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    xs = ctx.pods\n    xs.append(pod)\n    return float(len(xs))\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n";
    let prog = compile(mutator).unwrap();
    bundle::validate(&prog).unwrap();
    let (pod, node, ctx) = frozen_tiny_cluster();
    let err = bundle::score_batch(
        &prog,
        &pod,
        &[node.clone(), node.clone()],
        &ctx,
        Limits::default_score(),
    )
    .unwrap_err();
    assert!(
        matches!(err.1, CallError::Ps(PsError::Trap { .. })),
        "{}",
        err.1
    );

    // Batch equals scalar for a well-behaved scorer, in any candidate order.
    let ok = "POLICY_NAME: str = \"ok\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    return float(node.cpu_capacity - node.cpu_allocated)\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n";
    let prog = compile(ok).unwrap();
    let (scores, _) = bundle::score_batch(
        &prog,
        &pod,
        &[node.clone(), node.clone()],
        &ctx,
        Limits::default_score(),
    )
    .unwrap();
    let (s0, _) = bundle::score(&prog, &pod, &node, &ctx, Limits::default_score()).unwrap();
    assert_eq!(scores[0].to_bits(), s0.to_bits());
    assert_eq!(scores[1].to_bits(), s0.to_bits());
}

#[test]
fn string_literals_are_load_interned_not_charged_per_evaluation() {
    // R1.3: evaluating a string literal N times allocates nothing and charges no
    // invocation memory (it is interned at load, charged to the load budget).
    let src = "def f(n: int) -> str:\n    s = \"x\"\n    for i in range(n):\n        s = \"a-reasonably-long-string-literal\"\n    return s\n";
    let prog = compile(src).unwrap();
    let out = call(&prog, "f", &[Value::Int(10_000)], Limits::default_propose()).unwrap();
    assert_eq!(
        out.mem_used, 0,
        "string literal evaluation must charge 0 bytes"
    );
}

#[test]
fn structural_limits_enforced_at_load() {
    // R1.1: expression nesting depth (parse error).
    let deep = format!(
        "def f() -> int:\n    return {}1{}\n",
        "(".repeat(40),
        ")".repeat(40)
    );
    let e = compile(&deep).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );
    // At/below the limit parses fine.
    let ok = format!(
        "def f() -> int:\n    return {}1{}\n",
        "(".repeat(30),
        ")".repeat(30)
    );
    compile(&ok).unwrap();

    // Block nesting depth (parse error).
    let mut src = String::from("def f() -> int:\n");
    for d in 0..9 {
        src.push_str(&format!("{}if True:\n", "    ".repeat(d + 1)));
    }
    src.push_str(&format!("{}pass\n", "    ".repeat(10)));
    src.push_str("    return 0\n");
    let e = compile(&src).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("block nesting")),
        "{e}"
    );

    // Function count (type error). 65 tiny functions.
    let mut src = String::new();
    for i in 0..65 {
        src.push_str(&format!("def f{i}() -> int:\n    return {i}\n"));
    }
    let e = compile(&src).unwrap_err();
    assert!(
        matches!(&e, PsError::Type { message, .. } if message.contains("too many functions")),
        "{e}"
    );

    // Static call-chain depth (type error): f0 <- f1 <- ... <- f16 is depth 17.
    let mut src = String::from("def f0() -> int:\n    return 1\n");
    for i in 1..17 {
        src.push_str(&format!("def f{i}() -> int:\n    return f{}()\n", i - 1));
    }
    let e = compile(&src).unwrap_err();
    assert!(
        matches!(&e, PsError::Type { message, .. } if message.contains("call chain too deep")),
        "{e}"
    );
    // Depth 16 (f0..f15) is legal and runs on the default budget.
    let mut src = String::from("def f0() -> int:\n    return 1\n");
    for i in 1..16 {
        src.push_str(&format!("def f{i}() -> int:\n    return f{}()\n", i - 1));
    }
    let prog = compile(&src).unwrap();
    let out = call(&prog, "f15", &[], Limits::default_score()).unwrap();
    assert!(matches!(out.value, Value::Int(1)));

    // R1.1 repro (scaled): a huge helper module fails load *fuel*, structurally, not
    // with a native stack overflow. 5000 functions x ~7 AST nodes > 100k load fuel
    // (and > MAX_FUNCTIONS); the blunt function cap fires first with default limits,
    // so check the AST-size charge with an explicit generous function allowance.
    let mut src = String::new();
    for i in 0..64 {
        // Each function large enough that 64 of them exceed the default load fuel.
        src.push_str(&format!("def g{i}() -> int:\n    x = 1\n"));
        for _ in 0..400 {
            src.push_str("    x = x + 1 + 1 + 1\n");
        }
        src.push_str("    return x\n");
    }
    let e = compile(&src).unwrap_err();
    assert!(matches!(e, PsError::FuelExhausted), "{e}");
}

fn frozen_tiny_cluster() -> (Value, Value, Value) {
    // The same cluster as `tiny_cluster`, but built the way hosts build it: via the
    // frozen-list boundary constructors.
    let pod = Value::record(
        RecordType::Pod,
        vec![
            Value::Int(0),
            Value::Int(0),
            Value::Int(0),
            Value::Int(100),
            Value::Int(128),
            Value::Bool(false),
            Value::Bool(false),
            Value::Int(50),
            Value::Int(-1),
        ],
    )
    .unwrap();
    let node = Value::record(
        RecordType::NodeState,
        vec![
            Value::Int(0),
            Value::Int(2),
            Value::Int(4000),
            Value::Int(1000),
            Value::Int(8192),
            Value::Int(2048),
            Value::Float(3.0),
            Value::Float(8.0),
            Value::Bool(false),
            Value::Bool(true),
            Value::frozen_list(vec![Value::Float(0.0)]),
        ],
    )
    .unwrap();
    let graph = Value::record(
        RecordType::CallGraphCtx,
        vec![
            Value::frozen_list(vec![]),
            Value::frozen_list(vec![]),
            Value::frozen_list(vec![]),
            Value::frozen_list(vec![]),
        ],
    )
    .unwrap();
    let ctx = Value::record(
        RecordType::ClusterCtx,
        vec![
            Value::frozen_list(vec![node.clone()]),
            Value::frozen_list(vec![pod.clone()]),
            graph,
            Value::Int(0),
        ],
    )
    .unwrap();
    (pod, node, ctx)
}

#[test]
fn operator_chains_and_type_nesting_bounded_r2() {
    // Round-2 review (R1.1 re-open): a long left-associative operator chain builds a
    // deep tree; it must be a structured parse error, never a stack overflow (the
    // original repro was a 20k-term addition that SIGSEGV'd compile_check).
    let chain = format!("def f() -> int:\n    return {}1\n", "1 + ".repeat(19_999));
    let e = compile(&chain).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );
    // At the limit (31 operators → tree height 32) it still compiles and runs.
    let ok = format!("def f() -> int:\n    return {}1\n", "1 + ".repeat(31));
    let prog = compile(&ok).unwrap();
    let out = call(&prog, "f", &[], Limits::default_score()).unwrap();
    assert!(matches!(out.value, Value::Int(32)));
    // Boolean chains are charged the same way.
    let bools = format!(
        "def f() -> bool:\n    return {}True\n",
        "True and ".repeat(32)
    );
    let e = compile(&bools).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );

    // Type nesting (the second repro: 100k nested `list[` also SIGSEGV'd). The
    // original 600 KB repro now exhausts *load fuel during lexing* (R3: one fuel per
    // source byte) — an even earlier structured error — so use a source under the
    // byte budget to exercise the type-nesting limit itself.
    let deep_ty_oversized = format!(
        "def f(x: {}int{}) -> int:\n    return 0\n",
        "list[".repeat(100_000),
        "]".repeat(100_000)
    );
    let e = compile(&deep_ty_oversized).unwrap_err();
    assert!(matches!(&e, PsError::FuelExhausted), "{e}");
    let deep_ty = format!(
        "def f(x: {}int{}) -> int:\n    return 0\n",
        "list[".repeat(1_000),
        "]".repeat(1_000)
    );
    let e = compile(&deep_ty).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("type nesting too deep")),
        "{e}"
    );
    let ok_ty = format!(
        "def f(x: {}int{}) -> int:\n    return 0\n",
        "list[".repeat(8),
        "]".repeat(8)
    );
    compile(&ok_ty).unwrap();
}

#[test]
fn postfix_chains_bounded_post_r2() {
    // Post-Round-2 self-review (R1.1 residual): postfix suffixes (`[i]` / `.field`) also
    // build a deep left spine and were uncharged — 20k `[0]` suffixes SIGSEGV'd
    // compile_check. Must be a structured parse error, never a stack overflow.
    let deep_index = format!(
        "def f(xs: list[int]) -> int:\n    return xs{}\n",
        "[0]".repeat(20_000)
    );
    let e = compile(&deep_index).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );
    // Field chains are charged identically (parse error fires before type checking).
    let deep_field = format!("def f(p: Pod) -> int:\n    return p{}\n", ".id".repeat(40));
    let e = compile(&deep_field).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );
    // Just over the limit is already structural, without needing 20k suffixes.
    let over = format!(
        "def f(xs: list[int]) -> int:\n    return xs{}\n",
        "[0]".repeat(40)
    );
    let e = compile(&over).unwrap_err();
    assert!(
        matches!(&e, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{e}"
    );
    // A deep-but-legal chain still parses; structural misuse below the depth limit
    // stays a *type* error (over-indexing an int), proving parse accepts the chain.
    let ty_err = format!(
        "def f(xs: list[int]) -> int:\n    return xs{}\n",
        "[0]".repeat(20)
    );
    let e = compile(&ty_err).unwrap_err();
    assert!(matches!(&e, PsError::Type { .. }), "{e}");
    // And a realistic mixed chain compiles and runs.
    let ok = "def f(xs: list[list[int]]) -> int:\n    return xs[0][0]\n";
    let prog = compile(ok).unwrap();
    let inner = Value::frozen_list(vec![Value::Int(7)]);
    let outer = Value::frozen_list(vec![inner]);
    let out = call(&prog, "f", &[outer], Limits::default_score()).unwrap();
    assert!(matches!(out.value, Value::Int(7)));
}

#[test]
fn expr_height_bounded_across_precedence_r3() {
    // Round-3 review (R1.1 re-open): the open-context depth counter released each
    // precedence layer's charge before its caller wrapped the finished subtree, so
    // nested parenthesized operator chains legally stacked tree heights far past the
    // limit (a 30-deep paren/chain nest built an AST of height ~932; with the 16-deep
    // call chain that invalidated the 4 MiB wasm stack proof). Height is now computed
    // compositionally at node construction and over-limit nodes are never built.
    let mut e = "1".to_string() + &" + 1".repeat(8); // height 9
    for _ in 0..30 {
        e = format!("({e}) * 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1"); // +8 height per wrap
    }
    let src = format!("def f() -> int:\n    return {e}\n");
    let err = compile(&src).unwrap_err();
    assert!(
        matches!(&err, PsError::Parse { message, .. } if message.contains("nesting too deep")),
        "{err}"
    );
    // Exactly at the limit (height 32) still compiles and evaluates correctly.
    let mut e = "1".to_string() + &" + 1".repeat(7); // height 8
    for _ in 0..3 {
        e = format!("({e}) + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1"); // +8 height per wrap
    }
    let src = format!("def f() -> int:\n    return {e}\n");
    assert_eq!(run_i(&src, "f", &[]), 32);
}

#[test]
fn lexing_charges_load_fuel_r3() {
    // Round-3 review: load fuel was charged only at type-check, *after* unbounded
    // lexing and parsing — a token-dense multi-MiB source allocated hundreds of MiB
    // of tokens before failing. Source bytes are now charged during lexing, so an
    // over-budget source is a structured fuel error after O(budget) work.
    let dense = "x ".repeat(150_000); // 300 KB > the default 100,000 load fuel
    let e = compile(&dense).unwrap_err();
    assert!(matches!(e, PsError::FuelExhausted), "{e}");
    // Exact accounting boundary: source bytes + AST nodes + constants.
    let src = "def f() -> int:\n    return 1\n"; // 29 bytes, 2 AST nodes, 0 constants
    let total = src.len() as u64 + 2;
    let mem = Limits::default_load().memory_bytes;
    placementscript_core::compile_with_limits(
        src,
        Limits {
            fuel: total,
            memory_bytes: mem,
        },
    )
    .unwrap();
    let e = placementscript_core::compile_with_limits(
        src,
        Limits {
            fuel: total - 1,
            memory_bytes: mem,
        },
    )
    .unwrap_err();
    assert!(matches!(e, PsError::FuelExhausted), "{e}");
    // The byte charge lands in the *lexer*: under a tiny budget, a source whose only
    // grammar problem is at the very end (unclosed bracket) exhausts fuel first.
    let src2 = "def f() -> int:\n    return (\n";
    let e = placementscript_core::compile_with_limits(
        src2,
        Limits {
            fuel: 20,
            memory_bytes: mem,
        },
    )
    .unwrap_err();
    assert!(matches!(e, PsError::FuelExhausted), "{e}");
}

#[test]
fn long_single_tokens_and_unicode_strings_bounded_r4() {
    // Round-4 review: the between-token fuel checks let ONE long token (identifier,
    // number, string, comment) scan and allocate far past the budget (a 64 MiB
    // identifier cost ~67 MiB RSS before failing). Over-length sources are now
    // rejected before scanning begins, so each of these is an immediate structured
    // fuel error with O(1) work:
    let long_ident = format!("def f() -> int:\n    return {}\n", "x".repeat(1 << 20));
    let long_number = format!("def f() -> int:\n    return {}\n", "1".repeat(1 << 20));
    let long_string = format!("S: str = \"{}\"\n", "a".repeat(1 << 20));
    let long_comment = format!("# {}\ndef f() -> int:\n    return 1\n", "c".repeat(1 << 20));
    for src in [long_ident, long_number, long_string, long_comment] {
        let e = compile(&src).unwrap_err();
        assert!(matches!(e, PsError::FuelExhausted), "{e}");
    }
    // Non-ASCII string decoding is linear (Round-4 review: re-validating the whole
    // remaining suffix per character was quadratic — 128 KiB took seconds). A large
    // in-budget Unicode string constant compiles quickly and round-trips exactly.
    let n = 100_000; // 200 KB of 2-byte chars
    let uni = format!(
        "S: str = \"{}\"\ndef f() -> str:\n    return S\n",
        "é".repeat(n)
    );
    let prog = placementscript_core::compile_with_limits(
        &uni,
        Limits {
            fuel: 1 << 20,
            memory_bytes: 1 << 20,
        },
    )
    .unwrap();
    let out = call(&prog, "f", &[], Limits::default_score()).unwrap();
    match out.value {
        Value::Str(s) => assert_eq!(s.chars().count(), n),
        other => panic!("expected str, got {other:?}"),
    }
}

#[test]
fn careless_rust_host_values_are_frozen_at_the_bundle_boundary_r2() {
    // Round-2 review (R1.2 partial): a Rust host that builds Value trees by hand
    // (mutable `Value::list` + raw `RecordVal`, bypassing `Value::record`) must not
    // reintroduce alias contamination — `bundle::*` deep-freezes at entry.
    use std::rc::Rc;
    let mutator = "POLICY_NAME: str = \"m\"\nPOLICY_VERSION: str = \"1\"\n\
def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n    xs = ctx.pods\n    xs.append(pod)\n    return float(len(xs))\n\
def propose_moves(state: ClusterState, graph: CallGraphCtx, budget: int) -> list[Move]:\n    out: list[Move] = []\n    return out\n";
    let prog = compile(mutator).unwrap();
    let (pod, node, _) = frozen_tiny_cluster();
    // Careless ctx: mutable lists, raw record construction.
    let graph = Value::Record(Rc::new(placementscript_core::value::RecordVal {
        rt: RecordType::CallGraphCtx,
        fields: vec![
            Value::list(vec![]),
            Value::list(vec![]),
            Value::list(vec![]),
            Value::list(vec![]),
        ],
    }));
    let ctx = Value::Record(Rc::new(placementscript_core::value::RecordVal {
        rt: RecordType::ClusterCtx,
        fields: vec![
            Value::list(vec![node.clone()]),
            Value::list(vec![pod.clone()]),
            graph,
            Value::Int(0),
        ],
    }));
    let err = bundle::score_batch(
        &prog,
        &pod,
        &[node.clone(), node.clone()],
        &ctx,
        Limits::default_score(),
    )
    .unwrap_err();
    assert!(
        matches!(err.1, CallError::Ps(PsError::Trap { .. })),
        "append to host state must trap even for hand-built mutable inputs: {}",
        err.1
    );
}
