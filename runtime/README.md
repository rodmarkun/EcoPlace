# runtime/ — PlacementScript interpreter

Rust workspace for PlacementScript, the small statically typed, capability-free language in which
every placement policy (hand-written or evolved) is written. The normative definition is
[`docs/placementscript-spec.md`](../docs/placementscript-spec.md). There is one interpreter,
compiled to a native library and to `wasm32-wasip1`.

## Crates

- `crates/placementscript-core` — lexer, parser, type checker and tree-walking interpreter, with
  fuel metering and memory accounting; structured errors (`parse`, `type`, `trap`, `fuel`, `oom`);
  policy-bundle validation; the wire types; and the AST API used by the GP operator (typed subtree
  enumeration, crossover, mutation, numeric-literal patching).
- `crates/placementscript-py` — PyO3 bindings (Python module `placementscript`). The batch API
  `Policy.score_batch(pod, nodes, ctx)` scores all nodes in one boundary crossing.
- `crates/placementscript-wasm-guest` — the deployable WebAssembly module: interpreter plus an
  embedded policy (`PS_EMBED_POLICY`), exporting a postcard-encoded ABI (`ps_load_policy`,
  `ps_score_batch`, `ps_propose_moves`, ...).
- `crates/placementscript-wasm-host` — wasmtime host with fuel and linear-memory limits and no WASI
  capabilities (WASI imports resolve to trapping stubs). Includes the `ps-wasm-bench`
  decision-latency benchmark.
- `crates/placementscript-conformance` — runs the cases in `conformance/cases/*.json` (every construct,
  numeric edge cases, fuel/oom traps, tie-breaking, static errors), run against the native and the
  wasm build, which must agree bit for bit on values, error classes, fuel and memory.

`policies/*.ps` are the hand-written reference policies (least/most allocated, latency- and
energy-greedy, kube-scheduler scoring analogues, first fit, network overhead, graph headroom).

The guest ABI uses postcard: serde-native (one definition of the records), deterministic and
compact, and small in the guest (~270 KB module).

## Build and test

```bash
uv run --no-sync maturin develop --release -m runtime/crates/placementscript-py/Cargo.toml
rustup target add wasm32-wasip1
cargo build --manifest-path runtime/Cargo.toml -p placementscript-wasm-guest --target wasm32-wasip1
cargo test  --manifest-path runtime/Cargo.toml --workspace
```

The conformance wasm tests need the guest built first, as above.
