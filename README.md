# EvolvingContinuum

Code, frozen registrations and analysis outputs for the paper

> **Evolving Energy-Efficient Placement Policies for the Computing Continuum: Language Models
> versus Genetic Programming.** Pablo Rodríguez Martín, Sergio Laso, Juan Luis Herrera,
> Adolfo Lozano-Tello, Javier Berrocal. University of Extremadura. Submitted to *Future
> Generation Computer Systems*.

Raw run archives: [10.5281/zenodo.23964427](https://doi.org/10.5281/zenodo.23964427).

Placement policies are short programs in a small typed language (PlacementScript) that plug into
the scoring step of a Kubernetes-style scheduler. An LLM or genetic programming (GP) proposes
edits to these programs, and a discrete-event simulator of a fog/cloud cluster evaluates them for
QoS and modeled energy. Champions are selected on held-out traffic and judged on fresh traffic.

## Repository layout

| Path | Contents |
|---|---|
| `runtime/` | PlacementScript interpreter (Rust crate `placementscript-core`, native and `wasm32-wasip1`), Python bindings, conformance suite, reference policies (`runtime/policies/*.ps`) |
| `sim/` | Discrete-event simulator (SimPy): application/queueing model, scheduling-cycle emulator, energy model, scenarios |
| `evolution/` | Search harness: evaluation ladder, islands, GP and LLM proposal operators, LLM engines (OpenRouter, Codex CLI) |
| `baselines/` | Hand-written heuristics and the GP baselines |
| `analysis/` | Seed derivation and run tracking |
| `prereg/` | Frozen registrations of every study (protocols, seeds, task/campaign configs, initialization, source hashes) |
| `experiments/` | One folder per study, with launch/analysis scripts and outputs (see below) |
| `figures/` | Scripts that regenerate the paper's figures from the committed outputs |
| `tests/` | Unit and regression tests |
| `docs/` | PlacementScript language specification, seeding convention, third-party data notes |

### Studies in the paper

| Paper | Registration (`prereg/`) | Code and outputs (`experiments/`) |
|---|---|---|
| Study F (main text: GP vs DeepSeek V4.1 Flash vs gpt-6.1-sol, 90 paired runs, three scenarios) | `STUDY_F_SOL_20261007` | `study_f_20261007` |
| Study E (three scenarios, DeepSeek vs GP, zero-shot arm) | `STUDY_E_MULTIFAMILY_20261005` | `study_e_20261005` |
| Study D (single search seed) | `STUDY_D_SINGLESEED_20261004` | `study_d_20261004` |
| Study C (four search seeds) | `STUDY_C_ROBUST_20261003` | `study_c_20261003` |
| Studies A and B (first registered family) | `CORRECTED_STUDIES_20260920_v3` | `corrected_ab_20260925` |
| Validation-grid re-selection (supplement) | – | `brittleness_20261002` |
| Execution cost of evolved policies | – | `execution_cost_20261002` |
| Evaluation-ladder fidelity check | – | `ladder_fidelity_20260924` |
| Simulator correctness gate | – | `scientific_gates_20260920` |

Each study folder has `analysis/PRIMARY.json` (registered tests and descriptive statistics) and
`analysis/RESULTS.md` (readable summary). The `runs/<root>/` subfolders keep, for each paired
run, the selection record (`SELECTION.json`), the labels of the programs judged on the final grid
(`final/LABELS.json`) and the evolved programs themselves (`*.ps`).

The other folders under `experiments/` hold modules and inputs that the study launchers import
(engine clients, the matched campaign runner, earlier calibration outputs). They stay at their
original paths because the registrations record file hashes against those paths.

### Raw run archives

The raw outputs of the six studies (per-candidate final-grid records `runs/<root>/final/<sha>.json`,
paired-run state `pair.json`, campaign manifests and event logs, LLM request/response journals and
search archives, Study F validation records) total about 26 GB and are not in this repository.
They are archived on Zenodo, one tarball per study, at [10.5281/zenodo.23964427](https://doi.org/10.5281/zenodo.23964427).
Extract each archive at the root of a checkout to re-run the `analyze.py` scripts or to replay the
LLM arms from their journals:

```bash
for f in /path/to/archives/*.tar.gz; do tar xzf "$f"; done
```

The Alibaba-derived call-graph sample used by the ingestion tests is a separate record,
[10.5281/zenodo.23964625](https://doi.org/10.5281/zenodo.23964625), because it keeps the trace maintainers' research-use terms
(see `docs/DATA_LICENSES.md`).

Approval fields in the registration receipts were edited for publication and the hashes that
cover them recomputed; see `prereg/README.md`.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a stable Rust toolchain.

The evolution engine is a pinned fork of [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve),
checked out next to this repository (`../ShinkaEvolve`). Rebuild the exact pinned tree from the
public base and the two patches in `evolution/shinkaevolve-patches/`:

```bash
git clone https://github.com/rodmarkun/ShinkaEvolve.git ../ShinkaEvolve
git -C ../ShinkaEvolve checkout --detach 4fba8f69a51a9da7970ac8cdd89de148b9696bea
git -C ../ShinkaEvolve apply --index "$PWD"/evolution/shinkaevolve-patches/0001-*.patch
git -C ../ShinkaEvolve apply --index "$PWD"/evolution/shinkaevolve-patches/0002-*.patch
test "$(git -C ../ShinkaEvolve write-tree)" = "710a2feb684aa9d934039256cbfa520efaa245cd"
```

Then install the environment and build the PlacementScript bindings:

```bash
uv venv --python 3.12
uv sync                      # add --extra rl for the study launchers and the PPO baseline
uv pip install maturin
uv run --no-sync maturin develop --release -m runtime/crates/placementscript-py/Cargo.toml

uv run --no-sync pytest      # Python tests

# Interpreter tests; the native/wasm parity suite needs the wasm guest built first
rustup target add wasm32-wasip1
cargo build --manifest-path runtime/Cargo.toml -p placementscript-wasm-guest --target wasm32-wasip1
cargo test --manifest-path runtime/Cargo.toml
```

Some Python tests need inputs that are not in the repository: the Alibaba call-graph sample
(`experiments/alibaba_ingestion_20260809/call_graph_sample.json`, from [10.5281/zenodo.23964625](https://doi.org/10.5281/zenodo.23964625)), the
checkouts made by `scripts/fetch_data.sh`, and the `rl` extra. `tests/conftest.py` skips those
tests, naming the missing input, until it is present. It also lists three stale tests of
development experiments as expected failures. The sealed energy-bundle registration reproduces
bit for bit only on macOS arm64, where the studies ran, so CI runs the Python suite there.

## Reproducing

**Figures** (read only committed outputs):

```bash
uv run --no-sync python figures/make_headline_f.py   # Study F outcomes
uv run --no-sync python figures/make_mechanism.py    # Study F mechanism
uv run --no-sync python figures/make_tradeoff.py     # energy/QoS trade-off, Studies A-D
uv run --no-sync python figures/make_topology.py     # scenario topology
tectonic figures/pipeline.tex                        # process diagram
```

**Analyses** (need the Zenodo archives extracted first), for example:

```bash
uv run --no-sync python experiments/study_f_20261007/analyze.py
```

**New searches** call paid LLM APIs. The DeepSeek arm uses OpenRouter (`OPENROUTER_API_KEY` in
the environment or a local `.env`); the gpt-6.1-sol arm uses the Codex CLI. Each study's
`launch.py` verifies its frozen registration before starting. The published searches ran on one
16 GB laptop.

## License

Apache-2.0 (see `LICENSE`). Third-party data: `docs/DATA_LICENSES.md`.
