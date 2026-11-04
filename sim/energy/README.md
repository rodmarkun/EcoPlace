# sim/energy/ — EnergyModelBundles + sensitivity (Phase 3)

**TODO items:** 3.1–3.5 (plan §4, §5, §6.3; corrections S4/S5, T4, U3/U4, V1/V2, W2/W3).
**Depends on:** Phase 2's boundary machinery (now `sim/energy/boundary.py`, moved verbatim
from `sim/energy.py`; every existing `from sim.energy import ...` still works via
`__init__.py` re-exports).
**Review status:** under the Phase-3 adversarial loop (`PHASE3_REVIEW.md`); R1-x/R2-x/R3-x
tags in code comments reference per-round findings.

## Module map

| Module | TODO | Contents |
|---|---|---|
| `boundary.py` | 2.4 + 3.4 | Phase 2's closed energy boundary (`EnergyMeter`, `EnergyReport`, the three never-silently-summed categories, `idle_floor_j` (V1), `ceiling_j` (W2), `ProvenanceRecord` (§4.3)) plus the Phase-3 **`BoundaryCertificate`** (R1-6/R2-1): `certify_boundary(...)` is the **only seal-producing constructor** — it computes I and M from one (cluster, coefficients, weights, horizon, bounds) tuple and binds the instance identity incl. the node count; a hand-built or field-tampered certificate fails at every use. `EnergyReport` is **meter-sealed** (R3-1): only `EnergyMeter.finish()` produces one, so a hand-assembled or field-edited report cannot mint tokens; all meter inputs and report fields are finite/non-negative, category totals must equal their metered bases, and every reporting/archive consumer re-verifies the seal (R4-2). `verify_covers(report)` verifies the full identity — cluster, horizon, node count, coefficients, transition/byte bounds, E ≤ M — and returns a sealed **`VerifiedInstanceEnergy` token** citing the **issuing certificate's seal** (which covers weights and bounds, so two certificates whose derived I/M coincide remain distinct boundaries); tokens are the only currency `InstanceOutcome` and the certificate's regret/penalty methods accept. Provenance records carry the bundle set from `EnergyCoefficients.bundle_set`, the **structured `source_records`** (id/URL/DOI/access-date/content-hash per pinned entry, filtered per category — R2-2), and state the network path-scope assumption (R1-5). |
| `sources.py` | 3.1 | Pinned published sources with a **per-number provenance record** (source, URL/DOI, access date, exact field transcribed, status, retained-artifact `content_sha256` + `artifact_path` where one exists). **Retained artifacts are re-hashed at every load** (R2-2): a record citing an artifact whose file is missing or whose hash diverges refuses to load; artifact paths must be plain relative paths that resolve inside the pinned data directory (R3-6). `SourceRegistry` is immutable and loader-issued; its seal covers every source object plus the exact pinned-data closure digest, and scientific consumers independently compare its full payload with the canonical repository data (R4-3), so caller-built objects cannot acquire authority by retaining a `VERIFIED` label. `SourceRegistry.verified()` is the only pool default bundle sets may draw from. SPECpower entries are **joint (throughput, power) tuples per published machine**, and the curve coordinate is the row's measured `ssj_ops/peak_ssj_ops` fraction, not the nominal target-load label (R1-1). Machines carry a `benchmark_domain`; throughput points are never compared across domains. |
| `data/` | 3.1–3.5 | `specpower.json` (machine-parsed from spec.org result pages by `fetch_specpower.py` — never hand-transcribed; content sha256 of each fetched page recorded); `pinned_sources.json` (ARM SBC characterizations incl. published HPL GFLOPS points, Grid'5000 off/on transition measurements, CCF/Aslan allocation coefficients — each with verbatim `field_transcribed` and the verification method); `energy_analysis_registration.json` (the sealed split-role and complete GSA-design registration, including bundle manifest hashes and `design_sha256`); `artifacts/` (retained raw fetches, e.g. the CCF methodology page with its sha256). `fetch_specpower.py` is **build-time tooling**: the simulator never imports it; web access exists only at pinning time. |
| `registration.py` | 3.3 + 3.5 | Loads `energy_analysis_registration.json` into immutable mappings, recomputes the split payload hash and full GSA `design_sha256` with strict canonical JSON, validates role/roster/bundle/stratum completeness, and cross-checks that the GSA bundle identities exactly equal the development split. Validators use this artifact as authority; builders are never consulted to decide completeness (R7-1/R7-2). |
| `capacity.py` | 3.2 | Workload→capacity mapping (T4): abstract capacity units (= sim millicores), **domain-scoped anchoring** (R1-1) — one `DomainAnchor` per benchmark domain (SPECpower ssj_ops; HPL GFLOPS), `capacity = anchor_capacity × peak/anchor_peak` within a domain only, cross-domain use rejected; core-count fallback only for boards publishing no throughput point at all. Anchors are **registry-backed and sealed** (R2-8/R3): `DomainAnchor.from_machine` is the only seal-producing factory (domain + peak from the cited entry's own published values — a hand-typed anchor cannot be constructed), and `CapacityMapping` re-verifies every anchor at construction and again at every pricing/payload use (R4-4), so even post-construction field mutation cannot price a capacity. `validate_anchor`/`validate_mapping` additionally check anchors against the registry (enforced by `validate_bundle`, the tuple factory, and the default-mapping builder), and each machine tuple's manifest materializes the anchor payload. **A stated modeling assumption, not "consistency by construction"** — `LIMITATIONS` enumerates the threats-to-validity inputs the paper must cite (incl. cross-domain anchor commensurability). |
| `bundles.py` | 3.3 | `MachineTuple` and `EnergyModelBundle` with **content seals** (R1-3): **`machine_tuple_from_registry()` and `bundle_from_registry()` are the only public factories** (R2-5/R3-2) — both accept registry entry *ids*, resolve every value from the VERIFIED registry internally, and validate; the raw constructors are private, so no public path seals caller-supplied source values. Every consumption path (`machine_ids`, `node_class_binding` — the sealed spec+capacity bridge for cluster construction, itself tied to its sealed tuple and not assemblable from loose values (R3-2), `node_class_spec(s)`, `energy_coefficients`, `stratum_for_bundle`) re-verifies the seal, so a field-swapped copy fails at the point of use (the seal is mistake-proofing, not cryptography — the docstring says so). Two curve families: `piecewise-joint` (published points, development) and `parametric-fan` (Fan et al. 2007, exponent fitted deterministically **within the physically monotone interval r ∈ [1, 2]** — R1-2; grid validated nondecreasing and within [idle, max]; fitted r + max residual recorded; fixed at the published r=1.4 when <4 points). `manifest()`/`manifest_sha256()` carry an explicit **schema tag** and cover every simulation-relevant number **including the Phase-2 defaults `image_pull_mib_per_s` and `include_image_pull_bytes`, the capacity-anchor payloads, and the structured source records**; canonical serialization refuses NaN (`allow_nan=False`) and every bundle field is validated (R2-6/R2-2/R2-8). `validate_bundle` re-derives every tuple from its cited entry and checks anchors + source records against the registry; fail-closed `validate_split` loads sealed role IDs/manifests/complete rosters, independently validates supplied bundles against the source registry, then checks disjoint machines, curve families, and allocation entries (R5-1/R6/R7-1). |
| `fitness.py` | 3.4 | The V1/V2/W2/W3 fitness aggregation (see below) — pure arithmetic; never runs the simulator. `InstanceOutcome` consumes a **sealed `VerifiedInstanceEnergy` token** (R2-1): identity, energy, I, and M all come from `BoundaryCertificate.verify_covers`, never as caller-typed bare floats. `evaluate_candidate` re-verifies the token and requires an exact P2/campaign certificate-seal registry over `(scenario, seed, bundle)`, so a valid token from a foreign coincident-I/M certificate is not scoreable (R4-1). Specs and outcomes **snapshot their mappings and validate physical domains and non-empty string component names** (R3-4/R4-7): satisfactions/goodput/rates are fractions in [0, 1], disruption components ≥ 0, caps/targets finite (a NaN cap raises rather than softening to a zero shortfall), and post-construction mutation of caller dicts cannot reach validated state. Requires the frozen `instance_keys` grid, rejects partial/extra grids and duplicate proposal sequence numbers (R1-7). |
| `gsa.py` | 3.5 | Stratified dependency-preserving GSA: within-stratum Morris + Sobol (SALib) sampled in **unit-quantile space** through each nuisance's registered inverse CDF (`uniform`/`loguniform` — R1-9); between-stratum categorical swap analysis (common quantile draws, per-stratum ppf); every nuisance carries stripped, meaningful string V6 basis/independence statements. Strata are **bound to sealed bundles** (R2-3), and `validate_strata_bound` requires exact equality with the independently loaded sealed design—full stratum payloads, registered development IDs/manifests, and recomputed `design_sha256`—so builder substitution and caller extras cannot authorize a design (R7-1/R7-2). Individual results contain immutable statistics but no authority field. Registered runners return an `AnalysisRun` whose `energy-analysis-envelope-v3` binds the complete result manifest and content-bearing model/producer/estimator identities; `verify_analysis_envelope` and `load_analysis_archive` report internal consistency only. Dirty-tree runs are permanently labeled `development_only`. The separate `analysis_postflight_receipt` boundary requires a receipt-candidate archive plus its registration, every registered executable source (primary model and nested Shapley samplers), every referenced PlacementScript policy/bundle artifact, and the `sim/energy` source closure to be exact blobs in one clean commit; `confirmatory_status` re-walks that same complete enumeration and grants eligibility only while every archive/commit/closure/content binding still matches. Every model call is scalar/finite checked, including Shapley conditional sampling (R5-3/R7-4). Shared runtime validators guard estimators and compute-cost paths (R2-9), whose `shapley_eval_count`/`as_g0_line_item()` report actual calls (R1-10). |
| `holdout_gate.py` | 3.3 | Run-once held-out evaluation driver: refuses unless HEAD is a tagged commit matching the P2 registration, the on-disk registration is **byte-identical to its blob committed at HEAD**, **the entire repository working tree matches the pinned commit** (R2-4 dependency closure — no path allowlist; a dirty `evolution/` module refuses the gate), the driver + full-manifest bundle-list hashes match, the registered entrypoint resolves **textually** (never via `importlib`, which would execute working-tree parent packages — R2-4) to a tracked/clean/hash-pinned repo file (R1-8), and `CONFIRM_HELD_OUT_GATE=1`; validates the entrypoint's `module:function` shape **before** the run-once marker can burn (R3-5). Before creating the `O_EXCL` marker it materializes the main repository and every out-of-repository editable (the ShinkaEvolve fork) as **detached worktrees at their registered commits**, frozen-syncs from the detached `uv.lock` with exactly the P2-registered dependency groups/extras, builds and installs a locked PlacementScript wheel, genuinely imports the native API plus the registered entrypoint dependency probes, and compares the actual interpreter/platform/toolchain/lock facts and canonical installed-distribution names+versions hash with the P2 pins (R5-4/R4-5/R4-6). Missing PPO/MILP dependencies therefore fail before the marker. The marker attests those facts plus lock/wheel/installed-native hashes. Live editable paths and the active development environment are never on the evaluation path. Held-out results serialize with `allow_nan=False` (R3-5). The module docstring states the honest enforcement claim (procedural control, not a technical barrier). |

Tests: `tests/test_energy_sources.py`, `test_energy_bundles.py`, `test_energy_fitness.py`,
`test_energy_gsa.py`, `test_energy_holdout_gate.py`, plus the certificate tests in
`test_sim_energy.py`.

## Source pinning and the data-integrity rule (3.1)

Every numeric entry carries provenance (source name, URL/DOI, access date, the exact
field transcribed, verification method, retained-artifact hash where one exists).
Current inventory — all entries **VERIFIED** (fetched from the cited public source;
SPECpower parsed by machine, the rest independently re-extracted from the fetched
documents in a second pass; records corrected 2026-07-18 where the original
verification description was inaccurate — see `PHASE3_REVIEW.md` R1-4):

- **SPECpower_ssj2008 (6 machines, all Single Node reports):** development —
  Lenovo ThinkSystem SR665 V3 (2023), Supermicro AS-1115SV-WTNRT (2023), Fujitsu
  PRIMERGY RX100 S7 (2011); held-out — ASUS ESC4000-E11 (2023), H3C UniServer R6900 G6
  (2023), IBM System x3550 M3 (2011). Ten joint (target load, ssj_ops, power) rows +
  active idle per machine; the curve coordinate is `ssj_ops/peak_ssj_ops`.
- **ARM SBC (3 boards, peer-reviewed):** Raspberry Pi 3B and Jetson TX-1 (Cloutier,
  Paradis & Weaver, *Electronics* 2016, DOI 10.3390/electronics5040061 — idle + HPL
  load power **and the published HPL GFLOPS points (3.7 stock-cooling / 16.0)**, used
  for HPL-domain capacity anchoring); Raspberry Pi B (Kaup et al., IEEE LCN 2014, DOI
  10.1109/LCN.2014.6925777 — published fitted model, no throughput point → core-count
  fallback; not in any default bundle).
- **Transitions (2 entries, Orgerie lineage):** Dell R720 "Taurus" (Benoit et al.,
  IJHPCA 2018, DOI 10.1177/1094342017714530) and Dell R630 "Paravance" (Raïs et al.,
  ICA3PP 2016). These are ACPI **S5 off/on** measurements used as the suspend/wake
  table — the lineage itself reports S3 unsupported on these servers (Raïs et al.,
  CCPE 2018); the `derivation` field states this and the pulse semantics.
- **Allocation coefficients (4 entries):** CCF AWS + GCP per-vCPU min/max watts
  (point-estimate allocation models; the "swept range" is the published min/max pair;
  raw methodology page retained at `data/artifacts/` with its sha256, **re-hashed and
  verified at every registry load** — R2-2); CCF network 0.001 kWh/GB and Aslan et
  al. 0.06 kWh/GB (DOI 10.1111/jiec.12630, verified from the Semantic Scholar Graph
  API abstract with the corrected `?fields=abstract` request) — point estimates with
  sweeps spanning **published** alternative estimates (CCF Appendix IV). **No
  confidence intervals are invented anywhere** (§4.1).

**Boavizta (documented scope deviation, R1-4/R2-7):** the plan's source menu (§4.1)
names "Cloud Carbon Footprint / Teads, Boavizta" for allocation coefficients.
Boavizta publishes an **instance-specific consumption-profile impact model** (host
power modeled nonlinearly from CPU profiles, allocated by vCPU *and memory* ratios),
*not* a scalar per-vCPU-hour coefficient and not the linear min/max-utilization
interpolation this package's `CloudAllocationSource` schema models (CCF's usage) —
pinning a scalar would model Boavizta through a functional form it does not publish.
An interim scalar entry derived by dividing BoaviztAPI's published m5.xlarge
avg-power figures per-vCPU was **removed** in Round 2 (R2-7: the division is
transparent arithmetic, but consuming it through the linear 50%-midpoint rule is not
Boavizta's model); the raw API response remains retained at
`data/artifacts/boaviztapi-aws-m5xlarge-2026-07-18.json` (sha256 recorded in
`pinned_sources.json` → `boavizta_scope_deviation`) as evidence for the explicit P2
bundle-list decision on whether a Boavizta consumption-profile family — a separate
structural family with its own model code — is added. The pinned coefficient
families are therefore CCF/Teads-lineage + Aslan.

Known compositional facts, stated rather than hidden: transition measurements come
from different physical machines than the power-curve machines (no public source
publishes both; the joint-tuple discipline governs the (power curve, capacity) pair);
dev and held-out ARM boards share one source paper (different machines/CPU families);
dev and held-out cloud coefficients share the CCF page lineage (different published
coefficient sets). Plan §4.1 explicitly frames the held-out set as *robustness across
specified model families*, never independent validation.

**Network path-scope assumption (stated, R1-5):** the cited network coefficients are
scoped by their sources to inter-data-center (CCF) or core+fixed-line-access (Aslan)
paths; the simulator applies one selected coefficient to **all** cross-node
transfers, including edge/fog LAN hops. This is a synthetic all-transfer allocation
assumption, stated on every network-allocated `ProvenanceRecord`; the coefficient
entry is a structural factor (bundles swap it) and its value is a GSA nuisance.

## Fitness aggregation (3.4) — the Phase 5 `evaluate.py` contract

Phase 5 verifies every episode's report through its instance certificate, wraps the
resulting sealed tokens into `InstanceOutcome` records (R2-1: no bare energy floats
anywhere), and calls `evaluate_candidate`:

```python
from sim.energy import certify_boundary
from sim.energy.fitness import ConstraintSpec, InstanceOutcome, evaluate_candidate

spec = ConstraintSpec(                      # frozen in P2
    slo_targets={"app": 0.95}, goodput_target=0.9,
    drop_cap=0.02, timeout_cap=0.02,
    disruption_caps={"moves": 20, "bytes_mb": 500, "unavailability_s": 0.0},
    zero_cap_scales={"unavailability_s": 60.0},   # V2 zero-cap scaling constants
)
outcomes = []
for (s, k, b) in scenario_seed_bundle_grid:      # EVERY instance x EVERY dev bundle
    cert = certify_boundary(cluster, coeffs, horizon, weights, bounds,
                            scenario_id=s, seed=k, bundle_id=b)  # I > 0 enforced; sealed
    token = cert.verify_covers(episode.energy)   # sealed VerifiedInstanceEnergy:
    outcomes.append(InstanceOutcome(             # identity + E + I + M, all verified
        verified=token,
        slo_achieved=...,                    # EpisodeResult.slo_satisfaction per app
        goodput=...,                         # EpisodeResult.goodput
        drop_rate=..., timeout_rate=...,     # from offered_outcomes
        disruption={...},                    # EpisodeResult.rebalancer counters
    ))
fitness = evaluate_candidate(spec, outcomes,
                             development_bundle_ids=bundle_ids,
                             instance_keys=frozen_grid,   # P2-frozen scenario x seed grid
                             expected_certificate_seals=p2_certificate_seals,
                             proposal_seq=seq)   # canonical creation sequence (W3)
key = fitness.sort_key()                          # total order; lower is better
```

Order semantics (verbatim from plan §4.1/§6.3):

1. **Feasible** ⇔ every C20 constraint satisfied in **every** development bundle on
   **every** frozen instance (the evaluator rejects grids that miss a bundle or an
   instance, cite instances outside the grid, or duplicate an outcome — R1-7).
2. Infeasible < feasible, always. Infeasible candidates: lexicographic on the
   4-vector (SLO, goodput, drop/timeout, disruption) built by *max across bundles →
   max within category → mean across scenario×seed instances*; ties by the literal
   **total sum** of primitive shortfalls (before within-category reduction — W3),
   then `proposal_seq` (duplicates rejected at ranking).
3. Feasible fitness = mean over development scenario×bundle instances of
   `energy_j / idle_floor_j` (V1 policy-independent normalizer).
4. Regret/penalty (§6.3 1b): `feasible_regret(E, ref, I)` and
   `penalty_regret(M, ref, I) = (M − ref)/I + 1` are the bare arithmetic kernels.
   The **common-boundary identity** behind the dominance algebra is enforced by
   `BoundaryCertificate` (R1-6/R2-1): I and M are computed together by
   `certify_boundary` (sealed — hand-built certificates are rejected), every
   episode's report must pass `verify_covers` (same cluster digest, same horizon,
   same coefficients, transition counts and bytes within the certified bounds,
   E ≤ M) which returns a sealed `VerifiedInstanceEnergy` token, and the
   certificate's `feasible_regret(token, ref_token)`/`penalty_regret(ref_token)`
   accept only tokens of the same certificate — a reference verified under another
   boundary is refused. The dominance property is asserted in tests
   (`penalty_dominates`, including E = M).

Zero-cap scaling constants, boundary weights, per-instance bounds, targets/caps, the
reference-policy rule, the instance grid, and the bundle lists are all **P2-frozen
inputs** — this module validates and consumes them; it never chooses them.

## Stratified GSA (3.5)

`Stratum` = one machine-tuple level (categorical factor) + its nuisances' conditional
distributions (each with a `conditional_basis` string naming the published spread,
a separately required `within_stratum_independence` attestation (V6), and a
registered `distribution` family). Both V6 prose fields must be strings, are
stripped before sealing, and require at least eight non-whitespace characters;
truthy scalars and placeholders fail at construction. Sampling is in unit-quantile
space through each nuisance's `ppf` (identical to direct bounds sampling for
uniform; correct for non-uniform registrations — R1-9). Morris/Sobol run within strata only;
`between_stratum_swap` reports tuple swaps separately under common quantile draws
(no stratum is ever evaluated outside its own spread); `stratum_for_bundle` builds
**sealed** strata from sealed bundles, every sampling entry point re-verifies the
seal, and `validate_strata_bound` (always run by the production entry points because
`bundles=` is mandatory) loads and hash-verifies
`data/energy_analysis_registration.json`. Supplied bundles are evidence checked
against the canonical source registry plus the registered IDs/manifests, never the
authority defining completeness; every supplied stratum payload and the recomputed
full `design_sha256` must equal the sealed design. Invented bounds, extra nuisances,
and caller-selected subsets are rejected before sampling (R2-3/R6-1/R7-1/R7-2).
Explicitly named `nonconfirmatory_*` harnesses are isolated from the registered
path. Standalone one-stratum Morris/Sobol calls also carry no authority. Individual
`MorrisResult`, `SobolResult`, `SwapResult`, and `ShapleyResult` objects deliberately
hold only immutable scientific values: there is no `_issuer`, object capability,
seal, or per-result authority flag to copy or replay.

### Two-layer analysis contract

Layer 1 is internal consistency and content identity. A registered runner returns
one `AnalysisRun` containing `results` and an `energy-analysis-envelope-v3`.
The envelope records the registered `design_sha256`, split-registration payload
hash, every development bundle ID + manifest hash, explicit seeds and estimator
parameters, and the canonical ordered hash of every required result. For stratified
GSA the set is exactly Morris + Sobol for every registered stratum; swap and
registered Shapley have their own exact run-level sets. Producer and estimator
identities carry the source-closure hash over every `sim/energy/**/*.py` file plus
the canonical enumeration of the primary model and every nested executable
registration (currently both registered Shapley samplers), including each
registration's repository content bindings and callable span in its pinned source.

Every production-path model must implement `analysis_model_descriptor()` and return
exactly:

```python
{
    "kind": "python-callable",            # or "placementscript"
    "parameters": {...},                  # canonical constructor/closure state
    "policy_path": None,                  # repository path for placementscript
    "policy_sha256": None,                # required SHA-256 for placementscript
    "bundle_path": None,                  # repository path for placementscript
    "bundle_sha256": None,                # required SHA-256 for placementscript
}
```

The implementation independently adds the callable's module/qualname, inspected
source hash, repository-relative source path, and whole source-file hash, then seals
that descriptor as `registration_sha256`. PlacementScript adapters must provide
repository-relative policy/bundle paths and their content hashes; Python callables
must expose every semantic constructor/closure parameter in `parameters`. Registered
Shapley samplers use the same descriptor contract. Missing, noncanonical,
noninspectable, or out-of-repository model code is rejected before production
sampling. Consequently two instances of the same callable class with opposite
registered `sign` state have different model, source-closure, and envelope hashes.

`verify_analysis_envelope(envelope, results)`, `AnalysisRun.envelope_verified`, and
`load_analysis_archive` prove only this layer. They recompute public hashes, reload
the design registration, enforce exact result completeness/types, and cross-check
seeds, parameters, nuisance coverage, and evaluation counts. `archive_analysis`
writes one canonical, non-overwriting `energy-analysis-archive-v3` whose own
`payload_sha256` detects corruption. Genuine bytes re-verify in a fresh process;
partial, foreign, or tampered bytes fail. A caller can still hand-assemble arbitrary
finite statistics and recompute every public hash. Such an archive may be internally
consistent, but no Layer-1 API calls it confirmatory or eligible. A dirty-tree run
is labeled `development_only` in its envelope, `AnalysisRun`, archive, and reload,
and that label can never receive a receipt.

Layer 2 is production eligibility. `analysis_postflight_receipt(archive, receipt)`
fails closed unless the archive was created as `receipt_pending` and the repository
is clean at one full commit. The exact archive bytes, the exact
`data/energy_analysis_registration.json`, every executable registration's source
file (primary model plus nested Shapley marginal/conditional samplers), and each
PlacementScript registration's policy/bundle files must all match their recorded
hashes and be committed blobs at that commit. The current `sim/energy` + complete
executable-registration source closure must equal the envelope. It then re-loads
and re-verifies the archive from its canonical bytes and exclusively writes a
durable `energy-analysis-postflight-receipt-v2` containing:

- archive repository path and archive `payload_sha256`;
- envelope `payload_sha256`, `design_sha256`, and registration payload hash;
- repository commit, combined source-closure hash, and model-registration hash;
- UTC issue time, `eligible` status, and a canonical `receipt_sha256`.

`confirmatory_status(archive, receipt)` returns true only when those bindings still
match the current repository commit, committed blobs, registration, every enumerated
executable/artifact file, and source closure. A missing receipt, `development_only`
archive, changed commit, closure mismatch, copied/uncommitted archive, stale sampler,
false PlacementScript artifact hash, or altered receipt returns false.

This is the same honest-operator boundary used by the campaign postflight and
held-out gates, not cryptographic proof against a malicious committer. Public hashes
do not prove that code ran. A hand-built archive can obtain eligibility only if an
operator deliberately commits the forged archive bytes (and any forged
code/registration) to the repository before receipt issuance. That action is outside
the honest-operator model and leaves durable reviewable evidence in repository
history; no secrecy or inaccessible in-process capability is claimed.

Non-finite or nonscalar model outputs fail before analysis/result construction,
including every conditional Shapley call (R5-3/R7-4);
`strata_design_sha256` is the P2 pin over the full design payload.
`exact_shapley_effects` is the named
dependent-input path (cited: Song, Nelson & Staum, SIAM/ASA JUQ 2016), tested against
analytic values on independent and correlated Gaussian models. All entry points
require explicit seeds (`analysis.tracking.derive_seed` streams; omission is a
`TypeError`).

`estimate_compute_cost(...)` returns exact evaluation counts —
`shapley_eval_count` matches the estimator's true model-call count (cache-aware,
tested by instrumentation — R1-10) — and `as_g0_line_item()` emits the markdown row
for the G0 capacity table, including the 30%-reserve figure (plan §8).
`seconds_per_eval` must be a measured pilot number, and the cost calculator shares
the estimators' runtime input validators (R2-9): every design it prices is a design
the estimators will actually execute.

## P2 pre-registration touchpoints (drafted here, frozen in P2 — plan §6.3/W2)

The P2 registration must pin, using this package's artifacts:

1. **Bundle lists** — final development + held-out sets (materialized with role
   rosters and manifest hashes in `data/energy_analysis_registration.json`), with
   inclusion/exclusion rules:
   *inclusion*: complete published joint table, Single Node report, VERIFIED
   provenance; *exclusion*: multi-node reports, provisional entries, machines whose
   idle ≥ any load-level power; plus the Boavizta-family decision (see above).
   `holdout_gate.bundle_list_sha256()` (full manifests) is the hash to pin.
2. **Capacity mapping** — the per-domain anchors (`default_capacity_mapping`:
   SPECpower + HPL; registry-validated via `validate_mapping`, payloads via
   `CapacityMapping.payload()` — R2-8) and the fallback constant.
3. **Constraint spec** — targets/caps + the V2 zero-cap scaling constants.
4. **Instance grid** — the frozen scenario × seed `instance_keys` every candidate is
   scored on (R1-7).
5. **Boundary certificates** — `BoundaryWeights` + per-instance `BoundaryBounds`
   (transition and byte bounds with their derivations), certified per instance by
   `certify_boundary`; I > 0 verification is the factory's positivity check,
   recorded over every registered instance (W2).
6. **GSA design** — the complete bundle-bound strata in
   `data/energy_analysis_registration.json`, pinned as `design_sha256` over schema,
   canonical development bundle IDs/manifests, every nuisance bound/basis/family/V6
   attestation, and every stratum payload; trajectory/sample counts and the
   compute-cost line item remain P2 analysis inputs.
7. **Held-out gate registration** — `prereg/P2_HELD_OUT_BUNDLES.json` with
   `pinned_commit`, `driver_sha256`, `bundle_list_sha256`, `entrypoint`,
   `entrypoint_sha256`, `dependency_groups`, `dependency_extras`,
   `entrypoint_dependency_imports`, the installed-distribution names+versions hash
   in `execution_environment`, and
   `one_shot_assumption: honest-operator-local-marker`,
   committed at the tagged commit. The driver refuses to run without it; the local
   `O_EXCL` marker is an accidental-rerun control, not append-only storage (the
   module docstring carries the full honest enforcement claim).

## Seeds and determinism

No stochastic entry point has a default seed (guard-tested). The only
`datetime.now` usages are tz-aware operational timestamps in the two build/gate
tools (`fetch_specpower.py`, `holdout_gate.py`) — recorded, never results-feeding
(docs/SEEDS.md non-goal), and exempted by name in the grep guard.

## Known limitations (stated)

- The T4 mapping limitations (`capacity.LIMITATIONS`) — benchmark-domain transfer,
  load-coordinate identification, cross-domain anchor commensurability, core-count
  fallback — go into threats-to-validity.
- ARM boards have idle + one load point only; their piecewise curve is a two-point
  interpolation and their parametric fit uses the published Fan et al. exponent.
- Parametric fits are constrained to the monotone r ∈ [1, 2] family; machines whose
  published curves are strongly concave (the 4-socket held-out machines) show large
  recorded residuals — that is the point of a *structurally different* held-out
  family, and the residuals are part of the artifact.
- The held-out "suspend" table models ACPI S5 off/on (S3 unsupported per the cited
  lineage); classes without published transition data are non-suspendable (§3.4).
- The network path-scope assumption above (R1-5).
- Boavizta scope deviation above (R1-4/R2-7).
- SPECpower result pages are machine-parsed and their fetch-time `content_sha256`
  values recorded, but the raw HTML pages themselves are not retained in-repo; those
  hashes are attestations of the pinning fetch, re-verifiable only against spec.org.
  Entries that *do* retain an artifact (`artifact_path` set: the CCF methodology
  page) are re-hashed and verified at every registry load (R2-2).
