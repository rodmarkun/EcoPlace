# Phase 4 — workloads and scenarios

This package implements the v2.5 Phase-4 contracts without turning missing evidence
into synthetic “measurements.” The completed adversarial implementation audit is recorded
in [`../PHASE4_REVIEW.md`](../PHASE4_REVIEW.md). Its boundaries are:

- `sources.py` loads only the exact hash-pinned production calibration/face-validity
  split; caller-supplied registry paths are not a production API. Alibaba v2021
  and DSB Hotel Reservation publications are calibration sources; Alibaba v2022 and
  TeaStore publications are face-validity only. A source whose evidence arrives as
  several distinct parser inputs registers one **payload component** per parser input,
  each with its own status and exact digest: a single `payload_sha256` cannot
  authenticate two parser inputs, so component-aware sources leave that field null and
  `require_component` binds every consumer to the exact component it parsed, which also
  stops one component's bytes being accepted in another's role. Alibaba v2021 is
  registered `AVAILABLE` with its `metrics` and `call_graph` chunk-0 components
  (calibration corpus v1, adopted 2026-08-09 — see the source notes). All other raw
  traces and extracted published curve points remain explicitly `UNAVAILABLE`. The
  held-out G3 access receipt still binds one digest per source, so a component-aware
  *face-validity* source fails closed until that receipt schema is extended.
- `alibaba.py` has version-specific exact-byte CSV parsers that compute payload hashes
  and issue sealed corpora. v2021 signed RPC/HTTP halves collapse into one logical call
  only when their identities agree. Each RT cell is joined to the unique same-instance,
  same-time, same-family MCR cell: MCR=0 marks inactivity and excludes the RT cell;
  an active v2021 `rt=0` remains left-censored below 1 ms, while an active zero without
  a source-documented censoring rule fails closed. Call graphs retain their documented
  0.5% sampling status. v2022 cannot produce
  absolute RPS. Root MCR is summed across selected instances by timestamp, preserves
  explicit zero-load cells, rejects missing 60-second source cells, is normalized by
  the positive-bin mean, converted from trace ms to relative seconds, and ends with a
  sealed terminal-zero support boundary before becoming an `ArrivalSpec`. Fitted Alibaba RT is
  explicitly the distribution of **service-instance × 60-second interval-mean**
  response times, not per-request latency or CPU service time. G3 requires simulator
  service-call records to be aggregated to that identical estimand. The P1 selector
  also freezes linear type-7 quantiles and the v2021 left-censoring rule: a zero marked
  below 1 ms is retained as 0.5 ms (half the detection limit), never discarded.
- `alibaba_stream.py` runs the same parsers over a **complete** archive member without
  materialising it. Byte truncation is not an option — the dump is not time ordered, so
  a head slice silently severs RT-MCR pairs — so memory is bounded by grouping on disk
  instead: every byte is streamed once and hashed, accepted rows fan out to shard files
  keyed by service (metrics) or trace id (call graph), and each shard is reduced alone.
  Both grouping keys are closed under their shard key, so no pair or call half is lost.
  Row acceptance, typing, rejection taxonomy, fit sealing, and rate-shape sealing are
  imported from `alibaba.py` rather than re-implemented, and the tests assert that the
  two paths produce identical seals on identical bytes. A sampled corpus produced this
  way cites the sealed complete-member scan (`parent_scan_seal`) instead of a parent
  corpus object that never existed.
- `topology.py` releases hierarchical and BA generators plus a hash-checking GraphML
  transform. Topology-Zoo-derived artifacts carry the exact label
  **“empirically grounded network-structure generalization”**, state that tiers,
  capacities, and link latencies are synthetic, and carry no placement
  external-validity claim. BA is exactly **“synthetic stress test.”** Confirmatory
  sizes require a production-pinned literal G0 capacity table. Status, reserve, and
  confirmatory cap are recomputed from its numerical ceiling/trial rows; the production
  loader currently fails closed because G0 has not run. Synthetic table parsing is a
  visibly private, test-only API.
- `drift.py` generates diurnal changes, conservatively certified **pre-scaled**
  HPA-feasible flash crowds (blocking/backend/retry/network residence and packing of
  **every** initially active replica included; peak demand must fit those initial replicas
  because stochastic cold starts have no finite readiness bound), seed-bearing planned
  cordon/uncordon schedules, frozen synthetic
  category-prior onboarding trees, and mix shifts including their declared start.
  Diurnal schedules include both declared endpoints; onboarding rejects any template
  whose arrivals begin before its application-wide activation.
  Onboarding is application/template-level: every service in the generated call graph
  activates together, so no child pods or image pulls leak in early. The runtime
  models planned capacity churn—not crash failure or eviction.
- `artifacts.py` requires an exact workload-model closure before sealing. Fully
  synthetic workloads may cite sealed calibration evidence, but the label explicitly
  says **evidence-citing synthetic assumption**: citation does not prove that the
  `WorkloadSpec` was fitted or calibrated. `alibaba_workload.py` provides the typed,
  seeded builder required by **calibration-informed-synthetic-workload**: it revalidates
  the exact retained calibration and call-sample bytes plus every nested fit/scan seal,
  then binds the derived `WorkloadSpec`, transform assumptions, and selected fits in the
  closure. The empirical label remains fail-closed because no retained evidence supports
  that stronger claim. Evidence-citing closures enumerate sealed
  source transforms, selection, sampling seeds/counts, configuration, and code versions.
  The package then seals the topology/workload/drift/source closure and
  materializes exact arrivals for an episode seed. The finite structural network-byte
  bound uses exact offered counts, worst-case bounded retries, image pulls, and the
  registered migration-byte cap; startup pulls are bounded per possible pod start, so
  concurrent cache misses remain below the ceiling. Public, private, and pilot roles
  are orthogonal to topology evidence roles. Split descriptors are derived from
  verified artifacts and compare generator-component, onboarding/churn, and episode
  seed commitments plus recipe hashes. Artifacts have canonical JSON save/load; loaders
  recompute config, workload, arrival-count, byte-bound, and artifact hashes, and a
  fresh-process replay test locks the contract. Normal manifests consume private
  **commitments**, not private artifact objects. Public loaders and the ordinary runner
  reject private payloads. Confirmation requires the production-pinned external P2
  receipt, an exact matching **clean** Git checkout (including no untracked files), a
  gate-issued timestamp, and an authority-bearing opened-private wrapper. Split
  manifests, P2 protocols, access receipts, and already-opened private wrappers have
  canonical contextual loaders, including fresh-process revalidation.
- `g3.py` freezes outcome-independent selectors and typed prediction artifacts through
  a production-pinned external P1 receipt before loading held-out bytes. Prediction
  APIs accept only authority-sealed `Simulation.run()` results whose exact configs and
  workload/source closures match. Runs are issued by one module-owned, versioned
  least-allocated reference scorer with no proposer and round-robin load balancing;
  the execution commit, runner code, policy contract, and episode seeds are bound into
  each prediction. Published-curve y values are derived as linear-type-7
  p95 completed-root latency in milliseconds, while x is derived (not caller supplied)
  as the stationary configured total root rate in requests/s; Alibaba predictions come
  from retained per-service-instance call timings aggregated into complete 60-second
  cells after excluding warm-up and drain windows. Curve cohorts likewise require every
  offered request in the frozen measurement window to finish successfully; right-censored,
  dropped, or timed-out cohorts cannot become a prediction.
  Curve points require at least 30 completed requests each. RT shapes require at least
  20 active instance-cells on both sides and currently map only abstract non-root child
  calls to Alibaba `PROVIDERRPC_RT`; other communication families fail closed until the
  simulator carries a typed protocol-kind field.
  Prediction generation itself requires a clean checkout and records its exact commit;
  P1 refuses predictions from any commit other than the externally registered one. Caller
  outcome arrays, claimed result hashes, and caller-supplied generator code are not an
  API. Published curves come from canonical extraction manifests whose selector fixes
  units, configuration, and the **all visible points** rule; the runtime curve evaluator
  currently fails closed unless the observed units are requests/s and milliseconds. P1 access
  also requires the exact clean frozen checkout. Face-validity evaluation requires the
  gate-issued receipt; results enforce every series coordinate/count, so dropping one
  point cannot pass. Protocols, predictions, receipts, residuals, and result digests
  round-trip via canonical contextual loaders. Evaluated result reload additionally
  requires the exact source-evaluator-issued residuals and never grants residual authority
  to self-hashed JSON. Missing evidence or registrations return/fail
  `UNAVAILABLE`, never `PASS`.
  A G3 PASS supports face plausibility only for the exact P1-frozen, evidence-citing
  synthetic model. It does not prove that model was empirically calibrated. G3 never
  validates energy and is separate from DES verification.

## Evidence debt

The repository now retains, and registers, the complete published Alibaba v2021 chunk-0
metric and call-graph members — calibration corpus v1, adopted 2026-08-09 — so Alibaba
RT fitting, root-rate shapes, and trace-atomic call-graph sampling have been executed
(`experiments/alibaba_ingestion_20260809/`). Adequacy is bounded to that chunk: nothing
here establishes representativeness of all services, clusters, or chunks, and the
call-graph stream is itself an upstream 0.5% sample.

Still outstanding: every other Alibaba chunk, all of v2022 (face-validity, deliberately
unfetched until G3-cal), and exact publication/replication curve artifacts with
page/figure/series/configuration provenance. The acceptance G3 run therefore remains
blocked on the published-curve extraction. README example RPS values, workload scripts,
and Kubernetes manifests are not observed latency-load curves and are rejected as such.

A fitted model is still not an empirical scenario. The typed Alibaba builder can now
seal a **calibration-informed-synthetic-workload** for quarantined development use, with
the retained interval-mean RT fits transformed through an explicit synthetic service-time
attribution. `seal_empirical_workload` remains fail-closed, and the development builder
does not broaden chunk-0 coverage or support an empirical/confirmatory label.

No code in this package inspects Train-Ticket.

The split and access machinery is mistake-proofing under the preregistration protocol's
honest-operator assumption, not a claim of cryptographic secrecy from a researcher with
arbitrary filesystem access.

P1/P2 access markers must be written outside the checkout (or to a path already ignored
by Git). Receipt verification rechecks the clean frozen checkout during every protected
evaluation, so a marker that itself dirties the tree correctly makes the gate fail closed.
