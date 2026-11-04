# Capacity preparation; not G0 certification

The planned primary final grids contain at most8,640A and34,560B label-episodes
before exact source/grid deduplication. Nine fixed references, two champions and the
common initialization best give12labels per root. Deduplication cannot remove labels.

The completed independent calibration supplies324 measured pressure0.15 episodes,
mean11.5037617796worker seconds. Multiplying that mean by the B final label grid gives
110.44worker hours before deduplication. This is an extrapolation from fixed references,
not a measured evolved-policy or multi-scenario runtime bound. Search costs are additional.

For B, work weights imply a loose ceiling of56×max(4/0.3,12/1)=746.67simulated episodes
per arm when separately charging every L1/L2call (actual L0/ladder dependencies reduce it).
Across60B arms this is44,800episode-equivalents, about143.16hours at the calibration mean.
Neither value measures L0, model latency, orchestration, failure inflation or memory.

The combined B illustration alone is253.59hours, about362.27hours with30% unallocated
capacity. Do not represent this as a passing14-day wall-clock plan, nor as a reliable
worst-case estimate. A resource pilot and exact source-bound rung timings are required.
All real CPU, storage, wall and researcher-time allocations remain unapproved/unmeasured.

The proposed monetary envelope is$225:60LLM runs at$2.50 plus3pilot runs at$2.50,
$157.50allocated and$67.50reserve. This proposal is not current spending authorization.
A root's cost ceiling may interrupt search before56work units; no full-budget guarantee.
