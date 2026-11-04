# All-baselines development comparison

- Status: **DEVELOPMENT_ALL_ARMS_COMPLETE**
- Payload SHA-256: `8c4ebf58cc7c51ed8ece36a9e92f9b6abcb6f404d6af1fccb86fc0a23083816f`
- Scenario: `dev-continuum-tight`
- Root seed: `800000801`
- Matched normalized-work budget: `56.0`
- Selection authority: exact Phase-3 `CandidateFitness.sort_key` (lower is better)

## Initialization/search decomposition

| Arm | Role | Init best | Absolute best | Search delta | Best found at | Work | Calls | Healthy |
|---|---|---:|---:|---:|---|---:|---:|---|
| classic_gphh | adaptive-search | 0.647034 | 0.576425 | 0.070609 | `{"candidate_id":"classic_gphh:70","coordinate":"generation","index":6,"proposal_seq":70}` | 55.400000 | 0 | True |
| energy_greedy | hand-written-evaluation-only-reference | n/a | n/a | n/a | `{"candidate_id":"energy_greedy","coordinate":"evaluation_only","index":null,"proposal_seq":10001}` | 0.000000 | 0 | True |
| first_fit | hand-written-evaluation-only-reference | 0.647034 | 0.647034 | 0.000000 | `{"candidate_id":"first_fit","coordinate":"evaluation_only","index":null,"proposal_seq":10002}` | 0.000000 | 0 | True |
| gphh_gen0 | random-population-baseline-derived-no-additional-evaluation | 0.647034 | 0.647034 | 0.000000 | `{"candidate_id":"classic_gphh:5","coordinate":"generation","index":0,"proposal_seq":5}` | 0.000000 | 0 | True |
| latency_greedy | hand-written-evaluation-only-reference | 0.834547 | 0.834547 | 0.000000 | `{"candidate_id":"latency_greedy","coordinate":"evaluation_only","index":null,"proposal_seq":10000}` | 0.000000 | 0 | True |
| least_allocated | hand-written-evaluation-only-reference | 1.260594 | 1.260594 | 0.000000 | `{"candidate_id":"least_allocated","coordinate":"evaluation_only","index":null,"proposal_seq":10003}` | 0.000000 | 0 | True |
| llm | adaptive-search-development-a1-subscription | 0.834547 | 0.638961 | 0.195587 | `{"candidate_id":"h00:llm:00000024","coordinate":"proposal","index":24,"proposal_seq":24}` | 55.600000 | 0 | True |
| most_allocated | hand-written-evaluation-only-reference | n/a | n/a | n/a | `{"candidate_id":"most_allocated","coordinate":"evaluation_only","index":null,"proposal_seq":10004}` | 0.000000 | 0 | True |
| network_overhead | hand-written-evaluation-only-reference | 0.834547 | 0.834547 | 0.000000 | `{"candidate_id":"network_overhead","coordinate":"evaluation_only","index":null,"proposal_seq":10005}` | 0.000000 | 0 | True |
| ppo_frozen | frozen-learned-policy-evaluation-only-reference | n/a | n/a | n/a | `{"candidate_id":"ppo_frozen","coordinate":"evaluation_only","index":null,"proposal_seq":20000}` | 0.000000 | 0 | True |
| random_ast | adaptive-search | 0.647034 | 0.647034 | 0.000000 | `{"candidate_id":"random_ast:2","coordinate":"proposal","index":2,"proposal_seq":2}` | 55.200000 | 0 | True |
| same_dsl_gp | adaptive-search | 0.647034 | 0.647034 | 0.000000 | `{"candidate_id":"h00:gp:00000002","coordinate":"proposal","index":2,"proposal_seq":2}` | 55.000000 | 0 | True |

`gphh_gen0` is a derived random-population baseline. It performs no extra evaluation and exposes the best member of classic GPHH generation 0.

## Distinct feasible-energy plateaus

- `classic_gphh`: 9 plateaus; levels `[0.576425277527292,0.5800112977152151,0.5852706621742972,0.647033890008899,0.6612445155756882,0.6683172960724671,0.6683551275578353,0.7207372097999398,0.8226491909175208]`
- `energy_greedy`: 0 plateaus; levels `[]`
- `first_fit`: 1 plateaus; levels `[0.647033890008899]`
- `gphh_gen0`: 4 plateaus; levels `[0.647033890008899,0.6612445155756882,0.7207372097999398,0.8226491909175208]`
- `latency_greedy`: 1 plateaus; levels `[0.8345471976973723]`
- `least_allocated`: 1 plateaus; levels `[1.2605938305274138]`
- `llm`: 7 plateaus; levels `[0.6389605560168095,0.6473729653271001,0.6643205345462095,0.8345471976973723,1.227381519459488,1.2622257302764324,1.269011964590937]`
- `most_allocated`: 0 plateaus; levels `[]`
- `network_overhead`: 1 plateaus; levels `[0.8345471976973723]`
- `ppo_frozen`: 0 plateaus; levels `[]`
- `random_ast`: 6 plateaus; levels `[0.647033890008899,0.6612445155756882,0.8345471976973723,1.2605938305274138,1.2622257302764324,1.269011964590937]`
- `same_dsl_gp`: 6 plateaus; levels `[0.647033890008899,0.6519927030046789,0.8345471976973723,1.2622257302764324,1.2679377481773797,1.269011964590937]`

## Interpretation boundary

Quarantined development comparison only; never acceptance, P1/P2, held-out, or confirmatory evidence.
 Frozen PPO and hand-written policies receive no adaptation budget. This report supports development ranking only and carries no statistical or confirmatory claim.
