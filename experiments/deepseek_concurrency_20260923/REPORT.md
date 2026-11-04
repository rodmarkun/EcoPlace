# Three-root concurrency pilot (G0-concurrency) — 23–24 September 2026

Resource measurement under the frozen v3 treatment (DeepSeek V4.1 Flash, low reasoning,
32,000 output tokens). Three LLM roots ran simultaneously on fresh reserved seeds: A0, B0, B1.
No GP arm, comparison, final grid or protected fixture. Each root had its own durable journal
and a $0.10 ceiling. The v3 freeze verified unchanged before and after the run.

## Outcome

All three roots reached their six-request ceiling with exit code 0; no memory, wall or
dispatch failure. Wall time 1,487 s (the slowest root, B1, 1,482 s).

| | A0 | B0 | B1 |
|---|---|---|---|
| Peak process family RSS | 509 MiB | 589 MiB | 606 MiB |
| Mean evaluation, rejected / L2 (single-root pilot) | 7.9 / 16.7 s (6.6 / 15.9) | 45.0 / 135.2 s (48.3 / 138.3) | 35.9 / 136.1 s (48.3 / 138.3) |
| Adaptive proposals valid / attempts | 3/3, 5 attempts | 3/4, 6 attempts | 3/3, 6 attempts |
| Request latency, mean (range) | 39 s (8–66) | 34 s (11–74) | 40 s (20–70) |
| Finish reasons | all `stop` | all `stop` | all `stop` |

Machine: 10 cores, 16 GB, with unrelated user applications running. Across 737 two-second
samples, combined root RSS peaked at 1,419 MiB, the minimum available memory was 2,602 MiB,
and CPU averaged 35% (max 79%). Evaluation times show no measurable contention at three roots.
Accounted API cost: $0.119318.

## Revised projection from all five pilot roots

Repairs were more frequent here than in the single-root pilots: 18 requests produced 10
adaptive candidates (9 valid). The dominant repair cause is a recurring DSL error, a bare
expression statement; others were a reserved name and int/float typing. Pooling all five
roots (A, A0; B, B0, B1):

| | Requests per adaptive candidate | Projected API per LLM root |
|---|---|---|
| A | 1.50 | $0.42 |
| B | 1.64 | $0.61 |

30+30 roots project **$30.79 API** and **$44.12 project total** with a 40% reserve and prior
exposure ($1.01 including this pilot). This still fits the $60 ceiling, with less margin
than the $34.56 in the v3 PILOT_SUMMARY. The v3 per-root ceiling of $1.50 is about 2.5× the B
projection.

With no measured slowdown, three concurrent roots divide the ≈142 serial search hours to
roughly 48 hours before finals and peak-window waits. Four roots are plausible by CPU and
memory headroom but were not measured.

## Limits

Five roots in total, at most five adaptive candidates each. Repair rates are noisy at this
size and may change later in search. Promotion and validity counts are operational
measurements, not evidence about proposal quality. The recurring bare-expression error could
be reduced by a prompt clarification, but that would change the frozen treatment and require
a recorded amendment.
