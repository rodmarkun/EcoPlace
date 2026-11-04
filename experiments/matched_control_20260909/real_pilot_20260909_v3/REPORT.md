# Real initialization and interruption witness — complete

Twelve distinct archived policies were evaluated by the real simulator and archive
assessment in each arm. GP ran continuously. The LLM-labeled arm exited before evaluation
of candidate 5 after durable creation, then resumed. Exact evaluator inputs, completed
initialization records and the paired control receipt agree; each arm made exactly twelve
evaluations. A forbidden proposal operator enforces **zero proposal/API calls**.

The pilot root is 9113512321037399042 and evaluation seed 3246458936443790985;
the seed was checked against 181 historical evaluation seeds. These are pilot-only data.

| Worker | Seconds | Peak RSS MiB | Exit |
|---|---:|---:|---|
| GP initialization | 195.37 | 536.14 | 10: initialization barrier |
| LLM initialization, interrupted | 70.53 | 508.36 | 17: deliberate interruption |
| LLM initialization, resumed | 127.94 | 527.66 | 10: initialization barrier |

Each worker had a 2 GiB RSS and 1,200-second wall limit. All **209 source/native
archive members** independently match `snapshot.json`; result and snapshot seals verify.
The result binds the pair receipt and source snapshot hashes. This establishes real
initialization/replay behavior for this bounded fixture, not adaptive-search correctness,
provider billing, scientific gate acceptance, or a comparative search result.

The two preceding setup attempts remain preserved with failure records and source archives.
Both failed before evaluation: JSON tuple/list representation comparison, then the
ten-campaign-definition invariant. Version 3 canonicalized the representation and retained
all ten definitions while executing only the specified pilot. Neither guard was weakened.
