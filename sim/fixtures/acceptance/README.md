# Acceptance fixtures — FROZEN AT P1, DO NOT RUN CASUALLY

These fixtures are the **acceptance set** for the DES verification gate (plan §3.4/S6,
TODO 2.5, T6 fixture discipline):

- They are **disjoint from the development set** (`../dev/`): structurally analogous
  scenarios with different parameters, and gate executions use root seeds that never
  appear in development runs.
- The verification suite runs continuously on the *development* fixtures during
  construction. Running the suite against **this** directory constitutes a *gate
  execution* and is only meaningful **after P1 freezes the acceptance criteria**
  (numeric tolerances, pass/fail rules) — scheduled Aug 24, gate execution window
  Aug 25 – Sep 5, 2026.
- Until P1 freezes, the files here are **drafts of the acceptance set**; P1 pins their
  hashes together with the acceptance criteria and the gate root seeds. After P1, any
  change to this directory invalidates the gate and requires a new registration version.
- Do not point tests, benchmarks, notebooks, or ad-hoc scripts at these files. The
  one-shot gate run is executed by a dedicated driver, once. The fixed
  `prereg/P1_DES_GATE.json` registration must pin the criteria-file path/hash, driver
  hash, manifest hash, source commit, executed-source-closure hash (local Python plus
  the loaded PlacementScript artifact), and the canonical execution-environment hash
  (Python implementation/version, OS, architecture, and `uv.lock` hash). It must also
  register `one_shot_assumption: honest-operator-local-marker`. The driver refuses a
  dirty or untracked execution tree, atomically creates read-only
  `GATE_ATTEMPT.json` before loading any scenario, and then creates
  `GATE_RESULT.json`. `O_EXCL` prevents accidental overwrite/retry; because the parent
  directory is writable, this is an honest-operator procedural control rather than an
  append-only or adversary-resistant record. External append-only registration/storage
  is required if that stronger trust boundary is desired.

Regeneration (before P1 only): `uv run python sim/fixtures/make_fixtures.py`.
