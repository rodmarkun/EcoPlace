# Scoped study-A design freeze (v2)

Status: scientific design recorded; execution authorization blocked.

This version supersedes `CORRECTED_STUDIES_20260917_v1`. It registers **one** study — the
LLM-versus-GP proposal-operator comparison — over **twelve** paired roots at the unchanged
56-unit work ceiling. Study B is dropped. v1 remains frozen, verifiable and unmodified.

Read `PROTOCOL.md` for the decisions, `DEVIATION_FROM_V1.json` for exactly what changed
and what each change costs in claims, `ANALYSIS.json` for the single primary test and the
decision rule this sample size implies, `POWER.json` for what it can and cannot detect,
`MODEL_AND_RESOURCES.json` for the measured costs, and `GATES.json` for what still blocks
launch.

Why a new version rather than an amendment: the measured cost of a paired root puts v1's
60-root design at $198 against a $60 authorized ceiling, and the 2026-09-18 episode-ledger
retention remediation changed registered evaluation source. No root of either v1 study was
ever executed, so no outcome informed the reduction.

What this design **cannot** claim: anything about transfer to independently designed
conditions, or about whether search adds value beyond the discovered simple rule. That was
v1's study B and it is gone. It is also powered only for a large effect — 0.56 at a 0.8
per-root win probability — so a non-significant result is an underpowered non-rejection,
never an equivalence claim.

Verify from the repository root:

    .venv/bin/python -m experiments.prospective_20260919.build verify
    .venv/bin/python -m experiments.prospective_20260919.build check-launch

The second command must refuse launch. A changed scientific design needs a new version.
