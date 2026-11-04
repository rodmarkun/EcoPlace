"""Registered analysis of Study F: gpt-6.1-sol (via Codex) as the LLM engine on Study E's 90 roots,
with a validation step selecting each arm's final program from its search top-k.

Confirmatory endpoints, fixed sequence, feasibility-first sign test (exact, two-sided, alpha .05;
a directional claim needs p <= .05 and wins > losses; missing observations block inference):
  S1 (first):  sol_validated vs deepseek_validated, pooled over the 90 registered roots.
  P  (second): sol_validated vs gp_validated, tested only if S1 supports Sol; else descriptive.
Descriptive, unadjusted: sol_champion vs E's gp_champion (E's primary with the engine swapped)
and vs E's llm_champion (DeepSeek); each arm's validated program vs its search champion; every
validated program vs E's oracle-best reference (E's K1 rule) and initialization best; per family
T/W/R; feasibility fractions; jointly-feasible energy differences with bootstrap intervals (E's
analysis seeds); identical-program counts; S1 and P excluding the 12 E-01 rerun roots.

Statistics, outcome extraction, the oracle rule and the relative-saving summary are imported from
Study E's frozen analysis (statistics module sha256 b42bd4fd...); nothing is copied.

Layout read (experiments/study_f_20261007/runs/<campaign_id>/):
  ROOT_STOPPED.json  -> root MISSING
  SELECTION.json     arms.{sol,gp,deepseek}.champion, validated.{sol,gp,deepseek} (rec | null)
  final/LABELS.json  {"labels": {sha: [sol_champion|sol_validated|gp_validated|deepseek_validated]}}
  final/<sha>.json   records in E's format (evaluated in F, or byte-copied from E)
  final/REUSED.json  {"copied": {sha: "<path of E's runs/<id>/final/<sha>.json>"}}
  final/COMPLETE.json

Usage:
  .venv/bin/python -m experiments.study_f_20261007.analyze            # writes analysis/ once
  .venv/bin/python -m experiments.study_f_20261007.analyze --dry-run-from-e
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/study_f_20261007/analyze.py`
    sys.path.insert(0, str(ROOT))

from experiments.study_e_20261005 import analyze as E  # noqa: E402

stats = E.stats
ALPHA = 0.05
FAMILIES = E.FAMILIES
E_RUNS = E.RUNS
F_RUNS = HERE / "runs"
F_OUT = HERE / "analysis"
RERUN_ROOTS = frozenset([*(f"R{i}" for i in range(23, 27)), *(f"T{i}" for i in range(24, 28)),
                         *(f"W{i}" for i in range(24, 28))])  # amendment E-01

SOL_CH, SOL_V, GP_V, DS_V = "sol_champion", "sol_validated", "gp_validated", "deepseek_validated"
F_LABELS = (SOL_CH, SOL_V, GP_V, DS_V)
E_GP, E_DS = "e_gp_champion", "e_deepseek_champion"  # E's gp_champion / llm_champion
INIT, ORACLE = "initialization_best", "oracle_reference"
E_LABELS = {E_GP: "gp_champion", E_DS: "llm_champion", INIT: "initialization_best",
            **{r: r for r in E.REFERENCES}}
SHOWN = (*F_LABELS, E_GP, E_DS, INIT, ORACLE)
S1_PAIR, P_PAIR, E_STYLE = (SOL_V, DS_V), (SOL_V, GP_V), (SOL_CH, E_GP)
CONTRASTS = [S1_PAIR, P_PAIR, E_STYLE, (SOL_CH, E_DS),
             (SOL_V, SOL_CH), (GP_V, E_GP), (DS_V, E_DS),
             (SOL_V, ORACLE), (GP_V, ORACLE), (DS_V, ORACLE),
             (SOL_V, INIT), (GP_V, INIT), (DS_V, INIT)]
IDENTICAL = [S1_PAIR, P_PAIR, E_STYLE, (SOL_CH, E_DS), (SOL_V, SOL_CH), (GP_V, E_GP),
             (DS_V, E_DS)]
NAMES = {SOL_CH: "Sol champion", SOL_V: "Sol validated", GP_V: "GP validated",
         DS_V: "DeepSeek validated", E_GP: "E GP champion", E_DS: "E DeepSeek champion",
         INIT: "initialization best", ORACLE: "oracle-best reference"}


class LayoutError(ValueError):
    """F's layout contradicts itself or E (sha mismatch, altered copy, unknown provenance)."""


@contextlib.contextmanager
def _e_runs(path):
    """Point E's frozen outcome() at another runs directory for the duration of one call."""
    saved = E.RUNS
    E.RUNS = Path(path)
    try:
        yield
    finally:
        E.RUNS = saved


def _missing(reason):
    return {**E.MISSING, "missing_reason": reason}


def _check_rec(cid, where, rec):
    if rec is not None and hashlib.sha256(rec["source"].encode()).hexdigest() \
            != rec["source_sha256"]:
        raise LayoutError(f"{cid}: {where} source does not hash to its source_sha256")


def sol_root(f_runs, cid, family, index):
    """F's four labelled outcomes for one root, with missing reasons and provenance."""
    directory = Path(f_runs) / cid
    notes = {"arm_champion_mismatch": []}
    if (directory / "ROOT_STOPPED.json").exists():
        return {lab: _missing("root_stopped") for lab in F_LABELS}, notes
    sel_path = directory / "SELECTION.json"
    selection = json.loads(sel_path.read_text()) if sel_path.exists() else None
    if not selection or not selection.get("sealed_at"):
        return {lab: _missing("no_selection") for lab in F_LABELS}, notes
    if selection.get("study") != family or selection.get("index") != index:
        raise LayoutError(f"{cid}: SELECTION study/index {selection.get('study')}/"
                          f"{selection.get('index')} != registered {family}/{index}")
    sealed = {SOL_CH: selection["arms"]["sol"]["champion"],
              SOL_V: selection["validated"]["sol"], GP_V: selection["validated"]["gp"],
              DS_V: selection["validated"]["deepseek"]}
    for lab, rec in sealed.items():
        _check_rec(cid, lab, rec)
    # gp/deepseek arms are re-read from E's archives: their search champions should be E's.
    e_sel = json.loads((E_RUNS / cid / "SELECTION.json").read_text())["arms"]
    for arm, e_arm in (("gp", "gp"), ("deepseek", "llm")):
        ours, theirs = selection["arms"][arm]["champion"], e_sel[e_arm]["champion"]
        if (ours or {}).get("source_sha256") != (theirs or {}).get("source_sha256"):
            notes["arm_champion_mismatch"].append(arm)

    final = directory / "final"
    if not (final / "COMPLETE.json").exists():
        return {lab: (_missing("no_record") if sealed[lab] is None else _missing("no_final"))
                for lab in F_LABELS}, notes
    copied = {}
    if (final / "REUSED.json").exists():
        copied = json.loads((final / "REUSED.json").read_text())["copied"]
    e_final = (E_RUNS / cid / "final").resolve()
    e_labels = json.loads((e_final / "LABELS.json").read_text())["labels"]
    for sha, src in copied.items():
        src = Path(src) if Path(src).is_absolute() else ROOT / src
        if src.resolve() != e_final / f"{sha}.json" or not src.exists():
            raise LayoutError(f"{cid}: copy of {sha} does not come from E's final grid of "
                              "this root")
        if not e_labels.get(sha):
            raise LayoutError(f"{cid}: copied {sha} is not labelled in E's LABELS.json")
        mine = final / f"{sha}.json"
        if not mine.exists() or mine.read_bytes() != src.read_bytes():
            raise LayoutError(f"{cid}: copy of {sha} is not byte-identical to E's record")

    out = {}
    for lab, rec in sealed.items():
        if rec is None:
            out[lab] = _missing("no_record")
            continue
        with _e_runs(f_runs):
            result = E.outcome(cid, lab)  # E's own reader over F's final/
        if result["status"] == "missing":
            out[lab] = _missing("no_final")
            continue
        if result["source_sha256"] != rec["source_sha256"]:
            raise LayoutError(f"{cid}: {lab} final-grid sha differs from the sealed record")
        out[lab] = {**result, "missing_reason": None,
                    "final_source": "copied_from_e" if result["source_sha256"] in copied
                    else "evaluated_in_f"}
    return out, notes


def collect(f_runs):
    seeds = json.loads((E.REG / "SEEDS.json").read_text())
    out, family_of, campaign_of, notes = {}, {}, {}, {}
    for fam in FAMILIES:
        for index, row in enumerate(seeds["studies"][fam]):
            cid = row["campaign_id"]
            labels = {name: E.outcome(cid, e_name) for name, e_name in E_LABELS.items()}
            labels[ORACLE] = E.oracle([labels[r] for r in E.REFERENCES])
            f_out, notes[cid] = sol_root(f_runs, cid, fam, index)
            labels.update(f_out)
            out[row["root_seed"]] = labels
            family_of[row["root_seed"]] = fam
            campaign_of[row["root_seed"]] = cid
    return seeds, out, family_of, campaign_of, notes


def _claim(c, tested):
    p = c["p_two_sided"] if tested else None
    c["confirmatory"] = tested
    c["p_tested"] = p
    c["directional_claim"] = bool(p is not None and p <= ALPHA and c["wins"] > c["losses"])
    c["opposite_direction"] = bool(p is not None and p <= ALPHA and c["losses"] > c["wins"])
    c["blocked"] = bool(c["missing"]) if tested else None
    return c


def analyse(f_runs=F_RUNS):
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    if not actual.startswith("b42bd4fd"):
        raise ValueError("statistics module differs from the registered source")
    seeds, out, family_of, campaign_of, notes = collect(f_runs)
    every = list(out)

    def summary(left, right, roots, seed):
        r = stats.paired_report([{"root_seed": k, "left": out[k][left], "right": out[k][right]}
                                 for k in roots], roots, bootstrap_seed=seed)
        sav, n = E.saving([(out[k][left], out[k][right]) for k in roots])
        return {**{k: r["primary"][k] for k in ("wins", "losses", "ties", "missing",
                                                  "p_two_sided",
                                                  "conditional_win_probability_ci95")},
                "energy": r["energy"], "mean_relative_saving": sav, "jointly_feasible": n}

    def same(a, b, roots):
        return sum(out[k][a]["source_sha256"] is not None
                   and out[k][a]["source_sha256"] == out[k][b]["source_sha256"] for k in roots)

    def block(roots, seed):
        return {
            "contrasts": {f"{a} vs {b}": summary(a, b, roots, seed) for a, b in CONTRASTS},
            "feasibility": {lab: sum(out[k][lab]["feasible"] is True for k in roots) / len(roots)
                            for lab in SHOWN},
            "evaluated": {lab: sum(out[k][lab]["status"] == "evaluated" for k in roots)
                          for lab in SHOWN},
            "mean_energy_when_feasible": {
                lab: (lambda v: sum(v) / len(v) if v else None)(
                    [out[k][lab]["energy"] for k in roots if out[k][lab]["feasible"]])
                for lab in SHOWN},
            "identical_programs": {f"{a} = {b}": same(a, b, roots) for a, b in IDENTICAL},
        }

    pooled_seed = seeds["analysis"]["pooled"]
    s1 = _claim(summary(*S1_PAIR, every, pooled_seed), True)
    gate_open = s1["directional_claim"] and not s1["blocked"]
    p = _claim(summary(*P_PAIR, every, pooled_seed), gate_open)
    kept = [k for k in every if campaign_of[k] not in RERUN_ROOTS]
    excluding = {"roots": len(kept),
                 "S1": summary(*S1_PAIR, kept, pooled_seed),
                 "P": summary(*P_PAIR, kept, pooled_seed)}

    reasons = {}
    for lab in F_LABELS:
        for k in every:
            reason = out[k][lab]["missing_reason"]
            if reason:
                reasons.setdefault(lab, {}).setdefault(reason, []).append(campaign_of[k])
    execution = {
        "roots": len(every),
        "evaluated": {lab: sum(out[k][lab]["status"] == "evaluated" for k in every)
                      for lab in F_LABELS},
        "copied_from_e": {lab: sum(out[k][lab].get("final_source") == "copied_from_e"
                                   for k in every) for lab in F_LABELS},
        "evaluated_in_f": {lab: sum(out[k][lab].get("final_source") == "evaluated_in_f"
                                    for k in every) for lab in F_LABELS},
        "missing_by_reason": {lab: {r: sorted(c) for r, c in rs.items()}
                              for lab, rs in reasons.items()},
        "e_missing": {lab: sum(out[k][lab]["status"] == "missing" for k in every)
                      for lab in (E_GP, E_DS, INIT, ORACLE)},
        "arm_champion_mismatch_vs_e": {cid: n["arm_champion_mismatch"]
                                       for cid, n in notes.items() if n["arm_champion_mismatch"]},
    }
    return {"statistics_source_sha256": actual, "execution": execution,
            "confirmatory": {"S1 sol_validated vs deepseek_validated": s1,
                             "P sol_validated vs gp_validated": p, "gate_open_for_P": gate_open},
            "excluding_e01_rerun_roots": excluding,
            "pooled_descriptive": block(every, pooled_seed),
            "per_family": {fam: block([k for k in every if family_of[k] == fam],
                                      seeds["analysis"][fam]) for fam in FAMILIES},
            "outcomes": {str(k): {"family": family_of[k], "campaign_id": campaign_of[k], **v}
                         for k, v in out.items()}}


def _p(p):
    if p is None:
        return "n/a"
    return f"{p:.2g}" if p < 0.01 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}".rstrip("0")


def _wlt(c, bold=False):
    s = f"{c['wins']}–{c['losses']}–{c['ties']}"
    if c["missing"]:
        s += f" (+{c['missing']} missing)"
    s += f" ({_p(c['p_two_sided'])})"
    return f"**{s}**" if bold and c["p_two_sided"] is not None and c["p_two_sided"] <= ALPHA \
        else s


def _energy(c):
    e = c["energy"]
    if e["mean"] is None:
        return "n/a (no jointly feasible roots)"
    ci = e["bootstrap_ci95_descriptive"]
    ci_s = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    sav = c["mean_relative_saving"]
    return (f"{e['mean']:+.3f}{ci_s}, n = {e['jointly_feasible']}"
            + (f", mean relative saving {sav:+.1%}" if sav is not None else ""))


def _verdict(c, favoured, other):
    if not c["confirmatory"]:
        return "**descriptive only** (gate closed: S1 did not support Sol)"
    if c["blocked"]:
        return f"**blocked** ({c['missing']} missing observations)"
    if c["directional_claim"]:
        return f"**supported** ({favoured} over {other})"
    lean = (f"direction favours {other}" if c["losses"] > c["wins"] else
            f"direction favours {favoured}" if c["wins"] > c["losses"] else "no direction")
    return f"**not supported** ({lean}" + (
        "; significant in the opposite direction)" if c["opposite_direction"] else ")")


def render(result, title_date):
    conf = result["confirmatory"]
    s1 = conf["S1 sol_validated vs deepseek_validated"]
    p = conf["P sol_validated vs gp_validated"]
    ex, fam, pooled = result["execution"], result["per_family"], result["pooled_descriptive"]
    exc = result["excluding_e01_rerun_roots"]
    blocks = (*(fam[f] for f in FAMILIES), pooled)

    def miss_line():
        parts = []
        for lab, rs in sorted(ex["missing_by_reason"].items()):
            parts.append(f"{lab}: " + ", ".join(f"{r} {len(c)} ({' '.join(c)})"
                                                for r, c in sorted(rs.items())))
        return "; ".join(parts) or "none"

    def wlt_cell(c):
        return (f"**{c['wins']} / {c['losses']} / {c['ties']}**"
                + (f" (+{c['missing']} missing)" if c["missing"] else ""))

    mism = ex["arm_champion_mismatch_vs_e"]
    lines = [
        f"**Study F (Sol engine with validated selection): registered results, {title_date}**",
        "",
        "Engine gpt-6.1-sol via Codex on Study E's 90 registered roots; GP and DeepSeek arms "
        "re-read from E's archives; E-side outcomes are Study E's sealed final grids. "
        f"Statistics module sha256 `{result['statistics_source_sha256'][:8]}…` (imported from "
        "Study E).", "",
        f"**Execution.** {ex['roots']} registered roots. Evaluated on the final grid: "
        + ", ".join(f"{lab} {n} ({ex['evaluated_in_f'][lab]} in F, "
                    f"{ex['copied_from_e'][lab]} byte-copied from E)"
                    for lab, n in ex["evaluated"].items())
        + f". Missing: {miss_line()}. Missing E observations: "
        + (", ".join(f"{k} {v}" for k, v in ex["e_missing"].items() if v) or "none")
        + ". Re-read GP/DeepSeek search champions differing from E's: "
        + (", ".join(f"{c} ({'/'.join(a)})" for c, a in sorted(mism.items())) or "none")
        + ".", "",
        "## Confirmatory (registered, fixed sequence)", "",
        "| | S1: Sol validated vs DeepSeek validated | P: Sol validated vs GP validated |",
        "|---|---|---|",
        f"| Wins / losses / ties | {wlt_cell(s1)} | {wlt_cell(p)} |",
        f"| Exact two-sided p | **{_p(s1['p_two_sided'])}** | **{_p(p['p_two_sided'])}** |",
        f"| Directional claim | {_verdict(s1, 'Sol', 'DeepSeek')} | "
        f"{_verdict(p, 'Sol', 'GP')} |",
        f"| Energy difference (right − left) | {_energy(s1)} | {_energy(p)} |", "",
        f"Excluding the 12 E-01 rerun roots ({exc['roots']} roots): S1 {_wlt(exc['S1'])}; "
        f"P {_wlt(exc['P'])}. E-style comparison (descriptive), Sol champion vs E's GP "
        f"champion: {_wlt(pooled['contrasts'][f'{SOL_CH} vs {E_GP}'])}.", "",
        "## By family and pooled (descriptive, unadjusted)", "",
        "| | T (historical) | W (wide) | R (drift) | Pooled |", "|---|---|---|---|---|"]
    for a, b in CONTRASTS:
        lines.append(f"| {NAMES[a]} vs {NAMES[b]} | " + " | ".join(
            _wlt(blk["contrasts"][f"{a} vs {b}"], bold=True) for blk in blocks) + " |")
    feas = (SOL_V, GP_V, DS_V, SOL_CH, E_GP, E_DS, INIT, ORACLE)
    lines.append("| Feasible: " + " / ".join(NAMES[lab] for lab in feas) + " | " + " | ".join(
        " / ".join(f"{blk['feasibility'][lab]:.2f}" for lab in feas) for blk in blocks) + " |")
    for a, b in IDENTICAL:
        lines.append(f"| Identical program: {NAMES[a]} = {NAMES[b]} | " + " | ".join(
            str(blk["identical_programs"][f"{a} = {b}"]) for blk in blocks) + " |")
    lines += ["", "Jointly-feasible energy differences (right − left normalized energy; positive "
              "favours the left program; mean, descriptive bootstrap 95% interval with E's "
              "analysis seeds):", ""]
    for a, b in CONTRASTS:
        key = f"{a} vs {b}"
        lines.append(f"- {NAMES[a]} vs {NAMES[b]}: " + "; ".join(
            f"{f} {_energy(fam[f]['contrasts'][key])}" for f in FAMILIES)
            + f"; pooled {_energy(pooled['contrasts'][key])}.")
    lines += ["", "Families are related variants; per-family and descriptive contrasts are "
              "unadjusted. Non-significance does not establish equivalence.", ""]
    return "\n".join(lines)


def write(result, out_dir, title_date):
    out_dir = Path(out_dir)
    targets = [out_dir / "PRIMARY.json", out_dir / "RESULTS.md"]
    if any(t.exists() for t in targets):
        raise FileExistsError(f"refusing to overwrite sealed analysis outputs in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(result, title_date)
    with targets[0].open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True,
                  default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
        stream.write("\n")
    with targets[1].open("x") as stream:
        stream.write(text)
    return text


def _rec(champion, validated=False):
    if champion is None:
        return None
    rec = {k: champion[k] for k in ("candidate_id", "proposal_seq", "source", "source_sha256",
                                    "search_fitness")}
    if validated:
        rec["validation_sort_key"] = champion["search_fitness"]["sort_key"]
    return rec


def build_dry_run_layout(f_runs):
    """Synthetic F layout from E: the Sol arm is E's DeepSeek (llm) archive. Without validation,
    validated sol = E llm_champion, gp = E gp_champion, deepseek = E llm_champion. Every final
    record is byte-copied from E's final grid and listed in REUSED.json."""
    f_runs = Path(f_runs)
    seeds = json.loads((E.REG / "SEEDS.json").read_text())
    for fam in FAMILIES:
        for row in seeds["studies"][fam]:
            cid = row["campaign_id"]
            e_dir = E_RUNS / cid
            selection = json.loads((e_dir / "SELECTION.json").read_text())
            llm, gp = selection["arms"]["llm"], selection["arms"]["gp"]
            directory = f_runs / cid
            (directory / "final").mkdir(parents=True)
            search = {k: llm.get(k) for k in ("stage_exit", "terminal", "completed_candidates",
                                              "full_records")}
            arm = {"sol": {**search, "champion": _rec(llm["champion"]),
                           "top_k": [_rec(llm["champion"])]},
                   "gp": {"champion": _rec(gp["champion"]), "top_k": [_rec(gp["champion"])]},
                   "deepseek": {"champion": _rec(llm["champion"]),
                                "top_k": [_rec(llm["champion"])]}}
            validated = {"sol": _rec(llm["champion"], True), "gp": _rec(gp["champion"], True),
                         "deepseek": _rec(llm["champion"], True)}
            (directory / "SELECTION.json").write_text(json.dumps({
                "study": selection["study"], "index": selection["index"],
                "sealed_at": selection["sealed_at"], "arms": arm, "validated": validated,
                "rule": "dry run from Study E: validated = search champion"}))
            labels, copied = {}, {}
            for lab, rec in ((SOL_CH, arm["sol"]["champion"]), (SOL_V, validated["sol"]),
                             (GP_V, validated["gp"]), (DS_V, validated["deepseek"])):
                if rec is None:
                    continue
                sha = rec["source_sha256"]
                labels.setdefault(sha, []).append(lab)
                src = e_dir / "final" / f"{sha}.json"
                shutil.copyfile(src, directory / "final" / f"{sha}.json")
                copied[sha] = str(src.relative_to(ROOT))
            (directory / "final" / "LABELS.json").write_text(json.dumps({"labels": labels}))
            (directory / "final" / "REUSED.json").write_text(json.dumps({"copied": copied}))
            (directory / "final" / "COMPLETE.json").write_text(
                json.dumps({"at": "dry-run", "sources": len(labels)}))


def _table(result):
    conf = result["confirmatory"]
    exc = result["excluding_e01_rerun_roots"]
    rows = [("S1 sol_v vs deepseek_v (pooled)", conf["S1 sol_validated vs deepseek_validated"]),
            ("P sol_v vs gp_v (pooled)", conf["P sol_validated vs gp_validated"])]
    rows += [(f"P sol_v vs gp_v ({f})", result["per_family"][f]["contrasts"][
        f"{SOL_V} vs {GP_V}"]) for f in FAMILIES]
    rows += [("E-style sol_champion vs E gp (pooled)",
              result["pooled_descriptive"]["contrasts"][f"{SOL_CH} vs {E_GP}"]),
             ("S1 excl. E-01 reruns", exc["S1"]), ("P excl. E-01 reruns", exc["P"])]
    return "\n".join(["| contrast | W | L | T | missing | p |", "|---|---|---|---|---|---|"]
                     + [f"| {n} | {c['wins']} | {c['losses']} | {c['ties']} | {c['missing']} | "
                        f"{_p(c['p_two_sided'])} |" for n, c in rows])


def dry_run_from_e(work_dir=None):
    """Run the full analysis on a synthetic layout built from E; assert it reproduces E."""
    context = (tempfile.TemporaryDirectory(prefix="study_f_dry_") if work_dir is None
               else contextlib.nullcontext(work_dir))
    with context as tmp:
        tmp = Path(tmp)
        build_dry_run_layout(tmp / "runs")
        result = analyse(tmp / "runs")
        write(result, tmp / "analysis", "dry run from Study E")
        e = json.loads((E.HERE / "analysis" / "PRIMARY.json").read_text())
        conf = result["confirmatory"]
        s1 = conf["S1 sol_validated vs deepseek_validated"]
        p = conf["P sol_validated vs gp_validated"]
        assert (s1["wins"], s1["losses"], s1["ties"], s1["missing"]) == (0, 0, 90, 0), s1
        assert not s1["directional_claim"] and conf["gate_open_for_P"] is False
        assert p["confirmatory"] is False and p["p_tested"] is None
        pooled = result["pooled_descriptive"]["contrasts"]
        stat_keys = ("wins", "losses", "ties", "missing", "p_two_sided",
                     "conditional_win_probability_ci95", "energy", "mean_relative_saving",
                     "jointly_feasible")
        for c in (p, pooled[f"{SOL_CH} vs {E_GP}"]):
            assert (c["wins"], c["losses"], c["ties"], c["missing"]) == (27, 39, 24, 0), c
            assert abs(c["p_two_sided"] - 0.17528579444303533) < 1e-12, c["p_two_sided"]
            for key in stat_keys:
                assert c[key] == e["primary"][key], (key, c[key], e["primary"][key])
        expected = {"T": (9, 3, 18), "W": (9, 20, 1), "R": (9, 16, 5)}
        for fam, wlt in expected.items():
            ours = result["per_family"][fam]["contrasts"]
            theirs = e["per_family"][fam]["contrasts"]
            for key in (f"{SOL_V} vs {GP_V}", f"{SOL_CH} vs {E_GP}"):
                c = ours[key]
                assert (c["wins"], c["losses"], c["ties"]) == wlt, (fam, key, c)
                assert c == theirs["llm_champion vs gp_champion"], (fam, key)
            for mine, e_key in (((SOL_V, ORACLE), "llm_champion vs oracle_reference"),
                                ((GP_V, ORACLE), "gp_champion vs oracle_reference"),
                                ((SOL_V, INIT), "llm_champion vs initialization_best"),
                                ((GP_V, INIT), "gp_champion vs initialization_best")):
                assert ours[f"{mine[0]} vs {mine[1]}"] == theirs[e_key], (fam, mine)
        exc = result["excluding_e01_rerun_roots"]
        assert exc["roots"] == 78
        assert (exc["P"]["wins"], exc["P"]["losses"], exc["P"]["ties"]) == (24, 33, 21), exc
        assert (exc["S1"]["wins"], exc["S1"]["losses"], exc["S1"]["ties"]) == (0, 0, 78), exc
        idents = result["pooled_descriptive"]["identical_programs"]
        assert idents[f"{SOL_V} = {DS_V}"] == idents[f"{SOL_V} = {SOL_CH}"] == 90
        assert idents[f"{GP_V} = {E_GP}"] == idents[f"{DS_V} = {E_DS}"] == 90
        ex = result["execution"]
        assert ex["copied_from_e"] == {lab: 90 for lab in F_LABELS}, ex["copied_from_e"]
        assert not ex["missing_by_reason"] and not ex["arm_champion_mismatch_vs_e"]
        for root, row in e["outcomes"].items():
            for lab, e_lab in ((SOL_CH, "llm_champion"), (SOL_V, "llm_champion"),
                               (GP_V, "gp_champion"), (DS_V, "llm_champion")):
                assert {k: result["outcomes"][root][lab][k] for k in E.MISSING} \
                    == row[e_lab], (root, lab)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run-from-e", action="store_true",
                        help="validate on a synthetic layout built from Study E (temp dir)")
    parser.add_argument("--runs", type=Path, default=F_RUNS)
    parser.add_argument("--out", type=Path, default=F_OUT)
    parser.add_argument("--date", default=None, help="date shown in RESULTS.md")
    args = parser.parse_args(argv)
    if args.dry_run_from_e:
        result = dry_run_from_e()
        print("dry run from Study E: all reproduction checks passed")
        print(_table(result))
        return
    targets = [args.out / "PRIMARY.json", args.out / "RESULTS.md"]
    if any(t.exists() for t in targets):
        raise SystemExit(f"refusing to overwrite sealed analysis outputs in {args.out}")
    from datetime import date
    result = analyse(args.runs)
    write(result, args.out, args.date or date.today().strftime("%-d %B %Y"))
    print(_table(result))
    print("execution", json.dumps(result["execution"], indent=1))


if __name__ == "__main__":
    main()
