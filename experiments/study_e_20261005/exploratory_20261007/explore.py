"""EXPLORATORY (post hoc, not registered) analyses of Study E's sealed data, 7 October 2026.

Every analysis here was chosen after the registered results (analysis/RESULTS.md) were known.
Nothing here is confirmatory, and nothing here changes the registered results. The script only
reads sealed files; it runs no simulation and makes no API call. Output goes to stdout; the
report EXPLORATORY.md transcribes it.

Run from the repository root with the project environment (the frozen statistics module needs
numpy/scipy):
    .venv/bin/python -m experiments.study_e_20261005.exploratory_20261007.explore
or  .venv/bin/python experiments/study_e_20261005/exploratory_20261007/explore.py

Inputs (read-only): analysis/PRIMARY.json, runs/*/SELECTION.json, BUDGET.jsonl,
runs/*/{llm,zeroshot}/journal.jsonl, runs/*/superseded_E01/{llm,zeroshot}/journal.jsonl,
prereg/STUDY_E_MULTIFAMILY_20261005/SEEDS.json, and the PRIMARY.json files of studies A/B, C, D.
Memory: journals are streamed line by line; no final-grid record or pair.json is loaded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent
ROOT = E.parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import binomtest, fisher_exact  # noqa: E402

from experiments.prospective_20260917 import statistics as stats  # noqa: E402

FAMILIES = ("T", "W", "R")
RERUN = {f"{f}{i}" for f, idx in (("R", range(23, 27)), ("T", range(24, 28)),
                                     ("W", range(24, 28))) for i in idx}  # amendment E-01
CALIBRATION_REFERENCE = {  # PROTOCOL.md headroom gate: references feasible on calibration seeds
    "T": "reference:latency_greedy", "W": "reference:kube_least_allocated",
    "R": "reference:graph_headroom"}
SELECTION_PATHS = {
    "llm_champion": ("arms", "llm", "champion"),
    "llm_champion_atN": ("arms", "llm", "champion_atN"),
    "gp_champion": ("arms", "gp", "champion"),
    "initialization_best": ("initialization_best",),
    "zeroshot_champion": ("zeroshot", "champion"),
}


def load(path):
    with open(path) as stream:
        return json.load(stream)


def sign(w, l):
    return 1.0 if w + l == 0 else float(binomtest(w, w + l, 0.5).pvalue)


def fmt_p(p):
    return "—" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def freeman_halton(table):
    """Exact conditional test for an r x 2 table (sum of probabilities <= observed)."""
    rows = [sum(r) for r in table]
    col0 = sum(r[0] for r in table)
    n = sum(rows)

    def prob(cells):
        num = math.prod(math.comb(rows[i], cells[i]) for i in range(len(rows)))
        return num / math.comb(n, col0)

    observed = prob([r[0] for r in table])
    total = 0.0
    for cells in itertools.product(*(range(r + 1) for r in rows)):
        if sum(cells) == col0:
            p = prob(cells)
            if p <= observed * (1 + 1e-7):
                total += p
    return min(1.0, total)


# ------------------------------------------------------------------------------- data
def outcomes():
    primary = load(E / "analysis" / "PRIMARY.json")
    seeds = load(ROOT / "prereg/STUDY_E_MULTIFAMILY_20261005/SEEDS.json")
    cid = {str(r["root_seed"]): r["campaign_id"] for f in FAMILIES for r in seeds["studies"][f]}
    out = {cid[k]: v for k, v in primary["outcomes"].items()}
    return primary, out


def directions(out, left, right, roots):
    return [stats.direction(out[c][left], out[c][right]) for c in roots]


def wlt(out, left, right, roots):
    d = directions(out, left, right, roots)
    return d.count(1), d.count(-1), d.count(0)


def saving(out, left, right, roots):
    pairs = [(out[c][left]["energy"], out[c][right]["energy"]) for c in roots
             if out[c][left]["feasible"] and out[c][right]["feasible"]]
    return (sum((b - a) / b for a, b in pairs) / len(pairs) if pairs else None), len(pairs)


def roots_of(out, fam=None, drop_rerun=False):
    return sorted(c for c, v in out.items() if (fam is None or v["family"] == fam)
                  and not (drop_rerun and c in RERUN))


# ------------------------------------------------------------------- 1. heterogeneity
def heterogeneity(primary, out):
    print("\n## 1. Task-family heterogeneity of LLM vs GP")
    for drop in (False, True):
        tag = "excluding 12 rerun roots" if drop else "all 90 roots"
        counts = {f: wlt(out, "llm_champion", "gp_champion", roots_of(out, f, drop))
                  for f in FAMILIES}
        print(f"\n### {tag}")
        for f in FAMILIES:
            w, l, t = counts[f]
            same = sum(out[c]["llm_champion"]["source_sha256"] == out[c]["gp_champion"]["source_sha256"]
                       for c in roots_of(out, f, drop))
            print(f"{f}: {w}-{l}-{t} sign p={fmt_p(sign(w, l))}; same program in {same}")
            if not drop:
                reg = primary["per_family"][f]["contrasts"]["llm_champion vs gp_champion"]
                assert (reg["wins"], reg["losses"], reg["ties"]) == (w, l, t), "mismatch vs PRIMARY"
        wr = tuple(counts["W"][i] + counts["R"][i] for i in range(3))
        print(f"W+R: {wr[0]}-{wr[1]}-{wr[2]} sign p={fmt_p(sign(wr[0], wr[1]))}")
        table = [[counts["T"][0], counts["T"][1]], [wr[0], wr[1]]]
        odds, p = fisher_exact(table)
        print(f"Fisher exact 2x2 decisive pairs T vs W+R {table}: two-sided p={p:.4f}, "
              f"sample odds ratio={odds:.2f}")
        for a, b in (("T", "W"), ("T", "R"), ("W", "R")):
            t2 = [list(counts[a][:2]), list(counts[b][:2])]
            print(f"  Fisher {a} vs {b} {t2}: p={fisher_exact(t2)[1]:.4f}")
        fh = freeman_halton([list(counts[f][:2]) for f in FAMILIES])
        print(f"Freeman-Halton exact 3x2 (T, W, R decisive pairs): p={fh:.4f}")
        # ties are not ignorable: share of roots with a decisive outcome by family
        for f in FAMILIES:
            w, l, t = counts[f]
            n = w + l + t
            print(f"  {f}: decisive {w + l}/{n}; LLM wins as share of all roots {w}/{n}")


# --------------------------------------------------------------- 2. cross-study T
def cross_study(out):
    print("\n## 2. LLM vs GP on the historical task (dev-continuum-tight) across studies")
    ab = load(ROOT / "experiments/corrected_ab_20260925/analysis/PRIMARY.json")["studies"]
    a = ab["A"]["primary"]["primary"]
    b = ab["B"]["descriptive"]["llm_champion vs gp_champion"]["wins_losses_ties"]
    c = load(ROOT / "experiments/study_c_20261003/analysis/PRIMARY.json")["primary"]["primary"]
    d = load(ROOT / "experiments/study_d_20261004/analysis/PRIMARY.json")["descriptive"][
        "llm_champion@D vs gp_champion@D"]["wins_losses_ties"]
    et = wlt(out, "llm_champion", "gp_champion", roots_of(out, "T"))
    ew = wlt(out, "llm_champion", "gp_champion", roots_of(out, "W"))
    er = wlt(out, "llm_champion", "gp_champion", roots_of(out, "R"))
    rows = [("A (registered primary; 1 search seed)", "T", (a["wins"], a["losses"], a["ties"])),
            ("C (registered primary; 4 seeds)", "T", (c["wins"], c["losses"], c["ties"])),
            ("D (descriptive; C's roots, 1 seed)", "T", tuple(d)),
            ("E family T (descriptive per family)", "T", et),
            ("B (descriptive; distributed/edge-heavy task)", "B", tuple(b)),
            ("E family W", "W", ew), ("E family R", "R", er)]
    for name, task, (w, l, t) in rows:
        print(f"{name} [{task}]: {w}-{l}-{t}  (per-study sign p={fmt_p(sign(w, l))})")
    # Context: how often each operator improved on its own starting archive on task T.
    cd = load(ROOT / "experiments/study_c_20261003/analysis/PRIMARY.json")["descriptive"]
    dd = load(ROOT / "experiments/study_d_20261004/analysis/PRIMARY.json")["descriptive"]
    ad = ab["A"]["descriptive"]
    for name, gp, llm in (
            ("A", ad["gp_champion vs initialization_best"]["wins_losses_ties"],
             ad["llm_champion vs initialization_best"]["wins_losses_ties"]),
            ("C", cd["gp_champion vs initialization_best"]["wins_losses_ties"],
             cd["llm_champion vs initialization_best"]["wins_losses_ties"]),
            ("D", dd["gp_champion@D vs initialization_best@D"]["wins_losses_ties"],
             dd["llm_champion@D vs initialization_best@D"]["wins_losses_ties"]),
            ("E-T", wlt(out, "gp_champion", "initialization_best", roots_of(out, "T")),
             wlt(out, "llm_champion", "initialization_best", roots_of(out, "T"))),
            ("E-W", wlt(out, "gp_champion", "initialization_best", roots_of(out, "W")),
             wlt(out, "llm_champion", "initialization_best", roots_of(out, "W"))),
            ("E-R", wlt(out, "gp_champion", "initialization_best", roots_of(out, "R")),
             wlt(out, "llm_champion", "initialization_best", roots_of(out, "R")))):
        print(f"{name}: GP vs initialization best {'-'.join(map(str, gp))}; "
              f"LLM vs initialization best {'-'.join(map(str, llm))}")
    tot = [sum(x) for x in zip(*(r[2] for r in rows if r[0][0] in "ACE" and r[1] == "T"))]
    print(f"Sum over independent T root sets A + C + E-T (D omitted: same roots as C): "
          f"{tot[0]}-{tot[1]}-{tot[2]} (descriptive only)")


# ------------------------------------------------------------ 3. iteration vs zero-shot
def journals():
    """Per (campaign, arm, generation) request accounting from the transport journals."""
    acc = defaultdict(lambda: Counter())
    keys = {}
    # Superseded first: 87 rerun requests repeat a superseded request key (deterministic
    # requests), and the ledger keeps the rerun's settlement for them, so a shared key is
    # attributed to the current (rerun) journal.
    status = {}
    for gen, pattern in (("superseded", "runs/*/superseded_E01/{arm}/journal.jsonl"),
                         ("current", "runs/*/{arm}/journal.jsonl")):
        for arm in ("llm", "zeroshot"):
            for path in sorted(E.glob(pattern.format(arm=arm))):
                cid = path.relative_to(E / "runs").parts[0]
                a = acc[(cid, arm, gen)]
                with path.open() as stream:
                    for line in stream:
                        p = json.loads(line)["payload"]
                        kind = p["type"]
                        if kind != "configuration" and kind != "reserved":
                            status[p["request_sha256"]] = kind
                        if kind == "reserved":
                            a["requests"] += 1
                            keys[p["request_sha256"]] = (cid, arm, gen)
                        elif kind == "completed":
                            comp = p["completion"]
                            usage = comp["raw_response"].get("usage", {})
                            a["completed"] += 1
                            a["billed_microusd"] += comp["cost_microusd"]
                            a["prompt_tokens"] += comp["prompt_tokens"]
                            a["completion_tokens"] += comp["completion_tokens"]
                            a["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                        elif kind == "settled_empty":
                            s = p["settlement"]
                            a["empty"] += 1
                            a["billed_microusd"] += s["cost_microusd"]
                            a["prompt_tokens"] += s.get("prompt_tokens", 0)
                            a["completion_tokens"] += s.get("completion_tokens", 0)
                            a["reasoning_tokens"] += s.get("reasoning_tokens", 0)
                        elif kind == "settled_http_error":
                            a["http_error"] += 1
                            a["conservative_microusd"] += p["settlement"]["cost_microusd"]
    return acc, keys, status


def ledger():
    """BUDGET.jsonl in micro-USD: last settlement per request key, as the launcher counts it."""
    reserved, settled, label = {}, {}, {}
    with (E / "BUDGET.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            (reserved if r["t"] == "reserve" else settled)[r["key"]] = r["amount"]
            label[r["key"]] = r["label"]
    amount = {k: settled.get(k, v) for k, v in reserved.items()}
    for k, v in settled.items():
        amount.setdefault(k, v)
    return amount, label


def iteration(primary, out):
    print("\n## 3. Iteration vs zero-shot")
    for drop in (False, True):
        print(f"\n### {'excluding 12 rerun roots' if drop else 'all 90 roots'}")
        for fam in (*FAMILIES, None):
            r = roots_of(out, fam, drop)
            k2 = wlt(out, "llm_champion_atN", "zeroshot_champion", r)
            full = wlt(out, "llm_champion", "zeroshot_champion", r)
            s2, n2 = saving(out, "llm_champion_atN", "zeroshot_champion", r)
            print(f"{fam or 'pooled'} (n={len(r)}): K2 atN vs ZS {k2[0]}-{k2[1]}-{k2[2]} "
                  f"p={fmt_p(sign(*k2[:2]))}; full LLM vs ZS {full[0]}-{full[1]}-{full[2]} "
                  f"p={fmt_p(sign(*full[:2]))}; atN vs ZS saving when jointly feasible "
                  f"{'—' if s2 is None else f'{100 * s2:.1f}%'} (n={n2})")
    slots = {}
    for c in out:
        sel = load(E / "runs" / c / "SELECTION.json")
        slots[c] = (sel["arms"]["llm"]["atN_slots"], sel["zeroshot"]["proposals"],
                    sel["zeroshot"]["valid"], sel["zeroshot"]["promoted_to_l2"])
    for fam in FAMILIES:
        r = roots_of(out, fam)
        mean = [sum(slots[c][i] for c in r) / len(r) for i in range(4)]
        print(f"{fam}: mean N (atN slots) {mean[0]:.1f}; ZS proposals {mean[1]:.1f}, valid "
              f"{mean[2]:.1f}, promoted to L2 {mean[3]:.1f}")

    acc, keys, status = journals()
    amount, label = ledger()
    total = sum(amount.values())
    unmatched = [k for k in amount if k not in keys]
    print(f"\nLedger (micro-USD, last settlement per request key): Study E ${total / 1e6:.2f}; "
          f"{len(unmatched)} ledger keys without a journal reservation "
          f"(${sum(amount[k] for k in unmatched) / 1e6:.2f})")
    try:
        from experiments.study_e_20261005 import launch
        prior = launch.prior_exposure_microusd()
        print(f"Launcher prior exposure ${prior / 1e6:.2f}; prior + Study E = "
              f"${(prior + total) / 1e6:.2f} (RESULTS.md: $110.67)")
    except Exception as exc:  # pragma: no cover - reconciliation is informative only
        print("prior exposure unavailable:", exc)

    raw = 0
    with (E / "BUDGET.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            raw += r["amount"] if r["t"] == "settle" else 0
    in_ledger = Counter()
    for k, v in amount.items():
        in_ledger[status.get(k, "?")] += v
    print(f"Sum of every settle event (no de-duplication) ${raw / 1e6:.2f}; de-duplicated ledger by "
          f"final status: " + ", ".join(f"{k} ${v / 1e6:.2f}" for k, v in sorted(in_ledger.items())))

    led = Counter()
    for k, v in amount.items():
        if k in keys:
            led[keys[k]] += v
        else:
            lab = label[k]
            led[(lab.removesuffix("-zs"), "zeroshot" if lab.endswith("-zs") else "llm", "?")] += v

    def table(sel, title):
        print(f"\n{title}")
        print("family | arm | roots | ledger $ | billed $ | conservative $ | requests | "
              "prompt tok | completion tok (reasoning) | ledger $/root")
        for fam in (*FAMILIES, "all"):
            for arm in ("llm", "zeroshot"):
                rows = [k for k in acc if k[1] == arm and sel(k)
                        and (fam == "all" or k[0][0] == fam)]
                lroots = {k[0] for k in rows}
                a = Counter()
                for k in rows:
                    a.update(acc[k])
                lsum = sum(v for k, v in led.items() if k in rows)
                print(f"{fam} | {arm} | {len(lroots)} | {lsum / 1e6:.2f} | "
                      f"{a['billed_microusd'] / 1e6:.2f} | {a['conservative_microusd'] / 1e6:.2f} | "
                      f"{a['requests']} | {a['prompt_tokens'] / 1e6:.2f}M | "
                      f"{a['completion_tokens'] / 1e6:.2f}M ({a['reasoning_tokens'] / 1e6:.2f}M) | "
                      f"{lsum / 1e6 / max(1, len(lroots)):.3f}")

    table(lambda k: True, "Spend A: everything (current + superseded E-01 journals)")
    table(lambda k: k[2] == "current", "Spend B: current journals only (rerun arms counted once)")
    table(lambda k: k[2] == "current" and k[0] not in RERUN,
          "Spend C: current journals, excluding the 12 rerun roots (78 roots)")
    sup = Counter()
    for k in acc:
        if k[2] == "superseded":
            sup.update(acc[k])
    sup_led = sum(v for k, v in led.items() if k[2] == "superseded")
    print(f"\nSuperseded (E-01) journals: ledger ${sup_led / 1e6:.2f}, billed "
          f"${sup['billed_microusd'] / 1e6:.2f}, conservative ${sup['conservative_microusd'] / 1e6:.2f}, "
          f"{sup['requests']} requests, {sup['http_error']} HTTP errors")
    allc = Counter()
    for k in acc:
        allc.update(acc[k])
    print(f"All journals: conservative charges (HTTP errors) ${allc['conservative_microusd'] / 1e6:.2f} "
          f"over {allc['http_error']} requests; empty {allc['empty']}")


# ------------------------------------------------------------ 4. selection overfitting
def overfitting(primary, out):
    print("\n## 4. Search fitness vs final-grid feasibility")
    sel = {c: load(E / "runs" / c / "SELECTION.json") for c in out}
    for fam in (*FAMILIES, None):
        r = roots_of(out, fam)
        print(f"\n### {fam or 'pooled'} (n={len(r)})")
        print("label | search-feasible | final-feasible | final-feasible given search-feasible | "
              "search-infeasible but final-feasible | mean energy search->final (both feasible, n)")
        for lab, path in SELECTION_PATHS.items():
            sf = ff = both = rescue = 0
            es, ef = [], []
            for c in r:
                rec = sel[c]
                for p in path:
                    rec = rec[p]
                assert rec["source_sha256"] == out[c][lab]["source_sha256"], (c, lab)
                s_feas = bool(rec["search_fitness"]["feasible"])
                f_feas = out[c][lab]["feasible"] is True
                sf += s_feas
                ff += f_feas
                both += s_feas and f_feas
                rescue += (not s_feas) and f_feas
                if s_feas and f_feas:
                    es.append(rec["search_fitness"]["normalized_energy"])
                    ef.append(out[c][lab]["energy"])
            me = (f"{sum(es) / len(es):.3f}->{sum(ef) / len(ef):.3f} (n={len(es)})" if es else "—")
            print(f"{lab} | {sf}/{len(r)} | {ff}/{len(r)} | {both}/{sf} | {rescue}/{len(r) - sf} | {me}")

    print("\n### Transfer: family-T selections scored on 8 fresh dev-continuum-drift seeds")
    tr = primary["transfer_T_to_drift"]
    t_roots = roots_of(out, "T")
    for lab, frac in sorted(tr["feasibility"].items(), key=lambda kv: -kv[1]):
        n = sum(out[c][f"{lab}@drift"]["feasible"] is True for c in t_roots)
        assert abs(n / len(t_roots) - frac) < 1e-12
        native = ""
        if lab in primary["per_family"]["R"]["feasibility"]:
            native = f"; family-R native {primary['per_family']['R']['feasibility'][lab]:.2f}"
        print(f"{lab}: {n}/{len(t_roots)} ({frac:.2f}){native}")
    for name, c in tr["contrasts"].items():
        print(f"drift contrast {name}: {c['wins']}-{c['losses']}-{c['ties']} p={fmt_p(c['p_two_sided'])}")


# ------------------------------------------------------- 5. GP vs hand-written references
def references(primary, out):
    print("\n## 5. Champions vs hand-written references and initialization")
    for fam in (*FAMILIES, None):
        r = roots_of(out, fam)
        print(f"\n### {fam or 'pooled'} (n={len(r)})")
        comps = [("gp_champion", "oracle_reference"), ("gp_champion", "initialization_best"),
                 ("llm_champion", "oracle_reference"), ("llm_champion", "initialization_best"),
                 ("initialization_best", "oracle_reference")]
        if fam:
            comps += [("gp_champion", CALIBRATION_REFERENCE[fam]),
                      ("llm_champion", CALIBRATION_REFERENCE[fam])]
        for a, b in comps:
            w, l, t = wlt(out, a, b, r)
            s, n = saving(out, a, b, r)
            ff = (sum(out[c][a]["feasible"] is True for c in r), sum(out[c][b]["feasible"] is True for c in r))
            print(f"{a} vs {b}: {w}-{l}-{t} p={fmt_p(sign(w, l))}; feasible {ff[0]}/{ff[1]}; "
                  f"mean saving jointly feasible {'—' if s is None else f'{100 * s:.1f}%'} (n={n})")
        # Which reference was oracle-best, and how GP's wins split by kind
        oracle_names = Counter()
        for c in r:
            sha = out[c]["oracle_reference"]["source_sha256"]
            names = [k for k, v in out[c].items() if k.startswith("reference:")
                     and v["source_sha256"] == sha and "@" not in k]
            oracle_names[names[0] if len(names) == 1 else "/".join(sorted(names))] += 1
        print("oracle-best reference:", dict(oracle_names.most_common()))
        kinds = Counter()
        for c in r:
            g, o = out[c]["gp_champion"], out[c]["oracle_reference"]
            d = stats.direction(g, o)
            kinds[(d, "feasibility" if g["feasible"] != o["feasible"] else
                   ("energy" if g["feasible"] else "both infeasible"))] += 1
        print("GP vs oracle by decision kind:", {f"{k[0]:+d} {k[1]}": v for k, v in sorted(kinds.items())})


def main():
    actual = hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()
    assert actual.startswith("b42bd4fd"), "statistics module differs from the frozen source"
    primary, out = outcomes()
    assert len(out) == 90 and len(RERUN) == 12
    print("# Study E exploratory analyses (post hoc; not registered)")
    heterogeneity(primary, out)
    cross_study(out)
    iteration(primary, out)
    overfitting(primary, out)
    references(primary, out)


if __name__ == "__main__":
    main()
