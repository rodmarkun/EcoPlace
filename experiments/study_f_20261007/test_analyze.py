"""Validation of Study F's registered analysis against Study E's sealed results."""

from __future__ import annotations

import json
import shutil

import pytest

from experiments.study_f_20261007 import analyze as F

S1_KEY = "S1 sol_validated vs deepseek_validated"
P_KEY = "P sol_validated vs gp_validated"


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    work = tmp_path_factory.mktemp("dry")
    result = F.dry_run_from_e(work)  # asserts the reproduction of E internally
    return work, result


def wlt(c):
    return c["wins"], c["losses"], c["ties"]


def test_dry_run_reproduces_study_e(dry):
    work, result = dry
    conf = result["confirmatory"]
    assert wlt(conf[S1_KEY]) == (0, 0, 90) and conf[S1_KEY]["missing"] == 0
    p = conf[P_KEY]
    assert wlt(p) == (27, 39, 24) and p["missing"] == 0
    assert p["p_two_sided"] == pytest.approx(0.175, abs=5e-4)
    assert p["confirmatory"] is False and conf["gate_open_for_P"] is False
    assert [wlt(result["per_family"][f]["contrasts"]["sol_validated vs gp_validated"])
            for f in F.FAMILIES] == [(9, 3, 18), (9, 20, 1), (9, 16, 5)]
    assert wlt(result["pooled_descriptive"]["contrasts"]["sol_champion vs e_gp_champion"]) \
        == (27, 39, 24)
    text = (work / "analysis" / "RESULTS.md").read_text()
    assert "**0 / 0 / 90**" in text and "**27 / 39 / 24**" in text
    assert "descriptive only" in text
    print("\n" + F._table(result))


def test_save_once(dry):
    work, result = dry
    with pytest.raises(FileExistsError):
        F.write(result, work / "analysis", "again")
    with pytest.raises(SystemExit):
        F.main(["--runs", str(work / "runs"), "--out", str(work / "analysis")])


def _layout(tmp_path):
    runs = tmp_path / "runs"
    F.build_dry_run_layout(runs)
    return runs


def _edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_missing_roots_are_reported_and_block(tmp_path):
    runs = _layout(tmp_path)
    (runs / "T00" / "ROOT_STOPPED.json").write_text("{}")
    (runs / "W01" / "final" / "COMPLETE.json").unlink()
    (runs / "R02" / "SELECTION.json").unlink()
    _edit(runs / "R03" / "SELECTION.json",
          lambda s: s["validated"].__setitem__("deepseek", None))
    shutil.rmtree(runs / "R04")
    result = F.analyse(runs)
    reasons = result["execution"]["missing_by_reason"]
    assert reasons["sol_validated"] == {"root_stopped": ["T00"], "no_final": ["W01"],
                                        "no_selection": ["R02", "R04"]}
    assert reasons["deepseek_validated"]["no_record"] == ["R03"]
    s1 = result["confirmatory"][S1_KEY]
    assert s1["missing"] == 5 and sum(wlt(s1)) == 85
    assert s1["p_two_sided"] is None and s1["blocked"] and not s1["directional_claim"]
    assert result["confirmatory"]["gate_open_for_P"] is False
    assert result["confirmatory"][P_KEY]["missing"] == 4
    md = F.render(result, "test")
    assert "**blocked** (5 missing observations)" in md


def test_altered_copy_is_rejected(tmp_path):
    runs = _layout(tmp_path)
    labels = json.loads((runs / "T00" / "final" / "LABELS.json").read_text())["labels"]
    record = runs / "T00" / "final" / f"{next(iter(labels))}.json"
    record.write_bytes(record.read_bytes().replace(b"}", b" }", 1))
    with pytest.raises(F.LayoutError, match="byte-identical"):
        F.analyse(runs)


def test_copy_from_other_root_is_rejected(tmp_path):
    runs = _layout(tmp_path)
    _edit(runs / "T00" / "final" / "REUSED.json",
          lambda r: r["copied"].update({k: v.replace("/T00/", "/T02/")
                                        for k, v in r["copied"].items()}))
    with pytest.raises(F.LayoutError, match="this root"):
        F.analyse(runs)


def test_sealed_sha_mismatch_is_rejected(tmp_path):
    runs = _layout(tmp_path)
    for cid in sorted(d.name for d in runs.iterdir()):
        sel = json.loads((runs / cid / "SELECTION.json").read_text())
        gp_sha = sel["validated"]["gp"]["source_sha256"]
        if gp_sha != sel["validated"]["sol"]["source_sha256"]:
            break
    labels_path = runs / cid / "final" / "LABELS.json"

    def swap(d):  # point the sol_validated label at the GP program
        for names in d["labels"].values():
            if "sol_validated" in names:
                names.remove("sol_validated")
        d["labels"][gp_sha].append("sol_validated")
    _edit(labels_path, swap)
    with pytest.raises(F.LayoutError, match="sol_validated"):
        F.analyse(runs)


def test_tampered_source_is_rejected(tmp_path):
    runs = _layout(tmp_path)
    _edit(runs / "T00" / "SELECTION.json",
          lambda s: s["validated"]["sol"].__setitem__("source", "# altered\n"))
    with pytest.raises(F.LayoutError, match="hash"):
        F.analyse(runs)
